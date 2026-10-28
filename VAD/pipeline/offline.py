"""
VAD/pipeline/offline.py

Causal Offline Baseline Runner (VADOfflineRunner) for Multi-Band Adaptive VAD.
Processes a complete audio waveform in whole-file mode using the Phase 1 VADStreamingSession,
guaranteeing 100% mathematical parity with the streaming pipeline without duplicating
the Layer 1-4 algorithms.

Note on Architecture:
This is a causal offline baseline, not a non-causal offline-aware algorithm.
It preserves the current causal warm-up and thresholding behavior while providing
a convenient, leak-proof whole-waveform API.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.feature_extraction import DEFAULT_BANDS_HZ  # noqa: E402
from VAD.noise_estimator import NoiseConfig  # noqa: E402
from VAD.pipeline.contracts import (  # noqa: E402
    PipelineMode,
    VADDiagnostics,
    VADResult,
    VADSegment,
    validate_sample_rate,
)
from VAD.pipeline.streaming import VADStreamingSession  # noqa: E402
from VAD.sigmoid import SigmoidConfig  # noqa: E402
from VAD.state_machine import StateMachineConfig  # noqa: E402


class VADOfflineRunner:
    """
    Offline execution runner for Multi-Band Adaptive VAD.

    Accepts a complete mono audio waveform and produces a finalized VADResult
    by orchestrating a fresh, hermetic VADStreamingSession per task.

    Usage:
        runner = VADOfflineRunner(sample_rate=16000)
        result = runner.process_audio(waveform)
        for seg in result.segments:
            print(f"Speech: [{seg.start_time_s:.3f}, {seg.end_time_s:.3f})s")
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> None:
        self._sample_rate = validate_sample_rate(sample_rate)
        self._noise_config = noise_config
        self._sigmoid_config = sigmoid_config
        self._state_machine_config = state_machine_config
        self._debug = bool(debug)
        self._return_diagnostics = bool(return_diagnostics)
        self._bands_hz = bands_hz

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def hop_seconds(self) -> float:
        # Default VAD preset hop is 5 ms across 8k and 16k
        return 0.005

    @property
    def num_bands(self) -> int:
        return len(self._bands_hz)

    @property
    def mode(self) -> PipelineMode:
        return PipelineMode.OFFLINE

    def reset(self) -> None:
        """Reset runner state (no-op as tasks are hermetic, preserved for interface consistency)."""
        return None

    # ------------------------------------------------------------------------
    # Input Validation
    # ------------------------------------------------------------------------

    def _validate_waveform(self, audio: np.ndarray) -> np.ndarray:
        if audio is None:
            raise TypeError("Audio waveform cannot be None.")
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)

        if np.iscomplexobj(audio):
            raise ValueError("Audio waveform must be real-valued, got complex dtype.")
        if audio.ndim != 1:
            raise ValueError(
                f"Audio waveform must be a 1-D mono array, got ndim={audio.ndim} with shape {audio.shape}."
            )
        if not (np.issubdtype(audio.dtype, np.floating) or np.issubdtype(audio.dtype, np.integer)):
            raise TypeError(f"Audio waveform must have numeric dtype, got {audio.dtype}.")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        if self._debug and len(audio) > 0:
            if not np.isfinite(audio).all():
                raise ValueError("Audio waveform contains NaN or Inf.")

        return audio

    # ------------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------------

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        return_diagnostics: Optional[bool] = None,
    ) -> VADResult:
        """
        Process a complete mono audio waveform offline and return finalized speech segments.

        Args:
            audio: 1-D numpy array containing complete audio waveform.
            sample_rate: Optional sample rate override (must be 8000 or 16000 Hz integer).
                         Defaults to runner's sample_rate.
            return_diagnostics: Optional flag to attach VADDiagnostics to result.
                                Defaults to runner's return_diagnostics setting.

        Returns:
            Finalized VADResult containing speech segments, frame count,
            duration_s (exact audio waveform duration in seconds: len(audio) / sample_rate, or 0.0 for empty),
            and optional diagnostics.
        """
        sr = self._sample_rate if sample_rate is None else validate_sample_rate(sample_rate)

        audio = self._validate_waveform(audio)
        want_diag = self._return_diagnostics if return_diagnostics is None else bool(return_diagnostics)

        # Handle empty audio
        if len(audio) == 0:
            empty_diag = None
            if want_diag:
                empty_diag = VADDiagnostics(
                    frame_scores=np.empty(0, dtype=np.float32),
                    speech_mask=np.empty(0, dtype=bool),
                    states=np.empty(0, dtype=np.int8),
                    warmup_mask=np.empty(0, dtype=bool),
                    noise_floor=np.empty((self.num_bands, 0), dtype=np.float32),
                    snr_db=np.empty((self.num_bands, 0), dtype=np.float32),
                )
            return VADResult(
                segments=[],
                num_frames=0,
                duration_s=0.0,
                diagnostics=empty_diag,
            )

        # Create a fresh, hermetic streaming session for this offline task
        session = VADStreamingSession(
            sample_rate=sr,
            noise_config=self._noise_config,
            sigmoid_config=self._sigmoid_config,
            state_machine_config=self._state_machine_config,
            debug=self._debug,
            return_diagnostics=want_diag,
            bands_hz=self._bands_hz,
        )

        # Execute full-waveform processing followed immediately by flush
        res_chunk = session.process_chunk(audio, return_diagnostics=want_diag)
        res_flush = session.flush(return_diagnostics=want_diag)

        # Aggregate finalized segments from full chunk + flush
        all_segments = res_chunk.segments + res_flush.segments
        total_frames = session.total_frames
        total_duration = float(len(audio) / sr)

        # Merge diagnostics if requested
        merged_diag: Optional[VADDiagnostics] = None
        if want_diag:
            d1 = res_chunk.diagnostics
            d2 = res_flush.diagnostics
            if d1 is not None and d2 is not None and d2.num_frames > 0:
                merged_scores = np.concatenate([d1.frame_scores, d2.frame_scores])
                merged_mask = np.concatenate([d1.speech_mask, d2.speech_mask])
                merged_states = np.concatenate([d1.states, d2.states])
                merged_warmup = np.concatenate([d1.warmup_mask, d2.warmup_mask])
                merged_nf = (
                    np.concatenate([d1.noise_floor, d2.noise_floor], axis=1)
                    if d1.noise_floor is not None and d2.noise_floor is not None
                    else None
                )
                merged_snr = (
                    np.concatenate([d1.snr_db, d2.snr_db], axis=1)
                    if d1.snr_db is not None and d2.snr_db is not None
                    else None
                )
                merged_diag = VADDiagnostics(
                    frame_scores=merged_scores,
                    speech_mask=merged_mask,
                    states=merged_states,
                    warmup_mask=merged_warmup,
                    noise_floor=merged_nf,
                    snr_db=merged_snr,
                )
            else:
                merged_diag = d1

        return VADResult(
            segments=all_segments,
            num_frames=total_frames,
            duration_s=total_duration,
            diagnostics=merged_diag,
        )

    def process_file(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        return_diagnostics: Optional[bool] = None,
    ) -> VADResult:
        """Alias for process_audio() consistent with Phase 0 audit naming."""
        return self.process_audio(
            audio=audio,
            sample_rate=sample_rate,
            return_diagnostics=return_diagnostics,
        )

    def __repr__(self) -> str:
        return (
            f"VADOfflineRunner(sr={self._sample_rate}, hop={self.hop_seconds*1000:.1f}ms, "
            f"bands={self.num_bands}, mode={self.mode.value})"
        )
