"""
VAD/pipeline/streaming.py

Streaming orchestration session (VADStreamingSession) for Multi-Band Adaptive VAD.
Wraps the existing frozen Layers 1-4:
  FeatureExtractor.process_chunk() + flush()
  -> NoiseEstimator.update()
  -> SigmoidScorer.score()
  -> VADStateMachine.process_chunk() + flush()

Preserves the exact algorithm, decision thresholds, and timestamp semantics.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.feature_extraction import DEFAULT_BANDS_HZ, FeatureExtractor, VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseConfig, NoiseEstimator, NoiseOutput  # noqa: E402
from VAD.pipeline.contracts import (  # noqa: E402
    PipelineMode,
    VADDiagnostics,
    VADResult,
    VADSegment,
    validate_sample_rate,
)
from VAD.sigmoid import SigmoidConfig, SigmoidOutput, SigmoidScorer  # noqa: E402
from VAD.state_machine import StateMachineConfig, StateMachineOutput, VADStateMachine  # noqa: E402


class VADStreamingSession:
    """
    Streaming session manager orchestrating Layers 1-4 for online VAD processing.

    Lifecycle:
        1. session = VADStreamingSession(sample_rate=16000)
        2. for chunk in audio_chunks:
               res = session.process_chunk(chunk)
               # res.segments contains newly finalized segments in this chunk
        3. res_final = session.flush()
               # res_final.segments contains finalized segments from remaining buffer
        4. session.reset()
               # Clears all state across all layers for a new independent stream.
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
        self._debug = bool(debug)
        self._return_diagnostics = bool(return_diagnostics)

        # Initialize existing Layers 1-4
        self._fe = FeatureExtractor(
            sample_rate=self._sample_rate,
            bands_hz=bands_hz,
            debug=self._debug,
        )
        self._est = NoiseEstimator(
            num_bands=self._fe.num_bands,
            config=noise_config or NoiseConfig(),
            debug=self._debug,
        )
        self._scorer = SigmoidScorer(
            num_bands=self._fe.num_bands,
            config=sigmoid_config or SigmoidConfig(),
            debug=self._debug,
        )
        self._sm = VADStateMachine(
            hop_seconds=self._fe.hop_seconds,
            config=state_machine_config or StateMachineConfig(),
            debug=self._debug,
        )

        self.reset()

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def hop_seconds(self) -> float:
        return self._fe.hop_seconds

    @property
    def num_bands(self) -> int:
        return self._fe.num_bands

    @property
    def total_frames(self) -> int:
        """Total STFT frames produced and processed in this stream."""
        return self._total_frames

    @property
    def total_samples(self) -> int:
        """Total audio samples accepted in this stream."""
        return self._total_samples

    @property
    def total_duration_s(self) -> float:
        """Total audio duration accepted in this stream in seconds."""
        return self._total_samples / self._sample_rate

    @property
    def accumulated_segments(self) -> List[VADSegment]:
        """All finalized speech segments accumulated since session start or last reset."""
        return list(self._accumulated_segments)

    @property
    def is_flushed(self) -> bool:
        """Whether the session has been flushed."""
        return self._is_flushed

    @property
    def mode(self) -> PipelineMode:
        return PipelineMode.STREAMING

    # ------------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset all layers and session state to start a clean, independent stream.
        Ensures no state leaks across streams.
        """
        self._fe.reset()
        self._est.reset()
        self._scorer.reset()
        self._sm.reset()
        self._is_flushed = False
        self._total_samples = 0
        self._total_frames = 0
        self._accumulated_segments = []

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if chunk is None:
            raise TypeError("Audio chunk cannot be None.")
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk)

        if np.iscomplexobj(chunk):
            raise ValueError("Audio chunk must be real-valued, got complex dtype.")
        if chunk.ndim != 1:
            raise ValueError(
                f"Audio chunk must be a 1-D mono array, got ndim={chunk.ndim} with shape {chunk.shape}."
            )
        if not (np.issubdtype(chunk.dtype, np.floating) or np.issubdtype(chunk.dtype, np.integer)):
            raise TypeError(f"Audio chunk must have numeric dtype, got {chunk.dtype}.")

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32, copy=False)

        if self._debug and len(chunk) > 0:
            if not np.isfinite(chunk).all():
                raise ValueError("Audio chunk contains NaN or Inf.")

        return chunk

    def process_chunk(
        self,
        chunk: np.ndarray,
        return_diagnostics: Optional[bool] = None,
    ) -> VADResult:
        """
        Process the next chunk of mono audio.

        Args:
            chunk: 1-D numpy array of audio samples (float32 mono recommended).
            return_diagnostics: Optional override to attach VADDiagnostics to the result.

        Returns:
            VADResult containing newly finalized segments emitted in this chunk.
        """
        if self._is_flushed:
            raise RuntimeError(
                "Cannot call process_chunk() after session has been flushed. "
                "Call reset() to begin a new independent stream."
            )

        chunk = self._validate_chunk(chunk)
        chunk_len = len(chunk)
        self._total_samples += chunk_len
        chunk_dur = float(chunk_len / self._sample_rate)

        want_diag = self._return_diagnostics if return_diagnostics is None else bool(return_diagnostics)

        feats: VADFeatures = self._fe.process_chunk(chunk)
        new_segments: List[VADSegment] = []
        diag: Optional[VADDiagnostics] = None

        if feats.num_frames > 0:
            self._total_frames += feats.num_frames
            n_out: NoiseOutput = self._est.update(feats)
            s_out: SigmoidOutput = self._scorer.score(feats, n_out)
            sm_out: StateMachineOutput = self._sm.process_chunk(s_out)

            for s, e in sm_out.segments:
                seg = VADSegment(
                    start_time_s=round(float(s), 6),
                    end_time_s=round(float(e), 6),
                )
                new_segments.append(seg)
                self._accumulated_segments.append(seg)

            if want_diag:
                diag = VADDiagnostics(
                    frame_scores=s_out.score,
                    speech_mask=sm_out.speech_mask,
                    states=sm_out.states,
                    warmup_mask=s_out.warmup,
                    noise_floor=n_out.noise_floor,
                    snr_db=n_out.snr_db,
                )
        else:
            if want_diag:
                diag = VADDiagnostics(
                    frame_scores=np.empty(0, dtype=np.float32),
                    speech_mask=np.empty(0, dtype=bool),
                    states=np.empty(0, dtype=np.int8),
                    warmup_mask=np.empty(0, dtype=bool),
                    noise_floor=np.empty((self.num_bands, 0), dtype=np.float32),
                    snr_db=np.empty((self.num_bands, 0), dtype=np.float32),
                )

        return VADResult(
            segments=new_segments,
            num_frames=feats.num_frames,
            duration_s=chunk_dur,
            diagnostics=diag,
        )

    def flush(
        self,
        return_diagnostics: Optional[bool] = None,
    ) -> VADResult:
        """
        Finalize remaining buffered audio in STFT and pending segments in state machine.

        Invokes flush on FeatureExtractor first, pushes any zero-padded tail frame through
        Layers 2-4, and finally flushes VADStateMachine to close pending segments.

        Duration Contract:
            Because flush() processes existing buffered samples and accepts no new input audio,
            the returned VADResult always has duration_s=0.0. This guarantees that summing
            duration_s across process_chunk() calls and flush() equals the exact total audio duration.

        Subsequent calls to flush() return an empty VADResult safely (idempotent).
        """
        want_diag = self._return_diagnostics if return_diagnostics is None else bool(return_diagnostics)

        if self._is_flushed:
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

        self._is_flushed = True
        new_segments: List[VADSegment] = []
        tail_frames = 0
        diag: Optional[VADDiagnostics] = None

        # 1. Flush FeatureExtractor STFT buffer
        feats_tail: VADFeatures = self._fe.flush()
        if feats_tail.num_frames > 0:
            tail_frames = feats_tail.num_frames
            self._total_frames += tail_frames
            n_tail: NoiseOutput = self._est.update(feats_tail)
            s_tail: SigmoidOutput = self._scorer.score(feats_tail, n_tail)
            sm_tail: StateMachineOutput = self._sm.process_chunk(s_tail)

            for s, e in sm_tail.segments:
                seg = VADSegment(
                    start_time_s=round(float(s), 6),
                    end_time_s=round(float(e), 6),
                )
                new_segments.append(seg)
                self._accumulated_segments.append(seg)

            if want_diag:
                diag = VADDiagnostics(
                    frame_scores=s_tail.score,
                    speech_mask=sm_tail.speech_mask,
                    states=sm_tail.states,
                    warmup_mask=s_tail.warmup,
                    noise_floor=n_tail.noise_floor,
                    snr_db=n_tail.snr_db,
                )

        # 2. Flush VADStateMachine to finalize pending segment / open active segment
        sm_flush_segs = self._sm.flush()
        for s, e in sm_flush_segs:
            seg = VADSegment(
                start_time_s=round(float(s), 6),
                end_time_s=round(float(e), 6),
            )
            new_segments.append(seg)
            self._accumulated_segments.append(seg)

        if want_diag and diag is None:
            diag = VADDiagnostics(
                frame_scores=np.empty(0, dtype=np.float32),
                speech_mask=np.empty(0, dtype=bool),
                states=np.empty(0, dtype=np.int8),
                warmup_mask=np.empty(0, dtype=bool),
                noise_floor=np.empty((self.num_bands, 0), dtype=np.float32),
                snr_db=np.empty((self.num_bands, 0), dtype=np.float32),
            )

        return VADResult(
            segments=new_segments,
            num_frames=tail_frames,
            duration_s=0.0,
            diagnostics=diag,
        )

    def __repr__(self) -> str:
        return (
            f"VADStreamingSession(sr={self._sample_rate}, hop={self.hop_seconds*1000:.1f}ms, "
            f"frames={self._total_frames}, dur={self.total_duration_s:.3f}s, "
            f"segments={len(self._accumulated_segments)}, flushed={self._is_flushed})"
        )
