"""
VAD/pipeline/contracts.py

Shared Data Contracts for Multi-Band Adaptive VAD Pipelines.
Defines standardized data structures (VADSegment, VADDiagnostics, VADResult, PipelineMode)
serving as the shared contract across streaming and future offline execution modes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_SAMPLE_RATES: Tuple[int, ...] = (8000, 16000)


def validate_sample_rate(sample_rate: object) -> int:
    """
    Validate that sample_rate is an exact integer and belongs to SUPPORTED_SAMPLE_RATES.

    Raises:
        TypeError: If sample_rate is a bool, float, str, None, or any non-integer type.
        ValueError: If sample_rate is an integer not in SUPPORTED_SAMPLE_RATES (e.g. 44100, 0, -1).
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise TypeError(
            f"sample_rate must be an integer, got {type(sample_rate).__name__}"
        )
    sr = int(sample_rate)
    if sr not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported sample_rate={sr}. "
            f"Only sample rates in {list(SUPPORTED_SAMPLE_RATES)} Hz are supported."
        )
    return sr


class PipelineMode(str, Enum):
    """
    Execution pipeline mode identifier.
    - STREAMING: causal, chunk-by-chunk low-latency streaming pipeline (VADStreamingSession).
    - OFFLINE  : causal whole-waveform offline execution runner (VADOfflineRunner).
    """
    STREAMING = "streaming"
    OFFLINE = "offline"


@dataclass(frozen=True)
class VADSegment:
    """
    Represent a finalized speech segment detected by the VAD.

    Semantics:
        - Half-open time interval [start_time_s, end_time_s) in seconds.
        - Non-negative finite timestamps: 0.0 <= start_time_s < end_time_s.
        - Strictly positive duration: duration_s = end_time_s - start_time_s > 0.
    """
    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        if isinstance(self.start_time_s, bool) or not isinstance(
            self.start_time_s, (int, float, np.floating, np.integer)
        ):
            raise TypeError(
                f"start_time_s must be a real number, got {type(self.start_time_s).__name__}"
            )
        if isinstance(self.end_time_s, bool) or not isinstance(
            self.end_time_s, (int, float, np.floating, np.integer)
        ):
            raise TypeError(
                f"end_time_s must be a real number, got {type(self.end_time_s).__name__}"
            )

        s = float(self.start_time_s)
        e = float(self.end_time_s)

        if not (np.isfinite(s) and np.isfinite(e)):
            raise ValueError(f"Timestamps must be finite, got start={s}, end={e}")
        if s < 0.0 or e < 0.0:
            raise ValueError(
                f"Timestamps must be non-negative, got start={s}, end={e}"
            )
        if e <= s:
            raise ValueError(
                f"end_time_s ({e}) must be strictly greater than start_time_s ({s})"
            )

        object.__setattr__(self, "start_time_s", s)
        object.__setattr__(self, "end_time_s", e)

    @property
    def start_time(self) -> float:
        """Alias for start_time_s."""
        return self.start_time_s

    @property
    def end_time(self) -> float:
        """Alias for end_time_s."""
        return self.end_time_s

    @property
    def duration_s(self) -> float:
        """Segment duration in seconds."""
        return self.end_time_s - self.start_time_s

    @property
    def duration(self) -> float:
        """Alias for duration_s."""
        return self.duration_s

    def to_tuple(self) -> Tuple[float, float]:
        """Return interval as (start_time_s, end_time_s)."""
        return (self.start_time_s, self.end_time_s)

    def __iter__(self) -> Iterator[float]:
        """Support tuple unpacking: start, end = segment."""
        yield self.start_time_s
        yield self.end_time_s

    def __getitem__(self, index: int) -> float:
        return (self.start_time_s, self.end_time_s)[index]

    def __repr__(self) -> str:
        return f"VADSegment([{self.start_time_s:.3f}, {self.end_time_s:.3f})s, dur={self.duration_s:.3f}s)"


@dataclass(frozen=True)
class VADDiagnostics:
    """
    Optional diagnostic metadata and intermediate frame-level outputs.
    Avoids retaining large intermediate arrays unless explicitly requested.
    """
    frame_scores: np.ndarray          # (T,) float32: uncalibrated sigmoid speech scores p(t) ∈ [0, 1]
    speech_mask: np.ndarray           # (T,) bool: instantaneous state-machine speech mask (SPEECH or HANGOVER)
    states: np.ndarray                # (T,) int8: VADState codes (0=SILENCE, 1=ONSET, 2=SPEECH, 3=HANGOVER)
    warmup_mask: np.ndarray           # (T,) bool: warmup flag per frame
    noise_floor: Optional[np.ndarray] = None  # (n_bands, T) float32: linear noise floor per band
    snr_db: Optional[np.ndarray] = None       # (n_bands, T) float32: estimated SNR in dB per band

    def __post_init__(self) -> None:
        scores = np.asarray(self.frame_scores)
        if scores.ndim != 1:
            raise ValueError(f"frame_scores must be a 1-D array, got shape {scores.shape}")
        t = scores.shape[0]

        sp_mask = np.asarray(self.speech_mask)
        if sp_mask.ndim != 1 or sp_mask.shape[0] != t:
            raise ValueError(f"speech_mask must have shape ({t},), got {sp_mask.shape}")

        st = np.asarray(self.states)
        if st.ndim != 1 or st.shape[0] != t:
            raise ValueError(f"states must have shape ({t},), got {st.shape}")

        wu = np.asarray(self.warmup_mask)
        if wu.ndim != 1 or wu.shape[0] != t:
            raise ValueError(f"warmup_mask must have shape ({t},), got {wu.shape}")

        if self.noise_floor is not None:
            nf = np.asarray(self.noise_floor)
            if nf.ndim != 2 or nf.shape[1] != t:
                raise ValueError(f"noise_floor must have shape (n_bands, {t}), got {nf.shape}")

        if self.snr_db is not None:
            snr = np.asarray(self.snr_db)
            if snr.ndim != 2 or snr.shape[1] != t:
                raise ValueError(f"snr_db must have shape (n_bands, {t}), got {snr.shape}")

    @property
    def num_frames(self) -> int:
        """Number of frames in this diagnostics slice."""
        return int(self.frame_scores.shape[0])


@dataclass(frozen=True)
class VADResult:
    """
    Standardized result emitted by pipeline operations (process_chunk, flush, or process_audio).

    Fields:
        - segments: List of finalized VADSegment objects newly closed/emitted by this call.
          Aggregating segments across successive process_chunk() calls and flush() yields
          the complete, non-duplicate list of speech segments.
        - num_frames: Number of STFT analysis frames produced in this call.
        - duration_s: Audio waveform duration accepted in this call in seconds:
            * For process_chunk(chunk): len(chunk) / sample_rate.
            * For flush(): 0.0 (finalization call does not accept new audio waveform).
            * For process_audio(audio): len(audio) / sample_rate.
            * For empty audio: 0.0.
          Summing duration_s across all calls of a stream exactly equals total audio duration.
        - diagnostics: Optional VADDiagnostics if requested by caller.
    """
    segments: List[VADSegment]
    num_frames: int
    duration_s: float
    diagnostics: Optional[VADDiagnostics] = None

    def __post_init__(self) -> None:
        if not isinstance(self.segments, list):
            object.__setattr__(self, "segments", list(self.segments))
        for seg in self.segments:
            if not isinstance(seg, VADSegment):
                raise TypeError(f"All segments must be VADSegment instances, got {type(seg)}")

        if not isinstance(self.num_frames, (int, np.integer)) or self.num_frames < 0:
            raise ValueError(f"num_frames must be an integer >= 0, got {self.num_frames}")

        dur = float(self.duration_s)
        if not np.isfinite(dur) or dur < 0.0:
            raise ValueError(f"duration_s must be a non-negative finite float, got {dur}")
        object.__setattr__(self, "duration_s", dur)

    @property
    def speech_duration_s(self) -> float:
        """Total duration of finalized speech segments in this result slice."""
        return sum(seg.duration_s for seg in self.segments)
