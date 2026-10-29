"""
VAD/pipeline/ingest.py

Integration boundary between the VAD engine and an external streaming infrastructure
(ring buffer, windowing, transcript lifecycle). See
`VAD/report/tầng 4/PHASE_6A_AUDIO_ENGINE_INTEGRATION_CONTRACT.md`.

`VADIngestAdapter` wraps the existing `VADStreamingSession` (Layers 1-4, unchanged) and adds
only what the session cannot know:
  * an absolute, monotonic sample timeline supplied by the caller (`first_sample_index`);
  * explicit rejection of gaps (dropped audio) and overlaps (replayed audio) instead of
    silently shifting the timeline or double-counting samples;
  * PCM normalisation (int16 -> float32 in [-1, 1]) and input validation that does not depend
    on `debug=True`.

It deliberately imports nothing from the infrastructure (RingBuffer, SlidingWindowSlicer, ...):
the caller passes plain numpy samples and an integer index.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.pipeline.contracts import validate_sample_rate  # noqa: E402
from VAD.pipeline.streaming import VADStreamingSession  # noqa: E402


class SampleDiscontinuityError(ValueError):
    """`first_sample_index` does not equal the next expected sample index."""

    def __init__(self, expected: int, received: int) -> None:
        self.expected = int(expected)
        self.received = int(received)
        self.kind = "gap" if received > expected else "overlap"
        lost = received - expected
        detail = (
            f"{lost} sample(s) missing (dropped audio); call resync({received}) to acknowledge the loss"
            if self.kind == "gap"
            else f"{-lost} sample(s) already processed (replayed audio); each sample must be fed exactly once"
        )
        super().__init__(
            f"Sample discontinuity ({self.kind}): expected first_sample_index={self.expected}, "
            f"got {self.received}: {detail}."
        )


@dataclass(frozen=True)
class SpeechSpan:
    """
    Finalized speech interval on the caller's absolute sample timeline.

    Half-open [start_sample, end_sample). Seconds = sample / sample_rate. `epoch` increments on
    every `resync()`; spans of different epochs were produced by independent VAD states.
    """

    start_sample: int
    end_sample: int
    epoch: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.start_sample < self.end_sample):
            raise ValueError(
                f"Invalid SpeechSpan: need 0 <= start_sample < end_sample, "
                f"got [{self.start_sample}, {self.end_sample})."
            )

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample


class VADIngestAdapter:
    """
    Exactly-once, chronological feeding of one mono PCM stream into `VADStreamingSession`.

    Lifecycle:
        a = VADIngestAdapter(sample_rate=16000)
        spans = a.feed(chunk, first_sample_index=a.next_sample_index)   # repeat
        spans += a.flush()                                              # end of stream
        a.reset()                                                       # new independent stream
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        session: Optional[VADStreamingSession] = None,
        **session_kwargs,
    ) -> None:
        self._sample_rate = validate_sample_rate(sample_rate)
        if session is not None:
            if session_kwargs:
                raise TypeError("Pass either a pre-built session or session kwargs, not both.")
            if session.sample_rate != self._sample_rate:
                raise ValueError(
                    f"session.sample_rate={session.sample_rate} != adapter sample_rate={self._sample_rate}."
                )
            self._session = session
        else:
            self._session = VADStreamingSession(sample_rate=self._sample_rate, **session_kwargs)
        self.reset()

    # ------------------------------------------------------------------ properties
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def next_sample_index(self) -> int:
        """Absolute index the next `feed()` must start at."""
        return self._next_index

    @property
    def samples_processed(self) -> int:
        """Samples accepted by the underlying session in the current epoch (each exactly once)."""
        return self._session.total_samples

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def is_flushed(self) -> bool:
        return self._session.is_flushed

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Start a new independent stream at absolute sample index 0."""
        self._session.reset()
        self._base_index = 0
        self._next_index = 0
        self._epoch = 0

    def feed(self, samples: np.ndarray, first_sample_index: int) -> List[SpeechSpan]:
        """
        Feed the NEXT samples of the stream. Returns spans finalized by this call.

        Raises:
            SampleDiscontinuityError: first_sample_index != next_sample_index (gap or overlap).
            RuntimeError: stream already flushed.
            TypeError / ValueError: invalid index or audio (checked before any state changes).
        """
        if self._session.is_flushed:
            raise RuntimeError("Adapter is flushed; call reset() to start a new stream.")
        idx = self._validate_index(first_sample_index)
        audio = self._to_float32_mono(samples)
        if idx != self._next_index:
            raise SampleDiscontinuityError(self._next_index, idx)
        if len(audio) == 0:
            return []
        result = self._session.process_chunk(audio)
        self._next_index += len(audio)
        return self._to_spans(result.segments)

    def flush(self) -> List[SpeechSpan]:
        """
        End of stream: close pending speech. Accepts no audio and does NOT advance
        `next_sample_index` (flush never invents captured duration). Idempotent.
        """
        return self._to_spans(self._session.flush().segments)

    def resync(self, first_sample_index: int) -> List[SpeechSpan]:
        """
        Acknowledge a discontinuity (upstream dropped audio) and continue at `first_sample_index`.

        Pending speech from the old epoch is finalized and returned; the VAD state (including noise
        warm-up) restarts at the new index and `epoch` increments. Never moves the timeline backwards.
        """
        idx = self._validate_index(first_sample_index)
        if idx < self._next_index:
            raise ValueError(
                f"resync({idx}) would move the timeline backwards (next_sample_index={self._next_index})."
            )
        flushed = self.flush()
        self._session.reset()
        self._base_index = idx
        self._next_index = idx
        self._epoch += 1
        return flushed

    # ------------------------------------------------------------------ internals
    def _to_spans(self, segments) -> List[SpeechSpan]:
        return [
            SpeechSpan(
                start_sample=self._base_index + int(round(s.start_time_s * self._sample_rate)),
                end_sample=self._base_index + int(round(s.end_time_s * self._sample_rate)),
                epoch=self._epoch,
            )
            for s in segments
        ]

    @staticmethod
    def _validate_index(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"first_sample_index must be an integer, got {type(value).__name__}.")
        if value < 0:
            raise ValueError(f"first_sample_index must be >= 0, got {int(value)}.")
        return int(value)

    @staticmethod
    def _to_float32_mono(samples: object) -> np.ndarray:
        if samples is None:
            raise TypeError("samples cannot be None.")
        arr = np.asarray(samples)
        if np.iscomplexobj(arr):
            raise ValueError("samples must be real-valued.")
        if arr.ndim != 1:
            raise ValueError(f"samples must be a 1-D mono array, got shape {arr.shape}.")
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        else:
            raise TypeError(
                f"Unsupported dtype {arr.dtype}: pass float samples in [-1, 1] or int16 PCM."
            )
        if len(arr) and not np.isfinite(arr).all():
            raise ValueError("samples contain NaN or Inf.")
        return arr
