"""
tests/test_vad_streaming.py

Exact-parity and lifecycle tests for VADStreamingSession:
- Test A: 20 ms chunks across all 11 pilot WAV files (exact numerical parity with reference)
- Test B: 100 ms chunks across all 11 pilot WAV files (exact numerical parity with reference)
- Test C: One full-array chunk across all 11 pilot WAV files (exact numerical parity with reference)
- Test D: Session reset and state isolation between independent streams
- Test E: Flush idempotency, empty streams, invalid inputs, lifecycle error handling
- Test F: Arbitrary chunk boundaries across speech onset, offset, and silences
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VAD.pipeline.contracts import PipelineMode, VADDiagnostics, VADResult, VADSegment  # noqa: E402
from VAD.pipeline.streaming import VADStreamingSession  # noqa: E402

PILOT_DIR = os.path.join(ROOT, "VAD", "report", "tầng 4", "human_annotation_pilot")


def _load_pilot_module(name: str, filename: str) -> Any:
    path = os.path.join(PILOT_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pilot_tools = _load_pilot_module("pilot_tools", "pilot_tools.py")
preannotate_silero = _load_pilot_module("preannotate_silero", "preannotate_silero.py")
benchmark_runner = _load_pilot_module("benchmark_multiband_vs_silero", "benchmark_multiband_vs_silero.py")

PILOT_MANIFEST: Dict[str, Any] = pilot_tools.load_manifest()
PILOT_EXCERPT_IDS = sorted(PILOT_MANIFEST.keys())


def _load_excerpt_audio(excerpt_id: str) -> Tuple[np.ndarray, int, float]:
    spec = PILOT_MANIFEST[excerpt_id]
    wav_path = os.path.join(PILOT_DIR, spec.wav_rel_path.replace("/", os.sep))
    audio, sr, dur_s = preannotate_silero.read_pcm_wav_mono(wav_path)
    return audio, sr, dur_s


def _run_session_with_chunking(
    session: VADStreamingSession,
    audio: np.ndarray,
    chunk_duration_s: float | None,
) -> Tuple[List[VADSegment], VADResult]:
    sr = session.sample_rate
    if chunk_duration_s is None or chunk_duration_s <= 0.0:
        chunks = [audio]
    else:
        chunk_samples = max(1, int(round(chunk_duration_s * sr)))
        chunks = [audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

    emitted_segments: List[VADSegment] = []
    for ch in chunks:
        res = session.process_chunk(ch)
        emitted_segments.extend(res.segments)

    res_flush = session.flush()
    emitted_segments.extend(res_flush.segments)
    return emitted_segments, res_flush


# ============================================================================
# Test A — 20 ms chunks parity across all 11 pilot WAV files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_parity_20ms_chunks(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    # 1. Run reference implementation
    ref_segs, ref_frames = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=0.020
    )

    # 2. Run VADStreamingSession
    session = VADStreamingSession(sample_rate=sr)
    emitted_segs, _ = _run_session_with_chunking(session, audio, chunk_duration_s=0.020)

    # 3. Verify exact segment and timestamp parity
    session_tuples = [seg.to_tuple() for seg in emitted_segs]
    assert len(session_tuples) == len(ref_segs), (
        f"{excerpt_id}: segment count mismatch: {len(session_tuples)} != {len(ref_segs)}"
    )
    for (s_sess, e_sess), (s_ref, e_ref) in zip(session_tuples, ref_segs):
        assert math.isclose(s_sess, s_ref, abs_tol=1e-6), (
            f"{excerpt_id}: start timestamp mismatch: {s_sess} != {s_ref}"
        )
        assert math.isclose(e_sess, e_ref, abs_tol=1e-6), (
            f"{excerpt_id}: end timestamp mismatch: {e_sess} != {e_ref}"
        )

    # 4. Verify accumulated segments and frame count
    assert [s.to_tuple() for s in session.accumulated_segments] == session_tuples
    assert session.total_frames == ref_frames


# ============================================================================
# Test B — 100 ms chunks parity across all 11 pilot WAV files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_parity_100ms_chunks(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    ref_segs, ref_frames = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=0.100
    )

    session = VADStreamingSession(sample_rate=sr)
    emitted_segs, _ = _run_session_with_chunking(session, audio, chunk_duration_s=0.100)

    session_tuples = [seg.to_tuple() for seg in emitted_segs]
    assert session_tuples == ref_segs
    assert session.total_frames == ref_frames


# ============================================================================
# Test C — One full-array chunk parity across all 11 pilot WAV files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_parity_full_array_chunk(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    ref_segs, ref_frames = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=None
    )

    session = VADStreamingSession(sample_rate=sr)
    emitted_segs, _ = _run_session_with_chunking(session, audio, chunk_duration_s=None)

    session_tuples = [seg.to_tuple() for seg in emitted_segs]
    assert session_tuples == ref_segs
    assert session.total_frames == ref_frames


# ============================================================================
# Test D — Session reset and state isolation
# ============================================================================

def test_session_reset_and_isolation() -> None:
    audio1, sr1, _ = _load_excerpt_audio("E01_stat20_16k")
    audio2, sr2, _ = _load_excerpt_audio("E03_stat15_16k")
    assert sr1 == sr2 == 16000

    session = VADStreamingSession(sample_rate=16000)

    # Process first stream
    _run_session_with_chunking(session, audio1, chunk_duration_s=0.020)
    assert session.total_frames > 0
    assert session.total_samples > 0
    assert session.is_flushed is True
    assert len(session.accumulated_segments) > 0

    # Reset session
    session.reset()
    assert session.total_frames == 0
    assert session.total_samples == 0
    assert session.is_flushed is False
    assert len(session.accumulated_segments) == 0

    # Process second stream on reused session
    reused_segs, _ = _run_session_with_chunking(session, audio2, chunk_duration_s=0.020)
    reused_frames = session.total_frames

    # Process second stream on completely fresh session
    fresh_session = VADStreamingSession(sample_rate=16000)
    fresh_segs, _ = _run_session_with_chunking(fresh_session, audio2, chunk_duration_s=0.020)
    fresh_frames = fresh_session.total_frames

    # Verify exact match between reused session after reset and fresh session
    assert [s.to_tuple() for s in reused_segs] == [s.to_tuple() for s in fresh_segs]
    assert reused_frames == fresh_frames


# ============================================================================
# Test E — Flush and edge cases
# ============================================================================

class TestStreamingEdgeCases:
    def test_empty_stream(self) -> None:
        session = VADStreamingSession(sample_rate=16000)
        # Process an empty chunk
        res = session.process_chunk(np.empty(0, dtype=np.float32))
        assert res.segments == []
        assert res.num_frames == 0
        assert res.duration_s == 0.0

        res_flush = session.flush()
        assert res_flush.segments == []
        assert res_flush.num_frames == 0
        assert res_flush.duration_s == 0.0
        assert session.accumulated_segments == []
        assert session.total_frames == 0

    def test_repeated_flush_idempotency(self) -> None:
        audio, sr, _ = _load_excerpt_audio("E01_stat20_16k")
        session = VADStreamingSession(sample_rate=sr)

        # Feed some chunks
        c_samples = int(0.020 * sr)
        res1 = session.process_chunk(audio[:c_samples])
        res2 = session.process_chunk(audio[c_samples : 2 * c_samples])

        flush1 = session.flush()
        assert session.is_flushed is True

        # Second flush must not crash, must return empty result, and must not duplicate segments
        flush2 = session.flush()
        assert flush2.segments == []
        assert flush2.num_frames == 0
        assert flush2.duration_s == 0.0

        # Third flush
        flush3 = session.flush()
        assert flush3.segments == []

    def test_process_chunk_after_flush_raises(self) -> None:
        session = VADStreamingSession(sample_rate=16000)
        session.flush()
        with pytest.raises(RuntimeError, match="after session has been flushed"):
            session.process_chunk(np.zeros(320, dtype=np.float32))

        # But after reset(), it is allowed again
        session.reset()
        res = session.process_chunk(np.zeros(320, dtype=np.float32))
        assert res.duration_s == 320 / 16000

    def test_sample_rate_validation(self) -> None:
        # Type errors
        for bad_type in (16000.0, "16000", True, False, None, [16000]):
            with pytest.raises(TypeError, match="integer"):
                VADStreamingSession(sample_rate=bad_type)  # type: ignore

        # Value errors
        for bad_val in (0, -1, 44100, 22050, 48000, 32000):
            with pytest.raises(ValueError, match="Unsupported sample_rate"):
                VADStreamingSession(sample_rate=bad_val)

    def test_flush_duration_is_zero(self) -> None:
        session = VADStreamingSession(sample_rate=16000)
        audio = np.zeros(1600, dtype=np.float32)  # 100 ms
        res1 = session.process_chunk(audio)
        assert res1.duration_s == 0.100
        res_flush = session.flush()
        assert res_flush.duration_s == 0.0

    def test_duration_additivity_and_short_audio(self) -> None:
        session = VADStreamingSession(sample_rate=16000)
        # Audio shorter than 1 FFT frame (n_fft=256 for 16k)
        short_audio = np.zeros(100, dtype=np.float32)
        res_chunk = session.process_chunk(short_audio)
        assert res_chunk.duration_s == 100 / 16000
        assert res_chunk.num_frames == 0

        res_flush = session.flush()
        assert res_flush.duration_s == 0.0
        # FeatureExtractor zero-pads remaining samples < n_fft into 1 tail frame
        assert res_flush.num_frames == 1

        # Additive property:
        assert math.isclose(res_chunk.duration_s + res_flush.duration_s, 100 / 16000)
        assert math.isclose(session.total_duration_s, 100 / 16000)

    def test_chunk_duration_additivity_multichunk(self) -> None:
        audio, sr, dur_s = _load_excerpt_audio("E01_stat20_16k")
        session = VADStreamingSession(sample_rate=sr)

        c_samples = int(0.020 * sr)
        total_chunk_duration = 0.0
        for i in range(0, len(audio), c_samples):
            ch = audio[i : i + c_samples]
            res = session.process_chunk(ch)
            total_chunk_duration += res.duration_s

        res_flush = session.flush()
        assert res_flush.duration_s == 0.0

        total_emitted_duration = total_chunk_duration + res_flush.duration_s
        assert math.isclose(total_emitted_duration, dur_s, abs_tol=1e-6)
        assert math.isclose(session.total_duration_s, dur_s, abs_tol=1e-6)

    def test_invalid_chunk_type_and_shape(self) -> None:
        session = VADStreamingSession(sample_rate=16000)
        with pytest.raises(TypeError, match="cannot be None"):
            session.process_chunk(None)  # type: ignore

        with pytest.raises(ValueError, match="1-D mono array"):
            session.process_chunk(np.zeros((2, 160), dtype=np.float32))

        with pytest.raises(ValueError, match="real-valued"):
            session.process_chunk(np.zeros(160, dtype=np.complex64))

        with pytest.raises(TypeError, match="numeric dtype"):
            session.process_chunk(np.array(["a", "b"]))  # type: ignore

    def test_debug_nan_inf_rejected(self) -> None:
        session = VADStreamingSession(sample_rate=16000, debug=True)
        bad_chunk = np.zeros(160, dtype=np.float32)
        bad_chunk[10] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            session.process_chunk(bad_chunk)

    def test_diagnostics_returned_when_requested(self) -> None:
        session = VADStreamingSession(sample_rate=16000, return_diagnostics=True)
        chunk = np.zeros(640, dtype=np.float32)  # 40 ms -> 5 frames
        res = session.process_chunk(chunk)
        assert res.diagnostics is not None
        assert isinstance(res.diagnostics, VADDiagnostics)
        assert res.diagnostics.num_frames == res.num_frames
        assert res.diagnostics.frame_scores.shape == (res.num_frames,)
        assert res.diagnostics.noise_floor is not None

        res_flush = session.flush()
        assert res_flush.diagnostics is not None

    def test_timestamps_within_audio_duration(self) -> None:
        for eid in PILOT_EXCERPT_IDS:
            audio, sr, dur_s = _load_excerpt_audio(eid)
            session = VADStreamingSession(sample_rate=sr)
            segs, _ = _run_session_with_chunking(session, audio, chunk_duration_s=0.020)
            for seg in segs:
                assert 0.0 <= seg.start_time_s < seg.end_time_s <= (dur_s + 0.020)


# ============================================================================
# Test F — Chunk boundaries invariance
# ============================================================================

def test_arbitrary_chunk_boundaries_invariance() -> None:
    # Test with non-standard chunk sizes crossing speech onset and offsets
    audio, sr, _ = _load_excerpt_audio("E01_stat20_16k")

    # Baseline with standard 20 ms
    ref_segs, ref_frames = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=0.020
    )

    # Test irregular chunk durations: 7 ms, 13 ms, 33 ms
    for c_dur in (0.007, 0.013, 0.033):
        session = VADStreamingSession(sample_rate=sr)
        segs, _ = _run_session_with_chunking(session, audio, chunk_duration_s=c_dur)
        session_tuples = [s.to_tuple() for s in segs]
        assert session_tuples == ref_segs, f"Failed chunk invariance for c_dur={c_dur}"
        assert session.total_frames == ref_frames
