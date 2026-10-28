"""
tests/test_vad_offline.py

Exact-parity, state-isolation, and edge-case tests for VADOfflineRunner:
- Test A: Parity between VADOfflineRunner and VADStreamingSession on all 11 pilot WAV files
- Test B: Parity between VADOfflineRunner and reference run_multiband_streaming on all 11 pilot WAV files
- Test C: Runner reuse and state isolation between different files
- Test D: Repeated execution determinism on the same runner instance
- Test E: Input validation, boundary checks, empty streams, 8k/16k, and diagnostics
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VAD.pipeline.contracts import PipelineMode, VADDiagnostics, VADResult, VADSegment  # noqa: E402
from VAD.pipeline.offline import VADOfflineRunner  # noqa: E402
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


# ============================================================================
# Test A — Parity with VADStreamingSession across all 11 pilot WAV files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_offline_parity_with_streaming_session(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    # 1. Run VADOfflineRunner
    runner = VADOfflineRunner(sample_rate=sr)
    off_res = runner.process_audio(audio)

    # 2. Run VADStreamingSession as one complete chunk + flush
    session = VADStreamingSession(sample_rate=sr)
    res_chunk = session.process_chunk(audio)
    res_flush = session.flush()
    stream_segs = res_chunk.segments + res_flush.segments

    # 3. Assert exact segment and timestamp parity
    assert len(off_res.segments) == len(stream_segs)
    for seg_off, seg_str in zip(off_res.segments, stream_segs):
        assert math.isclose(seg_off.start_time_s, seg_str.start_time_s, abs_tol=1e-6)
        assert math.isclose(seg_off.end_time_s, seg_str.end_time_s, abs_tol=1e-6)

    assert off_res.num_frames == session.total_frames


# ============================================================================
# Test B — Reference parity with run_multiband_streaming on all 11 WAV files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_offline_parity_with_reference(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    ref_segs, ref_frames = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=None
    )

    runner = VADOfflineRunner(sample_rate=sr)
    off_res = runner.process_audio(audio)

    off_tuples = [seg.to_tuple() for seg in off_res.segments]
    assert off_tuples == ref_segs
    assert off_res.num_frames == ref_frames


# ============================================================================
# Test C — Runner reuse and state isolation
# ============================================================================

def test_runner_reuse_and_state_isolation() -> None:
    audio1, sr1, _ = _load_excerpt_audio("E01_stat20_16k")
    audio2, sr2, _ = _load_excerpt_audio("E03_stat15_16k")
    assert sr1 == sr2 == 16000

    runner = VADOfflineRunner(sample_rate=16000)

    # Process first file
    res1 = runner.process_audio(audio1)
    assert len(res1.segments) > 0

    # Process second file on the SAME runner instance
    res2_reused = runner.process_audio(audio2)

    # Process second file on a newly constructed runner
    fresh_runner = VADOfflineRunner(sample_rate=16000)
    res2_fresh = fresh_runner.process_audio(audio2)

    # Reused runner must produce identical result to fresh runner (zero state leak)
    assert [s.to_tuple() for s in res2_reused.segments] == [s.to_tuple() for s in res2_fresh.segments]
    assert res2_reused.num_frames == res2_fresh.num_frames
    assert res2_reused.duration_s == res2_fresh.duration_s


# ============================================================================
# Test D — Repeated execution determinism
# ============================================================================

def test_repeated_execution_determinism() -> None:
    audio, sr, _ = _load_excerpt_audio("E07_step_up_16k")
    runner = VADOfflineRunner(sample_rate=sr)

    # Run 3 consecutive times on the same instance
    res_a = runner.process_audio(audio)
    res_b = runner.process_audio(audio)
    res_c = runner.process_audio(audio)

    assert [s.to_tuple() for s in res_a.segments] == [s.to_tuple() for s in res_b.segments]
    assert [s.to_tuple() for s in res_b.segments] == [s.to_tuple() for s in res_c.segments]
    assert res_a.num_frames == res_b.num_frames == res_c.num_frames
    assert res_a.duration_s == res_b.duration_s == res_c.duration_s


# ============================================================================
# Test E — Inputs, boundaries, and edge cases
# ============================================================================

class TestOfflineEdgeCases:
    def test_mode_property(self) -> None:
        runner = VADOfflineRunner(sample_rate=16000)
        assert runner.mode == PipelineMode.OFFLINE
        assert runner.mode.value == "offline"

    def test_process_file_alias(self) -> None:
        audio, sr, _ = _load_excerpt_audio("E02_stat20_8k")
        runner = VADOfflineRunner(sample_rate=sr)
        res_audio = runner.process_audio(audio)
        res_file = runner.process_file(audio)
        assert [s.to_tuple() for s in res_audio.segments] == [s.to_tuple() for s in res_file.segments]
        assert res_audio.num_frames == res_file.num_frames

    def test_sample_rate_override_in_process_audio(self) -> None:
        audio, sr, _ = _load_excerpt_audio("E02_stat20_8k")
        runner = VADOfflineRunner(sample_rate=16000)  # default 16k
        res = runner.process_audio(audio, sample_rate=8000)  # override to 8k
        assert res.num_frames > 0

    def test_empty_audio_handling(self) -> None:
        runner = VADOfflineRunner(sample_rate=16000)
        res = runner.process_audio(np.empty(0, dtype=np.float32))
        assert res.segments == []
        assert res.num_frames == 0
        assert res.duration_s == 0.0
        assert res.diagnostics is None

    def test_empty_audio_with_diagnostics(self) -> None:
        runner = VADOfflineRunner(sample_rate=16000, return_diagnostics=True)
        res = runner.process_audio(np.empty(0, dtype=np.float32))
        assert res.segments == []
        assert res.diagnostics is not None
        assert res.diagnostics.num_frames == 0

    def test_sample_rate_validation(self) -> None:
        # TypeError on init
        for bad_type in (16000.0, "16000", True, False, None):
            with pytest.raises(TypeError, match="integer"):
                VADOfflineRunner(sample_rate=bad_type)  # type: ignore

        # ValueError on init
        for bad_val in (0, -1, 44100, 22050):
            with pytest.raises(ValueError, match="Unsupported sample_rate"):
                VADOfflineRunner(sample_rate=bad_val)

        # TypeError and ValueError in process_audio override
        runner = VADOfflineRunner(sample_rate=16000)
        audio = np.zeros(320, dtype=np.float32)
        with pytest.raises(TypeError, match="integer"):
            runner.process_audio(audio, sample_rate=8000.0)  # type: ignore
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            runner.process_audio(audio, sample_rate=44100)

    def test_offline_duration_contract(self) -> None:
        audio, sr, dur_s = _load_excerpt_audio("E01_stat20_16k")
        runner = VADOfflineRunner(sample_rate=sr)
        res = runner.process_audio(audio)
        assert math.isclose(res.duration_s, dur_s, abs_tol=1e-6)
        assert math.isclose(res.duration_s, len(audio) / sr, abs_tol=1e-6)

        # Short audio < n_fft (256 samples at 16k)
        short_audio = np.zeros(100, dtype=np.float32)
        res_short = runner.process_audio(short_audio)
        assert math.isclose(res_short.duration_s, 100 / sr, abs_tol=1e-6)
        # FeatureExtractor zero-pads remaining samples < n_fft into 1 tail frame
        assert res_short.num_frames == 1

    def test_invalid_waveform_shape_and_type(self) -> None:
        runner = VADOfflineRunner(sample_rate=16000)
        with pytest.raises(TypeError, match="cannot be None"):
            runner.process_audio(None)  # type: ignore

        with pytest.raises(ValueError, match="1-D mono array"):
            runner.process_audio(np.zeros((2, 1600), dtype=np.float32))

        with pytest.raises(ValueError, match="real-valued"):
            runner.process_audio(np.zeros(1600, dtype=np.complex64))

        with pytest.raises(TypeError, match="numeric dtype"):
            runner.process_audio(np.array(["test"]))  # type: ignore

    def test_diagnostics_coverage(self) -> None:
        audio, sr, _ = _load_excerpt_audio("E01_stat20_16k")
        runner = VADOfflineRunner(sample_rate=sr, return_diagnostics=True)
        res = runner.process_audio(audio)
        assert res.diagnostics is not None
        assert res.diagnostics.num_frames == res.num_frames
        assert res.diagnostics.frame_scores.shape == (res.num_frames,)
        assert res.diagnostics.states.shape == (res.num_frames,)
        assert res.diagnostics.speech_mask.shape == (res.num_frames,)
        assert res.diagnostics.warmup_mask.shape == (res.num_frames,)
        assert res.diagnostics.noise_floor.shape == (4, res.num_frames)
        assert res.diagnostics.snr_db.shape == (4, res.num_frames)

    def test_timestamps_within_audio_duration(self) -> None:
        for eid in PILOT_EXCERPT_IDS:
            audio, sr, dur_s = _load_excerpt_audio(eid)
            runner = VADOfflineRunner(sample_rate=sr)
            res = runner.process_audio(audio)
            for seg in res.segments:
                assert 0.0 <= seg.start_time_s < seg.end_time_s <= (dur_s + 0.020)

    def test_speech_at_start_excerpts(self) -> None:
        # E10 and E11 have speech at start t=0
        for eid in ("E10_start_16k", "E11_start_8k"):
            audio, sr, _ = _load_excerpt_audio(eid)
            runner = VADOfflineRunner(sample_rate=sr)
            res = runner.process_audio(audio)
            assert len(res.segments) > 0
            # Matches documented first segment at 0.625s due to 200ms warm-up + noise floor contamination
            assert math.isclose(res.segments[0].start_time_s, 0.625, abs_tol=1e-3)
