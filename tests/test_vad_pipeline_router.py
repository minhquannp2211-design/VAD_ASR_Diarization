"""
tests/test_vad_pipeline_router.py

Tests for VADPipelineRouter factory:
- Test A: Mode selection (PipelineMode and string identifiers)
- Test B: Streaming usability via router
- Test C: Offline usability via router
- Test D: No state leakage between separately constructed instances
- Test E: Validation of modes, sample rates, and types
- Test F: Parity preservation between router-created pipelines and direct construction
- Test G: Package import contract verification
- Test H: Timestamp bounds and duration integrity
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

from VAD.pipeline import (  # noqa: E402
    PipelineMode,
    VADDiagnostics,
    VADOfflineRunner,
    VADPipelineRouter,
    VADResult,
    VADSegment,
    VADStreamingSession,
)

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

PILOT_MANIFEST: Dict[str, Any] = pilot_tools.load_manifest()


def _load_excerpt_audio(excerpt_id: str) -> Tuple[np.ndarray, int, float]:
    spec = PILOT_MANIFEST[excerpt_id]
    wav_path = os.path.join(PILOT_DIR, spec.wav_rel_path.replace("/", os.sep))
    audio, sr, dur_s = preannotate_silero.read_pcm_wav_mono(wav_path)
    return audio, sr, dur_s


# ============================================================================
# Test A — Mode selection
# ============================================================================

class TestModeSelection:
    def test_streaming_mode_enum(self) -> None:
        pipe = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=16000)
        assert isinstance(pipe, VADStreamingSession)
        assert pipe.mode == PipelineMode.STREAMING
        assert pipe.sample_rate == 16000

    def test_streaming_mode_string(self) -> None:
        pipe = VADPipelineRouter.create(mode="streaming", sample_rate=16000)
        assert isinstance(pipe, VADStreamingSession)
        assert pipe.mode == PipelineMode.STREAMING

    def test_streaming_mode_case_insensitive(self) -> None:
        pipe = VADPipelineRouter.create(mode="Streaming", sample_rate=8000)
        assert isinstance(pipe, VADStreamingSession)
        assert pipe.sample_rate == 8000

    def test_offline_mode_enum(self) -> None:
        pipe = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=16000)
        assert isinstance(pipe, VADOfflineRunner)
        assert pipe.mode == PipelineMode.OFFLINE
        assert pipe.sample_rate == 16000

    def test_offline_mode_string(self) -> None:
        pipe = VADPipelineRouter.create(mode="offline", sample_rate=16000)
        assert isinstance(pipe, VADOfflineRunner)
        assert pipe.mode == PipelineMode.OFFLINE

    def test_offline_mode_case_insensitive(self) -> None:
        pipe = VADPipelineRouter.create(mode="OFFLINE", sample_rate=8000)
        assert isinstance(pipe, VADOfflineRunner)
        assert pipe.sample_rate == 8000

    def test_direct_call_syntax(self) -> None:
        pipe_stream = VADPipelineRouter(mode=PipelineMode.STREAMING, sample_rate=16000)
        assert isinstance(pipe_stream, VADStreamingSession)

        pipe_off = VADPipelineRouter(mode=PipelineMode.OFFLINE, sample_rate=16000)
        assert isinstance(pipe_off, VADOfflineRunner)


# ============================================================================
# Test B — Streaming usability via router
# ============================================================================

def test_streaming_usability() -> None:
    session = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=16000)
    audio, sr, _ = _load_excerpt_audio("E01_stat20_16k")

    c_samples = int(0.020 * sr)
    chunks = [audio[i : i + c_samples] for i in range(0, len(audio), c_samples)]

    emitted_segments: list[VADSegment] = []
    for ch in chunks:
        res = session.process_chunk(ch)
        assert isinstance(res, VADResult)
        emitted_segments.extend(res.segments)

    res_flush = session.flush()
    assert isinstance(res_flush, VADResult)
    emitted_segments.extend(res_flush.segments)

    assert len(emitted_segments) == 3
    assert session.is_flushed is True
    assert session.total_frames == 1238


# ============================================================================
# Test C — Offline usability via router
# ============================================================================

def test_offline_usability() -> None:
    runner = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=16000)
    audio, sr, dur_s = _load_excerpt_audio("E01_stat20_16k")

    res = runner.process_audio(audio)
    assert isinstance(res, VADResult)
    assert len(res.segments) == 3
    assert res.num_frames == 1238
    assert math.isclose(res.duration_s, dur_s, abs_tol=1e-3)


# ============================================================================
# Test D — No state leakage
# ============================================================================

def test_streaming_session_isolation() -> None:
    audio1, sr1, _ = _load_excerpt_audio("E01_stat20_16k")
    audio2, sr2, _ = _load_excerpt_audio("E03_stat15_16k")

    sess1 = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=sr1)
    sess2 = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=sr2)

    # Process audio1 on sess1
    sess1.process_chunk(audio1)
    sess1.flush()

    assert sess1.total_frames > 0
    assert sess2.total_frames == 0
    assert sess2.is_flushed is False
    assert len(sess2.accumulated_segments) == 0

    # Process audio2 on sess2
    sess2.process_chunk(audio2)
    sess2.flush()

    fresh = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=sr2)
    fresh.process_chunk(audio2)
    fresh.flush()

    assert [s.to_tuple() for s in sess2.accumulated_segments] == [s.to_tuple() for s in fresh.accumulated_segments]
    assert sess2.total_frames == fresh.total_frames


def test_offline_runner_isolation() -> None:
    audio1, sr1, _ = _load_excerpt_audio("E01_stat20_16k")
    audio2, sr2, _ = _load_excerpt_audio("E03_stat15_16k")

    runner1 = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=sr1)
    runner2 = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=sr2)

    res1 = runner1.process_audio(audio1)
    res2 = runner2.process_audio(audio2)

    # Re-running audio2 on runner1 must match runner2 exactly
    res2_on_runner1 = runner1.process_audio(audio2)
    assert [s.to_tuple() for s in res2.segments] == [s.to_tuple() for s in res2_on_runner1.segments]
    assert res2.num_frames == res2_on_runner1.num_frames


# ============================================================================
# Test E — Validation
# ============================================================================

class TestRouterValidation:
    def test_invalid_mode_string(self) -> None:
        with pytest.raises(ValueError, match="Invalid pipeline mode"):
            VADPipelineRouter.create(mode="realtime", sample_rate=16000)

    def test_invalid_mode_type(self) -> None:
        with pytest.raises(TypeError, match="mode must be a PipelineMode or str"):
            VADPipelineRouter.create(mode=123, sample_rate=16000)  # type: ignore
        with pytest.raises(TypeError, match="mode must be a PipelineMode or str"):
            VADPipelineRouter.create(mode=None, sample_rate=16000)  # type: ignore

    def test_unsupported_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=44100)
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=22050)
        with pytest.raises(ValueError, match="Unsupported sample_rate"):
            VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=0)

    def test_sample_rate_type_validation(self) -> None:
        for bad_type in (16000.0, "16000", True, False, None, [16000]):
            with pytest.raises(TypeError, match="integer"):
                VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=bad_type)  # type: ignore
            with pytest.raises(TypeError, match="integer"):
                VADPipelineRouter(mode=PipelineMode.STREAMING, sample_rate=bad_type)  # type: ignore
            with pytest.raises(TypeError, match="integer"):
                VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=bad_type)  # type: ignore
            with pytest.raises(TypeError, match="integer"):
                VADPipelineRouter(mode=PipelineMode.OFFLINE, sample_rate=bad_type)  # type: ignore


# ============================================================================
# Test F — Parity preservation
# ============================================================================

@pytest.mark.parametrize("excerpt_id", ["E01_stat20_16k", "E02_stat20_8k", "E07_step_up_16k", "E11_start_8k"])
def test_parity_preservation_with_direct_construction(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    # 1. Compare streaming path
    sess_router = VADPipelineRouter.create(mode=PipelineMode.STREAMING, sample_rate=sr)
    sess_direct = VADStreamingSession(sample_rate=sr)

    c_samples = int(0.020 * sr)
    segs_router = []
    segs_direct = []

    for i in range(0, len(audio), c_samples):
        ch = audio[i : i + c_samples]
        segs_router.extend(sess_router.process_chunk(ch).segments)
        segs_direct.extend(sess_direct.process_chunk(ch).segments)

    segs_router.extend(sess_router.flush().segments)
    segs_direct.extend(sess_direct.flush().segments)

    assert [s.to_tuple() for s in segs_router] == [s.to_tuple() for s in segs_direct]
    assert sess_router.total_frames == sess_direct.total_frames

    # 2. Compare offline path
    off_router = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=sr)
    off_direct = VADOfflineRunner(sample_rate=sr)

    res_router = off_router.process_audio(audio)
    res_direct = off_direct.process_audio(audio)

    assert [s.to_tuple() for s in res_router.segments] == [s.to_tuple() for s in res_direct.segments]
    assert res_router.num_frames == res_direct.num_frames


# ============================================================================
# Test G — Import contract
# ============================================================================

def test_pipeline_package_exports() -> None:
    import VAD.pipeline as vp

    assert hasattr(vp, "PipelineMode")
    assert hasattr(vp, "SUPPORTED_SAMPLE_RATES")
    assert hasattr(vp, "validate_sample_rate")
    assert hasattr(vp, "VADSegment")
    assert hasattr(vp, "VADDiagnostics")
    assert hasattr(vp, "VADResult")
    assert hasattr(vp, "VADStreamingSession")
    assert hasattr(vp, "VADOfflineRunner")
    assert hasattr(vp, "VADPipelineRouter")


# ============================================================================
# Test H — Timestamp bounds and duration integrity audit
# ============================================================================

def test_timestamp_duration_bounds_across_pilot_files() -> None:
    # Audit: Every segment emitted must satisfy 0.0 <= start < end <= dur_s
    for eid in PILOT_MANIFEST.keys():
        audio, sr, dur_s = _load_excerpt_audio(eid)
        runner = VADPipelineRouter.create(mode=PipelineMode.OFFLINE, sample_rate=sr)
        res = runner.process_audio(audio)

        for seg in res.segments:
            # Mathematical invariant: start non-negative, end > start
            assert 0.0 <= seg.start_time_s < seg.end_time_s
            # Endpoint must not exceed audio duration
            assert seg.end_time_s <= dur_s, (
                f"{eid}: segment end {seg.end_time_s:.6f} exceeds audio duration {dur_s:.6f}"
            )
