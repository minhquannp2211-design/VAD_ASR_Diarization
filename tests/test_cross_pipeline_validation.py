"""
tests/test_cross_pipeline_validation.py

Phase 3 Cross-Pipeline Validation Tests:
1. 5-Way Parity across all 11 pilot WAV files:
   - VADStreamingSession (1 single whole-file chunk)
   - VADStreamingSession (20 ms chunks)
   - VADStreamingSession (100 ms chunks)
   - VADOfflineRunner.process_audio() (whole waveform)
   - run_multiband_streaming() (reference helper)
2. Cross-Pipeline Parity on Dynamic Benchmark Scenarios:
   - Stationary noise (+20 dB, +15 dB, +10 dB)
   - Step-up / step-down noise (+15 dB jump)
   - Speech at start (t = 0 ms)
   - 8 kHz and 16 kHz inputs
3. Diagnostics and Timestamp Integrity:
   - Exact STFT framing alignment (center=False contract)
   - Time indexing and mask alignment with segments
   - Half-open interval semantics and duration bounds
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

from benchmark.run_vad_benchmark import (  # noqa: E402
    build_benchmark_stream,
    collect_benchmark_materials,
)
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
benchmark_runner = _load_pilot_module(
    "benchmark_multiband_vs_silero", "benchmark_multiband_vs_silero.py"
)

PILOT_MANIFEST: Dict[str, Any] = pilot_tools.load_manifest()
PILOT_EXCERPT_IDS = sorted(PILOT_MANIFEST.keys())


def _load_excerpt_audio(excerpt_id: str) -> Tuple[np.ndarray, int, float]:
    spec = PILOT_MANIFEST[excerpt_id]
    wav_path = os.path.join(PILOT_DIR, spec.wav_rel_path.replace("/", os.sep))
    audio, sr, dur_s = preannotate_silero.read_pcm_wav_mono(wav_path)
    return audio, sr, dur_s


def _run_stream_chunks(
    audio: np.ndarray,
    sr: int,
    chunk_duration_s: float | None,
) -> Tuple[List[Tuple[float, float]], int]:
    sess = VADStreamingSession(sample_rate=sr)
    if chunk_duration_s is None or chunk_duration_s <= 0.0:
        chunks = [audio]
    else:
        cs = max(1, int(round(chunk_duration_s * sr)))
        chunks = [audio[i : i + cs] for i in range(0, len(audio), cs)]

    emitted: List[VADSegment] = []
    for c in chunks:
        emitted.extend(sess.process_chunk(c).segments)
    emitted.extend(sess.flush().segments)
    return [seg.to_tuple() for seg in emitted], sess.total_frames


# ============================================================================
# 1. 5-Way Parity Matrix Across All 11 Pilot WAV Files
# ============================================================================

@pytest.mark.parametrize("excerpt_id", PILOT_EXCERPT_IDS)
def test_5way_parity_pilot_dataset(excerpt_id: str) -> None:
    audio, sr, _ = _load_excerpt_audio(excerpt_id)

    # Path 1: Streaming full chunk
    segs_p1, frames_p1 = _run_stream_chunks(audio, sr, None)

    # Path 2: Streaming 20 ms chunks
    segs_p2, frames_p2 = _run_stream_chunks(audio, sr, 0.020)

    # Path 3: Streaming 100 ms chunks
    segs_p3, frames_p3 = _run_stream_chunks(audio, sr, 0.100)

    # Path 4: Offline runner
    runner = VADOfflineRunner(sample_rate=sr)
    res_off = runner.process_audio(audio)
    segs_p4 = [s.to_tuple() for s in res_off.segments]
    frames_p4 = res_off.num_frames

    # Path 5: Pilot reference runner
    segs_p5, frames_p5 = benchmark_runner.run_multiband_streaming(
        audio, sr, chunk_duration_s=0.020
    )

    # Assert exact numerical parity across all 5 paths
    assert segs_p1 == segs_p2, f"{excerpt_id}: Path 1 != Path 2"
    assert segs_p2 == segs_p3, f"{excerpt_id}: Path 2 != Path 3"
    assert segs_p3 == segs_p4, f"{excerpt_id}: Path 3 != Path 4"
    assert segs_p4 == segs_p5, f"{excerpt_id}: Path 4 != Path 5"

    assert frames_p1 == frames_p2 == frames_p3 == frames_p4 == frames_p5, (
        f"{excerpt_id}: Frame count mismatch across paths: "
        f"{frames_p1}, {frames_p2}, {frames_p3}, {frames_p4}, {frames_p5}"
    )


# ============================================================================
# 2. Cross-Pipeline Parity on Dynamic Benchmark Scenarios (16 kHz and 8 kHz)
# ============================================================================

BENCHMARK_SCENARIOS = [
    "stationary_20db",
    "stationary_15db",
    "stationary_10db",
    "step_up_down",
    "speech_at_start",
]


@pytest.mark.parametrize("scenario", BENCHMARK_SCENARIOS)
def test_cross_pipeline_parity_16k_benchmark_scenarios(scenario: str) -> None:
    clean_cores, noise_pieces = collect_benchmark_materials(sr=16000)
    _, _, mix_wave, _ = build_benchmark_stream(clean_cores, noise_pieces, 16000, scenario, seed=100)

    # 1. Offline runner
    runner = VADOfflineRunner(sample_rate=16000)
    res_off = runner.process_audio(mix_wave)
    segs_off = [s.to_tuple() for s in res_off.segments]

    # 2. Streaming 20 ms
    segs_20, frames_20 = _run_stream_chunks(mix_wave, 16000, 0.020)

    # 3. Streaming 100 ms
    segs_100, frames_100 = _run_stream_chunks(mix_wave, 16000, 0.100)

    assert segs_off == segs_20 == segs_100, f"Scenario {scenario}: segments mismatch across pipelines"
    assert res_off.num_frames == frames_20 == frames_100


@pytest.mark.parametrize("scenario", ["stationary_15db", "step_up_down", "speech_at_start"])
def test_cross_pipeline_parity_8k_benchmark_scenarios(scenario: str) -> None:
    clean_cores, noise_pieces = collect_benchmark_materials(sr=8000)
    _, _, mix_wave, _ = build_benchmark_stream(clean_cores, noise_pieces, 8000, scenario, seed=100)

    runner = VADOfflineRunner(sample_rate=8000)
    res_off = runner.process_audio(mix_wave)
    segs_off = [s.to_tuple() for s in res_off.segments]

    segs_20, frames_20 = _run_stream_chunks(mix_wave, 8000, 0.020)
    segs_100, frames_100 = _run_stream_chunks(mix_wave, 8000, 0.100)

    assert segs_off == segs_20 == segs_100, f"8k Scenario {scenario}: segments mismatch across pipelines"
    assert res_off.num_frames == frames_20 == frames_100


# ============================================================================
# 3. Diagnostics and Timestamp Integrity
# ============================================================================

def test_diagnostics_stft_contract_and_indexing_integrity() -> None:
    # Test on a known audio excerpt
    audio, sr, dur_s = _load_excerpt_audio("E01_stat20_16k")

    runner = VADOfflineRunner(sample_rate=sr, return_diagnostics=True)
    res = runner.process_audio(audio)
    diag = res.diagnostics
    assert diag is not None

    # Verify frame counts match exactly
    total_frames = res.num_frames
    assert diag.num_frames == total_frames
    assert diag.frame_scores.shape == (total_frames,)
    assert diag.speech_mask.shape == (total_frames,)
    assert diag.states.shape == (total_frames,)
    assert diag.warmup_mask.shape == (total_frames,)
    assert diag.noise_floor.shape == (4, total_frames)
    assert diag.snr_db.shape == (4, total_frames)

    # Verify warmup frames (first 40 frames = 200 ms @ 5 ms) have score 0.0
    assert diag.warmup_mask[:40].all()
    assert np.allclose(diag.frame_scores[:40], 0.0)

    # Verify segment boundaries are within [0.0, dur_s + 0.020]
    for seg in res.segments:
        assert 0.0 <= seg.start_time_s < seg.end_time_s <= (dur_s + 0.020)
        assert seg.duration_s > 0.0


def test_chunk_streaming_diagnostics_concatenation_integrity() -> None:
    audio, sr, _ = _load_excerpt_audio("E02_stat20_8k")
    session = VADStreamingSession(sample_rate=sr, return_diagnostics=True)

    c_len = int(0.020 * sr)
    chunks = [audio[i : i + c_len] for i in range(0, len(audio), c_len)]

    diag_scores: List[np.ndarray] = []
    diag_masks: List[np.ndarray] = []
    total_chunk_frames = 0

    for ch in chunks:
        r = session.process_chunk(ch)
        if r.diagnostics is not None and r.num_frames > 0:
            diag_scores.append(r.diagnostics.frame_scores)
            diag_masks.append(r.diagnostics.speech_mask)
            total_chunk_frames += r.num_frames

    r_flush = session.flush()
    if r_flush.diagnostics is not None and r_flush.num_frames > 0:
        diag_scores.append(r_flush.diagnostics.frame_scores)
        diag_masks.append(r_flush.diagnostics.speech_mask)
        total_chunk_frames += r_flush.num_frames

    assert total_chunk_frames == session.total_frames
    merged_scores = np.concatenate(diag_scores)
    assert len(merged_scores) == session.total_frames

    # Compare with whole-file offline runner diagnostics
    runner = VADOfflineRunner(sample_rate=sr, return_diagnostics=True)
    res_off = runner.process_audio(audio)
    assert np.allclose(merged_scores, res_off.diagnostics.frame_scores, atol=1e-5)
