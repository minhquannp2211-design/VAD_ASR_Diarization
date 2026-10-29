"""
tests/test_vad_ingest_adapter.py

Focused, deterministic tests for VAD.pipeline.ingest.VADIngestAdapter (integration boundary).
Audio is SYNTHETIC (harmonic bursts over low noise). These tests prove interface properties
(chunk invariance, exactly-once, absolute timeline, flush, explicit errors). They say nothing
about VAD accuracy on real speech.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAD.pipeline.ingest import (  # noqa: E402
    SampleDiscontinuityError,
    SpeechSpan,
    VADIngestAdapter,
)
from VAD.pipeline.streaming import VADStreamingSession  # noqa: E402

SR = 16000
BURSTS_S = [(1.0, 1.8), (3.0, 3.6)]  # bursts at 1.0-1.8 s and 3.0-3.6 s


def synth(seconds: float = 5.0, bursts=BURSTS_S, seed: int = 0, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = (rng.standard_normal(n) * 0.002).astype(np.float32)
    t = np.arange(n) / sr
    for a, b in bursts:
        m = (t >= a) & (t < b)
        ph = 2 * np.pi * np.cumsum(np.full(int(m.sum()), 120.0)) / sr
        sig = sum(np.sin(k * ph) / k for k in range(1, 20))
        env = np.minimum(1.0, np.minimum(t[m] - a, b - t[m]) / 0.03)
        x[m] += (0.1 * env * sig).astype(np.float32)
    return x


def run(x: np.ndarray, chunk: int, sr: int = SR):
    a = VADIngestAdapter(sample_rate=sr)
    spans, pos = [], 0
    while pos < len(x):
        c = x[pos:pos + chunk]
        spans += a.feed(c, first_sample_index=pos)
        pos += len(c)
    spans += a.flush()
    return spans, a


@pytest.fixture(scope="module")
def wave() -> np.ndarray:
    return synth()


@pytest.fixture(scope="module")
def reference(wave):
    spans, _ = run(wave, 320)
    return spans


def test_reference_detects_synthetic_bursts(reference):
    # Sanity of the fixture only (not an accuracy claim): one span per burst.
    assert len(reference) == len(BURSTS_S)
    for sp, (a, b) in zip(reference, BURSTS_S):
        assert sp.start_sample < sp.end_sample
        assert abs(sp.start_sample / SR - a) < 0.1


@pytest.mark.parametrize("chunk", [1, 7, 80, 160, 333, 1600, 12345, 24000, 32000, 80000])
def test_chunk_size_invariance(wave, reference, chunk):
    x = wave[: 2 * SR] if chunk < 80 else wave  # tiny chunks: shorter signal compares vs its own ref
    ref = run(x, 320)[0] if chunk < 80 else reference
    got, _ = run(x, chunk)
    assert got == ref


def test_state_continuity_matches_underlying_session(wave):
    """Adapter must equal driving VADStreamingSession directly with the same chunks (no state reset)."""
    s = VADStreamingSession(sample_rate=SR)
    direct = []
    for i in range(0, len(wave), 1000):
        direct += [(g.start_time_s, g.end_time_s) for g in s.process_chunk(wave[i:i + 1000]).segments]
    direct += [(g.start_time_s, g.end_time_s) for g in s.flush().segments]
    got, _ = run(wave, 1000)
    assert [(sp.start_sample, sp.end_sample) for sp in got] == [
        (round(a * SR), round(b * SR)) for a, b in direct
    ]


def test_timestamps_across_chunk_boundary(wave, reference):
    """Chunk edge placed inside a burst: spans must still equal the reference."""
    edge = int(1.4 * SR)  # inside burst 1 (1.0-1.8 s)
    a = VADIngestAdapter(sample_rate=SR)
    spans = a.feed(wave[:edge], 0) + a.feed(wave[edge:], edge) + a.flush()
    assert spans == reference
    assert all(sp.start_sample % 80 == 0 and sp.end_sample % 80 == 0 for sp in spans)  # 5 ms hop grid


def test_no_sample_processed_twice(wave):
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(wave[:32000], 0)
    assert a.samples_processed == 32000 == a.next_sample_index
    # Overlapping window replay (what a sliding window would do) must be rejected, state untouched.
    with pytest.raises(SampleDiscontinuityError) as e:
        a.feed(wave[24000:56000], 24000)
    assert e.value.kind == "overlap" and e.value.expected == 32000 and e.value.received == 24000
    assert a.samples_processed == 32000 and a.next_sample_index == 32000
    a.feed(wave[32000:], 32000)  # still usable and exactly-once afterwards
    assert a.samples_processed == len(wave)


def test_exactly_once_total_over_many_chunks(wave):
    _, a = run(wave, 997)
    assert a.samples_processed == len(wave) == a.next_sample_index


def test_flush_does_not_invent_duration(reference):
    # Speech runs to the very last sample: flush must close it without extending the timeline.
    x = synth(2.0, bursts=[(1.0, 2.0)])
    a = VADIngestAdapter(sample_rate=SR)
    spans = a.feed(x, 0)
    n_before = a.next_sample_index
    spans += a.flush()
    assert a.next_sample_index == n_before == len(x) and a.samples_processed == len(x)
    assert spans, "open speech at end of stream must be finalized by flush"
    assert all(sp.end_sample <= len(x) for sp in spans)
    assert a.flush() == []  # idempotent
    with pytest.raises(RuntimeError):
        a.feed(np.zeros(10, np.float32), len(x))


def test_empty_stream_and_empty_chunk():
    a = VADIngestAdapter(sample_rate=SR)
    assert a.feed(np.empty(0, np.float32), 0) == []
    assert a.next_sample_index == 0
    assert a.flush() == []
    assert a.next_sample_index == 0


@pytest.mark.parametrize("bad", [0, 44100, -8000, 48000])
def test_invalid_sample_rate_value(bad):
    with pytest.raises(ValueError):
        VADIngestAdapter(sample_rate=bad)


@pytest.mark.parametrize("bad", [16000.0, "16000", None, True])
def test_invalid_sample_rate_type(bad):
    with pytest.raises(TypeError):
        VADIngestAdapter(sample_rate=bad)


def test_supported_sample_rates_8k_and_16k():
    for sr in (8000, 16000):
        x = synth(2.0, bursts=[(0.6, 1.2)], sr=sr)
        spans, a = run(x, 400, sr=sr)
        assert a.samples_processed == len(x)
        assert all(sp.end_sample <= len(x) for sp in spans)


def test_gap_is_explicit_and_resync_keeps_absolute_timeline(wave):
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(wave[:40000], 0)  # 2.5 s
    with pytest.raises(SampleDiscontinuityError) as e:
        a.feed(wave[44000:60000], 44000)  # 4000 samples were dropped upstream
    assert e.value.kind == "gap" and e.value.expected == 40000 and e.value.received == 44000
    assert a.next_sample_index == 40000  # rejected feed changed nothing
    a.resync(44000)  # 2.75 s: silence, 250 ms before the 3.0 s burst (warm-up needs 200 ms)
    assert a.epoch == 1 and a.next_sample_index == 44000 and a.samples_processed == 0
    spans = a.feed(wave[44000:], 44000) + a.flush()
    # Spans after the gap are on the ABSOLUTE timeline, not relative to the resync point.
    assert len(spans) == 1 and spans[0].epoch == 1
    assert abs(spans[0].start_sample / SR - 3.0) < 0.1
    assert spans[0].end_sample <= len(wave)


def test_resync_restarts_noise_warmup_characterization(wave):
    """
    KNOWN LIMITATION (documented, not a defect of the adapter): resync() restarts the VAD state, so
    the first init_frames (200 ms) are warm-up. Speech that starts AT the resync point is learned as
    noise and not detected (same behaviour as a fresh session; see PHASE_6A contract).
    """
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(wave[:32000], 0)
    a.resync(48000)  # burst 3.0-3.6 s starts exactly at sample 48000
    spans = a.feed(wave[48000:], 48000) + a.flush()
    assert spans == []



def test_resync_cannot_move_backwards(wave):
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(wave[:32000], 0)
    with pytest.raises(ValueError):
        a.resync(1000)
    assert a.next_sample_index == 32000 and a.epoch == 0


def test_resync_finalizes_pending_speech_of_old_epoch():
    x = synth(2.0, bursts=[(1.0, 2.0)])
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(x, 0)
    flushed = a.resync(len(x) + 16000)
    assert flushed and all(sp.epoch == 0 and sp.end_sample <= len(x) for sp in flushed)


@pytest.mark.parametrize("idx,exc", [(1.5, TypeError), ("0", TypeError), (None, TypeError),
                                    (True, TypeError), (-1, ValueError)])
def test_invalid_first_sample_index(idx, exc):
    with pytest.raises(exc):
        VADIngestAdapter(sample_rate=SR).feed(np.zeros(10, np.float32), idx)


def test_invalid_audio_rejected_without_state_change():
    a = VADIngestAdapter(sample_rate=SR)
    for bad, exc in [
        (None, TypeError),
        (np.zeros((2, 10), np.float32), ValueError),
        (np.array([0.0, np.nan], np.float32), ValueError),
        (np.array([0.0, np.inf], np.float32), ValueError),
        (np.zeros(10, np.int32), TypeError),
        (np.zeros(10, np.complex64), ValueError),
    ]:
        with pytest.raises(exc):
            a.feed(bad, 0)
    assert a.next_sample_index == 0 and a.samples_processed == 0


def test_int16_pcm_is_normalised_like_float(wave):
    pcm = np.clip(np.round(wave * 32768.0), -32768, 32767).astype(np.int16)
    as_float = (pcm.astype(np.float32) / 32768.0)
    a = run(pcm, 1600)[0]
    b = run(as_float, 1600)[0]
    assert a == b and len(a) == len(BURSTS_S)


def test_reset_starts_independent_stream(wave, reference):
    a = VADIngestAdapter(sample_rate=SR)
    a.feed(wave, 0)
    a.flush()
    a.reset()
    assert a.next_sample_index == 0 and a.epoch == 0 and not a.is_flushed
    spans = a.feed(wave, 0) + a.flush()
    assert spans == reference


def test_speech_span_validation():
    with pytest.raises(ValueError):
        SpeechSpan(5, 5)
    with pytest.raises(ValueError):
        SpeechSpan(-1, 5)
    assert SpeechSpan(0, 80).num_samples == 80


def test_session_sample_rate_mismatch_rejected():
    with pytest.raises(ValueError):
        VADIngestAdapter(sample_rate=16000, session=VADStreamingSession(sample_rate=8000))
