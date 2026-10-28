import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAD.feature_extraction import (  # noqa: E402
    FeatureExtractor, build_band_masks, bin_frequencies,
)

SR = 16000


def _bin_counts():
    W = build_band_masks(256, SR)
    return [(W[i] > 0).sum() for i in range(W.shape[0])]


def test_bin_counts_and_first_band_starts_at_125hz():
    assert _bin_counts() == [3, 11, 16, 32]
    f = bin_frequencies(256, SR)
    W = build_band_masks(256, SR)
    assert f[W[0] > 0][0] == 125.0


def test_band_power_is_mean_per_bin():
    ex = FeatureExtractor(debug=True)
    spec = np.full((129, 4), 2.0, dtype=np.float32)
    fe = ex.extract(spec)
    np.testing.assert_allclose(fe.band_power, 2.0, rtol=1e-5)


def test_empty_contract():
    ex = FeatureExtractor(debug=True)
    fe = ex.process_chunk(np.zeros(10, dtype=np.float32))  # chua du 1 frame
    assert fe.band_power.shape == (4, 0) and fe.band_power.dtype == np.float32
    assert fe.sfm.shape == (0,) and fe.sfm.dtype == np.float32


def test_invalid_ranges():
    with pytest.raises(ValueError):
        FeatureExtractor(bands_hz=[(1000.0, 9000.0)])      # > Nyquist
    with pytest.raises(ValueError):
        FeatureExtractor(sfm_range_hz=(80.0, 9000.0))
    with pytest.raises(ValueError):
        FeatureExtractor(sfm_range_hz=(100.0, 120.0))      # < 2 bin


def test_debug_rejects_nan_inf_negative():
    ex = FeatureExtractor(debug=True)
    for bad in (np.nan, np.inf, -1.0):
        spec = np.ones((129, 2), dtype=np.float32)
        spec[5, 0] = bad
        with pytest.raises(ValueError):
            ex.extract(spec)


def test_outputs_finite_and_sfm_range_streaming():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32) * 0.1
    ex = FeatureExtractor(debug=True)
    fe = ex.process_chunk(x)
    assert np.isfinite(fe.band_power).all() and (fe.band_power >= 0).all()
    assert ((fe.sfm >= 0) & (fe.sfm <= 1)).all()
    # silence tuyet doi -> SFM = 1, khong NaN
    z = ex.extract(np.zeros((129, 3), dtype=np.float32))
    assert np.isfinite(z.sfm).all()


def test_batch_matches_streaming_any_chunking():
    from Dictionary.STFT import StreamingSTFT
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR // 2 + 123).astype(np.float32) * 0.1
    b = FeatureExtractor(debug=True).process_batch(x)
    for size in (1, 37, 80, 255, 1000, len(x)):
        ex = FeatureExtractor(debug=True)
        parts = [ex.process_chunk(x[i:i + size]) for i in range(0, len(x), size)]
        bp = np.concatenate([p.band_power for p in parts], axis=1)
        sf = np.concatenate([p.sfm for p in parts])
        assert bp.shape == b.band_power.shape, size
        np.testing.assert_allclose(bp, b.band_power, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(sf, b.sfm, rtol=1e-4, atol=1e-6)


def test_batch_short_audio_contract():
    fe = FeatureExtractor().process_batch(np.zeros(50, dtype=np.float32))
    assert fe.band_power.shape == (4, 0) and fe.sfm.shape == (0,)


def test_external_stft_must_match_sample_rate():
    from Dictionary.STFT import StreamingSTFT, create_vad_streamer
    with pytest.raises(ValueError):
        FeatureExtractor(sample_rate=8000, stft=create_vad_streamer(16000))
    with pytest.raises(ValueError):
        FeatureExtractor(stft=StreamingSTFT(n_fft=512, hop_length=160, win_length=400))
    FeatureExtractor(sample_rate=8000, stft=create_vad_streamer(8000))


# ---------------------------------------------------------------------------
# Ða sample rate: 8 kHz (tho?i) và 16 kHz dùng chung engine STFT + T?ng 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sr, n_fft, hop, n_bins", [(8000, 128, 40, 65), (16000, 256, 80, 129)])
def test_vad_preset_per_sample_rate(sr, n_fft, hop, n_bins):
    from Dictionary.STFT import vad_stft_params
    p = vad_stft_params(sr)
    assert p == {"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}
    ex = FeatureExtractor(sample_rate=sr)
    assert ex.n_bins == n_bins
    assert sr / ex.n_fft == 62.5                     # cùng d? phân gi?i Hz
    assert abs(ex.hop_seconds - 0.005) < 1e-12


@pytest.mark.parametrize("sr", [8000, 16000])
def test_band_mapping_identical_across_sample_rates(sr):
    ex = FeatureExtractor(sample_rate=sr)
    f = bin_frequencies(ex.n_fft, sr)
    got = [(int((w > 0).sum()), f[w > 0][0], f[w > 0][-1]) for w in ex.band_weights]
    assert got == [(3, 125.0, 250.0), (11, 312.5, 937.5),
                   (16, 1000.0, 1937.5), (32, 2000.0, 3937.5)]
    assert int(ex.sfm_mask.sum()) == 62


@pytest.mark.parametrize("sr", [8000, 16000])
def test_batch_streaming_parity_per_sample_rate(sr):
    rng = np.random.default_rng(sr)
    x = rng.standard_normal(sr // 3 + 17).astype(np.float32) * 0.1
    b = FeatureExtractor(sample_rate=sr, debug=True).process_batch(x)
    for size in (1, 33, 40, 127, 500):
        ex = FeatureExtractor(sample_rate=sr, debug=True)
        parts = [ex.process_chunk(x[i:i + size]) for i in range(0, len(x), size)]
        bp = np.concatenate([p.band_power for p in parts], axis=1)
        sf = np.concatenate([p.sfm for p in parts])
        assert bp.shape == b.band_power.shape == (4, 1 + (len(x) - ex.n_fft) // ex.stft.hop_length)
        np.testing.assert_allclose(bp, b.band_power, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(sf, b.sfm, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_same_tone_lands_in_same_band(sr):
    # Tone 1500 Hz ph?i roi vào d?i 3 (1000-2000 Hz) ? c? hai sample rate
    t = np.arange(sr // 2) / sr
    x = (0.5 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
    fe = FeatureExtractor(sample_rate=sr).process_batch(x)
    assert (fe.band_power.argmax(axis=0) == 2).all()
    assert fe.sfm.mean() < 0.1


def test_8k_band_above_nyquist_rejected():
    with pytest.raises(ValueError):
        FeatureExtractor(sample_rate=8000, bands_hz=[(2000.0, 4500.0)])
