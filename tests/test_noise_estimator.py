import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAD.feature_extraction import FeatureExtractor, VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseConfig, NoiseEstimator  # noqa: E402

BINS = np.array([3, 11, 16, 32])


def noise_bp(T, level, rng):
    level = np.broadcast_to(np.asarray(level, dtype=float), (4,))
    g = rng.gamma(BINS[:, None], 1.0 / BINS[:, None], size=(4, T))
    return (level[:, None] * g).astype(np.float32)


def feats(bp, sfm=None):
    T = bp.shape[1]
    if sfm is None:
        sfm = np.full(T, 0.8, dtype=np.float32)
    return VADFeatures(band_power=bp.astype(np.float32), sfm=np.asarray(sfm, np.float32))


LEVEL = np.array([1e-4, 5e-5, 2e-5, 1e-5])


def test_stationary_convergence():
    rng = np.random.default_rng(0)
    est = NoiseEstimator()
    out = est.update(feats(noise_bp(4000, LEVEL, rng)))
    final = out.noise_floor[:, -500:].mean(axis=1)
    assert np.all(np.abs(final / LEVEL - 1) < 0.15)
    assert abs(out.snr_db[:, 500:].mean()) < 1.5


def test_warmup_flags_and_median():
    rng = np.random.default_rng(1)
    est = NoiseEstimator()
    bp = noise_bp(100, LEVEL, rng)
    out = est.update(feats(bp))
    assert out.warmup[:40].all() and not out.warmup[40:].any()
    assert (out.snr_db[:, :40] == 0).all()
    np.testing.assert_allclose(out.noise_floor[:, 40], np.median(bp[:, :40], axis=1), rtol=1e-5)


def test_speech_in_one_band_freezes_only_that_band():
    rng = np.random.default_rng(2)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    bp = noise_bp(200, LEVEL, rng)
    bp[1] *= 1000.0  # +30 dB chỉ ở băng 1, phổ không phẳng
    f_before = est.noise_floor.copy()
    out = est.update(feats(bp, np.full(200, 0.1)))
    assert not out.update_mask[1].any()
    assert out.update_mask[[0, 2, 3]].mean() > 0.8
    assert est.noise_floor[1] == pytest.approx(f_before[1])
    assert out.snr_db[1].mean() > 20


def test_high_sfm_with_high_snr_is_ambiguous_no_update():
    rng = np.random.default_rng(3)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    bp = noise_bp(100, LEVEL, rng) * 50.0  # ~+17 dB, SFM cao
    out = est.update(feats(bp, np.full(100, 0.9)))
    assert not out.update_mask[:, 1:].any() or out.update_mask[:, 1:].mean() < 0.05


def test_noise_floor_used_is_before_update():
    rng = np.random.default_rng(4)
    est = NoiseEstimator()
    bp = noise_bp(300, LEVEL, rng)
    out = est.update(feats(bp))
    expected = np.clip(10 * np.log10(bp[:, 40:] / out.noise_floor[:, 40:]), -10, 40)
    np.testing.assert_allclose(out.snr_db[:, 40:], expected, atol=1e-3)
    # floor tại frame t+1 phải khác t khi băng được cập nhật (cập nhật xảy ra sau khi tính SNR)
    changed = out.noise_floor[:, 41:] != out.noise_floor[:, 40:-1]
    np.testing.assert_array_equal(changed, out.update_mask[:, 40:-1])


def test_continuous_speech_does_not_raise_floor():
    rng = np.random.default_rng(5)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    bp = noise_bp(1500, LEVEL, rng) * 300.0  # 7.5 s nói liên tục, SFM thấp
    out = est.update(feats(bp, np.full(1500, 0.1)))
    assert out.stuck[:, -1].all()
    assert not out.update_mask.any()
    assert np.all(est.noise_floor < LEVEL * 1.5)


def test_speech_from_start_does_not_runaway_and_recovers_on_silence():
    rng = np.random.default_rng(50)
    est = NoiseEstimator()
    speech_level = LEVEL * 100.0  # +20 dB ngay từ frame 0
    out_sp = est.update(feats(noise_bp(600, speech_level, rng), np.full(600, 0.15)))
    init_floor = out_sp.noise_floor[:, 40]
    # Warm-up bằng median khởi tạo cao theo tiếng nói (~speech_level), nhưng không leo vượt mức tiếng nói
    assert np.all(init_floor > LEVEL * 50.0)
    assert np.all(est.noise_floor <= speech_level * 1.2)
    # Khi hết câu nói, quay về nền nhiễu thật -> hạ nhanh về LEVEL trên từng băng
    est.update(feats(noise_bp(400, LEVEL, rng), np.full(400, 0.8)))
    assert np.all(np.abs(est.noise_floor / LEVEL - 1.0) < 0.35)


def test_prolonged_high_sfm_speech_does_not_raise_floor():
    rng = np.random.default_rng(51)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    floor_ref = est.noise_floor.copy()

    # Kịch bản A: 450 frame nguyên âm (SFM thấp) + 120 frame (0.6 s) âm xát SFM cao ổn định (+20 dB)
    # Tổng đóng băng 570 frame (> max_freeze_frames=500), nhưng phần SFM cao sau khi stuck < 100 frame
    bp_a = noise_bp(570, LEVEL * 100.0, rng)
    sfm_a = np.concatenate([np.full(450, 0.1), np.full(120, 0.75)])
    out_a = est.update(feats(bp_a, sfm_a))
    assert out_a.stuck[:, -1].all()
    assert np.all(est.noise_floor < floor_ref * 1.5)

    # Kịch bản B: 1500 frame (7.5 s) tiếng nói liên tục xen kẽ nguyên âm (40 frame, SFM=0.15)
    # và âm xát có SFM cao + công suất ổn định (80 frame = 0.4 s, SFM=0.75)
    est.reset()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    bp_b = noise_bp(1500, LEVEL * 100.0, rng)
    sfm_b = np.tile(np.concatenate([np.full(40, 0.15), np.full(80, 0.75)]), 13)[:1500]
    out_b = est.update(feats(bp_b, sfm_b))
    assert out_b.stuck[:, -1].all()
    # Theo dõi riêng từng băng: không băng nào bị kéo noise floor lên theo tiếng nói
    assert np.all(est.noise_floor < LEVEL * 1.5)

    # Kịch bản C: Tín hiệu có SFM cao (0.8) và công suất +20 dB suốt max_freeze_frames (500 frame):
    # Trước khi stuck, _ref KHÔNG được chạy theo tín hiệu công suất cao
    est.reset()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    floor_before_c = est.noise_floor.copy()
    est.update(feats(np.broadcast_to((LEVEL * 100.0)[:, None], (4, 500)).astype(np.float32),
                     np.full(500, 0.8)))
    np.testing.assert_allclose(est._ref, floor_before_c, rtol=1e-5)


def test_noise_step_up_recovers_when_stable_flat():
    for seed in (6, 44, 100):
        rng = np.random.default_rng(seed)
        est = NoiseEstimator()
        est.update(feats(noise_bp(1000, LEVEL, rng)))
        new = LEVEL * 100.0  # +20 dB, phẳng và ổn định
        # Kiểm tra trong 1500 frame (7.5 s) trên nhiều seed
        out = est.update(feats(noise_bp(1500, new, rng)))
        assert out.stuck.any()
        # Trước khi đủ max_freeze_frames (500 frame đầu), chưa băng nào tự ý nhảy lên nền mới
        assert np.all(out.noise_floor[:, 500] < LEVEL * 2.0)
        # Sau 7.5 s nhiễu phẳng ổn định, cả 4 băng (kể cả băng 0 có 3 bin) đều phục hồi về mức mới
        assert np.all((est.noise_floor > new * 0.6) & (est.noise_floor < new * 1.5))


def test_noise_step_down_tracks_fast():
    rng = np.random.default_rng(7)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    out = est.update(feats(noise_bp(300, LEVEL / 10, rng)))
    assert np.all(out.noise_floor[:, -1] < LEVEL * 0.3)


def test_moderate_snr_15db_speech_gamma_multi_seed():
    """Hồi quy: Tiếng nói +15 dB (31.62x), SFM=0.1, biến thiên Gamma trên 10 seed cố định.
    Băng 0 (3 bin) không được lọt cổng strong đơn lẻ làm leo noise floor."""
    mult_15db = 10.0 ** (15.0 / 10.0)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        est = NoiseEstimator()
        est.update(feats(noise_bp(1000, LEVEL, rng)))
        floor_before = est.noise_floor.copy()
        bp_sp = noise_bp(1500, LEVEL * mult_15db, rng)
        out = est.update(feats(bp_sp, np.full(1500, 0.1)))
        assert out.update_mask[0].sum() == 0, f"seed {seed} updated band 0"
        assert np.all(est.noise_floor <= floor_before * 1.05)
        assert out.stuck[:, -1].all()


def test_noise_bands_update_after_confirm_during_single_band_speech():
    """Hồi quy: Khi băng 1 có tiếng nói làm SFM toàn frame thấp (0.1), các băng còn lại
    chứa nhiễu nền thật vẫn bắt đầu cập nhật ngay sau strong_confirm_frames."""
    rng = np.random.default_rng(22)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    # Đưa trước 1 frame SNR cao ở cả 4 băng để reset _strong_run về 0
    est.update(feats( (LEVEL[:, None] * 20.0).astype(np.float32), np.array([0.1], np.float32)))

    # Băng 1 có tiếng nói (+30 dB), băng 0, 2, 3 ở mức nhiễu nền thấp (0.5 * LEVEL) với SFM=0.1
    bp = np.broadcast_to((LEVEL * 0.5)[:, None], (4, 50)).copy().astype(np.float32)
    bp[1, :] = LEVEL[1] * 1000.0
    out = est.update(feats(bp, np.full(50, 0.1)))
    k = est.config.strong_confirm_frames
    # Trong k-1 frame đầu: chờ xác nhận nên chưa cập nhật; từ frame k-1 trở đi: cập nhật liên tục
    assert not out.update_mask[[0, 2, 3], :k - 1].any()
    assert out.update_mask[[0, 2, 3], k - 1:].all()
    assert not out.update_mask[1].any()


def test_step_down_confirm_delay_is_minimal():
    """Hồi quy: Nhiễu giảm bậc vẫn bám xuống nhanh, chỉ trễ đúng strong_confirm_frames - 1 frame."""
    rng = np.random.default_rng(77)
    est = NoiseEstimator()
    est.update(feats(noise_bp(1000, LEVEL, rng)))
    # Reset _strong_run bằng 1 frame công suất cao
    est.update(feats((LEVEL[:, None] * 20.0).astype(np.float32), np.array([0.1], np.float32)))

    low_bp = noise_bp(60, LEVEL / 10.0, rng)
    out = est.update(feats(low_bp, np.full(60, 0.8)))
    k = est.config.strong_confirm_frames
    # Ngay tại frame index k-1 (frame thứ 3, tức 15 ms), cổng strong mở trên cả 4 băng
    assert out.update_mask[:, k - 1].all()
    # Sau 60 frame (300 ms), noise floor đã giảm sâu dưới 25% mức cũ
    assert np.all(est.noise_floor < LEVEL * 0.25)


@pytest.mark.parametrize("chunk", [1, 7, 80, 1000])
def test_chunk_invariance(chunk):
    rng = np.random.default_rng(8)
    bp = noise_bp(3000, LEVEL, rng)
    bp[2, 1500:1800] *= 500
    sfm = rng.uniform(0, 1, 3000).astype(np.float32)
    ref = NoiseEstimator().update(feats(bp, sfm))
    est = NoiseEstimator()
    parts = [est.update(feats(bp[:, i:i + chunk], sfm[i:i + chunk]))
             for i in range(0, 3000, chunk)]
    for name in ("noise_floor", "snr_db", "update_mask", "stuck", "warmup", "is_noise"):
        got = np.concatenate([getattr(p, name) for p in parts], axis=-1)
        np.testing.assert_array_equal(got, getattr(ref, name))


def test_empty_and_dtypes():
    est = NoiseEstimator()
    out = est.update(feats(np.zeros((4, 0), np.float32)))
    assert out.noise_floor.shape == (4, 0) and out.noise_floor.dtype == np.float32
    assert out.snr_db.shape == (4, 0) and out.snr_db.dtype == np.float32
    assert out.update_mask.shape == (4, 0) and out.update_mask.dtype == bool
    assert out.warmup.shape == (0,) and out.is_noise.shape == (0,)
    full = est.update(feats(noise_bp(50, LEVEL, np.random.default_rng(9))))
    assert full.noise_floor.dtype == np.float32 and full.snr_db.dtype == np.float32


def test_reset_repeatable():
    rng = np.random.default_rng(10)
    bp = noise_bp(500, LEVEL, rng)
    est = NoiseEstimator()
    a = est.update(feats(bp))
    est.reset()
    b = est.update(feats(bp))
    np.testing.assert_array_equal(a.noise_floor, b.noise_floor)


def test_snr_clipped_and_zero_power_safe():
    est = NoiseEstimator()
    rng = np.random.default_rng(11)
    est.update(feats(noise_bp(100, LEVEL, rng)))
    bp = np.zeros((4, 10), np.float32)
    bp[:, 5:] = 1.0
    out = est.update(feats(bp, np.full(10, 0.1)))
    assert np.isfinite(out.snr_db).all()
    assert out.snr_db.min() >= -10.0 and out.snr_db.max() <= 40.0


def test_invalid_inputs():
    est = NoiseEstimator()
    with pytest.raises(ValueError):
        est.update(VADFeatures(np.zeros((3, 5), np.float32), np.zeros(5, np.float32)))
    with pytest.raises(ValueError):
        est.update(VADFeatures(np.zeros((4, 5), np.float32), np.zeros(4, np.float32)))
    with pytest.raises(ValueError):
        NoiseEstimator(config=NoiseConfig(alpha_slow=0.5, alpha_fast=0.1))
    dbg = NoiseEstimator(debug=True)
    with pytest.raises(ValueError):
        dbg.update(VADFeatures(np.full((4, 5), np.nan, np.float32), np.zeros(5, np.float32)))


def test_end_to_end_with_feature_extractor():
    sr = 16000
    rng = np.random.default_rng(12)
    audio = (rng.standard_normal(sr * 3) * 0.01).astype(np.float32)
    t = np.arange(sr) / sr
    audio[sr:2 * sr] += 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    fe = FeatureExtractor(sample_rate=sr)
    est = NoiseEstimator(num_bands=fe.num_bands)
    outs = []
    for i in range(0, len(audio), 1600):
        f = fe.process_chunk(audio[i:i + 1600])
        outs.append(est.update(f))
    snr = np.concatenate([o.snr_db for o in outs], axis=1)
    warm = np.concatenate([o.warmup for o in outs])
    assert warm[:40].all()
    hop = int(round(1 / fe.hop_seconds))
    tone = snr[1, int(1.2 * hop):int(1.8 * hop)].mean()
    quiet = snr[1, int(2.5 * hop):].mean()
    assert tone > 20 and abs(quiet) < 3


def test_noise_step_up_recovers_on_colored_noise_with_realistic_sfm():
    """Hồi quy: Nhiễu màu thực tế có SFM dao động trong [0.16, 0.32] (dưới 0.5), khi tăng bậc
    +17 dB (50x) và giữ ổn định, cơ chế phục hồi không bị reset về 0 liên tục mà phải tích lũy
    và kéo noise floor ở tất cả các băng (kể cả băng 0) phục hồi về mức nền mới."""
    rng = np.random.default_rng(42)
    est = NoiseEstimator()
    bp_init = noise_bp(1000, LEVEL, rng)
    sfm_init = rng.uniform(0.16, 0.32, 1000).astype(np.float32)
    est.update(feats(bp_init, sfm_init))

    new_level = LEVEL * 50.0  # +17 dB
    bp_step = noise_bp(1500, new_level, rng)
    sfm_step = rng.uniform(0.16, 0.32, 1500).astype(np.float32)
    out = est.update(feats(bp_step, sfm_step))

    assert out.stuck.any()
    # Sau 1500 frame (7.5 s) nhiễu mới, băng 0 và các băng khác đều phải phục hồi về mức nền mới (> 60% new_level)
    assert np.all(est.noise_floor > new_level * 0.6)
    assert np.all(est.noise_floor < new_level * 1.6)

