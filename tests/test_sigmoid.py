import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAD.feature_extraction import FeatureExtractor, VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseEstimator, NoiseOutput       # noqa: E402
from VAD.sigmoid import (                                         # noqa: E402
    SigmoidConfig,
    SigmoidOutput,
    SigmoidScorer,
    normalize_snr_db,
    stable_sigmoid,
)


def test_snr_normalization_scale_and_clipping():
    """Kiểm tra chuẩn hóa SNR từ dB về đúng đoạn [0, 1] theo ngưỡng cấu hình."""
    snr_db = np.array([[-10.0, 0.0, 10.0, 20.0, 40.0]], dtype=np.float32)
    norm = normalize_snr_db(snr_db, snr_low_db=0.0, snr_high_db=20.0)
    expected = np.array([[0.0, 0.0, 0.5, 1.0, 1.0]], dtype=np.float32)
    assert norm.dtype == np.float32
    np.testing.assert_allclose(norm, expected, atol=1e-6)


def test_linear_combination_and_sigmoid_exact_formula():
    """Kiểm tra công thức z(t) = b + sum(w_j * SNR_norm_j) + w_f * (1 - SFM) và p(t) = sigmoid(z)."""
    cfg = SigmoidConfig(
        band_weights=(1.0, 2.0, 2.0, 1.0),
        sfm_weight=2.0,
        bias=-4.0,
        snr_low_db=0.0,
        snr_high_db=20.0,
    )
    scorer = SigmoidScorer(num_bands=4, config=cfg)

    snr_db = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
        ],
        dtype=np.float32,
    )
    sfm = np.array([0.85, 0.50, 0.10], dtype=np.float32)

    out = scorer.score_arrays(snr_db, sfm)
    expected_z = np.array([-3.7, 0.0, 3.8], dtype=np.float32)
    expected_p = 1.0 / (1.0 + np.exp(-expected_z.astype(np.float64)))

    np.testing.assert_allclose(out.logit, expected_z, atol=1e-6)
    np.testing.assert_allclose(out.score, expected_p.astype(np.float32), atol=1e-6)
    assert out.score[0] < 0.03
    assert out.score[1] == pytest.approx(0.5, abs=1e-6)
    assert out.score[2] > 0.97


def test_warmup_contract_handling():
    """Bắt buộc: Khi warmup=True, không được tính z(t) từ snr_db=0.0 giả và SFM thấp."""
    scorer = SigmoidScorer()
    snr_db = np.zeros((4, 5), dtype=np.float32)
    snr_db[:, 3:] = 20.0
    sfm = np.zeros(5, dtype=np.float32)
    warmup = np.array([True, True, True, False, False], dtype=bool)

    out = scorer.score_arrays(snr_db, sfm, warmup=warmup)

    assert np.array_equal(out.warmup, warmup)
    np.testing.assert_array_equal(out.score[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(out.snr_norm[:, :3], np.zeros((4, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        out.logit[:3], np.full(3, scorer.config.warmup_logit, dtype=np.float32)
    )
    assert (out.score[3:] > 0.98).all()


def test_warmup_with_extreme_sfm_and_placeholder_or_corrupted_snr():
    """Phase 3: Kiểm tra các frame warmup có SFM cực đoan (-10, 0, +10, NaN khi debug=False)
    và SNR placeholder (0 dB hoặc giá trị cực lớn/NaN): đầu ra warmup vẫn phải bị khóa an toàn."""
    scorer = SigmoidScorer(debug=False)
    snr_db = np.array(
        [
            [0.0, 100.0, -50.0, np.nan, 20.0],
            [0.0, 100.0, -50.0, np.inf, 20.0],
            [0.0, 100.0, -50.0, -np.inf, 20.0],
            [0.0, 100.0, -50.0, 0.0, 20.0],
        ],
        dtype=np.float32,
    )
    sfm = np.array([0.0, -5.0, 10.0, np.nan, 0.1], dtype=np.float32)
    warmup = np.array([True, True, True, True, False], dtype=bool)

    out = scorer.score_arrays(snr_db, sfm, warmup=warmup)
    # 4 frame đầu là warmup: score, logit, snr_norm phải hữu hạn và đúng giá trị khóa
    assert np.isfinite(out.score[:4]).all()
    assert np.isfinite(out.logit[:4]).all()
    assert np.isfinite(out.snr_norm[:, :4]).all()
    np.testing.assert_array_equal(out.score[:4], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(
        out.logit[:4], np.full(4, scorer.config.warmup_logit, dtype=np.float32)
    )
    np.testing.assert_array_equal(out.snr_norm[:, :4], np.zeros((4, 4), dtype=np.float32))
    # Frame thứ 5 (index 4) ngoài warmup hoạt động bình thường
    assert out.score[4] > 0.97


def test_nan_and_inf_handling_when_debug_false():
    """Phase 3: Khi debug=False, kiểm tra hành vi với NaN và +/-Inf mà không gây RuntimeWarning."""
    scorer = SigmoidScorer(debug=False)
    # Frame 0: +Inf SNR -> clip về snr_norm = 1.0
    # Frame 1: -Inf SNR -> clip về snr_norm = 0.0
    # Frame 2: NaN SNR ở 1 băng -> frame 2 có score NaN, nhưng KHÔNG làm hỏng frame 0, 1, 3
    # Frame 3: bình thường (SNR = 15 dB, SFM = 0.2)
    snr_db = np.array(
        [
            [np.inf, -np.inf, np.nan, 15.0],
            [np.inf, -np.inf, 10.0, 15.0],
            [np.inf, -np.inf, 10.0, 15.0],
            [np.inf, -np.inf, 10.0, 15.0],
        ],
        dtype=np.float32,
    )
    sfm = np.array([0.1, 0.9, 0.2, 0.2], dtype=np.float32)

    out = scorer.score_arrays(snr_db, sfm)
    # Frame 0 (+Inf SNR) -> snr_norm = 1.0, hữu hạn
    np.testing.assert_array_equal(out.snr_norm[:, 0], np.ones(4, dtype=np.float32))
    assert np.isfinite(out.score[0]) and out.score[0] > 0.97
    # Frame 1 (-Inf SNR) -> snr_norm = 0.0, hữu hạn
    np.testing.assert_array_equal(out.snr_norm[:, 1], np.zeros(4, dtype=np.float32))
    assert np.isfinite(out.score[1]) and out.score[1] < 0.03
    # Frame 2 (NaN) -> NaN chỉ cô lập tại frame 2
    assert np.isnan(out.logit[2]) and np.isnan(out.score[2])
    # Frame 3 hoàn toàn không bị ảnh hưởng bởi frame 2
    assert np.isfinite(out.score[3]) and out.score[3] > 0.85


def test_invalid_snr_and_sfm_outside_unit_interval():
    """Phase 3: Kiểm tra SNR ngoài [-10, 40] dB và SFM ngoài [0, 1] ở chế độ debug=False vs debug=True."""
    scorer_prod = SigmoidScorer(debug=False)
    snr_extreme = np.array(
        [[-1000.0, 0.0, 1000.0], [-50.0, 10.0, 80.0], [-20.0, 10.0, 60.0], [-5.0, 10.0, 45.0]],
        dtype=np.float32,
    )
    sfm_out_of_bounds = np.array([-2.5, 0.5, 3.0], dtype=np.float32)

    out = scorer_prod.score_arrays(snr_extreme, sfm_out_of_bounds)
    # Khi debug=False: SFM được clip an toàn về [0, 1] -> flatness_complement ∈ [0, 1]
    np.testing.assert_allclose(
        out.flatness_complement, np.array([1.0, 0.5, 0.0], dtype=np.float32), atol=1e-6
    )
    assert np.all((out.snr_norm >= 0.0) & (out.snr_norm <= 1.0))
    assert np.all((out.score >= 0.0) & (out.score <= 1.0))
    assert np.isfinite(out.score).all()

    # Khi debug=True: SFM ngoài [0, 1] phải lập tức báo ValueError
    scorer_dbg = SigmoidScorer(debug=True)
    with pytest.raises(ValueError, match="sfm"):
        scorer_dbg.score_arrays(snr_extreme, sfm_out_of_bounds)


def test_monotonicity_with_snr_and_sfm():
    """Điểm số phải tăng đơn điệu khi SNR tăng (từng băng hoặc toàn băng) và giảm đơn điệu khi SFM tăng."""
    scorer = SigmoidScorer()
    T = 25
    snr_sweep = np.broadcast_to(np.linspace(-10.0, 30.0, T, dtype=np.float32), (4, T))
    sfm_const = np.full(T, 0.3, dtype=np.float32)
    out_snr = scorer.score_arrays(snr_sweep, sfm_const)
    assert np.all(np.diff(out_snr.score) >= -1e-7)
    assert out_snr.score[-1] > out_snr.score[0]

    # Kiểm tra tăng đơn điệu trên riêng từng băng j = 0, 1, 2, 3
    for band_idx in range(4):
        snr_single = np.zeros((4, T), dtype=np.float32)
        snr_single[band_idx, :] = np.linspace(0.0, 20.0, T, dtype=np.float32)
        out_b = scorer.score_arrays(snr_single, sfm_const)
        assert np.all(np.diff(out_b.logit) > 0.0)
        assert np.all(np.diff(out_b.score) >= 0.0)

    snr_const = np.full((4, T), 10.0, dtype=np.float32)
    sfm_sweep = np.linspace(0.0, 1.0, T, dtype=np.float32)
    out_sfm = scorer.score_arrays(snr_const, sfm_sweep)
    assert np.all(np.diff(out_sfm.score) <= 1e-7)
    assert out_sfm.score[0] > out_sfm.score[-1]


@pytest.mark.parametrize("chunk", [1, 7, 19, 40, 250])
def test_chunk_invariance_and_layer2_integration(chunk):
    """Kiểm tra tính bất biến theo kích thước chunk (kể cả chunk cắt ngang biên warm-up 40 frame)."""
    rng = np.random.default_rng(42)
    T = 400
    bp = rng.uniform(1e-5, 1e-4, size=(4, T)).astype(np.float32)
    bp[:, 150:280] *= 100.0
    sfm = rng.uniform(0.6, 0.95, size=T).astype(np.float32)
    sfm[150:280] = rng.uniform(0.05, 0.2, size=130).astype(np.float32)

    est_full = NoiseEstimator()
    scorer_full = SigmoidScorer()
    feats_full = VADFeatures(band_power=bp, sfm=sfm)
    noise_full = est_full.update(feats_full)
    out_full = scorer_full.score(feats_full, noise_full)

    est_stream = NoiseEstimator()
    scorer_stream = SigmoidScorer()
    parts = []
    # Chèn thêm các chunk rỗng T=0 xen kẽ để kiểm tra biên streaming thực tế
    for i in range(0, T, chunk):
        if i % (chunk * 3) == 0:
            f_empty = VADFeatures(
                band_power=np.zeros((4, 0), np.float32), sfm=np.zeros(0, np.float32)
            )
            n_empty = est_stream.update(f_empty)
            parts.append(scorer_stream.score(f_empty, n_empty))
        f_c = VADFeatures(band_power=bp[:, i:i + chunk], sfm=sfm[i:i + chunk])
        n_c = est_stream.update(f_c)
        parts.append(scorer_stream.score(f_c, n_c))

    for field in ("score", "logit", "snr_norm", "flatness_complement", "warmup"):
        got = np.concatenate([getattr(p, field) for p in parts], axis=-1)
        np.testing.assert_allclose(got, getattr(out_full, field), atol=1e-6)


def test_empty_frames_and_dtypes():
    """Kiểm tra biên T=0 (cả khi warmup=None và warmup rỗng) và kiểu dữ liệu đầu ra chuẩn."""
    scorer = SigmoidScorer()
    for warm_arg in (None, np.zeros(0, dtype=bool)):
        out = scorer.score_arrays(
            np.zeros((4, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            warm_arg,
        )
        assert out.num_frames == 0
        assert out.score.shape == (0,) and out.score.dtype == np.float32
        assert out.logit.shape == (0,) and out.logit.dtype == np.float32
        assert out.snr_norm.shape == (4, 0) and out.snr_norm.dtype == np.float32
        assert out.flatness_complement.shape == (0,) and out.flatness_complement.dtype == np.float32
        assert out.warmup.shape == (0,) and out.warmup.dtype == bool


def test_unconditional_and_debug_invalid_input_checks():
    """Kiểm tra đầy đủ các điều kiện bắt lỗi vô điều kiện (production) và khi debug=True."""
    with pytest.raises(ValueError):
        SigmoidScorer(num_bands=0)
    with pytest.raises(ValueError):
        SigmoidScorer(num_bands=4, config=SigmoidConfig(band_weights=(1.0, 1.0)))
    with pytest.raises(ValueError):
        SigmoidScorer(config=SigmoidConfig(snr_low_db=20.0, snr_high_db=10.0))
    with pytest.raises(ValueError):
        SigmoidScorer(config=SigmoidConfig(band_weights=(-1.0, 1.0, 1.0, 1.0)))
    with pytest.raises(ValueError):
        SigmoidScorer(config=SigmoidConfig(band_weights=(0.0, 0.0, 0.0, 0.0), sfm_weight=0.0))
    with pytest.raises(ValueError):
        SigmoidScorer(config=SigmoidConfig(warmup_score=1.5))

    # Kiểm tra vô điều kiện (ngay cả khi debug=False)
    scorer_prod = SigmoidScorer(num_bands=4, debug=False)
    with pytest.raises(ValueError):
        scorer_prod.score_arrays(np.zeros((3, 5), np.float32), np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        scorer_prod.score_arrays(np.zeros((4, 5), np.float32), np.zeros(4, np.float32))
    with pytest.raises(ValueError):
        scorer_prod.score_arrays(
            np.zeros((4, 5), np.float32),
            np.zeros(5, np.float32),
            warmup=np.zeros(3, dtype=bool),
        )
    with pytest.raises(ValueError):
        scorer_prod.score_arrays(
            np.zeros((4, 5), dtype=np.complex64), np.zeros(5, dtype=np.float32)
        )
    with pytest.raises(TypeError):
        scorer_prod.score("invalid_features", "invalid_noise")  # type: ignore[arg-type]

    # Kiểm tra khi debug=True
    scorer_dbg = SigmoidScorer(num_bands=4, debug=True)
    with pytest.raises(ValueError):
        scorer_dbg.score_arrays(np.full((4, 5), np.nan, np.float32), np.zeros(5, np.float32))
    with pytest.raises(ValueError):
        scorer_dbg.score_arrays(np.zeros((4, 5), np.float32), np.full(5, 1.5, np.float32))


def test_stable_sigmoid_extreme_logits():
    """Kiểm tra ổn định số học của hàm sigmoid với logit cực lớn/cực nhỏ, Inf và NaN."""
    z = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0, -np.inf, np.inf, np.nan], dtype=np.float32)
    p = stable_sigmoid(z)
    assert p.dtype == np.float32
    assert np.isfinite(p[:7]).all()
    assert p[0] == 0.0 and p[1] == pytest.approx(0.0, abs=1e-40)
    assert p[2] == pytest.approx(0.5)
    assert p[3] == 1.0 and p[4] == 1.0
    assert p[5] == 0.0 and p[6] == 1.0
    assert np.isnan(p[7])


def test_end_to_end_layer1_layer2_layer3_8khz_and_16khz():
    """Tích hợp Tầng 1 -> Tầng 2 -> Tầng 3 ở cả 8 kHz và 16 kHz trên tín hiệu tổng hợp có kiểm soát."""
    for sr in (16000, 8000):
        rng = np.random.default_rng(123)
        # 3 giây nhiễu nền nhỏ, chèn tổ hợp hài giả lập tiếng nói trong [1.0s, 2.0s]
        audio = (rng.standard_normal(sr * 3) * 0.005).astype(np.float32)
        t = np.arange(sr) / sr
        speech_like = (
            0.15 * np.sin(2 * np.pi * 200 * t)
            + 0.20 * np.sin(2 * np.pi * 600 * t)
            + 0.15 * np.sin(2 * np.pi * 1500 * t)
            + 0.10 * np.sin(2 * np.pi * 2500 * t)
        ).astype(np.float32)
        audio[sr:2 * sr] += speech_like

        fe = FeatureExtractor(sample_rate=sr)
        est = NoiseEstimator(num_bands=fe.num_bands)
        scorer = SigmoidScorer(num_bands=fe.num_bands)

        feats = fe.process_batch(audio)
        noise_out = est.update(feats)
        sig_out = scorer.score(feats, noise_out)

        assert sig_out.num_frames == feats.num_frames
        assert (sig_out.score[:40] == 0.0).all()
        hop_fps = int(round(1.0 / fe.hop_seconds))
        noise_region_score = sig_out.score[int(0.3 * hop_fps):int(0.8 * hop_fps)].mean()
        speech_region_score = sig_out.score[int(1.2 * hop_fps):int(1.8 * hop_fps)].mean()
        tail_noise_score = sig_out.score[int(2.4 * hop_fps):int(2.9 * hop_fps)].mean()

        assert noise_region_score < 0.15
        assert speech_region_score > 0.90
        assert tail_noise_score < 0.15
