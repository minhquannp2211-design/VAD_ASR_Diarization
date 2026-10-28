"""
VAD/sigmoid.py - TẦNG 3: CHẤM ĐIỂM TIẾNG NÓI THÍCH NGHI (ADAPTIVE SIGMOID SCORING)

Đầu vào :
    - snr_db: (n_bands, T) float32 từ NoiseOutput (Tầng 2)
    - sfm   : (T,) float32 từ VADFeatures (Tầng 1), trong [0, 1]
    - warmup: (T,) bool từ NoiseOutput (Tầng 2)
Đầu ra  :
    - SigmoidOutput (score, logit, snr_norm, flatness_complement, warmup)

CÔNG THỨC TOÁN HỌC & ĐỒNG BỘ THANG ĐO
    1. Chuẩn hóa SNR theo từng băng về [0, 1]:
       SNR_norm[j, t] = clip((snr_db[j, t] - snr_low_db) / (snr_high_db - snr_low_db), 0.0, 1.0)
    2. Độ có cấu trúc phổ (bù của độ phẳng phổ):
       flatness_complement[t] = clip(1.0 - sfm[t], 0.0, 1.0)   ∈ [0, 1]
    3. Tổ hợp tuyến tính (logit):
       z(t) = bias + sum_{j=1..n_bands} band_weights[j] * SNR_norm[j, t]
                   + sfm_weight * flatness_complement[t]
    4. Điểm Sigmoid (ổn định số học cho mọi z ∈ R):
       p(t) = 1 / (1 + exp(-z(t)))   ∈ [0, 1]

XỬ LÝ CỜ WARM-UP (HỢP ĐỒNG BẮT BUỘC TỪ TẦNG 2)
    Khi warmup[t] == True, snr_db[:, t] = 0.0 chỉ là giá trị giữ chỗ trong lúc học nền ban đầu,
    KHÔNG phải phép đo SNR = 0 dB thật. Tại các frame warmup:
      - Không đưa snr_db[:, t] = 0.0 vào công thức tổ hợp tuyến tính như phép đo hợp lệ.
      - Gán snr_norm[:, t] = 0.0, logit[t] = warmup_logit, score[t] = warmup_score (mặc định 0.0).
      - Truyền nguyên vẹn cờ warmup sang SigmoidOutput để Tầng 4 (State Machine) biết rõ.

LƯU Ý QUAN TRỌNG VỀ Ý NGHĨA ĐIỂM SỐ
    - score p(t) là ĐIỂM PHÁT HIỆN (detection score) trong [0, 1], KHÔNG mặc nhiên là xác suất
      hậu nghiệm P(speech | frame) đã hiệu chỉnh (calibrated probability).
    - Các trọng số và ngưỡng mặc định trong SigmoidConfig là BASELINE thiết kế từ quy ước vật lý
      của Tầng 1 & Tầng 2, chưa được huấn luyện hoặc hiệu chỉnh tối ưu trên tập audio có nhãn.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.feature_extraction import VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseOutput     # noqa: E402


@dataclass(frozen=True)
class SigmoidConfig:
    """
    Cấu hình cho bộ chấm điểm Adaptive Sigmoid (Tầng 3).
    Tất cả trọng số và ngưỡng đều có thể cấu hình độc lập.
    """
    band_weights: Tuple[float, ...] = (1.0, 2.0, 2.0, 1.0)
    sfm_weight: float = 2.0
    bias: float = -4.0
    snr_low_db: float = 0.0          # SNR <= snr_low_db  -> SNR_norm = 0.0
    snr_high_db: float = 20.0        # SNR >= snr_high_db -> SNR_norm = 1.0
    warmup_score: float = 0.0        # Điểm ép cố định cho các frame warmup=True
    warmup_logit: float = -20.0      # Logit đặt cho các frame warmup=True

    def validate(self, num_bands: Optional[int] = None) -> None:
        if len(self.band_weights) < 1:
            raise ValueError("band_weights không được rỗng")
        if num_bands is not None and len(self.band_weights) != num_bands:
            raise ValueError(
                f"Độ dài band_weights ({len(self.band_weights)}) phải khớp num_bands ({num_bands})"
            )
        w_arr = np.asarray(self.band_weights, dtype=np.float64)
        if not np.isfinite(w_arr).all():
            raise ValueError("band_weights chứa NaN hoặc Inf")
        if (w_arr < 0.0).any():
            raise ValueError("band_weights phải >= 0")
        if not np.isfinite(self.sfm_weight) or self.sfm_weight < 0.0:
            raise ValueError("sfm_weight phải hữu hạn và >= 0")
        if (w_arr.sum() + self.sfm_weight) <= 0.0:
            raise ValueError("Tổng trọng số (sum(band_weights) + sfm_weight) phải > 0")
        if not np.isfinite(self.bias):
            raise ValueError("bias phải hữu hạn")
        if not (np.isfinite(self.snr_low_db) and np.isfinite(self.snr_high_db)):
            raise ValueError("snr_low_db và snr_high_db phải hữu hạn")
        if self.snr_low_db >= self.snr_high_db:
            raise ValueError("snr_low_db phải < snr_high_db")
        if not np.isfinite(self.warmup_score) or not (0.0 <= self.warmup_score <= 1.0):
            raise ValueError("warmup_score phải hữu hạn và nằm trong [0, 1]")
        if not np.isfinite(self.warmup_logit):
            raise ValueError("warmup_logit phải hữu hạn")


@dataclass(frozen=True)
class SigmoidOutput:
    """
    Kết quả chấm điểm Tầng 3 cho chuỗi T frame.
    Lưu ý: `score` là điểm phát hiện chưa hiệu chỉnh xác suất (uncalibrated detection score).
    """
    score: np.ndarray                # (T,) float32 ∈ [0, 1]: điểm phát hiện tiếng nói p(t)
    logit: np.ndarray                # (T,) float32: giá trị tổ hợp tuyến tính z(t)
    snr_norm: np.ndarray             # (n_bands, T) float32 ∈ [0, 1]: SNR đã chuẩn hóa (=0 khi warmup)
    flatness_complement: np.ndarray  # (T,) float32 ∈ [0, 1]: 1 - clip(sfm, 0, 1)
    warmup: np.ndarray               # (T,) bool: truyền tiếp từ Tầng 2 sang Tầng 4

    @property
    def num_frames(self) -> int:
        return int(self.score.shape[0])


def normalize_snr_db(
    snr_db: np.ndarray,
    snr_low_db: float = 0.0,
    snr_high_db: float = 20.0,
) -> np.ndarray:
    """
    Chuẩn hóa tuyến tính từng phần (piecewise-linear) SNR (dB) về đoạn [0, 1]:
        SNR_norm = clip((snr_db - snr_low_db) / (snr_high_db - snr_low_db), 0.0, 1.0)
    Trả về mảng cùng shape với snr_db, dtype float32.
    """
    if not (np.isfinite(snr_low_db) and np.isfinite(snr_high_db)):
        raise ValueError("snr_low_db và snr_high_db phải hữu hạn")
    if snr_low_db >= snr_high_db:
        raise ValueError("snr_low_db phải < snr_high_db")
    snr_raw = np.asarray(snr_db)
    if np.iscomplexobj(snr_raw):
        raise ValueError("snr_db phải là mảng số thực")
    snr = snr_raw.astype(np.float64, copy=False)
    scale = float(snr_high_db - snr_low_db)
    norm = (snr - float(snr_low_db)) / scale
    np.clip(norm, 0.0, 1.0, out=norm)
    return norm.astype(np.float32, copy=False)


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Hàm Sigmoid ổn định số học 1 / (1 + exp(-z)), không tràn số (overflow) với |z| lớn.
    Bảo toàn NaN nếu đầu vào là NaN; z = +Inf -> 1.0, z = -Inf -> 0.0.
    Trả về mảng float32 trong [0, 1] (hoặc NaN tại vị trí z là NaN).
    """
    z_raw = np.asarray(z)
    if np.iscomplexobj(z_raw):
        raise ValueError("z phải là mảng số thực")
    z64 = z_raw.astype(np.float64, copy=False)
    out = np.empty_like(z64, dtype=np.float64)
    nan_mask = np.isnan(z64)
    pos_mask = (~nan_mask) & (z64 >= 0.0)
    neg_mask = (~nan_mask) & (z64 < 0.0)

    if nan_mask.any():
        out[nan_mask] = np.nan
    if pos_mask.any():
        exp_neg = np.exp(-z64[pos_mask])
        out[pos_mask] = 1.0 / (1.0 + exp_neg)
    if neg_mask.any():
        exp_pos = np.exp(z64[neg_mask])
        out[neg_mask] = exp_pos / (1.0 + exp_pos)
    return out.astype(np.float32, copy=False)


class SigmoidScorer:
    """
    Bộ chấm điểm Adaptive Sigmoid (Tầng 3).
    Hoạt động phi trạng thái (stateless) theo từng frame, đảm bảo bất biến hoàn toàn
    theo kích thước chunk (chunk-invariant), nhưng vẫn cung cấp reset() đồng bộ giao diện pipeline.
    """

    def __init__(
        self,
        num_bands: int = 4,
        config: Optional[SigmoidConfig] = None,
        debug: bool = False,
    ) -> None:
        if num_bands < 1:
            raise ValueError("num_bands phải >= 1")
        self.num_bands = int(num_bands)
        if config is None:
            if self.num_bands == 4:
                self.config = SigmoidConfig()
            else:
                self.config = SigmoidConfig(
                    band_weights=tuple(6.0 / self.num_bands for _ in range(self.num_bands))
                )
        else:
            self.config = config
        self.config.validate(num_bands=self.num_bands)
        self.debug = bool(debug)
        self._weights_col = np.asarray(self.config.band_weights, dtype=np.float64)

    def reset(self) -> None:
        """Tầng 3 không lưu trạng thái nối tiếp giữa các frame; giữ hàm để đồng bộ API pipeline."""
        return None

    def score_arrays(
        self,
        snr_db: np.ndarray,
        sfm: np.ndarray,
        warmup: Optional[np.ndarray] = None,
    ) -> SigmoidOutput:
        """
        Chấm điểm trực tiếp từ mảng snr_db (n_bands, T), sfm (T,) và warmup (T,).
        Cho phép kiểm thử độc lập Tầng 3 mà không bắt buộc khởi tạo Tầng 1 hay Tầng 2.
        """
        cfg = self.config
        n = self.num_bands

        snr_arr = np.asarray(snr_db)
        sfm_arr = np.asarray(sfm)

        if np.iscomplexobj(snr_arr) or np.iscomplexobj(sfm_arr):
            raise ValueError("snr_db và sfm phải là mảng số thực")
        if snr_arr.ndim != 2 or snr_arr.shape[0] != n:
            raise ValueError(f"snr_db phải có shape ({n}, T), nhận {snr_arr.shape}")
        T = int(snr_arr.shape[1])
        if sfm_arr.ndim != 1 or sfm_arr.shape[0] != T:
            raise ValueError(f"sfm phải có shape ({T},) khớp với snr_db, nhận {sfm_arr.shape}")

        if warmup is None:
            warm_arr = np.zeros(T, dtype=bool)
        else:
            warm_raw = np.asarray(warmup)
            if np.iscomplexobj(warm_raw):
                raise ValueError("warmup không được là mảng số phức")
            if warm_raw.ndim != 1 or warm_raw.shape[0] != T:
                raise ValueError(
                    f"warmup phải có shape ({T},) khớp với snr_db, nhận {warm_raw.shape}"
                )
            if self.debug and T > 0 and not np.isfinite(warm_raw).all():
                raise ValueError("warmup chứa NaN hoặc Inf")
            warm_arr = warm_raw.astype(bool, copy=True)

        if self.debug and T > 0:
            if not np.isfinite(snr_arr).all():
                raise ValueError("snr_db chứa NaN hoặc Inf")
            if not np.isfinite(sfm_arr).all():
                raise ValueError("sfm chứa NaN hoặc Inf")
            if (sfm_arr < 0.0).any() or (sfm_arr > 1.0).any():
                raise ValueError("sfm chứa giá trị ngoài đoạn [0, 1]")

        if T == 0:
            return SigmoidOutput(
                score=np.empty(0, dtype=np.float32),
                logit=np.empty(0, dtype=np.float32),
                snr_norm=np.empty((n, 0), dtype=np.float32),
                flatness_complement=np.empty(0, dtype=np.float32),
                warmup=warm_arr,
            )

        # 1. Chuẩn hóa SNR về [0, 1]
        snr_norm = normalize_snr_db(snr_arr, cfg.snr_low_db, cfg.snr_high_db)

        # 2. Tính phần bù độ phẳng phổ 1 - SFM ∈ [0, 1]
        sfm_clipped = np.clip(sfm_arr.astype(np.float64), 0.0, 1.0)
        flat_comp = (1.0 - sfm_clipped).astype(np.float32)

        # 3. Xử lý tách biệt các frame warm-up: khóa hoàn toàn cả snr_norm để không nhiễm NaN/0dB giả
        if warm_arr.any():
            snr_norm[:, warm_arr] = 0.0

        # 4. Tổ hợp tuyến tính z(t) và điểm Sigmoid p(t)
        z64 = (
            float(cfg.bias)
            + (self._weights_col @ snr_norm.astype(np.float64))
            + float(cfg.sfm_weight) * flat_comp.astype(np.float64)
        )
        p32 = stable_sigmoid(z64)
        z32 = z64.astype(np.float32)

        if warm_arr.any():
            z32[warm_arr] = np.float32(cfg.warmup_logit)
            p32[warm_arr] = np.float32(cfg.warmup_score)

        return SigmoidOutput(
            score=p32,
            logit=z32,
            snr_norm=snr_norm,
            flatness_complement=flat_comp,
            warmup=warm_arr,
        )

    def score(self, features: VADFeatures, noise_out: NoiseOutput) -> SigmoidOutput:
        """
        Chấm điểm từ đầu ra chuẩn của Tầng 1 (VADFeatures) và Tầng 2 (NoiseOutput).
        """
        if not isinstance(features, VADFeatures):
            raise TypeError("features phải là đối tượng VADFeatures")
        if not isinstance(noise_out, NoiseOutput):
            raise TypeError("noise_out phải là đối tượng NoiseOutput")
        return self.score_arrays(
            snr_db=noise_out.snr_db,
            sfm=features.sfm,
            warmup=noise_out.warmup,
        )

    def __repr__(self) -> str:
        return (
            f"SigmoidScorer(num_bands={self.num_bands}, "
            f"debug={self.debug}, config={self.config})"
        )
