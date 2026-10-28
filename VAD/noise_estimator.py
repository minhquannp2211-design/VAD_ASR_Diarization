"""
VAD/noise_estimator.py - TẦNG 2: ƯỚC LƯỢNG NỀN NHIỄU THÍCH NGHI

Đầu vào : VADFeatures (band_power (n_bands, T), sfm (T,)) từ Tầng 1.
Đầu ra  : NoiseOutput (noise_floor, snr_db, update_mask, stuck, warmup, is_noise).

ĐỊNH NGHĨA
    - noise_floor cùng định nghĩa với band_power: công suất TRUNG BÌNH TRÊN MỖI BIN.
    - noise_floor[:, t] là nền nhiễu ĐÃ DÙNG để tính snr_db[:, t]; việc cập nhật
      (nếu có) chỉ ảnh hưởng frame t+1 trở đi.
    - snr_db = 10*log10(P / max(N, floor_min)), tính từ tỷ số công suất tuyến tính, clip [snr_min_db, snr_max_db].

WARM-UP
    Với tối đa `init_frames` frame đầu, nền nhiễu khởi tạo = median theo từng băng.
    Trong warm-up: snr_db = 0 và warmup = True. 0 dB KHÔNG phải phép đo SNR đáng tin cậy,
    Tầng 3 phải xử lý cờ warmup riêng. Median không cứu được trường hợp phần lớn
    đoạn đầu là tiếng nói.

CỔNG CẬP NHẬT (theo TỪNG BĂNG; SFM là đặc trưng chung theo frame)
    Phân loại từng băng j tại frame t theo snr_db[j,t] (tính với nền cũ) và sfm[t]:
        snr <  snr_noise_db (liên tiếp >= strong_confirm_frames) -> NHIỄU MẠNH : cập nhật alpha_fast
        snr <  snr_noise_db (chưa đủ strong_confirm_frames)      -> CHỜ XÁC NHẬN: chưa cập nhật
        snr_noise_db <= snr < snr_speech_db và sfm >  sfm_thresh -> NHIỄU YẾU  : cập nhật alpha_slow
        snr >= snr_noise_db                  và sfm <= sfm_thresh -> TIẾNG NÓI : đóng băng băng đó
        snr >= snr_speech_db                 và sfm >  sfm_thresh -> MƠ HỒ     : không cập nhật
    alpha_fast/alpha_slow thể hiện độ tin cậy của quyết định, không phụ thuộc dấu (P - N).
    Việc yêu cầu `strong_confirm_frames` frame liên tiếp dưới `snr_noise_db` ngăn các hõm công suất
    ngẫu nhiên đơn lẻ (đặc biệt ở băng 80-300 Hz chỉ có 3 bin) kích hoạt `alpha_fast` và gây hiệu ứng
    tuyết lở kéo noise floor lên theo tiếng nói ở mức SNR trung bình (+15 dB).
    Lưu ý kỹ thuật:
    - Cổng `strong` KHÔNG bị chặn bởi SFM thấp để giữ tính độc lập theo từng băng (khi 1 băng có tiếng nói
      làm SFM toàn frame thấp, các băng chỉ chứa nhiễu nền vẫn cập nhật được). Đổi lại, nếu một băng trong
      đoạn tiếng nói có SNR < snr_noise_db kéo dài >= strong_confirm_frames (do tương quan cửa sổ STFT
      chồng lấn 68.75% hoặc âm vị ít năng lượng ở băng đó), băng đó vẫn sẽ cập nhật nhanh. Đây là trade-off
      cần đánh giá trên audio thực, không suy diễn từ xác suất độc lập p^3 của mô hình Gamma.
    - SFM cao đơn lẻ KHÔNG đủ để kết luận là nhiễu (khi SNR >= snr_speech_db và chưa kẹt thì rơi vào vùng
      MƠ HỒ, không cập nhật). SFM > sfm_thresh chỉ là một điều kiện cần trong cổng phục hồi khi băng đã kẹt
      lâu, bắt buộc kết hợp với điều kiện ổn định công suất.

CHỐNG KẸT (bản đầu, không có min-tracking)
    - frozen_count[j] tăng khi băng j không cập nhật thường; giữ nguyên khi cập nhật `weak` và chỉ reset
      về 0 khi đã xác nhận `strong`. Khi frozen_count > max_freeze_frames -> bật cờ `stuck` (chẩn đoán).
    - Trước khi `stuck`, tham chiếu ngắn hạn `_ref` được neo tại `noise_floor` (KHÔNG chạy theo tín hiệu
      có SFM cao trước khi stuck). Chỉ sau khi đã `stuck` và `sfm > sfm_thresh`, `_ref` mới bắt đầu bám
      theo công suất băng và tích lũy `_stable` khi độ lệch trong +-recover_stable_db đủ `recover_frames`.
    - Giới hạn lý thuyết: khi chỉ quan sát (band_power, sfm), một tín hiệu có SFM > sfm_thresh và công suất
      dừng kéo dài đủ lâu sau khi stuck (> max_freeze_frames + thời gian hội tụ _ref + recover_frames) sẽ
      có đặc trưng giống hệt nhiễu nền dừng mức mới và sẽ kích hoạt phục hồi chậm.

HIỆU NĂNG & THAM SỐ BASELINE
    Vòng lặp theo frame (phụ thuộc tuần tự), vector hóa trên số băng. Các tham số mặc định
    (đặc biệt snr_noise_db = 6.0 dB và strong_confirm_frames = 3 chọn từ test tổng hợp) chỉ
    mang tính BASELINE, chưa hiệu chỉnh trên audio thực; cần đánh giá trên audio 8 kHz và 16 kHz.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.feature_extraction import VADFeatures  # noqa: E402


@dataclass(frozen=True)
class NoiseConfig:
    init_frames: int = 40            # warm-up tối đa (40 frame = 200 ms @ 5 ms)
    alpha_fast: float = 0.15         # bằng chứng nhiễu mạnh
    alpha_slow: float = 0.01         # bằng chứng nhiễu yếu / phục hồi
    snr_noise_db: float = 6.0        # baseline (từ test tổng hợp); dưới ngưỡng: nhiễu mạnh
    snr_speech_db: float = 9.0       # từ ngưỡng: tiếng nói / mơ hồ
    strong_confirm_frames: int = 3   # số frame liên tiếp < snr_noise_db để xác nhận strong (15 ms @ 5 ms)
    sfm_thresh: float = 0.15         # SFM trên ngưỡng: phổ phẳng (điều kiện cần khi recover / weak)
    max_freeze_frames: int = 500     # 2.5 s @ 5 ms -> đánh dấu stuck
    recover_frames: int = 100        # điểm ổn định cần tích lũy khi stuck mới phục hồi
    recover_stable_db: float = 3.0   # biên ổn định quanh trung bình ngắn hạn
    recover_ref_alpha: float = 0.05  # EMA ngắn hạn làm tham chiếu ổn định (chỉ cập nhật khi stuck & flat)
    snr_min_db: float = -10.0
    snr_max_db: float = 40.0
    floor_min: float = 1e-10

    def validate(self) -> None:
        if self.init_frames < 1:
            raise ValueError("init_frames phải >= 1")
        if not (0.0 < self.alpha_slow <= self.alpha_fast < 1.0):
            raise ValueError("Yêu cầu 0 < alpha_slow <= alpha_fast < 1")
        if not (0.0 < self.recover_ref_alpha < 1.0):
            raise ValueError("recover_ref_alpha phải trong (0, 1)")
        if self.snr_noise_db >= self.snr_speech_db:
            raise ValueError("snr_noise_db phải < snr_speech_db")
        if self.strong_confirm_frames < 1:
            raise ValueError("strong_confirm_frames phải >= 1")
        if not (0.0 <= self.sfm_thresh <= 1.0):
            raise ValueError("sfm_thresh phải trong [0, 1]")
        if self.max_freeze_frames < 0 or self.recover_frames < 1:
            raise ValueError("max_freeze_frames >= 0 và recover_frames >= 1")
        if self.snr_min_db >= self.snr_max_db:
            raise ValueError("snr_min_db phải < snr_max_db")
        if self.floor_min <= 0:
            raise ValueError("floor_min phải > 0")


@dataclass(frozen=True)
class NoiseOutput:
    noise_floor: np.ndarray  # (n_bands, T) float32: nền nhiễu ĐÃ DÙNG tính SNR frame t
    snr_db: np.ndarray       # (n_bands, T) float32, đã clip; =0 khi warmup
    update_mask: np.ndarray  # (n_bands, T) bool: băng được dùng cập nhật nền ở frame t
    stuck: np.ndarray        # (n_bands, T) bool: băng đang bị đóng băng quá max_freeze_frames
    warmup: np.ndarray       # (T,) bool: True -> snr_db KHÔNG đáng tin
    is_noise: np.ndarray     # (T,) bool: cờ chẩn đoán = update_mask.all(axis=0), không thay update_mask

    @property
    def num_frames(self) -> int:
        return self.warmup.shape[0]


class NoiseEstimator:
    def __init__(self, num_bands: int = 4, config: Optional[NoiseConfig] = None,
                 debug: bool = False) -> None:
        if num_bands < 1:
            raise ValueError("num_bands phải >= 1")
        self.config = config or NoiseConfig()
        self.config.validate()
        self.num_bands = int(num_bands)
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        n = self.num_bands
        self._buf: list = []                          # warm-up buffer, mỗi phần tử (n,)
        self._ready = False
        self._floor = np.full(n, self.config.floor_min, dtype=np.float64)
        self._strong_run = np.zeros(n, dtype=np.int64)
        self._frozen = np.zeros(n, dtype=np.int64)
        self._stable = np.zeros(n, dtype=np.int64)
        self._ref = np.zeros(n, dtype=np.float64)    # EMA ngắn hạn của band_power khi stuck
        self._ref_init = False

    @property
    def noise_floor(self) -> np.ndarray:
        """Nền nhiễu hiện tại (n_bands,); trong warm-up là median tạm thời."""
        if not self._ready and self._buf:
            return np.maximum(np.median(np.stack(self._buf), axis=0),
                              self.config.floor_min).astype(np.float32)
        return self._floor.astype(np.float32)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def update(self, features: VADFeatures) -> NoiseOutput:
        cfg = self.config
        n = self.num_bands
        bp = np.asarray(features.band_power)
        sfm = np.asarray(features.sfm)
        if bp.ndim != 2 or bp.shape[0] != n:
            raise ValueError(f"band_power phải có shape ({n}, T), nhận {bp.shape}")
        if sfm.ndim != 1 or sfm.shape[0] != bp.shape[1]:
            raise ValueError("sfm phải có shape (T,) khớp band_power")
        if self.debug and (not np.isfinite(bp).all() or (bp < 0).any()
                           or not np.isfinite(sfm).all()):
            raise ValueError("band_power/sfm chứa NaN, Inf hoặc giá trị âm")

        T = bp.shape[1]
        out_floor = np.empty((n, T), dtype=np.float32)
        out_snr = np.zeros((n, T), dtype=np.float32)
        out_upd = np.zeros((n, T), dtype=bool)
        out_stuck = np.zeros((n, T), dtype=bool)
        out_warm = np.zeros(T, dtype=bool)

        P_all = bp.astype(np.float64)
        floor_min = cfg.floor_min
        for t in range(T):
            P = P_all[:, t]
            if not self._ready:
                self._buf.append(P.copy())
                out_floor[:, t] = np.maximum(
                    np.median(np.stack(self._buf), axis=0), floor_min)
                out_warm[t] = True
                if len(self._buf) >= cfg.init_frames:
                    self._floor = np.maximum(
                        np.median(np.stack(self._buf), axis=0), floor_min)
                    self._ref = self._floor.copy()
                    self._ref_init = True
                    self._ready = True
                    self._buf = []
                continue

            N = self._floor
            out_floor[:, t] = N
            snr_db = 10.0 * np.log10(np.maximum(P / np.maximum(N, floor_min), 1e-10))
            snr_db = np.clip(snr_db, cfg.snr_min_db, cfg.snr_max_db)
            out_snr[:, t] = snr_db

            flat = sfm[t] > cfg.sfm_thresh
            strong_candidate = snr_db < cfg.snr_noise_db
            self._strong_run = np.where(strong_candidate, self._strong_run + 1, 0)
            strong = strong_candidate & (self._strong_run >= cfg.strong_confirm_frames)
            weak = (~strong_candidate) & (snr_db < cfg.snr_speech_db) & flat
            normal_upd = strong | weak                       # speech/mơ hồ: không cập nhật

            stuck = self._frozen > cfg.max_freeze_frames

            # Tham chiếu ổn định ngắn hạn: _ref bám theo P khi stuck; _stable tích lũy khi công suất ổn định và phổ phẳng
            active_stuck = stuck & (~strong)
            ref = self._ref
            dev_db = np.abs(10.0 * np.log10(
                np.maximum(P, floor_min) / np.maximum(ref, floor_min)))
            self._ref = np.where(
                active_stuck, ref + cfg.recover_ref_alpha * (P - ref), N)
            good_frame = flat & (dev_db <= cfg.recover_stable_db)
            inc = np.where(good_frame,
                           self._stable + 1,
                           np.maximum(self._stable - 2, 0))
            self._stable = np.where(active_stuck & (~normal_upd), inc, 0)

            recover = (~normal_upd) & stuck & (self._stable >= cfg.recover_frames)
            upd = normal_upd | recover
            alpha = np.where(strong, cfg.alpha_fast, cfg.alpha_slow)
            self._floor = np.where(
                upd, np.maximum(N + alpha * (P - N), floor_min), N)

            # frozen_count chỉ reset về 0 khi đã xác nhận strong; giữ nguyên khi weak
            self._frozen = np.where(
                strong, 0, np.where(weak, self._frozen, self._frozen + 1))
            out_upd[:, t] = upd
            out_stuck[:, t] = stuck

        return NoiseOutput(
            noise_floor=out_floor, snr_db=out_snr, update_mask=out_upd,
            stuck=out_stuck, warmup=out_warm,
            is_noise=out_upd.all(axis=0) if T else np.zeros(0, dtype=bool),
        )

    def __repr__(self) -> str:
        return (f"NoiseEstimator(num_bands={self.num_bands}, ready={self._ready}, "
                f"config={self.config})")

