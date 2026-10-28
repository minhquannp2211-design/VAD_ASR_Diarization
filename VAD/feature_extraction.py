from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

# Cho phép import Dictionary/STFT.py khi chạy từ bất kỳ đâu trong repo
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Dictionary.STFT import (  # noqa: E402
    StreamingSTFT, create_vad_streamer, stft_for_vad, vad_stft_params,
)
# 1. CẤU HÌNH DẢI TẦN
# 4 dải trong khoảng 80-4000 Hz, chọn theo đặc tính tiếng nói:
#   F0/harmonic thấp | F1 | F2 | phụ âm / F3
DEFAULT_BANDS_HZ: Tuple[Tuple[float, float], ...] = (
    (80.0, 300.0),
    (300.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
)

_EPS = np.float32(1e-10)


@dataclass(frozen=True)
class VADFeatures:
    band_power: np.ndarray   # (n_bands, T) float32, công suất tuyến tính
    sfm: np.ndarray          # (T,) float32

    @property
    def num_frames(self) -> int:
        return self.sfm.shape[0]


def bin_frequencies(n_fft: int, sample_rate: int) -> np.ndarray:
    """Tần số thực (Hz) của từng bin rfft: f_k = k * sr / n_fft."""
    return np.arange(n_fft // 2 + 1, dtype=np.float64) * sample_rate / n_fft


def _validate_freq_range(
        rng: Tuple[float, float], sample_rate: int, name: str = "band"
) -> Tuple[float, float]:
    """Yêu cầu 0 <= lo < hi <= Nyquist (không cắt ngầm theo bin có sẵn)."""
    lo, hi = float(rng[0]), float(rng[1])
    nyquist = sample_rate / 2.0
    if not (0.0 <= lo < hi):
        raise ValueError(f"{name}: dai tan khong hop le ({lo}, {hi}).")
    if hi > nyquist:
        raise ValueError(f"{name}: ({lo}, {hi}) Hz vuot Nyquist {nyquist} Hz.")
    return lo, hi


def build_band_masks(
        n_fft: int,
        sample_rate: int,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
) -> np.ndarray:
    """
    Ánh xạ dải tần -> bin theo tần số thực. Bin k thuộc dải [lo, hi) nếu lo <= f_k < hi.
    Trả về ma trận trọng số (n_bands, n_bins) float32 đã chuẩn hóa theo số bin,
    để band_power = W @ spec là CÔNG SUẤT TRUNG BÌNH MỖI BIN của dải (không phụ thuộc độ rộng dải).
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate phai > 0, nhan duoc {sample_rate}.")
    if n_fft <= 0:
        raise ValueError(f"n_fft phai > 0, nhan duoc {n_fft}.")
    if len(bands_hz) == 0:
        raise ValueError("bands_hz khong duoc rong.")
    freqs = bin_frequencies(n_fft, sample_rate)
    W = np.zeros((len(bands_hz), freqs.size), dtype=np.float32)
    for i, band in enumerate(bands_hz):
        lo, hi = _validate_freq_range(band, sample_rate, f"bands_hz[{i}]")
        mask = (freqs >= lo) & (freqs < hi)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(
                f"Dai ({lo}, {hi}) Hz khong chua bin nao "
                f"(bin spacing = {sample_rate / n_fft:.2f} Hz)."
            )
        W[i, mask] = 1.0 / n
    W.flags.writeable = False
    return W

# 2. CÁC HÀM ĐẶC TRƯNG (STATELESS, VECTOR HÓA)

def _check_spec(spec: np.ndarray,n_bins: int,debug: bool = False,) -> np.ndarray:
    spec = np.asarray(spec)
    if np.iscomplexobj(spec):
        raise ValueError("spec must be a real-valued power spectrogram.")
    if spec.ndim != 2 or spec.shape[0] != n_bins:
        raise ValueError(f"spec must have shape ({n_bins}, T), "f"got {spec.shape}.")
    spec = np.asarray(spec, dtype=np.float32)
    if debug and spec.size:
        if not np.isfinite(spec).all():
            raise ValueError("spec chứa NaN/Inf.")
        if (spec < 0).any():
            raise ValueError(
                "Power spectrogram không được có giá trị âm."
            )
    return spec
def compute_band_power(spec: np.ndarray, band_weights: np.ndarray) -> np.ndarray:
    """
    Sub-band Power: (n_bands, n_bins) @ (n_bins, T) -> (n_bands, T), float32.
    Giá trị = công suất TRUNG BÌNH TRÊN MỖI BIN của dải. T=0 -> shape (n_bands, 0).
    """
    # Không kiểm tra lại: caller (FeatureExtractor.extract) đã validate spec 1 lần.
    return (band_weights @ spec).astype(np.float32, copy=False)


def compute_sfm(spec: np.ndarray, bin_mask: np.ndarray) -> np.ndarray:
    """
    Spectral Flatness Measure trên các bin được chọn (mặc định 80-4000 Hz):
        SFM = exp(mean(log P)) / mean(P)      ∈ [0, 1]
    Tính trong miền log để tránh underflow của tích hình học.
    Frame im lặng tuyệt đối (P ≈ 0) -> SFM = 1 (coi như phẳng / không phải tiếng nói).
    """
    # Không kiểm tra lại: caller (FeatureExtractor.extract) đã validate spec 1 lần.
    if spec.shape[1] == 0:
        return np.empty(0, dtype=np.float32)
    p = spec[bin_mask] + _EPS                         # (n_sel, T), mảng mới
    arith = p.mean(axis=0)
    np.log(p, out=p)                                  # in-place
    geo = np.exp(p.mean(axis=0))
    sfm = geo / arith
    np.clip(sfm, 0.0, 1.0, out=sfm)
    return sfm.astype(np.float32, copy=False)

# 3. BỘ TRÍCH XUẤT (WRAPPER STREAMING / BATCH)

class FeatureExtractor:
    """
    Tầng 1 của VAD.
      - extract(spec)        : nhận Power Spectrogram CÓ SẴN (n_bins, T), stateless. KHÔNG nhận waveform.
      - process_chunk(audio) : nhận waveform chunk, gọi StreamingSTFT nội bộ (stateful).
      - flush() / reset()    : kết thúc / làm mới phiên streaming.

    debug=True: kiểm tra spec không chứa NaN/Inf/giá trị âm (tốn thêm một lượt quét dữ liệu,
    nên tắt mặc định trong streaming).
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
            sfm_range_hz: Tuple[float, float] = (80.0, 4000.0),
            stft: StreamingSTFT | None = None,
            debug: bool = False,
    ):
        self.sample_rate = sample_rate
        self.debug = debug
        self.stft = stft if stft is not None else create_vad_streamer(sample_rate)
        if self.stft.output_type != "power":
            raise ValueError("FeatureExtractor yeu cau StreamingSTFT output_type='power'.")

        # STFT truyền từ ngoài phải khớp preset VAD ở sample_rate đã khai báo,
        # nếu không ánh xạ bin -> Hz (và batch/streaming) sẽ sai.
        expected = vad_stft_params(sample_rate)
        actual = {"n_fft": self.stft.n_fft, "hop_length": self.stft.hop_length,
                  "win_length": self.stft.win_length}
        if actual != expected:
            raise ValueError(
                f"Cau hinh STFT {actual} khong khop preset VAD o sample_rate={sample_rate} "
                f"(can {expected})."
            )

        self.n_fft = self.stft.n_fft
        self.n_bins = self.n_fft // 2 + 1
        self.bands_hz = tuple(tuple(map(float, b)) for b in bands_hz)
        self.band_weights = build_band_masks(self.n_fft, sample_rate, self.bands_hz)

        freqs = bin_frequencies(self.n_fft, sample_rate)
        lo, hi = _validate_freq_range(sfm_range_hz, sample_rate, "sfm_range_hz")
        self.sfm_mask = (freqs >= lo) & (freqs < hi)
        if self.sfm_mask.sum() < 2:
            raise ValueError(f"sfm_range_hz {sfm_range_hz} chua qua it bin (can >= 2).")

    @property
    def num_bands(self) -> int:
        return self.band_weights.shape[0]

    @property
    def hop_seconds(self) -> float:
        return self.stft.hop_length / self.sample_rate

    def extract(self, spec: np.ndarray) -> VADFeatures:
        spec = _check_spec(spec, self.n_bins, debug=self.debug)
        return VADFeatures(
            band_power=compute_band_power(spec, self.band_weights),
            sfm=compute_sfm(spec, self.sfm_mask),
        )

    def process_batch(self, audio: np.ndarray, workers: int = 1) -> VADFeatures:
        """
        Batch/Patch: nhận waveform hoàn chỉnh (1-D mono), gọi stft_for_vad() rồi dùng lại extract().
        Stateless — không đụng đến state của StreamingSTFT.
        Quy ước biên: số frame = 1 + (N - n_fft) // hop, không đệm frame cuối
        (khác streaming sau flush()). Trên các frame đầy đủ, kết quả khớp với process_chunk().
        """
        spec = stft_for_vad(audio, self.sample_rate, workers=workers)
        return self.extract(spec)

    def process_chunk(self, chunk: np.ndarray) -> VADFeatures:
        return self.extract(self.stft.process_chunk(chunk))

    def flush(self) -> VADFeatures:
        return self.extract(self.stft.flush())

    def reset(self) -> None:
        self.stft.reset()

    def __repr__(self) -> str:
        return (f"FeatureExtractor(sr={self.sample_rate}, n_fft={self.n_fft}, "
                f"bands={self.bands_hz}, hop={self.hop_seconds * 1000:.1f}ms)")
