"""
Dictionary/STFT.py
Kiến trúc STFT tối ưu hóa sâu về bộ nhớ (In-place operations, LRU Cache)
và hỗ trợ Stateful Buffer cho Streaming.
"""

import functools
import threading
import numpy as np
import scipy.signal
import scipy.fft


# ============================================================================
# 1. TỐI ƯU HÓA KHỞI TẠO: CACHING HÀM CỬA SỔ
# ============================================================================

_VALID_OUTPUT_TYPES = {"power", "magnitude", "complex"}


@functools.lru_cache(maxsize=16)
def _get_cached_window(
        window: str,
        win_length: int,
        n_fft: int,
        dtype_str: str = "float32"
) -> np.ndarray:
    """
    Chỉ tính toán hàm cửa sổ đúng 1 lần cho mỗi tổ hợp tham số.
    Đánh dấu writeable=False để đảm bảo an toàn bộ nhớ khi dùng chung.

    [FIX #2] Thêm kiểm tra win_length > n_fft để tránh shape mismatch.
    """
    # [FIX #2] Validate win_length <= n_fft trước khi tính toán
    if win_length > n_fft:
        raise ValueError(
            f"win_length ({win_length}) không được lớn hơn n_fft ({n_fft}). "
            f"Hãy tăng n_fft hoặc giảm win_length."
        )

    dtype = np.dtype(dtype_str)
    win = scipy.signal.get_window(window, win_length, fftbins=True).astype(dtype)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = np.pad(win, (pad_left, pad_right), mode="constant")

    win.flags.writeable = False
    return win


# ============================================================================
# 2. CORE ENGINE: TỐI ƯU HÓA IN-PLACE & MEMORY BANDWIDTH
# ============================================================================

def _core_stft_optimized(
        x: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = False,
        pad_mode: str = "reflect",
        output_type: str = "power",
        workers: int = 1
) -> np.ndarray:
    """
    Engine STFT tối ưu:
    - Cố định float32 -> complex64 để giảm 50% băng thông bộ nhớ so với float64.
    - Sử dụng overwrite_x=True trong scipy.fft.rfft.
    - Sử dụng In-place Ufuncs (np.abs, np.add) triệt tiêu các mảng tạm.

    [FIX #1] Power spectrum không còn ghi đè Z.real bằng Z.imag (Undefined Behavior).
    """
    # Đảm bảo mảng đầu vào là float32 liên tục trong RAM (C-contiguous)
    x = np.asarray(x, dtype=np.float32)

    if center:
        pad_width = n_fft // 2
        x = np.pad(x, pad_width, mode=pad_mode)

    if len(x) < n_fft:
        return np.empty((n_fft // 2 + 1, 0), dtype=np.float32)

    # Lấy cửa sổ từ bộ nhớ đệm (O(1) time complexity)
    win = _get_cached_window(window, win_length, n_fft, "float32")

    # frames là non-contiguous view (zero-copy). np.multiply buộc cấp phát 1 buffer
    # C-contiguous mới để RFFT hoạt động hiệu quả — đây là lần cấp phát duy nhất.
    frames = np.lib.stride_tricks.sliding_window_view(x, window_shape=n_fft)[::hop_length]
    windowed_frames = np.multiply(frames, win, dtype=np.float32)

    # RFFT với overwrite_x=True: Cho phép pocketfft tận dụng/hủy buffer windowed_frames
    # Đầu ra Z chắc chắn là complex64 vì đầu vào là float32
    Z = scipy.fft.rfft(
        windowed_frames,
        n=n_fft,
        axis=-1,
        overwrite_x=True,
        workers=workers
    )

    # Trích xuất đặc trưng bằng In-place Memory Operations
    if output_type == "power":
        # Z là complex64 — layout bộ nhớ: [Re0, Im0, Re1, Im1, ...] với stride=8 bytes,
        # mỗi phần tử 4 bytes. Z.real và Z.imag xen kẽ nhau nhưng KHÔNG chồng lấn,
        # nên ghi đè tại chỗ (out=Z.real, out=Z.imag) là an toàn tuyệt đối.
        # Lỗi cũ là lỗi logic: ghi Im² vào Z.real rồi đọc lại Z.real thay vì Re².
        # Cách này không cấp phát mảng tạm nào — Z.real và Z.imag tái dụng làm bộ nhớ đích.
        np.square(Z.real, out=Z.real)  # Z.real = Re²  (ghi đè tại chỗ, an toàn)
        np.square(Z.imag, out=Z.imag)  # Z.imag = Im²  (ghi đè tại chỗ, an toàn)
        spec = np.add(Z.real, Z.imag)  # spec   = Re² + Im²  (1 mảng float32 mới liên tục)
        return spec.T

    elif output_type == "magnitude":
        # np.abs trên complex64 trả về float32, ghi trực tiếp vào mảng đích
        spec = np.empty(Z.shape, dtype=np.float32)
        np.abs(Z, out=spec)
        return spec.T

    elif output_type == "complex":
        return Z.T
    else:
        raise ValueError(f"output_type '{output_type}' không hợp lệ.")


# ============================================================================
# 3. STREAMING ENGINE: GIẢI QUYẾT BÀI TOÁN MẤT MẪU BIÊN GIỮA CÁC CHUNK
# ============================================================================

class StreamingSTFT:
    """
    Bộ xử lý STFT có nhớ trạng thái (Stateful) dành riêng cho VAD và ASR Streaming.
    Tự động lưu phần dư (tail buffer) của chunk trước nối vào chunk sau,
    đảm bảo không bao giờ bị rớt frame ở biên.

    [FIX #4] Thread-safe: dùng threading.Lock bảo vệ _buffer.
    [FIX #7] Validate output_type ngay tại __init__, không đợi đến lúc chạy.
    """

    def __init__(
            self,
            n_fft: int = 512,
            hop_length: int = 160,
            win_length: int = 400,
            window: str = "hann",
            output_type: str = "power"
    ):
        # [FIX #7] Validate output_type ngay khi tạo đối tượng,
        # tránh lỗi typo chỉ phát hiện được sau khi đã stream một lúc.
        if output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"output_type '{output_type}' không hợp lệ. "
                f"Chọn một trong: {_VALID_OUTPUT_TYPES}"
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.output_type = output_type
        # Bộ đệm lưu các mẫu chưa đủ 1 frame từ chunk trước
        self._buffer = np.empty(0, dtype=np.float32)
        # [FIX #4] Lock bảo vệ _buffer khi nhiều thread cùng gọi process_chunk
        self._lock = threading.Lock()

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Nhận 1 chunk âm thanh mới, nối với phần dư cũ và trả về các frame STFT."""
        chunk = np.asarray(chunk, dtype=np.float32)

        # [FIX #4] Toàn bộ thao tác đọc/ghi _buffer được bảo vệ bởi lock
        with self._lock:
            # Nối phần dư của chunk trước vào đầu chunk hiện tại
            if len(self._buffer) > 0:
                x = np.concatenate((self._buffer, chunk))
            else:
                x = chunk

            if len(x) < self.n_fft:
                self._buffer = x
                return np.empty((self.n_fft // 2 + 1, 0), dtype=np.float32)

            # Tính số lượng frame hoàn chỉnh có thể cắt được
            num_frames = 1 + (len(x) - self.n_fft) // self.hop_length

            # Vị trí mẫu bắt đầu của frame tiếp theo (dành cho chunk sau)
            consumed_samples = num_frames * self.hop_length
            self._buffer = x[consumed_samples:]

        # Chỉ đưa đúng phần dữ liệu vừa khít các frame vào tính STFT
        valid_length = (num_frames - 1) * self.hop_length + self.n_fft
        return _core_stft_optimized(
            x=x[:valid_length],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            output_type=self.output_type,
            workers=1  # Streaming chunk nhỏ không nên bật multi-thread vì tốn overhead tạo thread
        )

    def reset(self):
        """Xóa bộ đệm khi bắt đầu phiên hội thoại mới."""
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"StreamingSTFT(n_fft={self.n_fft}, hop={self.hop_length}, "
            f"win={self.win_length}, buffer_len={len(self._buffer)})"
        )


# ============================================================================
# 4. CÁC HÀM PRESET CHO TỪNG TÁC VỤ TRONG REPO
# ============================================================================

def _next_power_of_two(n: int) -> int:
    """Trả về lũy thừa của 2 nhỏ nhất >= n (dùng cho n_fft)."""
    return 1 << (n - 1).bit_length()


def create_vad_streamer(sample_rate: int = 16000) -> StreamingSTFT:
    """Khởi tạo bộ STFT liên tục cho VAD (hop 5ms, win 16ms).

    [FIX #6] n_fft được tự động tính theo sample_rate thay vì hardcode 256,
    đảm bảo win_length luôn <= n_fft dù sample_rate thay đổi.
    """
    win_length = int(sample_rate * 0.016)
    n_fft = _next_power_of_two(win_length)
    return StreamingSTFT(
        n_fft=n_fft,
        hop_length=int(sample_rate * 0.005),
        win_length=win_length,
        output_type="power"
    )


def create_asr_streamer(sample_rate: int = 16000) -> StreamingSTFT:
    """Khởi tạo bộ STFT liên tục cho ASR Streaming (hop 10ms, win 25ms)."""
    win_length = int(sample_rate * 0.025)
    n_fft = _next_power_of_two(win_length)
    return StreamingSTFT(
        n_fft=n_fft,
        hop_length=int(sample_rate * 0.010),
        win_length=win_length,
        output_type="power"
    )


def stft_for_asr_patch(x: np.ndarray, sample_rate: int = 16000, workers: int = -1) -> np.ndarray:
    """
    STFT cho ASR Patch Processing (Offline):
    - Bật center=True để khớp nhãn thời gian.
    - Bật workers=-1 để tận dụng đa nhân CPU khi xử lý đoạn âm thanh dài.

    Lưu ý: workers=-1 dùng toàn bộ CPU logical core. Trong môi trường serving
    đa tiến trình, nên đặt workers = os.cpu_count() // số_worker_process.
    """
    win_length = int(sample_rate * 0.025)
    n_fft = _next_power_of_two(win_length)
    return _core_stft_optimized(
        x=x,
        n_fft=n_fft,
        hop_length=int(sample_rate * 0.010),
        win_length=win_length,
        center=True,
        pad_mode="reflect",
        output_type="power",
        workers=workers
    )


def stft_for_diarization(
        x: np.ndarray,
        sample_rate: int = 16000,
        n_fft: int = 512,
        return_magnitude: bool = True,
        workers: int = -1
) -> np.ndarray:
    """
    STFT cho Diarization (DTW / Speaker Embedding):
    - Phân giải tần số cao, tận dụng đa luồng CPU.

    [FIX #3] n_fft được truyền vào ngoài thay vì hardcode, win_length
    tự động clamp <= n_fft để không bao giờ mất sync giữa hai tham số.

    Lưu ý: workers=-1 dùng toàn bộ CPU logical core. Trong môi trường serving
    đa tiến trình, nên đặt workers = os.cpu_count() // số_worker_process.
    """
    win_length = min(int(sample_rate * 0.032), n_fft)
    return _core_stft_optimized(
        x=x,
        n_fft=n_fft,
        hop_length=int(sample_rate * 0.010),
        win_length=win_length,
        center=False,
        output_type="magnitude" if return_magnitude else "power",
        workers=workers
    )
