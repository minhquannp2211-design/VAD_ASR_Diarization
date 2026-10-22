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

# Ánh xạ output_type sang dtype tương ứng (dùng cho mảng rỗng và kiểm tra)
_OUTPUT_DTYPE = {
    "power": np.float32,
    "magnitude": np.float32,
    "complex": np.complex64,
}


def _empty_spec(n_fft: int, output_type: str) -> np.ndarray:
    """
    Trả về mảng rỗng đúng shape và dtype cho output_type đã cho.
    Shape luôn là (n_fft // 2 + 1, 0) — nhất quán với đầu ra bình thường.

    Helper nội bộ, tránh lặp code ở nhiều nhánh early-return.
    """
    return np.empty((n_fft // 2 + 1, 0), dtype=_OUTPUT_DTYPE[output_type])


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
    """
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
    - Sử dụng In-place Ufuncs (np.square, np.abs, np.add) triệt tiêu các mảng tạm.

    Đầu vào: audio mono một chiều (1-D float array).
    Đầu ra: shape (n_fft // 2 + 1, num_frames), dtype theo output_type.

    Ghi chú power spectrum:
        Z là complex64 — layout bộ nhớ [Re0, Im0, Re1, Im1, ...] với stride=8 bytes.
        Z.real và Z.imag xen kẽ nhưng KHÔNG chồng lấn (xác nhận bằng probe test),
        nên ghi đè in-place (out=Z.real, out=Z.imag) là an toàn tuyệt đối.
        Kết quả bằng |Z|^2 reference với max_diff < 1e-5 (xác nhận bằng probe test).
    """
    # ------------------------------------------------------------------
    # A. Validate sớm — trước bất kỳ phép tính nào
    # ------------------------------------------------------------------
    if output_type not in _VALID_OUTPUT_TYPES:
        raise ValueError(
            f"output_type '{output_type}' khong hop le. "
            f"Chon mot trong: {sorted(_VALID_OUTPUT_TYPES)}"
        )
    if n_fft <= 0:
        raise ValueError(f"n_fft phai > 0, nhan duoc {n_fft}.")
    if hop_length <= 0:
        raise ValueError(f"hop_length phai > 0, nhan duoc {hop_length}.")
    if win_length <= 0:
        raise ValueError(f"win_length phai > 0, nhan duoc {win_length}.")
    if win_length > n_fft:
        raise ValueError(
            f"win_length ({win_length}) khong duoc lon hon n_fft ({n_fft}). "
            f"Hay tang n_fft hoac giam win_length."
        )

    # ------------------------------------------------------------------
    # B. Chuẩn bị đầu vào
    # ------------------------------------------------------------------
    # Chỉ hỗ trợ audio mono 1-D. Báo lỗi rõ ràng nếu sai shape.
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(
            f"Chi ho tro audio mono 1-D, nhan duoc mang {x.ndim}-D shape={x.shape}."
        )

    # center=True: padding đối xứng để frame đầu tiên căn giữa tại mẫu 0
    # Xử lý audio rỗng trước np.pad để tránh crash với mode='reflect'
    if center:
        if len(x) == 0:
            return _empty_spec(n_fft, output_type)
        pad_width = n_fft // 2
        x = np.pad(x, pad_width, mode=pad_mode)

    if len(x) < n_fft:
        return _empty_spec(n_fft, output_type)

    # ------------------------------------------------------------------
    # C. Tính STFT
    # ------------------------------------------------------------------
    # Lấy cửa sổ từ bộ nhớ đệm (O(1) time complexity)
    win = _get_cached_window(window, win_length, n_fft, "float32")

    # frames là non-contiguous view (zero-copy). np.multiply buộc cấp phát 1 buffer
    # C-contiguous mới để RFFT hoạt động hiệu quả — đây là lần cấp phát duy nhất.
    frames = np.lib.stride_tricks.sliding_window_view(x, window_shape=n_fft)[::hop_length]
    windowed_frames = np.multiply(frames, win, dtype=np.float32)

    # RFFT với overwrite_x=True: cho phép pocketfft tận dụng/hủy buffer windowed_frames.
    # Đầu ra Z chắc chắn là complex64 vì đầu vào là float32.
    Z = scipy.fft.rfft(
        windowed_frames,
        n=n_fft,
        axis=-1,
        overwrite_x=True,
        workers=workers
    )

    # ------------------------------------------------------------------
    # D. Trích xuất đặc trưng
    # ------------------------------------------------------------------
    if output_type == "power":
        # In-place: Z.real = Re², Z.imag = Im², rồi cộng lại.
        # Không cấp phát mảng tạm — Z.real/Z.imag tái dụng làm bộ nhớ đích.
        np.square(Z.real, out=Z.real)   # Z.real = Re²
        np.square(Z.imag, out=Z.imag)   # Z.imag = Im²
        spec = np.add(Z.real, Z.imag)   # spec   = Re² + Im²  (1 mảng float32 mới)
        return spec.T                   # (n_bins, num_frames)

    elif output_type == "magnitude":
        spec = np.empty(Z.shape, dtype=np.float32)
        np.abs(Z, out=spec)
        return spec.T

    else:  # output_type == "complex"
        return Z.T


# ============================================================================
# 3. STREAMING ENGINE: GIẢI QUYẾT BÀI TOÁN MẤT MẪU BIÊN GIỮA CÁC CHUNK
# ============================================================================

class StreamingSTFT:
    """
    Bộ xử lý STFT có nhớ trạng thái (Stateful) dành riêng cho VAD và ASR Streaming.
    Tự động lưu phần dư (tail buffer) của chunk trước nối vào chunk sau,
    đảm bảo không bao giờ bị rớt frame ở biên.

    Giới hạn thread-safety:
        threading.Lock bảo vệ state nội bộ (_buffer, _has_produced_frames) khi
        nhiều thread truy cập cùng lúc. Tuy nhiên, class này được thiết kế cho
        mô hình SINGLE-CONSUMER: mỗi stream chỉ có một goroutine/thread gọi
        process_chunk() và flush() theo thứ tự. Nếu nhiều thread gọi process_chunk()
        đồng thời, thứ tự frame đầu ra KHÔNG được bảo đảm — thread nào hoàn thành
        FFT trước sẽ trả kết quả trước.
    """

    def __init__(
            self,
            n_fft: int = 512,
            hop_length: int = 160,
            win_length: int = 400,
            window: str = "hann",
            output_type: str = "power"
    ):
        # Validate output_type ngay khi tạo đối tượng,
        # tránh lỗi typo chỉ phát hiện được sau khi đã stream một lúc.
        if output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"output_type '{output_type}' khong hop le. "
                f"Chon mot trong: {sorted(_VALID_OUTPUT_TYPES)}"
            )
        if n_fft <= 0:
            raise ValueError(f"n_fft phai > 0, nhan duoc {n_fft}.")
        if hop_length <= 0:
            raise ValueError(f"hop_length phai > 0, nhan duoc {hop_length}.")
        if win_length <= 0:
            raise ValueError(f"win_length phai > 0, nhan duoc {win_length}.")
        if win_length > n_fft:
            raise ValueError(
                f"win_length ({win_length}) khong duoc lon hon n_fft ({n_fft})."
            )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.output_type = output_type

        # Bộ đệm lưu các mẫu chưa đủ 1 frame từ chunk trước.
        # Luôn là mảng độc lập (không phải view vào chunk đầu vào).
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)

        # Theo dõi xem đã có frame nào được tạo ra chưa.
        # Cần để flush() phân biệt "buffer toàn là overlap" vs "buffer có dữ liệu mới".
        self._has_produced_frames: bool = False

        # Lock bảo vệ _buffer và _has_produced_frames khi đọc/ghi từ nhiều thread.
        self._lock = threading.Lock()

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Nhận 1 chunk âm thanh mới, nối với phần dư cũ và trả về các frame STFT đầy đủ.

        Đảm bảo:
        - Không mất frame do chia chunk với kích thước bất kỳ.
        - Buffer nội bộ luôn là mảng độc lập — sửa đổi `chunk` sau khi gọi hàm
          này không ảnh hưởng đến state nội bộ.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(
                f"Chunk phai la mang 1-D, nhan duoc {chunk.ndim}-D shape={chunk.shape}."
            )

        with self._lock:
            # Nối phần dư của chunk trước vào đầu chunk hiện tại.
            # np.concatenate luôn tạo mảng mới — x là độc lập với chunk.
            if len(self._buffer) > 0:
                x = np.concatenate((self._buffer, chunk))
            else:
                # Dù không có buffer, vẫn cần copy để x độc lập với chunk đầu vào.
                x = chunk.copy()

            if len(x) < self.n_fft:
                # Chưa đủ một frame — lưu toàn bộ vào buffer.
                # Dùng copy() để buffer không phụ thuộc vào mảng x hoặc chunk.
                self._buffer = x
                return _empty_spec(self.n_fft, self.output_type)

            # Tính số lượng frame hoàn chỉnh có thể cắt được
            num_frames = 1 + (len(x) - self.n_fft) // self.hop_length

            # Vị trí mẫu bắt đầu của frame tiếp theo (phần dư cho chunk sau).
            consumed_samples = num_frames * self.hop_length

            # .copy() bắt buộc: tránh giữ view vào x (và qua đó giữ chunk lớn sống).
            self._buffer = x[consumed_samples:].copy()
            self._has_produced_frames = True

        # Chỉ đưa đúng phần dữ liệu vừa khít các frame vào tính STFT.
        # Gọi ngoài lock để tránh block thread khác trong thời gian FFT.
        valid_length = (num_frames - 1) * self.hop_length + self.n_fft
        return _core_stft_optimized(
            x=x[:valid_length],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            output_type=self.output_type,
            workers=1  # chunk nhỏ không nên bật multi-thread vì tốn overhead tạo thread
        )

    def flush(self) -> np.ndarray:
        """
        Xử lý phần âm thanh cuối phiên chưa được bao phủ bởi frame hoàn chỉnh.

        Sau khi gọi flush(), state được reset — stream sẵn sàng cho phiên mới.
        Gọi flush() nhiều lần liên tiếp chỉ tạo frame ở lần đầu (nếu có dữ liệu mới);
        các lần sau trả về mảng rỗng ngay lập tức.

        Phân biệt streaming vs batch (center=False):
            - Batch: frame cuối cùng không được tạo nếu tín hiệu không đủ n_fft mẫu.
            - Streaming + flush: frame cuối được tạo với zero-padding nếu có ít nhất
              1 mẫu mới chưa được frame nào bao phủ. Đây là hành vi cố ý khác nhau:
              batch ưu tiên tính chính xác thời gian, streaming ưu tiên không bỏ mẫu.

        Điều kiện tạo frame cuối (zero-padded):
            - Nếu chưa có frame nào được tạo: buffer > 0 mẫu.
            - Nếu đã có frame: buffer phải có hơn (n_fft - hop_length) mẫu, tức là
              có ít nhất 1 mẫu mới chưa nằm trong frame nào trước đó.
        """
        with self._lock:
            buf = self._buffer.copy()          # snapshot độc lập
            has_frames = self._has_produced_frames
            # Reset state — stream sẵn sàng cho phiên mới
            self._buffer = np.empty(0, dtype=np.float32)
            self._has_produced_frames = False

        # Không có dữ liệu tồn đọng
        if len(buf) == 0:
            return _empty_spec(self.n_fft, self.output_type)

        if has_frames:
            # Sau khi process_chunk(), buffer luôn chứa (n_fft - hop_length) mẫu
            # overlap với frame cuối, cộng thêm r mẫu mới (0 <= r < hop_length).
            # Chỉ flush nếu có ít nhất 1 mẫu mới (r > 0).
            overlap = self.n_fft - self.hop_length
            new_samples = len(buf) - overlap
            if new_samples <= 0:
                # Buffer chỉ chứa overlap — không có gì mới để xử lý
                return _empty_spec(self.n_fft, self.output_type)

        # Tạo frame cuối: zero-pad buffer lên n_fft và tính STFT
        padded = np.zeros(self.n_fft, dtype=np.float32)
        padded[:len(buf)] = buf
        return _core_stft_optimized(
            x=padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,  # padded = n_fft mẫu -> luôn đúng 1 frame
            win_length=self.win_length,
            window=self.window,
            center=False,
            output_type=self.output_type,
            workers=1
        )

    def reset(self):
        """Xóa bộ đệm khi bắt đầu phiên hội thoại mới."""
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)
            self._has_produced_frames = False

    def __repr__(self) -> str:
        return (
            f"StreamingSTFT(n_fft={self.n_fft}, hop={self.hop_length}, "
            f"win={self.win_length}, output={self.output_type}, "
            f"buffer_len={len(self._buffer)})"
        )


# ============================================================================
# 4. CÁC HÀM PRESET CHO TỪNG TÁC VỤ TRONG REPO
# ============================================================================

def _next_power_of_two(n: int) -> int:
    """Trả về lũy thừa của 2 nhỏ nhất >= n (dùng cho n_fft)."""
    if n <= 0:
        raise ValueError(f"_next_power_of_two yeu cau n > 0, nhan duoc {n}.")
    return 1 << (n - 1).bit_length()


def create_vad_streamer(sample_rate: int = 16000) -> StreamingSTFT:
    """Khởi tạo bộ STFT liên tục cho VAD (hop 5ms, win 16ms).

    n_fft được tự động tính theo sample_rate thay vì hardcode,
    đảm bảo win_length luôn <= n_fft dù sample_rate thay đổi.
    """
    win_length = int(sample_rate * 0.016)
    hop_length = int(sample_rate * 0.005)
    if win_length <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} qua nho: win_length={win_length}. "
            f"Yeu cau sample_rate >= 63 de co win_length >= 1."
        )
    if hop_length <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} qua nho: hop_length={hop_length}. "
            f"Yeu cau sample_rate >= 200 de co hop_length >= 1."
        )
    n_fft = _next_power_of_two(win_length)
    return StreamingSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        output_type="power"
    )


def create_asr_streamer(sample_rate: int = 16000) -> StreamingSTFT:
    """Khởi tạo bộ STFT liên tục cho ASR Streaming (hop 10ms, win 25ms)."""
    win_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    if win_length <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} qua nho: win_length={win_length}. "
            f"Yeu cau sample_rate >= 40 de co win_length >= 1."
        )
    if hop_length <= 0:
        raise ValueError(
            f"sample_rate={sample_rate} qua nho: hop_length={hop_length}. "
            f"Yeu cau sample_rate >= 100 de co hop_length >= 1."
        )
    n_fft = _next_power_of_two(win_length)
    return StreamingSTFT(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        output_type="power"
    )


def stft_for_asr_patch(x: np.ndarray, sample_rate: int = 16000, workers: int = -1) -> np.ndarray:
    """
    STFT cho ASR Patch Processing (Offline):
    - Bật center=True để khớp nhãn thời gian.
    - Bật workers=-1 để tận dụng đa nhân CPU khi xử lý đoạn âm thanh dài.

    Lưu ý: workers=-1 dùng toàn bộ CPU logical core. Trong môi trường serving
    đa tiến trình, nên đặt workers = os.cpu_count() // so_worker_process.
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

    n_fft được truyền vào ngoài thay vì hardcode, win_length
    tự động clamp <= n_fft để không bao giờ mất sync giữa hai tham số.

    Lưu ý: workers=-1 dùng toàn bộ CPU logical core. Trong môi trường serving
    đa tiến trình, nên đặt workers = os.cpu_count() // so_worker_process.
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
