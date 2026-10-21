import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

def stft(signal, fs=1000, nperseg=256, hop_size=64):
    # 1. Khởi tạo hàm cửa sổ (Ví dụ: cửa sổ Hann)
    win = hann(nperseg)

    # 2. Cấu hình bộ biến đổi ShortTimeFFT chuẩn mới của SciPy
    SFT = ShortTimeFFT(win, hop=hop_size, fs=fs, scale_to='magnitude')

    # 3. Thực hiện biến đổi để lấy ma trận số phức
    stft_matrix = SFT.stft(signal)

    # 4. Trích xuất các trục tọa độ thực tế (Tần số và Thời gian) nếu cần dùng
    frequencies = SFT.f
    times = SFT.t(len(signal))

    return stft_matrix, frequencies, times
