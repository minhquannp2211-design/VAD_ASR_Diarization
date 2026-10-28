# BÁO CÁO THI CÔNG & KIỂM CHỨNG PHASE 2: CAUSAL OFFLINE BASELINE (VADOFFLINERUNNER)

**Tài liệu**: `VAD/report/tầng 4/PHASE_2_OFFLINE_BASELINE_REPORT.md`  
**Giai đoạn**: Phase 2 — Implement `VADOfflineRunner`: Causal Offline Baseline  
**Ngày thực hiện**: 28/10/2026  
**Trạng thái hệ thống**:
- Toàn bộ 4 tầng VAD cốt lõi (`VAD/feature_extraction.py`, `VAD/noise_estimator.py`, `VAD/sigmoid.py`, `VAD/state_machine.py`) và `Dictionary/STFT.py` được **bảo toàn nguyên vẹn 100% (frozen core preserved)**.
- Các hợp đồng dữ liệu Phase 1 (`VAD/pipeline/contracts.py`) và wrapper streaming (`VAD/pipeline/streaming.py`) được **bảo toàn nguyên vẹn 100%**.
- Toàn bộ các file baseline, pilot audio, ground truth, log và script đối chiếu giữ nguyên vẹn.

---

## 1. Mục tiêu & Bản chất Kỹ thuật của `VADOfflineRunner`

Phase 2 triển khai runner xử lý toàn bộ file âm thanh (`VADOfflineRunner` trong `VAD/pipeline/offline.py`):
- Thiết lập đường ống thực thi `VAD_OFFLINE` hoàn chỉnh đầu tiên.
- **Phân định rõ ràng**: Đây là **Causal Offline Baseline (Baseline Offline nhân quả)** — một wrapper bao bọc lại chính xác thuật toán nhân quả hiện hành nhưng tiếp nhận toàn bộ waveform của file âm thanh cùng một lúc.
- **Không phải thuật toán offline phi nhân quả**: Chưa đưa vào các kỹ thuật ước lượng nhiễu toàn file (min-tracking toàn cục), làm mượt hai chiều (forward-backward scoring), hay hậu xử lý phi nhân quả trong giai đoạn này.
- **Tiêu chí nghiệm thu cốt lõi**: Đảm bảo **tương đương số học tuyệt đối (100% exact numerical parity)** với `VADStreamingSession` và hàm tham chiếu `run_multiband_streaming()` trên cùng một waveform hoàn chỉnh.

---

## 2. Các File Đã Tạo & Cập nhật

| STT | Đường dẫn file | Thao tác | Trách nhiệm chính |
|---|---|---|---|
| 1 | `VAD/pipeline/offline.py` | Tạo mới | Triển khai `VADOfflineRunner` tiếp nhận toàn bộ waveform, khởi tạo session độc lập (hermetic), gom kết quả và xuất `VADResult`. |
| 2 | `VAD/pipeline/__init__.py` | Cập nhật nhỏ | Export bổ sung `VADOfflineRunner` ra public package interface. |
| 3 | `tests/test_vad_offline.py` | Tạo mới | 34 test kiểm tra tương đương tuyệt đối với `VADStreamingSession` và reference `run_multiband_streaming` trên 11 file WAV pilot, kiểm tra tính cô lập trạng thái giữa các file, tính tất định khi chạy lặp lại, và các trường hợp biên. |
| 4 | `VAD/report/tầng 4/PHASE_2_OFFLINE_BASELINE_REPORT.md` | Tạo mới | Báo cáo chi tiết kỹ thuật Phase 2. |

---

## 3. Giao diện Công khai (Public API) & Ví dụ Sử dụng

### 3.1. Khởi tạo & Cấu hình
```python
from VAD.pipeline import VADOfflineRunner

runner = VADOfflineRunner(
    sample_rate: int = 16000,
    noise_config: Optional[NoiseConfig] = None,
    sigmoid_config: Optional[SigmoidConfig] = None,
    state_machine_config: Optional[StateMachineConfig] = None,
    debug: bool = False,
    return_diagnostics: bool = False,
    bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
)
```
- Hỗ trợ chính xác hai tần số lấy mẫu: `sample_rate ∈ {8000, 16000}` Hz.
- Từ chối mọi tần số không được hỗ trợ bằng `ValueError` (không tự ý resample ngầm).

### 3.2. Phương thức Thực thi Cốt lõi
```python
result = runner.process_audio(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    return_diagnostics: Optional[bool] = None,
) -> VADResult
```
- **`audio`**: Mảng 1-D mono (`float32` hoặc `int16`/`float64` tự động ép kiểu an toàn).
- **`sample_rate`**: Tùy chọn ghi đè sample rate (phải là 8000 hoặc 16000 Hz). Mặc định lấy theo runner.
- **`return_diagnostics`**: Tùy chọn yêu cầu sinh `VADDiagnostics` toàn file.
- **Alias tương thích**: `runner.process_file(audio, ...)` được cung cấp như một alias gọi trực tiếp đến `process_audio()`, tương thích hoàn toàn với quy ước trong báo cáo kiểm toán Phase 0.

### 3.3. Ví dụ Sử dụng
```python
import numpy as np
from VAD.pipeline import VADOfflineRunner

runner = VADOfflineRunner(sample_rate=16000)

# Xử lý toàn bộ waveform
waveform = np.random.randn(16000 * 5).astype(np.float32) * 0.05
result = runner.process_audio(waveform)

print(f"Tổng số frame STFT: {result.num_frames}")
print(f"Thời lượng: {result.duration_s:.3f} s")
for seg in result.segments:
    print(f"Đoạn tiếng nói: [{seg.start_time_s:.3f}, {seg.end_time_s:.3f})s - thời lượng {seg.duration_s:.3f}s")
```

---

## 4. Chiến lược Thực thi Tầng & Thứ tự Flush

`VADOfflineRunner` tái sử dụng trực tiếp `VADStreamingSession` đã được kiểm chứng ở Phase 1:

1. **Khởi tạo session độc lập (Hermetic session per task)**: Mỗi lần gọi `process_audio()`, một đối tượng `VADStreamingSession` mới tinh được tạo lập với đúng cấu hình. Điều này triệt tiêu hoàn toàn nguy cơ rò rỉ trạng thái (state leakage) của file trước đó sang file sau.
2. **Nạp toàn bộ waveform như một khối duy nhất (Whole-waveform chunk)**:
   ```python
   res_chunk = session.process_chunk(audio, return_diagnostics=want_diag)
   ```
   Waveform đi qua chuỗi:
   `FeatureExtractor.process_chunk(audio)` $\to$ `NoiseEstimator.update()` $\to$ `SigmoidScorer.score()` $\to$ `VADStateMachine.process_chunk()`.
3. **Thực hiện flush ngay lập tức**:
   ```python
   res_flush = session.flush(return_diagnostics=want_diag)
   ```
   Phần đệm dư cuối cùng của STFT (nếu có mẫu mới) được zero-pad thành 1 frame cuối qua Layers 2–4, sau đó `VADStateMachine.flush()` chốt đoạn đang chờ (`_pending_seg`) hoặc đoạn đang mở (`SPEECH`/`HANGOVER`).
4. **Hợp nhất kết quả & Chẩn đoán**:
   - `all_segments = res_chunk.segments + res_flush.segments`
   - `total_frames = session.total_frames`
   - `total_duration = len(audio) / sr`
   - Nếu có yêu cầu diagnostics, các mảng `frame_scores`, `speech_mask`, `states`, `warmup_mask`, `noise_floor`, `snr_db` từ `res_chunk` và `res_flush` được ghép nối liền mạch trên trục thời gian để phủ kín 100% `total_frames`.

---

## 5. Kết quả Kiểm thử & Bằng chứng Tương đương Số học (Parity Proof)

### 5.1. Danh sách 11 File WAV Pilot Đã Kiểm thử
Bộ test tự động `tests/test_vad_offline.py` chạy trên toàn bộ 11 file WAV pilot (tổng cộng 69.1 s, 13,820 frame 5 ms):

1. `E01_stat20_16k` (16 kHz, 6.000 s, stationary noise +20 dB)
2. `E02_stat20_8k` (8 kHz, 6.000 s, stationary noise +20 dB)
3. `E03_stat15_16k` (16 kHz, 6.000 s, stationary noise +15 dB)
4. `E04_stat15_8k` (8 kHz, 6.000 s, stationary noise +15 dB)
5. `E05_stat10_16k` (16 kHz, 6.000 s, stationary noise +10 dB)
6. `E06_stat10_8k` (8 kHz, 6.000 s, stationary noise +10 dB)
7. `E07_step_up_16k` (16 kHz, 7.000 s, noise step-up +15 dB)
8. `E08_step_down_16k` (16 kHz, 7.000 s, noise step-down -15 dB)
9. `E09_step_up_8k` (8 kHz, 7.000 s, noise step-up +15 dB)
10. `E10_start_16k` (16 kHz, 6.100 s, speech at start t=0)
11. `E11_start_8k` (8 kHz, 6.000 s, speech at start t=0)

### 5.2. Các Hạng mục Kiểm thử Cụ thể (34 Test Cases)
- **Test A (11 test)**: Đối chiếu từng cặp đoạn $[start, end)$ và số frame giữa `VADOfflineRunner` và `VADStreamingSession` (chạy 1 chunk + flush).
  - **Kết quả**: 11/11 file trùng khớp 100% đến từng microsecond (`abs_tol=1e-6`).
- **Test B (11 test)**: Đối chiếu với hàm tham chiếu `run_multiband_streaming(audio, sr, chunk_duration_s=None)` từ `benchmark_multiband_vs_silero.py`.
  - **Kết quả**: 11/11 file trùng khớp 100% về cả danh sách tuple đoạn và tổng số frame.
- **Test C (1 test)**: Xử lý file A (`E01`) rồi đến file B (`E03`) trên cùng một instance `VADOfflineRunner`. Đối chiếu kết quả của B với một runner mới tinh.
  - **Kết quả**: Hoàn toàn trùng khớp, không có bất kỳ rò rỉ trạng thái hay nhiễu nền nào giữa 2 file.
- **Test D (1 test)**: Chạy cùng một file 3 lần liên tiếp trên cùng một instance runner.
  - **Kết quả**: Đầu ra tất định 100%, không tích lũy thêm đoạn hay thay đổi kết quả.
- **Test E (10 test)**:
  - Kiểm tra `runner.mode == PipelineMode.OFFLINE`.
  - Kiểm tra alias `process_file` khớp với `process_audio`.
  - Kiểm tra ghi đè `sample_rate` trong lời gọi hàm.
  - Kiểm tra mảng rỗng: trả về `segments=[]`, `num_frames=0`, `duration_s=0.0`.
  - Kiểm tra từ chối tần số lấy mẫu không hợp lệ (44.1 kHz, 22.05 kHz).
  - Kiểm tra từ chối đầu vào không hợp lệ: `None`, mảng 2D, số phức, chuỗi ký tự.
  - Kiểm tra độ bao phủ chẩn đoán (`VADDiagnostics`) khớp chính xác độ dài `total_frames`.
  - Kiểm tra mốc thời gian mọi đoạn đều nằm trong biên độ âm thanh $0.0 \le start < end \le dur + 0.020$ s.
  - Kiểm tra hành vi trên file có tiếng nói ngay đầu (`E10`, `E11`): đoạn đầu chốt tại đúng `0.625 s` theo đúng thiết kế nhân quả.

---

## 6. Lệnh Test Đã Chạy & Kết quả Thực tế

### 6.1. Kiểm thử Pipeline Suites (Phase 1 + Phase 2)
```powershell
py -m pytest tests/test_vad_offline.py tests/test_vad_streaming.py tests/test_pipeline_contracts.py -q
```
**Kết quả**: `97 passed in 23.95s` (34 offline + 43 streaming + 20 contracts).

### 6.2. Kiểm thử Hồi quy Toàn bộ Hệ thống (Full Regression Suite)
```powershell
py -m pytest -q
```
**Kết quả**: `257 passed in 58.91s` (160 baseline ban đầu + 63 Phase 1 + 34 Phase 2, 0 fail, 0 warning).

---

## 7. Phân biệt Bản chất: Causal Offline Baseline vs Future Truly Offline Algorithm

| Đặc tính kỹ thuật | Causal Offline Baseline (`VADOfflineRunner` hiện tại) | Thuật toán Offline Thực thụ trong tương lai (Phase 4+) |
|---|---|---|
| **Hướng nhân quả (Causality)** | Nhân quả (Causal): chỉ dùng quá khứ và hiện tại $t' \le t$. | Phi nhân quả (Non-causal): dùng toàn bộ ngữ cảnh tương lai $t' > t$ của file âm thanh. |
| **Giai đoạn Khởi động (Warm-up)** | 200 ms đầu file (`init_frames = 40`) bị khóa `warmup=True`, `score=0.0` (SILENCE). | Không cần warm-up: phân vị tối thiểu (Minimum Statistics) hoặc phân tích phổ toàn file ước lượng ngay nền nhiễu chính xác tại $t=0$. |
| **Tiếng nói tại $t=0$ (`E10`, `E11`)** | Bỏ sót 625 ms đầu do median warm-up học nhầm công suất tiếng nói. | Phát hiện chính xác ngay từ âm tiết đầu câu $t=0$. |
| **Làm mượt điểm số** | Cửa sổ trượt nhân quả (Forward-only hystersis). | Làm mượt 2 chiều (Bidirectional smoothing / Forward-backward scoring). |
| **Mục đích thiết kế** | Cung cấp giao diện whole-file tiện lợi, bảo toàn 100% parity số học với luồng streaming. | Tối ưu hóa tối đa F1 và ranh giới đoạn trên file đã ghi âm hoàn chỉnh. |

---

## 8. Đề xuất Bước Tiếp theo (Phase 3)

### Mục tiêu & Tiêu chí Nghiệm thu Đề xuất cho Phase 3:
1. **Mục tiêu**: Tích hợp benchmark tự động và đánh giá parity quy mô lớn giữa `VAD_STREAMING` và `VAD_OFFLINE` trên toàn bộ tập dữ liệu (cả tập pilot và các file benchmark mở rộng).
2. **Tiêu chí nghiệm thu**:
   - Tự động hóa kiểm thử cross-pipeline: Chạy một script benchmark tổng hợp xác nhận 100% phân đoạn của `VADStreamingSession` và `VADOfflineRunner` khớp nhau trên toàn bộ các kịch bản nhiễu (dừng, nhảy bậc, tiếng nói đầu luồng) ở cả 8 kHz và 16 kHz.
   - Không làm thay đổi kết quả số học đã công bố của Stage C2 và Stage D1.
