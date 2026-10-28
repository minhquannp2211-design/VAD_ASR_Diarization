# BÁO CÁO THI CÔNG & ĐÁNH GIÁ PHASE 4: VAD PIPELINE ROUTER

**Tài liệu**: `VAD/report/tầng 4/PHASE_4_PIPELINE_ROUTER_REPORT.md`  
**Giai đoạn**: Phase 4 — Implement `VADPipelineRouter` & Timestamp-Duration Audit  
**Ngày thực hiện**: 28/10/2026  
**Trạng thái hệ thống**:
- Toàn bộ 4 tầng VAD cốt lõi (`VAD/feature_extraction.py`, `VAD/noise_estimator.py`, `VAD/sigmoid.py`, `VAD/state_machine.py`) và `Dictionary/STFT.py` được **bảo toàn nguyên vẹn 100% (frozen core preserved)**.
- Các hợp đồng dữ liệu Phase 1 (`VAD/pipeline/contracts.py`), wrapper streaming (`VAD/pipeline/streaming.py`), và wrapper offline (`VAD/pipeline/offline.py`) được **bảo toàn nguyên vẹn 100%**.
- Toàn bộ các file baseline, pilot audio, ground truth, log và script đối chiếu giữ nguyên vẹn.

---

## 1. Mục tiêu & Nguyên tắc Thiết kế của `VADPipelineRouter`

Phase 4 hoàn thiện tầng điều phối chế độ thực thi (Mode Selection) cho hệ thống VAD bằng việc xây dựng `VADPipelineRouter`:
- **Nguyên tắc Factory Thuần túy (Pure Stateless Factory)**: Router đóng vai trò là một nhà máy tĩnh (factory), chỉ kiểm tra tính hợp lệ và khởi tạo đúng đối tượng thực thi được yêu cầu (`PipelineMode.STREAMING` $\to$ `VADStreamingSession`, `PipelineMode.OFFLINE` $\to$ `VADOfflineRunner`).
- **Không nhân bản trạng thái (Zero State Overhead)**: Router không lưu giữ audio, không giữ trạng thái nội bộ của bộ ước lượng nhiễu hay máy trạng thái, và không tạo cả hai đường ống rồi mới chọn một.
- **Ranh giới rõ ràng**: Router này chỉ thực hiện chọn chế độ ở cấp độ **VAD Pipeline**, hoàn toàn chưa mở rộng sang router toàn hệ sinh thái ASR hay Diarization.

---

## 2. Các File Đã Tạo & Cập nhật

| STT | Đường dẫn file | Thao tác | Trách nhiệm chính |
|---|---|---|---|
| 1 | `VAD/pipeline/router.py` | Tạo mới | Triển khai `VADPipelineRouter` với phương thức `create()` có typing overload đầy đủ và hỗ trợ cú pháp gọi trực tiếp `VADPipelineRouter(...)`. |
| 2 | `VAD/pipeline/__init__.py` | Cập nhật nhỏ | Export `VADPipelineRouter` ra interface công khai của package. |
| 3 | `tests/test_vad_pipeline_router.py` | Tạo mới | 20 test bao phủ chọn chế độ, tính khả dụng streaming/offline, cô lập trạng thái, kiểm thực tham số, bảo toàn parity và audit mốc thời gian. |
| 4 | `VAD/report/tầng 4/PHASE_4_PIPELINE_ROUTER_REPORT.md` | Tạo mới | Báo cáo chi tiết kỹ thuật Phase 4 và kết quả audit mốc thời gian. |

---

## 3. Giao diện Công khai (Public API) & Ví dụ Sử dụng

### 3.1. Phương thức Khởi tạo Factory (`create`)
```python
from VAD.pipeline import PipelineMode, VADPipelineRouter

# 1. Khởi tạo phiên Streaming (trả về VADStreamingSession)
streaming_session = VADPipelineRouter.create(
    mode=PipelineMode.STREAMING, # hoặc chuỗi "streaming"
    sample_rate=16000,
)

# 2. Khởi tạo runner Offline (trả về VADOfflineRunner)
offline_runner = VADPipelineRouter.create(
    mode=PipelineMode.OFFLINE,   # hoặc chuỗi "offline"
    sample_rate=16000,
)
```
- **Hỗ trợ Overload Type Hints**: IDE và công cụ kiểm tra kiểu (Mypy, Pyright) tự động nhận diện kiểu trả về cụ thể (`VADStreamingSession` hoặc `VADOfflineRunner`) tương ứng với giá trị `mode` được truyền vào.
- **Cú pháp Gọi Trực tiếp**: Cung cấp `__new__` để hỗ trợ cả cú pháp `VADPipelineRouter(mode=..., sample_rate=...)` tương đương với `VADPipelineRouter.create(...)`.

### 3.2. Truyền Tham số Cấu hình (Configuration Propagation)
Cả `VADStreamingSession` và `VADOfflineRunner` đều có chung chữ ký khởi tạo:
```python
(
    sample_rate: int = 16000,
    noise_config: Optional[NoiseConfig] = None,
    sigmoid_config: Optional[SigmoidConfig] = None,
    state_machine_config: Optional[StateMachineConfig] = None,
    debug: bool = False,
    return_diagnostics: bool = False,
    bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
)
```
`VADPipelineRouter.create()` chuyển tiếp nguyên vẹn toàn bộ các tham số cấu hình tùy chọn này xuống đối tượng đích mà không làm thay đổi giá trị mặc định hay áp đặt cấu trúc dữ liệu trung gian mới.

### 3.3. Kiểm thực Tính Hợp lệ (Validation)
- **Kiểm thực Mode**: Chấp nhận `PipelineMode` hoặc chuỗi ký tự không phân biệt hoa thường (`"streaming"`, `"STREAMING"`, `"offline"`, `"Offline"`). Bất kỳ chuỗi nào ngoài danh mục này (như `"realtime"`) đều ném `ValueError`. Các kiểu dữ liệu không phải chuỗi hoặc enum ném `TypeError`.
- **Kiểm thực Sample Rate**: Kiểm tra `sample_rate ∈ {8000, 16000}` Hz ngay tại factory. Từ chối mọi tần số ngoài danh mục bằng `ValueError`.

---

## 4. Kết quả Audit Mốc Thời gian & Thời lượng (Timestamp-Duration Audit)

### 4.1. Câu hỏi & Bối cảnh Audit
Báo cáo Phase 3 có kiểm tra điều kiện biên:
$$0.0 \le start\_time\_s < end\_time\_s \le duration\_s + 0.020\text{ s}$$
Nhiệm vụ Phase 4 yêu cầu kiểm tra kỹ lưỡng: Liệu có bất kỳ đoạn tiếng nói nào thực sự có mốc kết thúc vượt quá thời lượng thực của file âm thanh ($end\_time\_s > duration\_s$) hay đây chỉ là một ngưỡng kiểm tra lỏng (loose test bound)?

### 4.2. Bằng chứng Thực nghiệm trên 11 File Pilot WAV
Khảo sát toàn bộ các đoạn tiếng nói phát ra trên 11 file WAV pilot thực tế:
- `E01_stat20_16k` ($dur = 6.200\text{ s}$): Đoạn cuối kết thúc tại $5.345\text{ s}$ (cách đuôi $-0.855\text{ s}$).
- `E02_stat20_8k` ($dur = 6.000\text{ s}$): Đoạn cuối kết thúc tại $5.255\text{ s}$ (cách đuôi $-0.745\text{ s}$).
- `E03_stat15_16k` ($dur = 6.000\text{ s}$): Đoạn cuối kết thúc tại $5.010\text{ s}$ (cách đuôi $-0.990\text{ s}$).
- `E04_stat15_8k` ($dur = 5.700\text{ s}$): Đoạn cuối kết thúc tại $4.225\text{ s}$ (cách đuôi $-1.475\text{ s}$).
- `E05_stat10_16k` ($dur = 6.000\text{ s}$): Đoạn cuối kết thúc tại $5.990\text{ s}$ (cách đuôi $-0.010\text{ s}$).
- `E06_stat10_8k` ($dur = 7.300\text{ s}$): Đoạn cuối kết thúc tại $5.740\text{ s}$ (cách đuôi $-1.560\text{ s}$).
- `E07_step_up_16k` ($dur = 7.200\text{ s}$): Đoạn cuối kết thúc tại $6.700\text{ s}$ (cách đuôi $-0.500\text{ s}$).
- `E08_step_down_16k` ($dur = 6.500\text{ s}$): Đoạn cuối kết thúc tại $5.590\text{ s}$ (cách đuôi $-0.910\text{ s}$).
- `E09_step_up_8k` ($dur = 7.200\text{ s}$): Đoạn cuối kết thúc tại $6.700\text{ s}$ (cách đuôi $-0.500\text{ s}$).
- `E10_start_16k` ($dur = 5.500\text{ s}$): Đoạn cuối kết thúc tại $3.965\text{ s}$ (cách đuôi $-1.535\text{ s}$).
- `E11_start_8k` ($dur = 5.500\text{ s}$): Đoạn cuối kết thúc tại $3.700\text{ s}$ (cách đuôi $-1.800\text{ s}$).

**Kết luận thực nghiệm**: **100% các đoạn tiếng nói trên toàn bộ 11 file WAV pilot đều có $end\_time\_s \le duration\_s$. Không có bất kỳ đoạn nào bị tràn ra ngoài thời lượng file.**

### 4.3. Chứng minh Toán học về Tính Kín của Ranh giới STFT (`center=False`)
Xét file âm thanh có $N$ mẫu, tần số lấy mẫu $f_s$, thời lượng $T_{\text{audio}} = N / f_s$.
Độ dịch bước nhảy $\Delta t = \text{hop\_length} / f_s = 0.005\text{ s}$ (với $\text{hop} = 80$ ở 16 kHz; $\text{hop} = 40$ ở 8 kHz).
Kích thước cửa sổ $n\_fft = 256$ ở 16 kHz; $n\_fft = 128$ ở 8 kHz.

Khi streaming toàn bộ $N$ mẫu rồi `flush()`:
- Số frame trước flush: $num\_frames_{\text{chunk}} = 1 + (N - n\_fft) // \text{hop}$.
- Số frame tối đa sau flush: $total\_frames \le num\_frames_{\text{chunk}} + 1$.
Ta có bất đẳng thức:
$$total\_frames \le 1 + \frac{N - n\_fft}{\text{hop}} + 1 = \frac{N - n\_fft + 2\cdot\text{hop}}{\text{hop}}$$
Mốc thời gian tối đa của frame cuối cùng là:
$$t_{\text{max}} = total\_frames \cdot \Delta t = total\_frames \cdot \frac{\text{hop}}{f_s} \le \frac{N - n\_fft + 2\cdot\text{hop}}{f_s} = T_{\text{audio}} + \frac{2\cdot\text{hop} - n\_fft}{f_s}$$
Tính toán giá trị sai số biên $\delta = \frac{2\cdot\text{hop} - n\_fft}{f_s}$:
- Ở **16 kHz**: $2\cdot\text{hop} - n\_fft = 2(80) - 256 = 160 - 256 = -96 < 0 \implies \delta = \frac{-96}{16000} = -0.006\text{ s}$.
- Ở **8 kHz**: $2\cdot\text{hop} - n\_fft = 2(40) - 128 = 80 - 128 = -48 < 0 \implies \delta = \frac{-48}{8000} = -0.006\text{ s}$.

Vì $\delta < 0$ trên cả 8 kHz và 16 kHz với mọi $N \ge n\_fft$, ta có hệ quả toán học bất biến:
$$total\_frames \cdot 0.005\text{ s} \le T_{\text{audio}} - 0.006\text{ s} < T_{\text{audio}}$$
Bất kỳ đoạn tiếng nói nào được chốt đều có $end\_time\_s \le total\_frames \cdot 0.005\text{ s}$. Do đó:
$$end\_time\_s < T_{\text{audio}}$$
**Kết luận bản chất**: Con số $+0.020\text{ s}$ trong các assertion kiểm thử trước đó chỉ là một biên dung sai bảo thủ (loose upper bound) đề phòng giả định padding, còn trên thực tế thuật toán VAD đảm bảo chặt chẽ $end\_time\_s < duration\_s$ mà không cần bất kỳ thao tác cắt gọt (clipping) nhân tạo nào.

---

## 5. Kết quả Kiểm thử & Nghiệm thu Kỹ thuật

### 5.1. Bộ Kiểm thử Mới của Router (`tests/test_vad_pipeline_router.py`)
Gồm 20 test cases bao phủ toàn bộ các yêu cầu của Section 6:
- **Test A**: Chọn đúng `VADStreamingSession` khi yêu cầu `STREAMING` và `VADOfflineRunner` khi yêu cầu `OFFLINE`.
- **Test B & C**: Kiểm tra tính khả dụng thực tế của đối tượng tạo ra từ router (chạy chunk, flush, chạy whole-waveform).
- **Test D**: Xác nhận không rò rỉ trạng thái giữa các instance độc lập được tạo từ router.
- **Test E**: Ném lỗi rõ ràng khi gặp mode không hợp lệ hoặc sample rate không hỗ trợ.
- **Test F**: Bảo toàn 100% parity so với khởi tạo trực tiếp trên các file đại diện 8 kHz và 16 kHz.
- **Test G**: Kiểm tra import contract từ `VAD.pipeline`.
- **Test H**: Kiểm tra xác thực mốc thời gian mọi đoạn đều thỏa mãn $0.0 \le start < end \le duration\_s$.

**Kết quả chạy**:
```powershell
py -m pytest tests/test_vad_pipeline_router.py -v
```
Output: `20 passed in 5.63s`.

### 5.2. Chạy Toàn bộ Hệ thống Kiểm thử Hồi quy (Full Regression Suite)
```powershell
py -m pytest -q
```
Output: `298 passed in 84.15s` (278 baseline ban đầu + 20 test Phase 4 mới, 0 fail, 0 warning).

---

## 6. Xác nhận Bảo toàn Mã Nguồn Cốt lõi & Dữ liệu Gốc

Kiểm tra bằng `git status --porcelain`:
- Không có bất kỳ file sản xuất cốt lõi nào bị thay đổi:
  - `Dictionary/STFT.py` (Unchanged)
  - `VAD/feature_extraction.py` (Unchanged)
  - `VAD/noise_estimator.py` (Unchanged)
  - `VAD/sigmoid.py` (Unchanged)
  - `VAD/state_machine.py` (Unchanged)
  - `VAD/pipeline/contracts.py` (Unchanged)
  - `VAD/pipeline/streaming.py` (Unchanged)
  - `VAD/pipeline/offline.py` (Unchanged)
- File duy nhất được cập nhật là `VAD/pipeline/__init__.py` (chỉ export bổ sung `VADPipelineRouter`).
- Không có file baseline CSV, ground truth hay audio WAV nào bị ảnh hưởng.

---

## 7. Đánh giá Mức độ Sẵn sàng cho Tích hợp ASR (ASR Integration Readiness)

Hệ thống VAD Multi-Band đã chính thức hoàn thiện toàn bộ tầng Pipeline Architecture (Phases 1–4):
1. **Shared Contracts**: `VADSegment`, `VADDiagnostics`, `VADResult`, `PipelineMode` hoạt động ổn định.
2. **Causal Streaming Pipeline**: `VADStreamingSession` xử lý trực tuyến chunk-by-chunk với độ trễ thấp và flush chuẩn xác.
3. **Causal Offline Baseline**: `VADOfflineRunner` xử lý whole-waveform với 100% parity số học so với luồng streaming.
4. **Pipeline Factory Router**: `VADPipelineRouter` cho phép caller chọn chế độ thực thi một cách tập trung, an toàn và tường minh về kiểu dữ liệu.

**Định hướng Tích hợp Tầng Kế tiếp (ASR Integration)**:
- Đối với đường ống **Streaming ASR**: Tầng ASR tiêu thụ trực tiếp các đoạn `VADSegment` phát ra tức thời từ `VADStreamingSession.process_chunk()`.
- Đối với đường ống **Offline ASR**: Tầng ASR nhận danh sách `VADSegment` hoàn chỉnh từ `VADOfflineRunner.process_audio()`, sau đó cắt các đoạn waveform tương ứng để đưa vào mô hình nhận dạng âm thanh.
