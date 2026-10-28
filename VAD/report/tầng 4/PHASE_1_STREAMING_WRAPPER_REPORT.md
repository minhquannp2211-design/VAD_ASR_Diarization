# BÁO CÁO THI CÔNG & KIỂM CHỨNG PHASE 1: SHARED CONTRACTS & VADSTREAMINGSESSION

**Tài liệu**: `VAD/report/tầng 4/PHASE_1_STREAMING_WRAPPER_REPORT.md`  
**Giai đoạn**: Phase 1 — Shared Contracts & Streaming Wrapper (`VADStreamingSession`)  
**Ngày thực hiện**: 28/10/2026  
**Trạng thái hệ thống**: 
- Toàn bộ 4 tầng VAD cốt lõi (`VAD/feature_extraction.py`, `VAD/noise_estimator.py`, `VAD/sigmoid.py`, `VAD/state_machine.py`) và `Dictionary/STFT.py` được **bảo toàn nguyên vẹn 100% (frozen core preserved)**.
- Toàn bộ các file baseline, pilot audio, ground truth, log và script đối chiếu giữ nguyên vẹn.

---

## 1. Mục tiêu & Phạm vi Thực hiện

Phase 1 triển khai các hợp đồng dữ liệu dùng chung (Shared Data Contracts) và lớp bao bọc trực tuyến mỏng (`VADStreamingSession`) cho chuỗi xử lý 4 tầng Multi-band Adaptive VAD:
- Thiết lập nền móng cho kiến trúc đường ống kép (`VAD_STREAMING` và `VAD_OFFLINE`).
- Giữ nguyên 100% thuật toán hiện hành, không can thiệp vào các tham số quyết định hay logic nội bộ của các tầng 1–4.
- Đảm bảo **tương đương số học tuyệt đối (exact numerical parity)** với hàm tham chiếu `run_multiband_streaming()` trong `VAD/report/tầng 4/human_annotation_pilot/benchmark_multiband_vs_silero.py` trên toàn bộ 11 file WAV của tập pilot ở tất cả các chế độ chia chunk (20 ms, 100 ms, toàn file).

---

## 2. Các File Đã Tạo & Trách nhiệm Từng File

| STT | Đường dẫn file | Trách nhiệm chính |
|---|---|---|
| 1 | `VAD/pipeline/__init__.py` | Export công khai: `PipelineMode`, `VADSegment`, `VADDiagnostics`, `VADResult`, `VADStreamingSession`. |
| 2 | `VAD/pipeline/contracts.py` | Định nghĩa các dataclass bất biến (`frozen=True`) và enum: `VADSegment` (đoạn tiếng nói nửa mở `[start_time_s, end_time_s)`), `VADDiagnostics` (thông tin chẩn đoán tùy chọn), `VADResult` (kết quả chuẩn hóa), `PipelineMode` (`STREAMING`, `OFFLINE`). |
| 3 | `VAD/pipeline/streaming.py` | Hiện thực `VADStreamingSession` điều phối 4 tầng VAD với vòng đời `process_chunk()` $\to$ `flush()` $\to$ `reset()`. |
| 4 | `tests/test_pipeline_contracts.py` | 20 unit test kiểm tra việc khởi tạo, kiểm thực tính hợp lệ (validation), khoảng nửa mở, mốc thời gian không âm, phát hiện lỗi biên và tính năng tích lũy kết quả của các hợp đồng. |
| 5 | `tests/test_vad_streaming.py` | 43 test kiểm tra tính tương đương tuyệt đối với `run_multiband_streaming` trên 11 file pilot WAV (20 ms, 100 ms, 1 khối), tính cô lập trạng thái qua `reset()`, tính lũy đẳng của `flush()`, xử lý stream rỗng, kiểm tra mốc thời gian và tính bất biến qua biên chunk bất kỳ. |
| 6 | `VAD/report/tầng 4/PHASE_1_STREAMING_WRAPPER_REPORT.md` | Báo cáo chi tiết kỹ thuật của Phase 1. |

---

## 3. Giao diện Công khai (Public API) của `VADStreamingSession`

### 3.1. Khởi tạo & Cấu hình
```python
session = VADStreamingSession(
    sample_rate: int = 16000,
    noise_config: Optional[NoiseConfig] = None,
    sigmoid_config: Optional[SigmoidConfig] = None,
    state_machine_config: Optional[StateMachineConfig] = None,
    debug: bool = False,
    return_diagnostics: bool = False,
    bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
)
```
- **Hỗ trợ tần số lấy mẫu**: Chỉ chấp nhận `sample_rate ∈ {8000, 16000}` Hz. Ném `ValueError` ngay lập tức nếu nhận tần số không được hỗ trợ (không tự ý resample ngầm).
- **Tự động liên kết tham số**: Tầng 2 và 3 nhận `num_bands = fe.num_bands`; Tầng 4 nhận `hop_seconds = fe.hop_seconds` (5 ms), triệt tiêu hoàn toàn rủi ro sai lệch hop giữa các tầng.

### 3.2. Vòng đời Xử lý & Các hàm Thành viên
- **`process_chunk(chunk: np.ndarray, return_diagnostics: Optional[bool] = None) -> VADResult`**:
  - Nhận mảng 1-D mono (float32/int16 thực).
  - Kiểm tra tính hợp lệ: từ chối None, mảng đa chiều, mảng số phức hoặc kiểu không phải số. Khi `debug=True`, kiểm tra NaN/Inf.
  - Trả về `VADResult` chứa các đoạn `VADSegment` **mới chốt** trong chunk này.
  - Ngăn chặn gọi `process_chunk()` sau khi đã `flush()` mà chưa `reset()` bằng `RuntimeError`.
- **`flush(return_diagnostics: Optional[bool] = None) -> VADResult`**:
  - Đẩy phần đệm còn lại của STFT (nếu có mẫu mới) qua Layers 2–4.
  - Chốt đoạn cuối cùng trong bộ đệm `_pending_seg` của `VADStateMachine`.
  - Trả về `VADResult` chứa các đoạn cuối cùng được chốt.
  - Hoạt động lũy đẳng (idempotent): các lần gọi `flush()` lặp lại trả về `VADResult` rỗng một cách an toàn mà không sinh trùng lặp đoạn hoặc báo lỗi.
- **`reset() -> None`**:
  - Làm mới toàn bộ 4 tầng (`_fe.reset()`, `_est.reset()`, `_scorer.reset()`, `_sm.reset()`).
  - Xóa bộ đếm mẫu, bộ đếm frame và danh sách đoạn tích lũy, đảm bảo stream tiếp theo hoàn toàn độc lập và không bị rò rỉ trạng thái.

### 3.3. Các Thuộc tính Truy vấn (Properties)
- `session.sample_rate`: Tần số lấy mẫu (8000 hoặc 16000 Hz).
- `session.hop_seconds`: Bước nhảy thời gian giữa 2 frame (0.005 s = 5 ms).
- `session.num_bands`: Số dải tần số (mặc định 4 dải).
- `session.total_frames`: Tổng số frame STFT đã được trích xuất và xử lý trong luồng.
- `session.total_samples`: Tổng số mẫu âm thanh đã nạp vào session.
- `session.total_duration_s`: Tổng thời lượng âm thanh đã nhận (`total_samples / sample_rate`).
- `session.accumulated_segments`: Danh sách toàn bộ các đoạn `VADSegment` đã chốt từ khi bắt đầu stream.
- `session.is_flushed`: Cờ đánh dấu session đã được flush hay chưa.
- `session.mode`: `PipelineMode.STREAMING`.

---

## 4. Định nghĩa Hợp đồng & Quy ước Mốc Thời gian

### 4.1. `VADSegment`
- **Quy ước**: Khoảng nửa mở `[start_time_s, end_time_s)` tính bằng giây.
- **Ràng buộc kiểm thực**:
  - `start_time_s` và `end_time_s` phải là số thực hữu hạn (`np.isfinite`), không âm ($\ge 0.0$).
  - `end_time_s > start_time_s` (thời lượng $\Delta t > 0$).
  - Không âm thầm kẹp (clamp) các giá trị lỗi; ném `ValueError` / `TypeError` tương thích với quy ước repo.
- **Độ chính xác số học**: Các mốc thời gian sinh ra từ `VADStateMachine` được làm tròn microsecond (`round(float, 6)`) khớp 100% với hàm tham chiếu `run_multiband_streaming`, triệt tiêu sai số trôi dấu phẩy động IEEE 754.
- **Hỗ trợ tương thích**: Cung cấp `start_time`, `end_time`, `duration_s`, `duration`, `to_tuple()`, và hỗ trợ tuple unpacking (`s, e = seg`).

### 4.2. `VADResult` & Ngữ nghĩa Tích lũy (Result Aggregation)
- `process_chunk()` và `flush()` trả về `VADResult.segments` chỉ chứa **các đoạn mới hoàn tất tại lần gọi đó**.
- Caller khi gom kết quả qua từng lần gọi (`all_segments.extend(res.segments)`) sẽ nhận được danh sách đoạn hoàn chỉnh **không có bất kỳ đoạn nào bị trùng lặp hay bỏ sót**.
- `VADStreamingSession.accumulated_segments` đồng thời lưu trữ danh sách tích lũy này để caller có thể truy vấn trực tiếp bất cứ lúc nào.

### 4.3. `VADDiagnostics`
- Chỉ được tạo khi caller yêu cầu (`return_diagnostics=True` hoặc truyền qua hàm gọi).
- Chứa các mảng 1-D đồng bộ độ dài: `frame_scores` (float32), `speech_mask` (bool), `states` (int8), `warmup_mask` (bool).
- Tùy chọn chứa `noise_floor` và `snr_db` (shape `(n_bands, T)`).

---

## 5. Chuỗi Gọi Hàm & Thứ tự Flush

### 5.1. Khi xử lý chunk (`process_chunk`)
```
Audio Chunk x[n] (mono float32)
   │
   ▼
1. FeatureExtractor.process_chunk(chunk) ──> feats (VADFeatures)
   │
   ├── Nếu feats.num_frames > 0:
   │      │
   │      ├── 2. NoiseEstimator.update(feats) ──> n_out (NoiseOutput)
   │      │
   │      ├── 3. SigmoidScorer.score(feats, n_out) ──> s_out (SigmoidOutput)
   │      │
   │      └── 4. VADStateMachine.process_chunk(s_out) ──> sm_out (StateMachineOutput)
   │             └── Chuyển đổi sm_out.segments thành List[VADSegment]
   │
   └── Nếu feats.num_frames == 0 (chunk nhỏ hơn độ dài n_fft cần thiết):
          └── Giữ mẫu trong buffer STFT, trả về VADResult(segments=[], num_frames=0)
```

### 5.2. Khi kết thúc luồng (`flush`)
Thứ tự flush được bảo toàn nghiêm ngặt theo đúng implementation của `benchmark_multiband_vs_silero.py`:
```
1. FeatureExtractor.flush() ──> feats_tail (VADFeatures)
   │
   ├── Nếu feats_tail.num_frames > 0 (buffer STFT có mẫu mới dư, zero-padded thành 1 frame cuối):
   │      ├── 2. NoiseEstimator.update(feats_tail) ──> n_tail
   │      ├── 3. SigmoidScorer.score(feats_tail, n_tail) ──> s_tail
   │      └── 4. VADStateMachine.process_chunk(s_tail) ──> sm_tail
   │             └── Thu thập sm_tail.segments
   │
   └── 5. VADStateMachine.flush() ──> sm_flush_segs
          └── Đóng đoạn đang mở nếu đang ở SPEECH/HANGOVER và chốt _pending_seg
                 (nếu speech_evidence_frames >= min_speech_frames)
```

---

## 6. Kết quả Kiểm thử & Bằng chứng Tương đương Tuyệt đối (Parity Proof)

### 6.1. Danh sách 11 File WAV Pilot Đã Kiểm thử
Bộ test tự động `tests/test_vad_streaming.py` chạy trên toàn bộ 11 file WAV của pilot dataset (tổng cộng 69.1 s, 13,820 frame 5 ms):

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

### 6.2. Kết quả Chạy Kiểm thử Độc lập (New Tests)
Lệnh thực thi:
```powershell
py -m pytest tests/test_pipeline_contracts.py tests/test_vad_streaming.py -v
```
Kết quả thực tế:
- **`tests/test_pipeline_contracts.py`**: 20/20 test PASS.
- **`tests/test_vad_streaming.py`**: 43/43 test PASS:
  - 11/11 test Test A (20 ms chunks): 100% exact parity.
  - 11/11 test Test B (100 ms chunks): 100% exact parity.
  - 11/11 test Test C (1 full-array chunk): 100% exact parity.
  - 1 test Test D (reset & isolation across streams): PASS.
  - 8 test Test E (empty stream, repeated flush, lifecycle errors, invalid inputs, bounds): PASS.
  - 1 test Test F (arbitrary chunk boundaries 7 ms, 13 ms, 33 ms across onsets/offsets): PASS.
- **Tổng cộng**: `63 passed in 15.05s`.

### 6.3. Kết quả Chạy Toàn bộ Bộ Test Hồi quy (Full Regression Suite)
Lệnh thực thi:
```powershell
py -m pytest -q
```
Kết quả thực tế:
- Baseline trước khi làm Phase 1: 160 test pass.
- Sau khi bổ sung Phase 1: **223 passed in 38.62s** (160 pre-existing + 20 contract + 43 streaming parity).
- Không có bất kỳ test hồi quy nào bị fail hoặc bị bỏ qua.

---

## 7. Xác nhận Bảo toàn Mã Nguồn Cốt lõi & Dữ liệu Gốc

Kiểm tra bằng `git status --porcelain`:
- Các file sản xuất cốt lõi **hoàn toàn không bị sửa đổi**:
  - `Dictionary/STFT.py` (giữ nguyên các thay đổi có sẵn trước Phase 1 của user)
  - `VAD/feature_extraction.py` (không đổi)
  - `VAD/noise_estimator.py` (không đổi)
  - `VAD/sigmoid.py` (không đổi)
  - `VAD/state_machine.py` (không đổi)
- Các file benchmark tham chiếu, CSV nhãn pilot, WAV audio giữ nguyên 100%.
- Không bổ sung bất kỳ dependency ngoài nào vào môi trường.

---

## 8. Hạn chế Hiện tại & Khuyến nghị cho Phase 2

### 8.1. Hạn chế Kỹ thuật Đã Xác nhận
1. **Hạn chế Warm-up 200 ms của thuật toán nhân quả**: Khi tiếng nói xuất hiện ngay tại `t=0` (như trên `E10_start_16k` và `E11_start_8k`), 200 ms warm-up đầu file vẫn bị khóa về SILENCE và noise floor bị học nhầm năng lượng câu đầu. Đây là đặc tính vốn có của thuật toán streaming nhân quả hiện tại, được bảo toàn nguyên vẹn trong Phase 1 này.
2. **`VAD_OFFLINE` chưa được thi công**: Identifier `PipelineMode.OFFLINE` hiện được khai báo trước trong hợp đồng nhưng chưa có runner.

### 8.2. Đề xuất Bước Tiếp theo (Phase 2)
- **Tiến hành Phase 2: Xây dựng Causal Offline Wrapper (`VADOfflineRunner` trong `VAD/pipeline/offline.py`)**.
- `VADOfflineRunner` sẽ tiếp nhận toàn bộ file âm thanh một lần và gọi đường ống nhất quán để đạt 100% parity với `VADStreamingSession`, hoàn tất giai đoạn Baseline Offline Wrapper trước khi tiến tới các thuật toán phi nhân quả trong tương lai.
