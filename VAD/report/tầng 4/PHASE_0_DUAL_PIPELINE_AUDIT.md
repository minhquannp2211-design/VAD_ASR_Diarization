# BÁO CÁO KIỂM TOÁN KIẾN TRÚC VAD HAI CHẾ ĐỘ THỰC THI (PHASE 0 DUAL-PIPELINE AUDIT)

**Tài liệu**: `VAD/report/tầng 4/PHASE_0_DUAL_PIPELINE_AUDIT.md`  
**Giai đoạn**: Phase 0 — Audit & Planning (Chỉ khảo sát và lập kế hoạch, không sửa code sản xuất)  
**Ngày thực hiện**: 27/10/2026  
**Trạng thái hệ thống**: Toàn bộ 4 tầng VAD (`VAD/feature_extraction.py`, `VAD/noise_estimator.py`, `VAD/sigmoid.py`, `VAD/state_machine.py`) và `Dictionary/STFT.py` được **đóng băng (frozen)** 100%.

---

## 1. Kiến trúc hiện hành & Chuỗi gọi hàm (Current Architecture & Call Sequence)

### 1.1. Luồng thực thi Streaming hiện tại
Hiện tại, toàn bộ hệ thống Multi-Band Adaptive VAD được thiết kế và vận hành theo mô hình chuỗi thời gian trực tuyến (causal streaming):

```
Audio Chunk x[n] (1-D mono float32)
   │
   ▼
[Tầng 1: FeatureExtractor.process_chunk()]
   │ ├── Dictionary/STFT.StreamingSTFT.process_chunk()  ──> Power Spectrogram (n_bins, T)
   │ ├── compute_band_power(spec, band_weights)          ──> band_power (4, T) [công suất trung bình/bin]
   │ └── compute_sfm(spec, sfm_mask)                     ──> sfm (T,) [80-4000 Hz, [0, 1]]
   ▼ VADFeatures(band_power, sfm)
[Tầng 2: NoiseEstimator.update()]
   │ ├── Giai đoạn Warm-up (tối đa init_frames = 40 frame = 200 ms @ 5ms):
   │ │   └── noise_floor = median(band_power), snr_db = 0.0, warmup = True
   │ ├── Cổng thích nghi theo từng băng sau warm-up (strong / weak / speech / freeze)
   │ └── Cơ chế chống kẹt (stuck detection sau 500 frame freeze, recovery sau 100 frame ổn định)
   ▼ NoiseOutput(noise_floor, snr_db, update_mask, stuck, warmup, is_noise)
[Tầng 3: SigmoidScorer.score()]
   │ ├── Xử lý hợp đồng warmup: nếu warmup[t] == True -> logit = -20.0, score = 0.0, snr_norm = 0.0
   │ ├── Chuẩn hóa SNR tuyến tính từng phần về [0, 1] trên dải [snr_low_db=0, snr_high_db=20]
   │ ├── z(t) = bias + sum(w_j * snr_norm_j) + w_f * (1 - sfm)
   │ └── p(t) = 1 / (1 + exp(-z(t))) (ổn định số học, uncalibrated score ∈ [0, 1])
   ▼ SigmoidOutput(score, logit, snr_norm, flatness_complement, warmup)
[Tầng 4: VADStateMachine.process_chunk()]
   │ ├── Chuỗi trạng thái: SILENCE (0) -> ONSET (1) -> SPEECH (2) -> HANGOVER (3) -> SILENCE (0)
   │ ├── Hợp đồng warm-up: ép về SILENCE, onset_count = 0
   │ ├── Hysteresis (onset_thresh = 0.55 / 30ms, offset_thresh = 0.35 / 150ms)
   │ └── Bộ đệm pending-segment: nối khoảng lặng (min_silence_ms = 100ms) & lọc độ dài (min_speech_ms = 100ms)
   ▼ StateMachineOutput(segments, states, speech_mask)

Khi kết thúc luồng âm thanh:
   FeatureExtractor.flush() ──> NoiseEstimator.update() ──> SigmoidScorer.score() ──> VADStateMachine.process_chunk()
   Cuối cùng: VADStateMachine.flush() ──> chốt các đoạn còn mở hoặc pending-segment.
```

---

## 2. Bản đồ API có thể tái sử dụng & Trách nhiệm từng tầng (Reusable APIs)

| Tầng | File & Class | API công khai cốt lõi | Quản lý trạng thái (State Ownership) |
|---|---|---|---|
| **STFT** | `Dictionary/STFT.py`<br>`StreamingSTFT` | `__init__(n_fft, hop_length, win_length, window, output_type)`<br>`process_chunk(chunk: np.ndarray) -> np.ndarray`<br>`flush() -> np.ndarray`<br>`reset() -> None` | Stateful: sở hữu `_buffer` (mẫu dư chưa đủ frame) và `_has_produced_frames`. Được bảo vệ bởi `threading.Lock` cho single-consumer. |
| **Preset STFT** | `Dictionary/STFT.py` | `create_vad_streamer(sample_rate)`<br>`stft_for_vad(audio, sample_rate, workers=1)`<br>`vad_stft_params(sample_rate)` | Stateless factory & batch processing. Hỗ trợ 8 kHz (`n_fft=128, hop=40, win=128`) và 16 kHz (`n_fft=256, hop=80, win=256`), cùng cho `hop_seconds = 0.005 s` (5 ms). |
| **Tầng 1** | `VAD/feature_extraction.py`<br>`FeatureExtractor` | `__init__(sample_rate, bands_hz, sfm_range_hz, stft, debug)`<br>`extract(spec: np.ndarray) -> VADFeatures`<br>`process_chunk(chunk: np.ndarray) -> VADFeatures`<br>`process_batch(audio: np.ndarray, workers=1) -> VADFeatures`<br>`flush() -> VADFeatures`<br>`reset() -> None`<br>Properties: `num_bands`, `hop_seconds` | Sở hữu instance `StreamingSTFT`. Stateless ở tầng trích xuất ma trận trọng số (`band_weights`, `sfm_mask`), stateful thông qua `stft`.<br>`process_batch` hoàn toàn độc lập với state của STFT streaming. |
| **Tầng 2** | `VAD/noise_estimator.py`<br>`NoiseEstimator` | `__init__(num_bands, config, debug)`<br>`update(features: VADFeatures) -> NoiseOutput`<br>`reset() -> None`<br>Properties: `noise_floor`, `is_ready` | Stateful: sở hữu `_buf` (warm-up buffer), `_floor` (vector nền hiện tại), `_strong_run`, `_frozen`, `_stable`, `_ref` (bộ đếm phục hồi kẹt). Cập nhật đệ quy tuần tự từng frame. |
| **Tầng 3** | `VAD/sigmoid.py`<br>`SigmoidScorer` | `__init__(num_bands, config, debug)`<br>`score(features, noise_out) -> SigmoidOutput`<br>`score_arrays(snr_db, sfm, warmup) -> SigmoidOutput`<br>`reset() -> None` (no-op) | **Hoàn toàn phi trạng thái (Stateless)** theo frame. Mỗi frame được tính toán độc lập, bất biến 100% qua mọi kích thước chunk. |
| **Tầng 4** | `VAD/state_machine.py`<br>`VADStateMachine` | `__init__(hop_seconds, config, debug)`<br>`process_chunk(sig_out: SigmoidOutput) -> StateMachineOutput`<br>`process_scores(score, warmup) -> StateMachineOutput`<br>`flush() -> List[Tuple[float, float]]`<br>`reset() -> None`<br>Properties: `state`, `frame_index` | Stateful: sở hữu `_state` (`VADState`), `_frame_idx`, `_onset_count`, `_hangover_count`, `_cand_start`, `_active_start`, `_active_speech_frames`, và `_pending_seg` (chờ nối khoảng lặng). |

### 2.1. Quy ước thời gian & Ánh xạ mốc chỉ số
- **Đơn vị thời gian cơ sở**: Frame hop $\Delta t = \text{hop\_seconds} = 0.005\text{ s}$ (5 ms).
- **Quy ước khoảng nửa mở**: Mọi đoạn phát hiện đều tuân thủ `[start_time, end_time)` với `start_time = start_frame * hop_seconds` (inclusive) và `end_time = end_frame * hop_seconds` (exclusive). Thời lượng đoạn là `(end_frame - start_frame) * hop_seconds`.
- **Hành vi STFT Buffering & `flush()`**:
  - `process_chunk(chunk)`: Gom mẫu vào buffer, cắt tối đa `num_frames = 1 + (len(x) - n_fft) // hop_length`. Phần dư giữ lại trong buffer.
  - `flush()`: Nếu buffer có mẫu mới chưa phủ kín frame (`new_samples > 0`), zero-pad lên `n_fft` để sinh ra đúng 1 frame cuối cùng. Sau đó tự động reset buffer về rỗng.
- **Biểu diễn và lan truyền Warm-up**:
  - Tầng 2 phát ra `NoiseOutput.warmup` (`bool` array). 40 frame đầu tiên (200 ms) mang giá trị `True`.
  - Tầng 3 nhận cờ này: nếu `warmup[t] == True`, khóa `score[t] = 0.0`, `logit[t] = -20.0`, `snr_norm[:, t] = 0.0` và truyền nguyên vẹn cờ `warmup` sang `SigmoidOutput`.
  - Tầng 4 nhận `SigmoidOutput.warmup`: cưỡng bức trạng thái về `SILENCE`, reset `onset_count = 0`.
- **Hành vi phát đoạn và chốt đoạn (Emission & Finalization)**:
  - Khi một đoạn kết thúc tự nhiên qua `HANGOVER`, nó được đưa vào `_pending_seg = (start_f, end_f, speech_f)`.
  - Đoạn chỉ được phát ra (`emitted`) khi khoảng lặng sau nó vượt quá `min_silence_frames`, hoặc khi gặp đoạn mới cách xa quá `min_silence_frames`.
  - Khi luồng dừng, gọi `flush()` trên `VADStateMachine` để chốt đoạn cuối cùng trong `_pending_seg` (nếu `speech_f >= min_speech_frames`).
- **Yêu cầu Reset giữa các phiên / file khác nhau**:
  - Khi chuyển sang file âm thanh hoặc phiên mới, **bắt buộc** phải gọi `.reset()` trên cả 3 đối tượng stateful: `FeatureExtractor.reset()`, `NoiseEstimator.reset()`, `VADStateMachine.reset()`.

---

## 3. Bản đồ Độ bao phủ Kiểm thử & Điểm thiếu hụt (Current Test Coverage & Gaps)

Hiện tại toàn bộ 160 unit/regression test trong thư mục `tests/` đều pass 100% (`160 passed in 24.44s`).

| Hạng mục kiểm tra | File kiểm thử hiện có | Các test case cụ thể | Trạng thái & Khoảng trống (Gaps) |
|---|---|---|---|
| **Trích xuất đặc trưng & Đa tần số lấy mẫu (8k/16k)** | `tests/test_stft.py`<br>`tests/test_feature_extraction.py` | `test_vad_preset_per_sample_rate`<br>`test_band_mapping_identical_across_sample_rates`<br>`test_batch_streaming_parity_per_sample_rate`<br>`test_same_tone_lands_in_same_band`<br>`test_8k_band_above_nyquist_rejected` | **Đã phủ kín**: Đảm bảo 8 kHz và 16 kHz đều có cùng độ phân giải 62.5 Hz/bin, 4 dải tần khớp nhau về số bin [3, 11, 16, 32]. |
| **Tính liên tục trạng thái & Bất biến kích thước chunk (Chunk Invariance)** | `tests/test_stft.py`<br>`tests/test_feature_extraction.py`<br>`tests/test_noise_estimator.py`<br>`tests/test_sigmoid.py`<br>`tests/test_state_machine.py` | `TestStreamingParity` (STFT)<br>`test_batch_matches_streaming_any_chunking` (L1)<br>`test_chunk_invariance[1, 7, 80, 1000]` (L2)<br>`test_chunk_invariance_and_layer2_integration[1, 7, 19, 40, 250]` (L3)<br>`test_chunk_invariance_across_onset_hangover_and_min_silence[1, 2, 3, 5, 7, 13, 40, 111]` (L4) | **Đã phủ kín ở cấp tầng đơn lẻ**: Đảm bảo chia chunk bất kỳ (từ 1 frame đến 111 frame) đều cho kết quả segments và state giống hệt nguyên khối. |
| **Hành vi Flush & Kết thúc phiên** | `tests/test_stft.py`<br>`tests/test_state_machine.py` | `TestFlush` (10 test cases trong STFT)<br>`test_empty_chunks_flush_and_reset` | **Đã phủ kín**: Kiểm tra idempotency, short signal zero-pad, giải phóng pending-segment. |
| **Tiếng nói xuất hiện ngay tại thời điểm 0 (Startup speech)** | `tests/test_noise_estimator.py`<br>`tests/test_state_machine.py` | `test_speech_from_start_does_not_runaway_and_recovers_on_silence`<br>`test_speech_at_stream_start_warmup_contract_and_end_to_end_constant_power` | **Đã xác nhận hạn chế kiến trúc**: Do warm-up lấy median 40 frame đầu (200 ms), nếu tiếng nói có công suất không đổi từ t=0, noise floor bị học nhầm theo tiếng nói, gây Miss câu đầu cho đến khi có khoảng lặng. |
| **Chuyển tiếp nhiễu (Noise transitions: step-up, step-down)** | `tests/test_noise_estimator.py` | `test_stationary_convergence`<br>`test_noise_step_up_recovers_when_stable_flat`<br>`test_noise_step_up_recovers_on_colored_noise_with_realistic_sfm`<br>`test_noise_step_down_tracks_fast`<br>`test_step_down_confirm_delay_is_minimal` | **Đã phủ kín ở Tầng 2**: Nhiễu giảm bậc thích ứng trong 3 frame; nhiễu tăng bậc cần tích lũy qua cờ `stuck` (500 frame) + ổn định (100 frame). |
| **Tiếng nói yếu & Khoảng lặng ngắn (Hysteresis, Hangover, Boundary)** | `tests/test_state_machine.py` | `test_state_transitions_and_hysteresis`<br>`test_hangover_boundary_exact_frame_counts`<br>`test_score_rising_at_hangover_expiry_boundary_starts_new_onset`<br>`test_score_rising_on_final_hangover_frame_before_expiry_resumes_speech`<br>`test_min_speech_ms_measures_qualifying_speech_evidence_not_hangover`<br>`test_short_score_dips_during_speech_held_by_hangover`<br>`test_short_noise_bursts_during_silence_rejected` | **Đã phủ kín**: Quy ước biên hangover và lọc min_speech được bảo vệ nghiêm ngặt bằng regression tests chuyên biệt. |
| **Độ hợp lệ mốc thời gian đoạn (Timestamp validity)** | `tests/test_benchmark_metrics.py`<br>`tests/test_preannotate_silero.py` | `test_one_frame_interval_and_interval_ending_at_final_frame`<br>`test_touching_non_overlapping_intervals_and_one_to_one_matching`<br>`test_speech_touching_start_and_end_boundaries`<br>`test_adjacent_and_overlapping_speech_segments_merged`<br>`test_full_duration_coverage_and_csv_serialization_contiguity` | **Đã phủ kín**: Kiểm tra mốc nửa mở, không chồng lấn, kiểm tra round-trip giữa segments và boolean mask. |
| **KHOẢNG TRỐNG CẦN BỔ SUNG (Gaps)** | Chưa có file test tích hợp 2 chế độ | *Chưa có test so sánh trực tiếp*: `VAD_STREAMING` (chạy chunk-by-chunk) vs `VAD_OFFLINE` (chạy whole-file wrapper) trên cùng một file âm thanh để khẳng định 100% tương đương toán học (parity). | **Cần bổ sung ở Phase 1**. |

---

## 4. Đề xuất Hợp đồng Dùng chung (Proposed Shared Contracts)

> **Lưu ý**: Đây là đề xuất giao diện cho các giai đoạn tiếp theo, **hoàn toàn không sửa đổi hoặc đổi tên code hiện tại**.

### 4.1. Cấu trúc dữ liệu dùng chung (`pipeline_contracts.py` dự kiến)

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

class PipelineMode(str, Enum):
    STREAMING = "streaming"
    OFFLINE = "offline"

@dataclass(frozen=True)
class VADSegment:
    """
    Đại diện một đoạn tiếng nói chuẩn hóa của hệ thống VAD.
    Quy ước: Khoảng nửa mở [start_time, end_time), tính bằng giây.
    """
    start_time: float      # Giây (inclusive)
    end_time: float        # Giây (exclusive)
    start_frame: int       # Chỉ số frame bắt đầu (inclusive, canonical internal time)
    end_frame: int         # Chỉ số frame kết thúc (exclusive, canonical internal time)
    speech_frames: int     # Số frame thực sự có bằng chứng tiếng nói (không tính hangover tail)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass(frozen=True)
class VADDiagnostics:
    """
    Thông tin chẩn đoán và đặc trưng trung gian (tùy chọn).
    Chỉ được tạo ra khi caller yêu cầu `return_diagnostics=True`.
    """
    frame_scores: np.ndarray          # (T,) float32, điểm sigmoid p(t)
    speech_mask: np.ndarray           # (T,) bool, mặt nạ frame tức thời
    states: np.ndarray                # (T,) int8, mã trạng thái VADState
    warmup_mask: np.ndarray           # (T,) bool, cờ warm-up
    noise_floor: Optional[np.ndarray] = None  # (4, T) float32
    snr_db: Optional[np.ndarray] = None       # (4, T) float32

@dataclass(frozen=True)
class VADResult:
    """
    Kết quả chuẩn hóa đầu ra của một lần xử lý VAD (cả streaming chunk, streaming flush, và offline file).
    """
    segments: List[VADSegment]
    num_frames: int
    duration_s: float
    diagnostics: Optional[VADDiagnostics] = None
```

### 4.2. Hợp đồng Streaming Session (`VADStreamingSession`)
- **Vòng đời chuẩn**:
  1. `session = VADStreamingSession(sample_rate=16000, config=...)`: Kiểm tra `sample_rate in (8000, 16000)`, khởi tạo 4 tầng.
  2. `res = session.process_chunk(chunk: np.ndarray) -> VADResult`: Nhận chunk âm thanh mono bất kỳ, trả về các segment đã chốt và số frame.
  3. `final_res = session.flush() -> VADResult`: Đẩy nốt phần đệm của STFT và chốt đoạn đang chờ trong máy trạng thái.
  4. `session.reset() -> None`: Đưa toàn bộ các tầng về trạng thái khởi điểm để dùng tiếp cho phiên sau.

### 4.3. Hợp đồng Offline Runner (`VADOfflineRunner`)
- **Giao diện chuẩn**:
  `runner = VADOfflineRunner(sample_rate=16000, config=...)`  
  `result = runner.process_file(audio: np.ndarray, return_diagnostics=False) -> VADResult`
- **Xác thực đầu vào**:
  - `audio` phải là mảng 1-D, số thực (`float32` hoặc `float64`), không rỗng, giá trị biên độ trong `[-1.0, 1.0]`.
  - Từ chối `sample_rate` ngoài `{8000, 16000}`.

---

## 5. Định nghĩa Chính xác Baseline Offline (Offline Baseline Formulation)

Bắt buộc phân biệt rõ hai khái niệm để tránh ngộ nhận kỹ thuật:

### 5.1. Baseline Offline Giai đoạn 1 (`VADOfflineRunner` — Causal Offline Wrapper)
- **Bản chất**: Là một wrapper bao bọc lại chính xác thuật toán nhân quả (causal streaming) hiện tại nhưng nhận đầu vào là toàn bộ waveform của file âm thanh cùng một lúc.
- **Cách thức hoạt động**:
  1. Nhận toàn bộ file `audio`.
  2. Truyền thẳng qua `fe.process_batch(audio)` (hoặc đẩy qua `fe.process_chunk` + `fe.flush`).
  3. Chạy qua `NoiseEstimator.update()`.
  4. Chạy qua `SigmoidScorer.score()`.
  5. Chạy qua `VADStateMachine.process_chunk()` + `VADStateMachine.flush()`.
  6. Đóng gói kết quả thành `VADResult`.
- **Mục đích**: Thiết lập mốc cơ sở (baseline) tuyệt đối nhất quán với pipeline streaming. Đảm bảo 100% parity số học giữa chạy từng chunk và chạy cả file.

### 5.2. Thuật toán Offline Thực thụ trong tương lai (Future Truly Offline-Aware Algorithm)
- **Bản chất**: Thuật toán phi nhân quả (non-causal) tận dụng toàn bộ ngữ cảnh tương lai của file âm thanh:
  - **Ước lượng nền nhiễu toàn file (Whole-file global noise statistics)**: Sử dụng phân vị tối thiểu (Minimum Statistics / Percentile Tracking) trên toàn bộ phổ của file, giải quyết triệt để vấn đề mất 200 ms warm-up và vấn đề tiếng nói xuất hiện ngay tại `t=0`.
  - **Làm mượt 2 chiều (Bidirectional smoothing / Forward-backward scoring)**: Giúp điểm số tại các âm xát, phụ âm đóng/mở mượt mà hơn.
  - **Hậu xử lý đoạn toàn cục**: Tối ưu hóa ranh giới đoạn dựa trên thống kê năng lượng toàn câu thay vì chỉ dựa vào cửa sổ trượt nhân quả.
- **Nguyên tắc**: **Không được nhảy cóc sang thuật toán Offline thực thụ nếu chưa hoàn thành và kiểm chứng xong Baseline Offline Wrapper Giai đoạn 1.**

---

## 6. Khảo sát Benchmark, Dữ liệu Tham chiếu & Lệnh Tái hiện (Benchmark & Reproducibility)

### 6.1. Xuất xứ & Giới hạn của `clean_ground_truth.csv`
- **Xuất xứ**: Được tạo tự động bởi `VAD/report/tầng 4/human_annotation_pilot/generate_clean_ground_truth.py` trên 11 file âm thanh sạch (`wav_clean/E01...clean.wav` đến `E11...clean.wav`).
- **Nguyên lý nhãn**: Dựa trên mức năng lượng của tín hiệu tiếng nói sạch $s[n]$ (trước khi trộn nhiễu): frame có công suất toàn dải $\ge -25\text{ dB}$ so với phân vị p95 của câu phát ngôn đó thì được gán nhãn `SPEECH`, kết hợp nối khoảng hở $\le 150\text{ ms}$.
- **Giới hạn quan trọng**:
  > [!WARNING]
  > Nhãn trong `clean_ground_truth.csv` là **nhãn giả lập dựa trên năng lượng của nguồn sạch (Energy-derived Pseudo-ground-truth)**, **KHÔNG PHẢI nhãn ngữ âm chuẩn được chuyên gia con người thẩm định (human-verified phonetic ground truth)**. Các khoảng lặng ngắn giữa các âm vị ngắt (stop closure) hoặc âm xát yếu có thể bị gán nhãn khác biệt so với tri giác của người nghe.

### 6.2. Các lệnh chạy tái hiện kiểm thử và benchmark an toàn

1. **Chạy toàn bộ 160 unit/regression test hiện tại**:
   ```powershell
   py -m pytest -v
   ```

2. **Chạy bộ test độc lập của Tầng 4 & Metric Benchmark**:
   ```powershell
   py -m pytest tests/test_state_machine.py tests/test_benchmark_metrics.py tests/test_sigmoid.py -v
   ```

3. **Tái hiện Benchmark so sánh Multi-band VAD vs Silero trên tập pilot (KHÔNG ghi đè file cũ)**:
   Để kiểm tra lại logic mà không ghi đè lên các file kết quả đã chốt, dùng script so sánh độc lập:
   ```powershell
   py "VAD/report/tầng 4/human_annotation_pilot/compare_with_clean_ground_truth.py"
   ```

---

## 7. Đề xuất Cấu trúc Thư mục & Bố cục Module (Proposed Module Layout)

Đề xuất tổ chức các wrapper mà không làm xáo trộn các file tầng 1–4 hiện có:

```
c:\file\VAD_ASR_Diarization\
├── VAD\
│   ├── feature_extraction.py    [Tầng 1 - Frozen]
│   ├── noise_estimator.py       [Tầng 2 - Frozen]
│   ├── sigmoid.py               [Tầng 3 - Frozen]
│   ├── state_machine.py         [Tầng 4 - Frozen]
│   │
│   ├── pipeline\                [THƯ MỤC MỚI CHO GIAI ĐOẠN TIẾP THEO]
│   │   ├── __init__.py          [Expose VADStreamingSession, VADOfflineRunner, VADSegment, VADResult]
│   │   ├── contracts.py         [Định nghĩa VADSegment, VADResult, VADDiagnostics, PipelineMode]
│   │   ├── streaming.py         [Triển khai VADStreamingSession bọc causal streaming]
│   │   └── offline.py           [Triển khai VADOfflineRunner bọc causal whole-file baseline]
│   │
│   └── report\                  [Tài liệu & Báo cáo kỹ thuật]
└── tests\
    ├── test_pipeline_contracts.py  [Test mới cho dataclass và validation]
    ├── test_vad_streaming.py       [Test mới cho VADStreamingSession]
    └── test_vad_offline.py         [Test mới cho VADOfflineRunner & Parity với Streaming]
```

---

## 8. Kế hoạch Triển khai theo từng Giai đoạn (Phase-by-Phase Plan)

### Phase 1: Shared Contracts & Streaming Wrapper (`VADStreamingSession`)
- **Mục tiêu**: Xây dựng `VAD/pipeline/contracts.py` và `VAD/pipeline/streaming.py`.
- **Phạm vi file sửa/tạo**:
  - Tạo mới: `VAD/pipeline/__init__.py`, `VAD/pipeline/contracts.py`, `VAD/pipeline/streaming.py`.
  - Tạo mới: `tests/test_pipeline_contracts.py`, `tests/test_vad_streaming.py`.
  - Không sửa bất kỳ file nào trong `VAD/feature_extraction.py`, `VAD/noise_estimator.py`, `VAD/sigmoid.py`, `VAD/state_machine.py`.
- **Tiêu chí nghiệm thu (Acceptance Criteria)**:
  - 100% test mới pass.
  - `VADStreamingSession` xử lý đúng chuỗi `process_chunk` -> `flush` -> `reset`.
  - Kết quả segments của `VADStreamingSession` khớp từng bit với `run_multiband_streaming` trong `benchmark_multiband_vs_silero.py`.

### Phase 2: Causal Offline Wrapper (`VADOfflineRunner`)
- **Mục tiêu**: Xây dựng `VAD/pipeline/offline.py` (Causal Offline Baseline).
- **Phạm vi file sửa/tạo**:
  - Tạo mới: `VAD/pipeline/offline.py`.
  - Tạo mới: `tests/test_vad_offline.py`.
- **Tiêu chí nghiệm thu (Acceptance Criteria)**:
  - Với cùng một file âm thanh, chạy qua `VADOfflineRunner.process_file(audio)` và chạy qua `VADStreamingSession.process_chunk()` (với các kích thước chunk khác nhau 10ms, 20ms, 100ms) kết thúc bằng `flush()` phải sinh ra **chính xác 100% cùng một danh sách `VADSegment`** (Parity tuyệt đối).

### Phase 3: Tích hợp Benchmark & Đánh giá Parity trên Pilot Dataset
- **Mục tiêu**: Đưa hai wrapper vào kiểm thử tự động trên 11 file WAV của pilot dataset.
- **Tiêu chí nghiệm thu**:
  - Không làm thay đổi kết quả số học của Stage C2 và Stage D1.
  - Xác nhận bằng test tự động rằng chế độ Streaming và Offline Baseline cho kết quả trùng khớp hoàn toàn trên toàn bộ 11 file pilot ở cả 8 kHz và 16 kHz.

### Phase 4 (Tương lai): Nghiên cứu Bộ ước lượng Nhiễu Toàn file (Truly Offline Algorithm)
- **Mục tiêu**: Thiết kế thuật toán offline tận dụng thống kê toàn file để xử lý hạn chế warm-up 200 ms.
- **Điều kiện tiên quyết**: Phase 1, 2, 3 đã hoàn tất và được nghiệm thu đầy đủ.

---

## 9. Rủi ro, Tương thích ngược & Câu hỏi chưa giải quyết (Risks & Unresolved Questions)

### 9.1. Rủi ro kỹ thuật & Yêu cầu tương thích ngược
1. **Rủi ro rớt frame do sai lệch hop**: Cần đảm bảo `VADStreamingSession` và `VADOfflineRunner` sử dụng đúng `hop_seconds` từ `FeatureExtractor.hop_seconds` thay vì nhận cấu hình rời rạc từ bên ngoài.
2. **Bảo tồn tính bất biến**: Mọi wrapper mới chỉ được tổ chức lại luồng gọi hàm, tuyệt đối không được chèn thêm logic heuristic, thay đổi thứ tự gọi `est.update()` hay thay đổi ngưỡng của `state_machine`.
3. **Quản lý bộ nhớ**: `VADOfflineRunner` khi nhận file lớn (ví dụ audio hàng giờ) không được ép clone nhiều mảng trung gian gây tràn RAM; cần tính toán theo dạng stream hoặc block nếu cần.

### 9.2. Câu hỏi chưa giải quyết (Unresolved Questions)
- *Câu hỏi 1*: Trong chế độ `VAD_OFFLINE`, khi caller truyền vào một file rất dài (> 30 phút), kiến trúc có cần hỗ trợ xử lý theo block song song (multi-worker) hay chỉ chạy tuần tự đơn luồng như hiện tại?  
  *(Khuyến nghị: Ở Phase 2 giữ nguyên đơn luồng để bảo toàn parity số học với streaming)*.
- *Câu hỏi 2*: Cờ `return_diagnostics=True` có nên lưu toàn bộ mảng `snr_db` và `noise_floor` của toàn bộ file không, hay chỉ cần lưu `score` và `speech_mask`?  
  *(Khuyến nghị: Mặc định chỉ trả về `score` và `speech_mask` để tiết kiệm bộ nhớ; chỉ lưu `snr_db` khi có cờ `debug=True`)*.

---

## 10. Kết luận Kiểm toán & Bước đi tiếp theo được đề xuất

- **Kết luận**: Kiến trúc hiện tại của 4 tầng VAD rất mạch lạc, có phân định rõ ràng giữa các tầng stateful và stateless, đã được kiểm chứng bằng 160 unit/regression test với độ bao phủ cao. Hoàn toàn khả thi và an toàn để xây dựng 2 wrapper `VAD_STREAMING` và `VAD_OFFLINE` mà không cần sửa đổi bất kỳ dòng code sản xuất nào của Tầng 1–4.
- **Bước tiếp theo được đề xuất**: **Tiến hành Phase 1 — Xây dựng Shared Contracts (`VAD/pipeline/contracts.py`) và Causal Streaming Session (`VAD/pipeline/streaming.py`) kèm bộ test kiểm thử tương đương.**
