# Báo cáo Đối chiếu Multi-band Adaptive VAD (Tầng 1–4) với Silero Pre-Annotations trên 11 File WAV Pilot

> [!IMPORTANT]
> **Trạng thái nhãn tham chiếu (`PROVISIONAL_SILERO_PREANNOTATION`)**: Do chưa có file `annotations_human_reviewed.csv`, toàn bộ phép đo dưới đây sử dụng `preannotation_silero.csv` làm **nhãn tham chiếu tạm thời do Silero VAD tạo tự động**. Các con số `TP, FP, FN, TN, Precision, Recall, F1, Miss Rate, False Alarm Rate` phản ánh **mức độ đồng thuận (agreement) giữa Multi-band VAD hiện tại và Silero VAD**, **KHÔNG PHẢI độ chính xác thực tế đã được con người kiểm chứng (NOT verified ground-truth accuracy)**.

## 1. Cấu hình Đóng băng đã sử dụng (Frozen Production Configuration)

Không có bất kỳ thay đổi mã nguồn hay tinh chỉnh ngưỡng/trọng số nào trên Tầng 1–4:
- **Giao diện thực thi**: Streaming qua `FeatureExtractor.process_chunk(chunk) + flush()` $\to$ `NoiseEstimator.update()` $\to$ `SigmoidScorer.score()` $\to$ `VADStateMachine.process_chunk() + flush()`. Đã kiểm chứng bất biến tuyệt đối giữa các kích thước chunk (`20 ms`, `100 ms`, và toàn khối).
- **Tầng 1 (`FeatureExtractor`)**: `sample_rate ∈ {8000, 16000} Hz` (không resample), `hop_seconds = 0.005 s` (`5 ms`), cửa sổ Hann `16 ms` (`n_fft = 256, hop = 80` ở `16 kHz`; `n_fft = 128, hop = 40` ở `8 kHz`; `center = False`), 4 dải tần `((80.0, 300.0), (300.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0))` Hz (số bin `[3, 11, 16, 32]`), SFM trên `80–4000 Hz`.
- **Tầng 2 (`NoiseConfig`)**: `init_frames = 40` (`200 ms` warm-up đầu file bằng median theo từng băng, đặt `warmup = True` và `snr_db = 0.0`), `alpha_fast = 0.15`, `alpha_slow = 0.01`, `snr_noise_db = 6.0 dB`, `snr_speech_db = 9.0 dB`, `strong_confirm_frames = 3` (`15 ms`), `sfm_thresh = 0.15`, `max_freeze_frames = 500` (`2.5 s`), `recover_frames = 100` (`0.5 s`), `recover_stable_db = 3.0 dB`, `recover_ref_alpha = 0.05`.
- **Tầng 3 (`SigmoidConfig`)**: `band_weights = (1.0, 2.0, 2.0, 1.0)`, `sfm_weight = 2.0`, `bias = -4.0`, `snr_low_db = 0.0 dB`, `snr_high_db = 20.0 dB`, `warmup_score = 0.0`, `warmup_logit = -20.0`.
- **Tầng 4 (`StateMachineConfig`)**: `onset_thresh = 0.55`, `offset_thresh = 0.35`, `onset_ms = 30.0 ms` (`6` frame), `hangover_ms = 150.0 ms` (`30` frame), `min_speech_ms = 100.0 ms` (`20` frame bằng chứng `ONSET + SPEECH` độc lập với hangover), `min_silence_ms = 100.0 ms` (`20` frame).

## 2. Kết quả Tổng hợp (Aggregate Frame-Level & Segment-Level Agreement)

Quy ước đánh giá:
- **Mức frame (5 ms grid)**: Cả nhãn tham chiếu và các đoạn dự đoán `[start_time_s, end_time_s)` được ánh xạ lên cùng lưới frame `5 ms` (`expected_frames_5ms = duration_s / 0.005`, tổng cộng `13,820` frame = `69.100 s`) theo quy tắc nửa mở `start_time_s <= t * 0.005 < end_time_s`.
- **Mức đoạn (Segment matching)**: Ghép cặp 1–1 tham lam theo `IoU >= 0.05` (`match_segments_by_time`), không bao giờ coi hai đoạn không chồng lấn là khớp chỉ vì có tổng thời lượng gần nhau.

| Tập con | Số file | Thời lượng (s) | TP | FP | FN | TN | Prec (%) | Rec (%) | F1 (%) | Miss (%) | FA (%) | Dur Pred/Ref (s) | Segs (Match/Pred/Ref) | Seg Prec/Rec/F1 (%) | Onset ±10/±20/±50ms | Offset ±10/±20/±50ms | Cả 2 đầu ±10/±20/±50ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toàn bộ (11 files)** | 11 | 69.1 | 7043 | 717 | 1877 | 4183 | 90.8 | 79.0 | **84.4** | 21.0 | 14.6 | 38.80 / 44.60 | 22 / 32 / 23 | 68.8 / 95.7 / 80.0 | 1/3/8 (/22) | 1/3/4 (/22) | 0/1/2 (/22) |
| Sample rate `16 kHz` | 6 | 37.4 | 3679 | 414 | 974 | 2413 | 89.9 | 79.1 | **84.1** | 20.9 | 14.6 | 20.46 / 23.26 | 12 / 16 / 13 | 75.0 / 92.3 / 82.8 | 0/1/5 (/12) | 1/1/2 (/12) | 0/0/1 (/12) |
| Sample rate `8 kHz` | 5 | 31.7 | 3364 | 303 | 903 | 1770 | 91.7 | 78.8 | **84.8** | 21.2 | 14.6 | 18.34 / 21.34 | 10 / 16 / 10 | 62.5 / 100.0 / 76.9 | 1/2/3 (/10) | 0/2/2 (/10) | 0/1/1 (/10) |
| Split `dev` | 5 | 31.0 | 2687 | 13 | 1292 | 2208 | 99.5 | 67.5 | **80.5** | 32.5 | 0.6 | 13.50 / 19.89 | 11 / 16 / 11 | 68.8 / 100.0 / 81.5 | 0/0/2 (/11) | 0/0/0 (/11) | 0/0/0 (/11) |
| Split `eval` | 6 | 38.1 | 4356 | 704 | 585 | 1975 | 86.1 | 88.2 | **87.1** | 11.8 | 26.3 | 25.30 / 24.71 | 11 / 16 / 12 | 68.8 / 91.7 / 78.6 | 1/3/6 (/11) | 1/3/4 (/11) | 0/1/2 (/11) |

## 3. Kết quả Chi tiết theo từng File WAV (Per-File Comparison)

| Excerpt ID | SR (Hz) | Split | Dur (s) | TP | FP | FN | TN | Prec (%) | Rec (%) | F1 (%) | Miss (%) | FA (%) | Dur Pred/Ref (s) | Segs (M/P/R) | Seg F1 (%) | Med \|OnErr\| (ms) | Med \|OffErr\| (ms) | Onset ±10/20/50 | Offset ±10/20/50 | Đoạn đầu (Pred vs Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `E01_stat20_16k` | 16000 | `dev` | 6.2 | 611 | 13 | 72 | 544 | 97.9 | 89.5 | **93.5** | 10.5 | 2.3 | 3.12/3.42 | 2/3/2 | 80.0 | 343 | 103 | 0/0/0 | 0/0/0 | `[1.220, 1.680)` vs `[1.186, 2.558)` |
| `E02_stat20_8k` | 8000 | `eval` | 6.0 | 744 | 42 | 9 | 405 | 94.7 | 98.8 | **96.7** | 1.2 | 9.4 | 3.93/3.77 | 2/2/2 | 100.0 | 18 | 109 | 1/1/2 | 0/1/1 | `[0.795, 2.860)` vs `[0.802, 2.878)` |
| `E03_stat15_16k` | 16000 | `eval` | 6.0 | 722 | 0 | 38 | 440 | 100.0 | 95.0 | **97.4** | 5.0 | 0.0 | 3.61/3.80 | 2/2/2 | 100.0 | 39 | 56 | 0/0/1 | 0/0/1 | `[0.855, 2.875)` vs `[0.802, 2.942)` |
| `E04_stat15_8k` | 8000 | `eval` | 5.7 | 731 | 0 | 190 | 219 | 100.0 | 79.4 | **88.5** | 20.6 | 0.0 | 3.65/4.60 | 1/3/1 | 50.0 | 1421 | 573 | 0/0/0 | 0/0/0 | `[0.240, 0.560)` vs `[0.194, 4.798)` |
| `E05_stat10_16k` | 16000 | `eval` | 6.0 | 630 | 3 | 319 | 248 | 99.5 | 66.4 | **79.6** | 33.6 | 1.2 | 3.17/4.75 | 2/5/2 | 57.1 | 737 | 726 | 0/1/1 | 1/1/1 | `[0.885, 1.915)` vs `[0.898, 3.358)` |
| `E06_stat10_8k` | 8000 | `dev` | 7.3 | 622 | 0 | 438 | 400 | 100.0 | 58.7 | **74.0** | 41.3 | 0.0 | 3.11/5.30 | 3/6/3 | 66.7 | 482 | 762 | 0/0/0 | 0/0/0 | `[0.465, 0.740)` vs `[0.418, 2.622)` |
| `E07_step_up_16k` | 16000 | `eval` | 7.2 | 675 | 398 | 27 | 340 | 62.9 | 96.2 | **76.1** | 3.8 | 53.9 | 5.37/3.51 | 2/2/3 | 80.0 | 620 | 759 | 0/0/1 | 0/0/0 | `[0.615, 2.270)` vs `[0.578, 2.366)` |
| `E08_step_down_16k` | 16000 | `dev` | 6.5 | 554 | 0 | 334 | 412 | 100.0 | 62.4 | **76.8** | 37.6 | 0.0 | 2.77/4.44 | 2/2/2 | 100.0 | 339 | 496 | 0/0/1 | 0/0/0 | `[1.390, 1.830)` vs `[0.738, 2.718)` |
| `E09_step_up_8k` | 8000 | `eval` | 7.2 | 854 | 261 | 2 | 323 | 76.6 | 99.8 | **86.7** | 0.2 | 44.7 | 5.58/4.28 | 2/2/2 | 100.0 | 592 | 68 | 0/1/1 | 0/1/1 | `[0.590, 2.455)` vs `[0.578, 2.334)` |
| `E10_start_16k` | 16000 | `dev` | 5.5 | 487 | 0 | 184 | 429 | 100.0 | 72.6 | **84.1** | 27.4 | 0.0 | 2.44/3.35 | 2/2/2 | 100.0 | 329 | 130 | 0/0/1 | 0/0/0 | `[0.625, 1.180)` vs `[0.002, 1.342)` |
| `E11_start_8k` | 8000 | `dev` | 5.5 | 413 | 0 | 264 | 423 | 100.0 | 61.0 | **75.8** | 39.0 | 0.0 | 2.06/3.38 | 2/3/2 | 80.0 | 746 | 273 | 0/0/0 | 0/0/0 | `[0.625, 1.190)` vs `[0.034, 1.374)` |

## 4. Phân tích Kỹ thuật Chuyên sâu (Technical Analysis)

### 4.1. Các đoạn trích có bất đồng lớn nhất với Silero Pre-Annotations
Xếp hạng theo mức độ bất đồng cấp frame (`100 - F1%`):
1. **`E06_stat10_8k` (`F1 = 74.0%`, `Miss = 41.3%`, `FA = 0.0%`) & `E05_stat10_16k` (`F1 = 79.6%`, `Miss = 33.6%`, `FA = 1.2%`) — Kịch bản nhiễu nặng `+10 dB` SNR**:
   - **Bản chất bất đồng**: Không phải lệch biên nhẹ mà là **phân mảnh nội câu (intra-utterance fragmentation) và mất các âm tiết yếu**. Ở `+10 dB` SNR, các nguyên âm/phụ âm yếu có điểm Sigmoid tụt xuống dưới `offset_thresh = 0.35` lâu hơn `hangover_ms (150 ms) + min_silence_ms (100 ms)`, khiến Multi-band VAD tách `3` đoạn của `E06` thành `6` đoạn ngắn (`3.11 s` vs `5.30 s` của Silero) và tách `2` đoạn của `E05` thành `5` đoạn ngắn (`3.17 s` vs `4.75 s` của Silero). Đổi lại, `False Alarm Rate` của Multi-band VAD ở hai file này gần như bằng `0%` (`0.0%` và `1.2%`).
2. **`E11_start_8k` (`F1 = 75.8%`, `Miss = 39.0%`, `FA = 0.0%`) & `E10_start_16k` (`F1 = 84.1%`, `Miss = 27.4%`, `FA = 0.0%`) — Kịch bản tiếng nói ngay đầu file (`speech_at_start`)**:
   - **Bản chất bất đồng**: Khóa warm-up `200 ms` kết hợp hiện tượng nhiễm bẩn `noise_floor` khởi tạo làm mất `625 ms` đầu câu 1 (chi tiết tại Mục 4.2), cộng thêm ở `E11_start_8k` câu thứ hai bị tách thành 2 đoạn tại khoảng nghỉ `140 ms` (`[2.060, 2.780)` và `[2.920, 3.700)`).
3. **`E07_step_up_16k` (`F1 = 76.1%`, `Miss = 3.8%`, `FA = 53.9%`) & `E09_step_up_8k` (`F1 = 86.7%`, `Miss = 0.2%`, `FA = 44.7%`) — Kịch bản nhiễu nhảy bậc tăng `+15 dB` (`step_up_down`)**:
   - **Bản chất bất đồng**: Tại `t = 3.000 s` trong đoạn trích (tức `t = 12.0 s` của stream gốc), nhiễu nền nhảy vọt từ `+20 dB` SNR lên `+5 dB` SNR (tăng `+15 dB` công suất nhiễu). Vì `noise_floor` của Tầng 2 đang neo ở mức nhiễu thấp trước đó, bước nhảy công suất nhiễu tạo ra SNR tức thời `~15 dB` trên cả 4 băng và kích hoạt trạng thái `SPEECH` tại `t = 2.990 s` — kéo dài thành một đoạn duy nhất `[2.990, 6.700)` xuyên qua `1.20 s` nhiễu thuần (`2.990..4.194 s`) và nối liền câu 4 với câu 5. Trong khi đó, Silero VAD giữ `NON_SPEECH` trong khoảng `[2.366, 4.194)` (16 kHz) và `[2.334, 4.162)` (8 kHz).
4. **`E08_step_down_16k` (`F1 = 76.8%`, `Miss = 37.6%`, `FA = 0.0%`) — Kịch bản nửa đầu nhiễu nặng `+5 dB` rồi giảm xuống `+20 dB`**:
   - **Bản chất bất đồng**: Trong 3 giây đầu (`+5 dB` SNR), câu nói đầu tiên (`[0.738, 2.718)` theo Silero) chỉ vượt ngưỡng của Multi-band VAD ở phần lõi năng lượng cao nhất `[1.390, 1.830)` (`0.44 s`), gây `Miss = 37.6%`. Ngay sau khi nhiễu giảm xuống `+20 dB` tại `t = 3.000 s`, câu nói thứ hai đạt độ đồng thuận rất cao (`[3.260, 5.590)` so với `[3.234, 5.694)` của Silero, lệch onset chỉ `+26 ms`).
5. **`E01_stat20_16k` (`F1 = 93.5%`), `E02_stat20_8k` (`F1 = 96.7%`), `E03_stat15_16k` (`F1 = 97.4%`) — Kịch bản nhiễu dừng `+20 dB` và `+15 dB`**:
   - **Mức độ đồng thuận rất cao (`F1 = 93.5% – 97.4%`)**: Ở `E02` và `E03`, số đoạn khớp tuyệt đối `2/2` (`Seg F1 = 100.0%`) và sai số onset trung vị chỉ `18–39 ms`.

### 4.2. Chẩn đoán hành vi Khởi động đầu luồng trên `E10_start_16k` và `E11_start_8k`
- Trên cả hai file `E10_start_16k` (16 kHz) và `E11_start_8k` (8 kHz), tiếng nói bắt đầu ngay từ `t = 0.000 s` (Silero bắt đầu tại `0.002 s` ở 16 kHz và `0.034 s` ở 8 kHz), nhưng đoạn dự đoán đầu tiên của Multi-band VAD đều bắt đầu tại đúng **`0.625 s`** (`[0.625, 1.180)` ở 16 kHz và `[0.625, 1.190)` ở 8 kHz), bỏ sót `625 ms` đầu câu 1.
- **Nguyên nhân hai tầng kết hợp**:
  1. **Khóa warm-up cố định `200 ms` (`init_frames = 40`)**: Trong `[0.000, 0.200) s`, Tầng 2 bật `warmup = True`, Tầng 3 xuất `score = 0.0` và Tầng 4 giữ cứng trạng thái `SILENCE`.
  2. **Nhiễm bẩn `noise_floor` khởi tạo (`200 ms – 625 ms`)**: Vì 40 frame đầu tiên là tiếng nói, phép lấy `median` theo băng trong warm-up của Tầng 2 khởi tạo `noise_floor` bằng chính mức công suất của âm tiết đầu câu. Do đó, từ `0.200 s` đến `0.620 s`, SNR ước lượng bị nén xuống gần `0 dB` (`score < onset_thresh = 0.55`), mãi tới âm tiết nhấn mạnh ở `0.625 s` mới vượt ngưỡng.
  3. **Tự phục hồi ở câu thứ hai**: Trong khoảng nghỉ `[1.25 s, 2.05 s]`, cổng `strong` (`alpha_fast = 0.15`) kéo `noise_floor` xuống mức nền thật, giúp câu thứ hai bắt đầu tại `2.085 s` (`E10`, lệch `+35 ms` so với Silero `2.050 s`) và `2.060 s` (`E11`, lệch `+42 ms` so với Silero `2.018 s`).

### 4.3. So sánh hành vi giữa `16 kHz` và `8 kHz`
- Tổng hợp trên 6 file `16 kHz` (`F1 = 84.1%`, `Miss = 20.9%`, `FA = 14.6%`, `Seg F1 = 82.8%`) và 5 file `8 kHz` (`F1 = 84.8%`, `Miss = 21.2%`, `FA = 14.6%`, `Seg F1 = 76.9%`) cho thấy mức đồng thuận cấp frame rất gần nhau (`84.1%` vs `84.8%`), nhưng ở **`8 kHz` có xu hướng tách đoạn nội câu nhiều hơn** (`Seg Precision = 62.5%` ở 8 kHz so với `75.0%` ở 16 kHz: `E04_stat15_8k` tách `1` câu dài thành `3` đoạn; `E11_start_8k` tách câu 2 thành 2 đoạn `[2.060, 2.780)` & `[2.920, 3.700)` tại một khoảng nghỉ `140 ms`, trong khi `E10_start_16k` giữ nguyên 1 đoạn `[2.085, 3.965)`).
- Điều này phù hợp với đặc tính vật lý đã ghi nhận: ở `8 kHz`, băng 4 (`2000–4000 Hz`) sát tần số Nyquist bị suy giảm năng lượng âm xát và làm SFM biến thiên khác nhẹ so với `16 kHz`.

## 5. Giới hạn Hiện tại & Đề xuất Bước Tiếp theo
- **Giới hạn**: Kết quả trên chỉ đo độ đồng thuận với nhãn nháp của Silero VAD (`preannotation_silero.csv`). Ở những vùng nhiễu nhảy bậc (`E07`, `E09`) hoặc phụ âm đuôi mờ trong nhiễu `+10 dB` (`E05`, `E06`), chính Silero VAD cũng có thể mắc lỗi (ví dụ gộp cả khoảng nghỉ giữa câu vào `SPEECH` ở `E04_stat15_8k` hoặc bỏ sót/báo nhầm biên). Tuyệt đối không kết luận bộ VAD nào chính xác hơn nếu chưa có nhãn người nghe.
- **Bước tiếp theo hợp lý nhất**: Người gán nhãn ưu tiên nghe và hiệu đính các đoạn có độ bất đồng lớn nhất (`E06, E11, E07, E08, E05, E10, E09, E04`) để xuất ra `annotations_human_reviewed.csv` (có đánh dấu vùng `UNCERTAIN`), sau đó chạy lại chính công cụ `benchmark_multiband_vs_silero.py` này để có phép đo chuẩn xác trên nhãn người thật.
