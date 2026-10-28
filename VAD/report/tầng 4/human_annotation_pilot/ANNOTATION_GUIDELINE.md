# HƯỚNG DẪN GÁN NHÃN THỦ CÔNG & GIAO THỨC ĐÁNH GIÁ (STAGE D1 PILOT)

## 1. Mục tiêu và nguyên tắc cốt lõi
- Gói Pilot gồm **11 đoạn trích âm thanh trộn (mixed audio)** (`E01`..`E11`, mỗi đoạn dài `5.5–7.3 giây`, tổng cộng `69.1 giây`) được cắt trực tiếp từ các luồng âm thanh trộn `mix_wave` mà bộ VAD thực sự nhận vào.
- **Tuyệt đối không dùng nhãn tự động (pseudo-label) hay văn bản transcript để điền sẵn biên thời gian.** Người gán nhãn phải nghe trực tiếp file WAV trong `wav/` và quan sát phổ âm (spectrogram/waveform trên Audacity hoặc Praat) để đánh dấu biên.

---

## 2. Quy ước 3 nhãn và mốc thời gian

Mọi khoảng thời gian `[start_time_s, end_time_s)` được ghi bằng **giây (lấy tới 3 chữ số thập phân, tức độ phân giải 1 ms)** theo quy ước **nửa mở (start-inclusive, end-exclusive)**:
- Các khoảng trong cùng một `excerpt_id` phải nối tiếp nhau liên tục (`end_time_s` của dòng trước bằng đúng `start_time_s` của dòng sau) và phủ kín từ `0.000` đến hết `duration_s` của đoạn trích.

### Ý nghĩa của 3 nhãn hợp lệ:
1. **`SPEECH`**:
   - Vùng có tiếng nói của người (bao gồm nguyên âm, phụ âm hữu thanh, âm xát/âm bật ở đầu và cuối từ còn nghe rõ trong nền nhiễu).
   - **Khoảng nghỉ ngắn giữa các âm tiết trong cùng một cụm từ liên tục**:
     - Nếu khoảng ngắt tắc âm (stop closure) hoặc nghỉ cực ngắn `< 150 ms` nằm giữa hai âm tiết liền nhau và nghe tự nhiên như một phát ngôn liên tục, giữ nguyên trong cùng khoảng `SPEECH`.
     - Nếu khoảng nghỉ lấy hơi giữa các vế câu `>= 150 ms` mà tai người nghe rõ là khoảng lặng/nhiễu nền, tách thành khoảng `NON_SPEECH` riêng.
2. **`NON_SPEECH`**:
   - Vùng chỉ chứa nhiễu nền, tiếng thở nhẹ tách rời lời nói, tiếng va chạm micro, hoặc khoảng lặng giữa các câu.
3. **`UNCERTAIN`**:
   - Dùng cho các vùng biên mờ (onset/offset của phụ âm yếu bị chìm trong nhiễu nặng `+5 dB` / `+10 dB`) hoặc đoạn mà người nghe không thể khẳng định chắc chắn là còn âm đuôi của lời nói hay chỉ là biến thiên của nhiễu nền.
   - Không lạm dụng `UNCERTAIN` cho toàn bộ câu nói khó; chỉ khoanh vùng biên hoặc âm tiết thực sự mơ hồ (thường `20–150 ms`).

---

## 3. Quy trình thực hiện và kiểm tra hợp lệ (Validation)

### Bước 1: Xuất 11 file WAV âm thanh trộn (nếu chưa xuất)
Chạy lệnh từ thư mục gốc của repo:
```powershell
py "VAD/report/tầng 4/human_annotation_pilot/pilot_tools.py" export-wavs
```
Các file WAV 16-bit PCM sẽ được tạo trong `VAD/report/tầng 4/human_annotation_pilot/wav/`.

### Bước 2: Gán nhãn trên file CSV (từ Template trắng hoặc từ Bản nháp Tiền gán nhãn)
1. **Cách A (Gán nhãn từ đầu)**: Sao chép `annotation_template.csv` thành `annotations_completed.csv`.
2. **Cách B (Dùng bản nháp tiền gán nhãn `preannotation_silero.csv` để tăng tốc rà soát)**:
   - Có thể chạy công cụ tiền gán nhãn độc lập bằng gói `silero-vad` trên 11 file WAV:
     ```powershell
     py "VAD/report/tầng 4/human_annotation_pilot/preannotate_silero.py"
     ```
   - Lệnh này tạo file mới `preannotation_silero.csv` (`annotator_id = silero_vad_preannotation`, không bao giờ ghi đè `annotation_template.csv`).
   - **CẢNH BÁO BẮT BUỘC**: `preannotation_silero.csv` **CHỈ LÀ BẢN NHÁP TIỀN GÁN NHÃN TỰ ĐỘNG (Automated Pre-Annotation), TUYỆT ĐỐI KHÔNG PHẢI NHÃN CHUẨN CỦA CON NGƯỜI (NOT Human Ground Truth)**. Người gán nhãn bắt buộc phải sao chép sang `annotations_completed.csv`, nghe trực tiếp từng file WAV để chỉnh lại biên, bổ sung nhãn `UNCERTAIN` ở các vùng biên mơ hồ (Silero chỉ xuất `SPEECH`/`NON_SPEECH`), và đổi `annotator_id` sang mã người gán nhãn thật sau khi đã kiểm tra xong.
3. Mở từng file WAV trong `wav/` (ví dụ `E01_stat20_16k.wav` dài `6.200 s`) và chia khoảng `[0.000, 6.200)` thành các dòng liên tiếp nhau với nhãn `SPEECH`, `NON_SPEECH`, hoặc `UNCERTAIN`.

Ví dụ minh họa định dạng hợp lệ cho một đoạn dài `6.200 s`:
```csv
excerpt_id,start_time_s,end_time_s,label,annotator_id,notes
E01_stat20_16k,0.000,1.210,NON_SPEECH,ann_01,Initial background noise
E01_stat20_16k,1.210,2.440,SPEECH,ann_01,First utterance
E01_stat20_16k,2.440,2.490,UNCERTAIN,ann_01,Weak trailing consonant in noise
E01_stat20_16k,2.490,3.260,NON_SPEECH,ann_01,Inter-utterance gap
E01_stat20_16k,3.260,5.420,SPEECH,ann_01,Second utterance
E01_stat20_16k,5.420,6.200,NON_SPEECH,ann_01,Trailing noise
```

### Bước 3: Chạy công cụ kiểm tra tự động (Validation)
```powershell
py "VAD/report/tầng 4/human_annotation_pilot/pilot_tools.py" validate "VAD/report/tầng 4/human_annotation_pilot/annotations_completed.csv"
```
Công cụ `validate` sẽ kiểm tra:
- Không còn nhãn giữ chỗ (`REPLACE_WITH_...`); chỉ chấp nhận `SPEECH`, `NON_SPEECH`, `UNCERTAIN`.
- Mọi khoảng đều có `start_time_s < end_time_s`.
- Không có khoảng trống (gap) hay chồng lấn (overlap) vượt quá `1 ms`, và tổng chiều dài khớp chính xác với `duration_s` trong `pilot_manifest.csv`.

---

## 4. Giao thức chất lượng gán nhãn & đánh giá (Phase D1.4 Protocol)

1. **Xử lý vùng `UNCERTAIN` trong tính toán chỉ số**:
   - Khi chuyển sang lưới frame 5 ms qua `intervals_to_frame_labels()`, các frame có `uncertain_mask == True` được đặt `valid_eval_mask = False`.
   - Trong `evaluate_predictions_against_human_labels()`, toàn bộ các chỉ số chính (**Miss Rate, False Alarm Rate, Precision, Recall, F1**) chỉ tính trên các frame có `valid_eval_mask == True` (loại trừ hoàn toàn frame `UNCERTAIN` khỏi cả tử số và mẫu số).
   - Đồng thời, hệ thống bắt buộc báo cáo riêng:
     - `uncertain_frames` & `uncertain_pct` (tỷ lệ % thời lượng được gán là `UNCERTAIN`);
     - `pred_pos_in_uncertain_pct` (tỷ lệ % frame trong vùng `UNCERTAIN` mà VAD dự đoán là có tiếng nói).
2. **Giải quyết bất đồng giữa nhiều người gán nhãn (Multi-Annotator Resolution)**:
   - Nếu có từ 2 người gán nhãn độc lập trở lên trên cùng một excerpt:
     - Frame mà cả hai cùng gán `SPEECH` -> `SPEECH`.
     - Frame mà cả hai cùng gán `NON_SPEECH` -> `NON_SPEECH`.
     - Frame mà hai người bất đồng (`SPEECH` vs `NON_SPEECH`) hoặc ít nhất một người gán `UNCERTAIN`:
       + Nếu độ lệch biên `<= 30 ms` (6 frame): tự động gán vào vùng biên `UNCERTAIN`.
       + Nếu bất đồng kéo dài `> 30 ms`: tổ chức phiên rà soát chung (adjudication) nghe lại đoạn đó để chốt nhãn hoặc đánh dấu `UNCERTAIN` nếu tín hiệu bị nhiễu che lấp hoàn toàn.
3. **Phân chia tập Development (`dev`) và Held-Out Evaluation (`eval`) theo nhóm câu nguồn (`source_group_id`)**:
   - Trong `pilot_manifest.csv`, 11 đoạn trích được phân chia theo **4 nhóm câu nói nguồn rời nhau tuyệt đối (`source_group_id`)** để đảm bảo: (a) hai phiên bản 8 kHz và 16 kHz của cùng dữ liệu nguồn luôn nằm trong cùng một tập, và (b) không có bất kỳ câu nói nguồn nào lặp lại ở nhiều kịch bản nhiễu bị rò rỉ giữa `dev` và `eval`:
     - **Tập `dev` (5 đoạn: `E01, E06, E08, E10, E11`, tổng `31.0 s` = `6200` frames)**: Thuộc 3 nhóm nguồn `SG1_U01_U02` (`000218.mp3|000872.mp3`, gồm cả cặp 16 kHz/8 kHz `E10`/`E11` và `E01`), `SG3_U07_U08` (`002614.mp3|002832.mp3`), và `SG4_U09_U10` (`003703.mp3|004139.mp3`). Dùng để phân tích lỗi và hiệu chỉnh ngưỡng ở các giai đoạn sau (nếu được phép).
     - **Tập `eval` (6 đoạn: `E02, E03, E04, E05, E07, E09`, tổng `38.1 s` = `7620` frames)**: Thuộc nhóm nguồn liên thông `SG2_U03_U06` (`001307.mp3|001743.mp3|001961.mp3|002396.mp3`, gồm cả hai cặp đối chiếu 8 kHz/16 kHz `E02`/`E03` và `E07`/`E09`). Giữ độc lập hoàn toàn (held-out) để kiểm định chéo, không dùng để chọn ngưỡng.
4. **Ánh xạ biên thời gian giữa 16 kHz và 8 kHz**:
   - Cả hai mức sample rate đều dùng cùng `hop_seconds = 0.005 s` (5 ms) và cùng cửa sổ `win_length = 16 ms` (`center=False`), nên thời điểm bắt đầu của frame `t` đều là `t * 0.005 s`.
   - Lưu ý quan trọng đã kiểm chứng trong Audit D1.1: Do bộ giải mã MP3/resampler ở 8 kHz làm câu 2 (`000872.mp3`) ngắn hơn `25 ms` (5 frame) so với ở 16 kHz, từ câu 2 trở đi trong luồng ghép `build_benchmark_stream()`, mốc thời gian tuyệt đối của các câu ở 8 kHz bị dịch sớm hơn `25 ms` (và từ câu 6 dịch sớm hơn `20 ms`) so với luồng 16 kHz.
   - Vì vậy:
     - Với câu 1 (`E10_start_16k` và `E11_start_8k`, `0.000..1.251 s`), mốc thời gian giữa 8 kHz và 16 kHz khớp nhau tuyệt đối (`0 ms` độ lệch).
     - Với các câu từ câu 2 trở đi (ví dụ cặp `E02` ở 8 kHz và `E03` ở 16 kHz, hoặc `E07` ở 16 kHz và `E09` ở 8 kHz), người gán nhãn nên nghe trực tiếp từng file WAV riêng hoặc nếu ánh xạ nhãn từ 16 kHz sang 8 kHz cho các câu `>= 2` thì phải bù đúng độ dịch tích lũy của `utt_intervals` (`-25 ms` cho câu 2..5, `-20 ms` cho câu 6..10).
