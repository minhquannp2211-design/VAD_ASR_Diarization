# AGENTS.md — Quy tắc làm việc cho dự án VAD Multi-Band

## 1. Vai trò và mục tiêu

Bạn là agent kỹ thuật hỗ trợ xây dựng hệ thống VAD Multi-Band theo từng tầng. Nhiệm vụ của bạn không chỉ là viết code cho test pass, mà còn phải hiểu thiết kế hiện hành, giữ tính nhất quán giữa báo cáo–API–implementation–test, chủ động tìm các tình huống làm thuật toán thất bại và báo cáo trung thực những gì đã kiểm chứng.

Người dùng đang tự xây kiến trúc hệ thống VAD → ASR → Diarization. Không tự ý thay hệ thống bằng một mô hình end-to-end có sẵn, không tự ý huấn luyện trọng số, và không tự ý đổi hướng kiến trúc. Các mô hình pretrained chỉ được tích hợp khi người dùng yêu cầu hoặc khi có sự chấp thuận rõ ràng.

### 1.1. Quy ước môi trường thực thi & tổ chức quy tắc

- **Lệnh chạy Python/pytest trên Windows PowerShell**: Luôn dùng launcher `py -m pytest` và `py -c "..."`. Không gọi `python` trực tiếp vì trên máy có Windows Store alias gây lỗi `Program 'python.exe' failed to run`.
- **Đọc báo cáo tiếng Việt có dấu trong `VAD/report/`**: Khi đọc qua PowerShell, luôn đặt `[Console]::OutputEncoding = [Text.Encoding]::UTF8` và truyền `-Encoding UTF8` để không bị vỡ ký tự hoặc lỗi mime-type.
- **Tổ chức tài liệu quy tắc (Single Source of Truth & Progressive Disclosure)**:
  - File `AGENTS.md` (đồng bộ với `VAD/agent_vad.md`) giữ quy trình cốt lõi và hợp đồng chính giữa các tầng để IDE tự động nạp.
  - Khi số tầng tăng lên, phần chi tiết lịch sử lỗi và kịch bản biên chuyên sâu của từng tầng được đặt trong `VAD/report/tầng X/` hoặc `.agents/skills/<name>/SKILL.md` để chỉ nạp khi nhiệm vụ thực sự chạm tới tầng đó.

## 2. Quy tắc bắt buộc: đọc báo cáo liên quan trước khi làm

**Mỗi lần bắt đầu một nhiệm vụ, trước khi sửa code, bạn PHẢI đọc tài liệu đã viết có liên quan trực tiếp đến phần được yêu cầu.** Không đọc toàn bộ báo cáo của cả hệ thống một cách máy móc.

Quy trình tìm và đọc tài liệu:

1. Xác định chính xác phạm vi nhiệm vụ: tầng nào, module nào, API nào, test nào.
2. Liệt kê các báo cáo/quick reference trong `VAD/report/` liên quan trực tiếp đến phạm vi đó. Dùng tìm kiếm tên file, tiêu đề, từ khóa trước; không mở tất cả tài liệu.
3. Đọc tối thiểu:
   - báo cáo thiết kế/thuật toán của tầng đang làm;
   - báo cáo tra cứu API hoặc quick reference của module liên quan, nếu đã có;
   - báo cáo của tầng liền kề chỉ khi cần hiểu hợp đồng đầu vào/đầu ra hoặc sự phụ thuộc.
4. Đọc code và test hiện có sau khi hiểu thiết kế. Báo cáo không thay thế việc kiểm tra implementation thực tế; code cũng không tự động thay thế quyết định thiết kế đã ghi.
5. Nếu báo cáo và code mâu thuẫn, **không tự âm thầm chọn một bên**. Nêu rõ khác biệt, xác định ảnh hưởng, rồi ưu tiên hợp đồng/API đã được người dùng chốt gần nhất. Nếu khác biệt làm thay đổi hành vi quan trọng và không thể suy ra ý định, hỏi một câu ngắn trước khi thay đổi diện rộng.
6. Trong câu trả lời cuối, nêu ngắn gọn tên các tài liệu liên quan đã đọc. Không tuyên bố đã đọc toàn bộ hệ thống nếu chỉ đọc phần liên quan.

Không yêu cầu người dùng gửi lại thông tin đã có trong repo, báo cáo, hội thoại hoặc file đính kèm. Hãy tự kiểm tra nguồn hiện có trước.

## 3. Quy trình làm việc mỗi nhiệm vụ

### Bước A — Xác định phạm vi

- Viết lại mục tiêu kỹ thuật thành 1–3 điều kiện có thể kiểm chứng.
- Xác định file code, test và báo cáo có thể bị ảnh hưởng.
- Không sửa các tầng/module không thuộc phạm vi chỉ để tiện triển khai.
- Nếu nhiệm vụ chỉ yêu cầu phân tích/đánh giá thì không tự ý sửa code.

### Bước B — Khảo sát trước khi sửa

- Đọc các báo cáo liên quan theo Mục 2.
- Đọc implementation và test thực tế.
- Chạy hoặc tái hiện test nền nếu môi trường cho phép.
- Xác định hành vi hiện tại bằng bằng chứng: output, test, log, mô phỏng hoặc phép tính. Không kết luận từ tên hàm hay comment đơn thuần.
- Với thuật toán thích nghi/stateful, theo dõi trạng thái theo thời gian và theo từng băng, không chỉ xem giá trị cuối hoặc trung bình gộp.

### Bước C — Đề xuất thay đổi nhỏ, có lý do

Trước khi thay đổi đáng kể, trình bày ngắn gọn:
- nguyên nhân được xác nhận hoặc giả thuyết còn cần kiểm chứng;
- thay đổi nhỏ nhất có thể xử lý vấn đề;
- các hành vi có nguy cơ bị ảnh hưởng;
- test nào sẽ chứng minh thay đổi có hiệu quả.

Không over-engineer. Không thêm heuristic, dependency, abstraction hoặc chế độ cấu hình mới nếu chưa có lý do được test hỗ trợ. Không đổi toàn bộ kiến trúc chỉ để giải quyết một trường hợp biên nếu có thể sửa cục bộ.

### Bước D — Triển khai theo phạm vi

- Giữ API và hành vi đã được người dùng chốt, trừ khi nhiệm vụ yêu cầu đổi.
- Tách thay đổi logic với refactor thuần túy; tránh trộn nhiều loại thay đổi trong một lần.
- Không giả định kết quả test. Chạy lệnh thực tế và ghi kết quả thực tế.
- Không tuyên bố test pass nếu chưa chạy được. Nếu môi trường không cho phép chạy, nêu rõ đó là kiểm tra tĩnh hoặc suy luận, không phải test runtime.
- Không thêm dependency mới nếu chưa được chấp thuận.

### Bước E — Kiểm thử

Luôn xem xét các nhóm test phù hợp:

1. **Happy path:** hành vi bình thường.
2. **Boundary:** audio/frame rỗng, rất ngắn, đầu/cuối stream, ngưỡng sát biên, giá trị cực nhỏ/lớn.
3. **Adversarial/regression:** tình huống trước đây đã làm hỏng thuật toán và biến thể gần đó.
4. **Stateful/streaming:** cùng đầu vào nhưng chia chunk ở nhiều ranh giới phải cho kết quả nhất quán; kiểm tra reset và nhiều chunk liên tiếp.
5. **Integration:** hợp đồng với tầng trước/sau không bị phá vỡ.
6. **Numerical:** dtype, shape, NaN/Inf, chia cho 0, log của giá trị không hợp lệ, sai lệch số học.

Với thuật toán stateful, ưu tiên ghi lại chuỗi diễn biến của từng trạng thái quan trọng (ví dụ từng band, `noise_floor`, `snr_db`, `update_mask`, counter, trạng thái `stuck/recover`) thay vì chỉ assert đầu ra cuối cùng.

Khi dùng mô phỏng ngẫu nhiên:
- cố định seed để tái lập;
- kiểm thử nhiều seed khi cần đo độ bền;
- ghi rõ giả định của mô phỏng và giới hạn khi chuyển sang audio thật;
- không coi mô hình phân phối lý tưởng là bằng chứng thay thế hoàn toàn cho waveform thực.

### Bước F — Cập nhật tài liệu

Nếu thay đổi thuật toán, ngưỡng, trạng thái, API hoặc hành vi quan sát được:
- cập nhật báo cáo thi công/thiết kế của đúng tầng;
- cập nhật quick reference/API nếu chữ ký hoặc ý nghĩa output thay đổi;
- cập nhật test plan hoặc changelog nếu repo đang dùng định dạng đó;
- không sửa báo cáo của các tầng không liên quan.

Tài liệu phải phân biệt rõ:
- **đã xác minh bằng code/test**;
- **kết quả mô phỏng tổng hợp**;
- **giả thuyết/chưa kiểm chứng**;
- **chưa đánh giá trên audio có nhãn/thực tế**.

### Bước G — Báo cáo kết quả

Kết thúc mỗi nhiệm vụ bằng một bản tóm tắt ngắn:
1. Tài liệu liên quan đã đọc.
2. File đã thay đổi.
3. Vấn đề và cách xử lý.
4. Lệnh test đã chạy và kết quả thực tế.
5. Giới hạn/chưa kiểm chứng.
6. Bước tiếp theo hợp lý nhất (tối đa một đề xuất chính).

Không dùng lời khen chung chung hoặc tuyên bố “ổn định/đã hoàn tất” nếu bằng chứng chỉ là unit test tổng hợp.

## 4. Nguyên tắc kiến trúc của dự án

- Hệ thống được chia tầng; giữ ranh giới giữa trích xuất đặc trưng, ước lượng nền nhiễu, scoring và state machine.
- Không tích hợp Tầng 2 trực tiếp vào `vad.py` nếu nhiệm vụ đang chỉ thi công hoặc kiểm thử riêng Tầng 2.
- Không tự ý sửa Tầng 1, đặc biệt quy ước STFT, dải tần, `band_power`, SFM, sample rate, hop hoặc chunk handling, khi đang làm tầng khác. Nếu phát hiện vấn đề ở Tầng 1, báo cáo bằng chứng và đề xuất nhiệm vụ riêng.
- Tầng sau tiêu thụ API tầng trước; không lặp lại hoặc tự tính lại logic của tầng trước trừ khi thiết kế yêu cầu rõ ràng.
- Không coi score sigmoid là xác suất đã hiệu chỉnh nếu chưa đánh giá calibration.
- Giữ hỗ trợ sample rate 8 kHz và 16 kHz ở nơi thiết kế hiện hành yêu cầu; không mặc định kết quả tốt ở 16 kHz cũng đúng ở 8 kHz:
  - Dù STFT preset ở 8 kHz (`n_fft=128`) và 16 kHz (`n_fft=256`) có cùng độ phân giải `62.5 Hz/bin`, cùng `hop=5 ms` và số bin mỗi băng `[3, 11, 16, 32]`, nhưng ở **8 kHz**, băng thứ 4 (`2000–4000 Hz`) chạm sát tần số Nyquist (`4000 Hz`).
  - Trên audio thực 8 kHz (telephony/codec hoặc sau bộ lọc chống chồng phổ anti-aliasing), vùng `3400–4000 Hz` thường bị suy giảm mạnh (roll-off), làm giảm `band_power[3]` của âm xát và làm **SFM trên dải `80–4000 Hz` ở 8 kHz thấp hơn so với cùng âm thanh ở 16 kHz**. Phải kiểm tra hiện tượng này khi đánh giá hoặc hiệu chỉnh ngưỡng trên audio 8 kHz.
- Không biến VAD thành ASR, không để LLM tự viết lại transcript nếu chỉ yêu cầu nhận dạng; mọi bước LLM hậu xử lý phải được đặt ở interface riêng và kiểm soát thay đổi nội dung.

## 5. Bài học cụ thể từ Tầng 1

Tầng 1 cung cấp `VADFeatures` từ `FeatureExtractor`, hiện có các đặc trưng cốt lõi như:
- `band_power`: `(n_bands, T)`, công suất trung bình trên mỗi bin;
- `sfm`: `(T,)`;
- `process_chunk()`, `flush()` và `reset()` cho luồng streaming.

Khi làm với Tầng 1 hoặc tầng phụ thuộc trực tiếp vào nó:
- giữ nhất quán định nghĩa công suất: không trộn công suất trung bình mỗi bin với tổng công suất của dải;
- xác minh quy ước bin/tần số ở sample rate và `n_fft` cụ thể;
- kiểm tra batch/chunk invariance, các chunk rỗng và `flush()`;
- không kết luận đặc trưng “tốt để phân biệt speech/noise” chỉ từ việc code khớp số học. Parity chứng minh tính nhất quán, không tự chứng minh chất lượng phân loại trên audio thật.

## 6. Bài học cụ thể từ Tầng 2 — NoiseEstimator

Đây là bối cảnh tham khảo nếu nhiệm vụ liên quan đến `VAD/noise_estimator.py`; không được áp các chi tiết này máy móc sang tầng khác.

### Hợp đồng hiện hành cần kiểm tra trước khi sửa

- Đầu vào: `VADFeatures.band_power` `(n_bands, T)` và `sfm` `(T,)`.
- Noise floor dùng cùng định nghĩa công suất trung bình trên mỗi bin.
- `noise_floor[:, t]` là nền đã dùng để tính `snr_db[:, t]`; update ở frame t chỉ ảnh hưởng frame sau.
- Warm-up dùng median theo từng band trên tối đa `init_frames`; trong warm-up, `snr_db=0` và phải có `warmup=True` để tầng sau không hiểu nhầm 0 dB là phép đo thật.
- Output hiện có `noise_floor`, `snr_db`, `update_mask`, `stuck`, `warmup`, `is_noise`. `is_noise` là cờ chẩn đoán, không thay thế `update_mask`.
- Cổng update là theo từng band; SFM là đặc trưng chung của frame.
- `strong_confirm_frames` xác nhận liên tiếp trước khi cho phép cập nhật nhanh từ nhánh `strong`; các frame chưa đủ xác nhận không được rơi sang nhánh `weak`.
- `weak` chỉ được cho phép trong vùng SNR chuyển tiếp và khi SFM vượt ngưỡng theo thiết kế hiện hành.
- `stuck/recover` cần tránh tích lũy bộ đếm trước khi stuck, tránh để `_ref` chạy theo voiced speech, và cần chịu được dao động thống kê ở band ít bin.
- `snr_noise_db=6 dB` và `strong_confirm_frames=3` là baseline chọn từ test tổng hợp, chưa được coi là tham số tối ưu trên audio thật.

### Các tình huống không được quên khi sửa NoiseEstimator

- Nhiễu dừng và noise floor hội tụ.
- Nhiễu giảm bậc: floor theo xuống trong thời gian hợp lý.
- Nhiễu tăng bậc, ổn định: tất cả band, đặc biệt band 0 ít bin, phải có cơ chế phục hồi có kiểm soát.
- Tiếng nói có SFM thấp và SNR trung bình/cao, có biến thiên công suất kiểu Gamma: tránh để các frame rơi đáy đơn lẻ kích hoạt runaway update.
- Tiếng nói xen kẽ voiced vowel và fricative: kiểm tra `stuck/recover` không học theo đoạn speech có SFM cao ngắn.
- Tín hiệu SFM cao, công suất phẳng lì kéo dài: thừa nhận giới hạn khả năng phân biệt khi chỉ quan sát `(band_power, sfm)`; không tuyên bố phân biệt hoàn hảo nếu hai loại tín hiệu cho đặc trưng giống nhau.
- Một band chứa speech trong khi các band khác chứa noise: update gate theo band không được làm đóng băng vô ích tất cả các band.
- Chunk invariance ở nhiều kích cỡ, `T=0`, `reset()`, shape/dtype và kiểm tra đầu vào không hợp lệ.

Nếu một test tổng hợp pass nhưng có mâu thuẫn với quy luật toán học, điều kiện vật lý hoặc hành vi đầu vào thực tế, hãy điều tra test và implementation; không ưu tiên “test pass” một cách máy móc.

### 6.5. Ràng buộc chuyển giao (Forward Contracts) cho Tầng 3 và Tầng 4

Khi thiết kế hoặc triển khai các tầng tiếp theo, bắt buộc đảm bảo các hợp đồng chuyển giao sau:

1. **Tầng 3 (`VAD/sigmoid.py` — Adaptive Sigmoid Scoring)**:
   - **Xử lý cờ `warmup`**: Khi `NoiseOutput.warmup[t] == True`, `snr_db[:, t] = 0.0` không phải phép đo SNR thật. Tầng 3 bắt buộc có cơ chế xử lý riêng cho frame `warmup` (ép điểm về mức nền/không kích hoạt speech hoặc truyền cờ `warmup` sang Tầng 4), tuyệt đối không đưa `0 dB` trực tiếp vào công thức tổ hợp tuyến tính như một phép đo hợp lệ.
   - **Đồng bộ thang đo (Scale normalization)**: Trong công thức $z(t) = b + \sum_{j=1}^4 w_j \cdot \mathrm{SNR}_{\text{norm}, j}(t) + w_f \cdot (1 - \mathrm{SFM}(t))$, đại lượng `1 - sfm` nằm trong $[0, 1]$ còn `snr_db` nằm trong $[-10, 40]\text{ dB}$. Phải định nghĩa tường minh phép chuẩn hóa $\mathrm{SNR}_{\text{norm}, j}(t) \in [0, 1]$ trước khi nhân trọng số $w_j$ để các trọng số cùng thang đo và SNR không lấn át SFM.
2. **Tầng 4 (`VAD/state_machine.py` — VAD State Machine & Post-Processing)**:
   - **Đơn vị cấu hình thời gian**: Các tham số thời gian (`onset`, `hangover`, `min_speech`, `min_silence`) phải được cấu hình theo đơn vị thời gian (ms hoặc s) và quy đổi sang số frame thông qua `hop_seconds` (5 ms), tránh hardcode số frame tách rời `hop_seconds`.
   - **Tính bất biến qua biên streaming (`process_chunk` vs `flush`)**: Một đoạn tiếng nói kéo dài qua ranh giới nhiều chunk phải sinh ra danh sách segment `[(start_time, end_time), ...]` nhất quán với khi chạy nguyên khối (chunk-invariance ở mức segment đầu ra).

## 7. Quy tắc về báo cáo và quyết định thiết kế

- Báo cáo là tài liệu quyết định và giải thích thiết kế, không phải danh sách mọi file của hệ thống. Đọc phần liên quan, không đọc tràn lan.
- Trước khi thêm tham số hoặc heuristic mới, chỉ ra vấn đề đã tái hiện, lý do giải pháp hiện tại không đáp ứng và test nào sẽ bảo vệ thay đổi mới.
- Không biến mọi trường hợp biên giả định thành heuristic trong production. Đầu tiên tạo test tái hiện; sau đó đánh giá mức độ thực tế và trade-off.
- Không đổi ngưỡng chỉ vì một seed hoặc một ví dụ. Với test ngẫu nhiên, dùng nhiều seed; với âm thanh thực, dùng cùng tập audio và metric nhất quán.
- Khi chưa có audio thật có nhãn, ghi rõ đây là giai đoạn baseline/tổng hợp; không khẳng định độ chính xác thực tế.
- Nếu thay đổi thông số mặc định, giải thích bằng kết quả so sánh trước/sau và cập nhật báo cáo liên quan.

## 8. Quy tắc giao tiếp với người dùng

- Nói rõ khi nào đang phân tích, khi nào đã sửa code, và khi nào đã chạy test.
- Nếu nhiệm vụ có nhiều bước, nêu kế hoạch ngắn trước khi làm; không gửi các thông báo tiến độ rỗng.
- Không hỏi người dùng lại điều đã có trong báo cáo hoặc code.
- Nếu có thể tiếp tục an toàn với giả định hợp lý, hãy làm trước và ghi rõ giả định; chỉ hỏi khi có một quyết định thiết kế quan trọng không thể suy ra.
- Không phóng đại kết quả, không bịa số liệu test, benchmark, thời gian chạy hoặc nội dung báo cáo.
- Ưu tiên thay đổi nhỏ, đo được, có thể rollback.

## 9. Definition of Done

Một nhiệm vụ chỉ được báo hoàn thành khi các mục áp dụng đều được đáp ứng:

- [ ] Đã đọc đúng các báo cáo liên quan trước khi sửa.
- [ ] Đã kiểm tra code và test đang tồn tại.
- [ ] Thay đổi nằm trong phạm vi được yêu cầu.
- [ ] Có test hồi quy cho lỗi đã sửa hoặc lý do rõ ràng vì sao không cần test mới.
- [ ] Đã chạy test liên quan và báo kết quả thực tế, hoặc nêu rõ vì sao không chạy được.
- [ ] Đã kiểm tra API/hợp đồng với tầng liền kề nếu có ảnh hưởng.
- [ ] Đã cập nhật báo cáo/quick reference của phần bị thay đổi nếu thuật toán, API hoặc hành vi thay đổi.
- [ ] Nêu rõ điều gì vẫn chưa được xác minh bằng audio thật.
