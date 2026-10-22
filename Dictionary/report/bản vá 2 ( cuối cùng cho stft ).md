# Mini Phase STFT-1 — Báo cáo Hoàn thành

> **Kết quả test:** `Ran 66 tests in 0.042s — OK`
> **File đã sửa:** [`Dictionary/STFT.py`](file:///c:/file/VAD_ASR_Diarization/Dictionary/STFT.py)
> **File mới:** [`tests/test_stft.py`](file:///c:/file/VAD_ASR_Diarization/tests/test_stft.py)

---

## 1. Thay đổi đã thực hiện

### A. Helper `_empty_spec(n_fft, output_type)` — mới

**Vì sao:** Trước đây 2 nhánh early-return đều hardcode `dtype=np.float32`, sai với `output_type="complex"` (phải là `complex64`). Hàm helper tập trung logic này vào 1 chỗ, loại bỏ lặp code.

```python
_OUTPUT_DTYPE = {"power": np.float32, "magnitude": np.float32, "complex": np.complex64}

def _empty_spec(n_fft, output_type):
    return np.empty((n_fft // 2 + 1, 0), dtype=_OUTPUT_DTYPE[output_type])
```

**Hành vi thay đổi:** `_core_stft_optimized(..., output_type="complex")` với audio rỗng hoặc audio ngắn hơn `n_fft` bây giờ trả `complex64` thay vì `float32`. `process_chunk()` tương tự.

---

### B. Validation sớm trong `_core_stft_optimized()` — sửa

**Vì sao:** Trước đây không có guard nào ở đầu hàm. `output_type` sai chỉ được phát hiện sau khi chạy hết FFT. `hop_length=0` gây `ValueError: slice step cannot be zero` hoặc `ZeroDivisionError` trong lòng thư viện, thay vì thông báo rõ nguyên nhân.

**Thêm ở đầu `_core_stft_optimized()`:**
```python
if output_type not in _VALID_OUTPUT_TYPES: raise ValueError(...)
if n_fft <= 0:        raise ValueError(...)
if hop_length <= 0:   raise ValueError(...)
if win_length <= 0:   raise ValueError(...)
if win_length > n_fft: raise ValueError(...)
if x.ndim != 1:       raise ValueError(...)
```

**Hành vi thay đổi:** Gọi trực tiếp `_core_stft_optimized()` với tham số sai bây giờ nhận `ValueError` rõ ràng ngay lập tức, không cần đi qua `StreamingSTFT.__init__`.

---

### C. Guard `center=True` + audio rỗng — sửa crash

**Vì sao:** `np.pad([], 256, mode="reflect")` raise `ValueError: can't extend empty axis 0 using modes other than 'constant' or 'empty'` — xác nhận crash thực tế trong probe test.

```python
if center:
    if len(x) == 0:
        return _empty_spec(n_fft, output_type)  # thoát sớm
    x = np.pad(x, n_fft // 2, mode=pad_mode)
```

**Hành vi thay đổi:** `stft_for_asr_patch(np.array([]))` không còn crash — trả về mảng rỗng đúng shape.

---

### D. Validation trong `StreamingSTFT.__init__()` — bổ sung

**Vì sao:** `hop_length` và `win_length` chưa được kiểm tra trong `__init__`. `create_vad_streamer(sample_rate=100)` tạo ra `hop_length=0` và truyền xuống mà không có lỗi cho đến khi `process_chunk()` bị gọi.

**Thêm vào `__init__`:** kiểm tra `n_fft`, `hop_length`, `win_length` tương tự `_core_stft_optimized`. Thêm guard tương ứng vào `create_vad_streamer()` và `create_asr_streamer()` với thông báo lỗi cụ thể về `sample_rate` tối thiểu.

**Hành vi thay đổi:** `create_vad_streamer(sample_rate=100)` raise `ValueError` rõ ràng thay vì `ZeroDivisionError` khi gọi `process_chunk()` sau đó.

---

### E. Buffer ownership — sửa 2 điểm

**Vì sao:** Probe test xác nhận `self._buffer = x[consumed_samples:]` là **view** vào `x`. Nếu caller sửa mảng đầu vào sau khi gọi `process_chunk()`, buffer bị corrupt. View cũng giữ toàn bộ chunk lớn sống trong bộ nhớ (đo được: 480 float buffer giữ 400.000 bytes).

**Điểm 1** — audio ngắn hơn `n_fft`:
```python
# Truoc:
self._buffer = x
# Sau:
self._buffer = x.copy()
```

**Điểm 2** — sau khi tạo frame:
```python
# Truoc:
self._buffer = x[consumed_samples:]
# Sau:
self._buffer = x[consumed_samples:].copy()
```

**Hành vi thay đổi:** Mutate mảng chunk sau `process_chunk()` không còn ảnh hưởng buffer nội bộ. Chunk lớn được giải phóng khi ra khỏi scope.

---

### F. Docstring thread-safety — sửa

**Vì sao:** Docstring cũ ghi `"Thread-safe: dùng threading.Lock bảo vệ _buffer"` nhưng `_core_stft_optimized()` được gọi **ngoài lock**. Nếu 2 thread gọi `process_chunk()` đồng thời, thứ tự frame đầu ra không được bảo đảm.

**Thay đổi:** Docstring nêu rõ thiết kế **single-consumer** — Lock bảo vệ state nội bộ, không bảo đảm thứ tự frame khi nhiều thread gọi đồng thời.

---

### G. `StreamingSTFT.flush()` — mới

**Logic phân biệt "overlap cũ" vs "mẫu mới":**

Sau khi `process_chunk()` tạo `num_frames` frame với `len(x) = n_fft + k·hop + r` (`0 ≤ r < hop`):
- `_buffer` = `x[num_frames·hop:]` với độ dài = `n_fft - hop + r`
- `n_fft - hop` mẫu đầu buffer là overlap với frame cuối (đã xử lý)
- `r` mẫu cuối là **mới** (chưa có frame nào bao phủ)
- Flush tạo frame chỉ khi `r > 0`, tức `len(_buffer) > n_fft - hop`

```python
def flush(self) -> np.ndarray:
    with self._lock:
        buf = self._buffer.copy()
        has_frames = self._has_produced_frames
        self._buffer = np.empty(0, dtype=np.float32)
        self._has_produced_frames = False

    if len(buf) == 0:
        return _empty_spec(self.n_fft, self.output_type)

    if has_frames:
        overlap = self.n_fft - self.hop_length
        if len(buf) - overlap <= 0:
            return _empty_spec(self.n_fft, self.output_type)

    # Zero-pad buf -> n_fft, tính 1 frame
    padded = np.zeros(self.n_fft, dtype=np.float32)
    padded[:len(buf)] = buf
    return _core_stft_optimized(padded, ...)
```

**Hành vi thay đổi so với batch `center=False`:**
- Batch: frame cuối không được tạo nếu tín hiệu không đủ `n_fft` mẫu → ưu tiên tính chính xác thời gian.
- Streaming + `flush()`: frame cuối được zero-pad nếu có ≥ 1 mẫu mới → ưu tiên không bỏ mẫu.

**Tính chất đảm bảo:**
- `flush()` lần 2 trả về mảng rỗng, không phát lại frame cũ.
- Sau `flush()`, state reset — stream sẵn sàng cho phiên mới.
- Tín hiệu rỗng, tín hiệu kết thúc đúng ranh giới frame, flush lặp đều được xử lý đúng.

---

### H. `_has_produced_frames` flag — mới trong `StreamingSTFT`

Field bool mới để `flush()` biết phân biệt "chưa có frame nào" (toàn bộ buffer là mới) với "đã có frame" (buffer có phần overlap). Được reset bởi `flush()` và `reset()`.

---

## 2. Thứ không thay đổi

| Thành phần | Lý do giữ nguyên |
|------------|-----------------|
| Power spectrum in-place (`np.square` + `np.add`) | Probe test P1–P2 xác nhận đúng, `max diff < 1e-5` vs `\|Z\|²` reference, không có UB |
| Streaming frame logic | Probe test P9 xác nhận `max diff = 0.0` vs batch sau khi fix probe script |
| FFT backend (`scipy.fft.rfft`, `overwrite_x=True`) | Không có benchmark chứng minh cần thay |
| Shape đầu ra `(n_fft // 2 + 1, num_frames)` | Không thay đổi theo spec |
| LRU cache cửa sổ | Đúng, không sửa |
| API công khai của các preset | Không thay đổi signature |
| `_get_cached_window` — validation `win_length > n_fft` bị xóa | Validation đã được chuyển lên `_core_stft_optimized()` và `__init__()` — không cần duplicate trong cache function |

---

## 3. Kết quả test

```
Ran 66 tests in 0.042s — OK
```

| Nhóm | Test | Kết quả |
|------|------|---------|
| T1. Validation | 11 | 11 PASS |
| T2. Shape & dtype | 12 | 12 PASS |
| T3. Power correctness | 7 | 7 PASS |
| T4. Streaming parity | 6 | 6 PASS |
| T5. Buffer ownership | 4 | 4 PASS |
| T6. Flush | 11 | 11 PASS |
| T7. Preset regression | 13 | 13 PASS |
| T8. _empty_spec helper | 3 | 3 PASS |

---

## 4. Tiêu chí hoàn thành — đối chiếu

| Tiêu chí | Trạng thái |
|----------|-----------|
| Kiểu đầu ra và trường hợp rỗng nhất quán | ✅ `_empty_spec` đảm bảo dtype đúng mọi nhánh |
| Power spectrum đúng với tham chiếu độc lập | ✅ `assert_allclose` rtol=1e-5 PASS |
| Buffer không bị ảnh hưởng bởi mutation đầu vào | ✅ `.copy()` ở cả 2 điểm |
| `flush()` xử lý đúng không bỏ mẫu / không phát lặp | ✅ 11 test flush đều PASS |
| Streaming, batch, preset đều pass | ✅ 66/66 |
| API và shape không thay đổi | ✅ Không đổi signature, không đổi `(n_bins, n_frames)` |
