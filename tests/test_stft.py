"""
tests/test_stft.py
Regression test suite cho Dictionary/STFT.py

Cau truc: unittest.TestCase, khong them dependency ngoai (chi numpy / scipy).
Chay:  py -m pytest tests/test_stft.py -v
   or: py -m unittest tests.test_stft -v
"""

import sys
import os
import unittest
import numpy as np

# Them root cua repo vao sys.path de import Dictionary.STFT
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Dictionary.STFT import (
    _core_stft_optimized,
    _empty_spec,
    _OUTPUT_DTYPE,
    StreamingSTFT,
    create_vad_streamer,
    create_asr_streamer,
    stft_for_asr_patch,
    stft_for_diarization,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_power(x, n_fft, hop_length, win_length, center=False):
    """Cong thuc tham chieu doc lap: |Z|^2 khong dung in-place."""
    import scipy.signal
    import scipy.fft as sfft

    x = np.asarray(x, dtype=np.float32)
    if center:
        if len(x) == 0:
            return np.empty((n_fft // 2 + 1, 0), dtype=np.float32)
        x = np.pad(x, n_fft // 2, mode="reflect")
    if len(x) < n_fft:
        return np.empty((n_fft // 2 + 1, 0), dtype=np.float32)

    win = scipy.signal.get_window("hann", win_length, fftbins=True).astype(np.float32)
    if win_length < n_fft:
        pad_l = (n_fft - win_length) // 2
        pad_r = n_fft - win_length - pad_l
        win = np.pad(win, (pad_l, pad_r))

    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop_length]
    windowed = np.multiply(frames, win, dtype=np.float32)
    Z_ref = sfft.rfft(windowed.copy(), n=n_fft, axis=-1)  # copy, khong overwrite
    power = (np.abs(Z_ref) ** 2).astype(np.float32)
    return power.T


def _make_audio(length: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length).astype(np.float32)


# ---------------------------------------------------------------------------
# T1. Validation: tham so sai phai raise loi ro rang
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def test_invalid_output_type_core(self):
        """_core_stft_optimized phai raise ValueError ngay khi output_type sai."""
        with self.assertRaises(ValueError, msg="output_type='spectrogram' phai raise"):
            _core_stft_optimized(
                np.ones(512, dtype=np.float32),
                n_fft=512, hop_length=160, win_length=400,
                output_type="spectrogram"
            )

    def test_invalid_output_type_streaming(self):
        with self.assertRaises(ValueError):
            StreamingSTFT(output_type="mel")

    def test_n_fft_zero(self):
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones(512, dtype=np.float32),
                                 n_fft=0, hop_length=160, win_length=400)

    def test_n_fft_negative(self):
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones(512, dtype=np.float32),
                                 n_fft=-1, hop_length=160, win_length=400)

    def test_hop_length_zero_core(self):
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones(512, dtype=np.float32),
                                 n_fft=512, hop_length=0, win_length=400)

    def test_hop_length_zero_streaming(self):
        with self.assertRaises(ValueError):
            StreamingSTFT(n_fft=512, hop_length=0, win_length=400)

    def test_win_length_zero(self):
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones(512, dtype=np.float32),
                                 n_fft=512, hop_length=160, win_length=0)

    def test_win_length_exceeds_n_fft(self):
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones(512, dtype=np.float32),
                                 n_fft=256, hop_length=80, win_length=400)

    def test_win_length_exceeds_n_fft_streaming(self):
        with self.assertRaises(ValueError):
            StreamingSTFT(n_fft=256, hop_length=80, win_length=400)

    def test_multidimensional_input_core(self):
        """Input 2-D phai raise ValueError."""
        with self.assertRaises(ValueError):
            _core_stft_optimized(np.ones((2, 512), dtype=np.float32),
                                 n_fft=512, hop_length=160, win_length=400)

    def test_multidimensional_input_streaming(self):
        streamer = StreamingSTFT()
        with self.assertRaises(ValueError):
            streamer.process_chunk(np.ones((2, 256), dtype=np.float32))


# ---------------------------------------------------------------------------
# T2. Shape va dtype: ca ba output_type, audio rong, audio ngan
# ---------------------------------------------------------------------------

class TestShapeAndDtype(unittest.TestCase):

    N_FFT = 512
    HOP = 160
    WIN = 400

    def _run(self, audio_len, output_type, center=False):
        x = _make_audio(audio_len) if audio_len > 0 else np.array([], dtype=np.float32)
        return _core_stft_optimized(
            x, self.N_FFT, self.HOP, self.WIN,
            center=center, output_type=output_type
        )

    def test_normal_power_shape(self):
        spec = self._run(5000, "power")
        n_bins = self.N_FFT // 2 + 1
        expected_frames = 1 + (5000 - self.N_FFT) // self.HOP
        self.assertEqual(spec.shape, (n_bins, expected_frames))
        self.assertEqual(spec.dtype, np.float32)

    def test_normal_magnitude_shape(self):
        spec = self._run(5000, "magnitude")
        self.assertEqual(spec.dtype, np.float32)
        self.assertEqual(spec.shape[0], self.N_FFT // 2 + 1)

    def test_normal_complex_shape(self):
        spec = self._run(5000, "complex")
        self.assertEqual(spec.dtype, np.complex64)
        self.assertEqual(spec.shape[0], self.N_FFT // 2 + 1)

    def test_empty_audio_power(self):
        spec = self._run(0, "power")
        self.assertEqual(spec.shape, (self.N_FFT // 2 + 1, 0))
        self.assertEqual(spec.dtype, np.float32)

    def test_empty_audio_magnitude(self):
        spec = self._run(0, "magnitude")
        self.assertEqual(spec.dtype, np.float32)
        self.assertEqual(spec.shape[1], 0)

    def test_empty_audio_complex(self):
        spec = self._run(0, "complex")
        self.assertEqual(spec.dtype, np.complex64)   # BUG truoc: float32
        self.assertEqual(spec.shape[1], 0)

    def test_short_audio_power(self):
        """Audio ngan hon n_fft -> 0 frame."""
        spec = self._run(100, "power")
        self.assertEqual(spec.shape, (self.N_FFT // 2 + 1, 0))
        self.assertEqual(spec.dtype, np.float32)

    def test_short_audio_complex(self):
        spec = self._run(100, "complex")
        self.assertEqual(spec.dtype, np.complex64)   # phai la complex64
        self.assertEqual(spec.shape[1], 0)

    def test_empty_audio_center_true(self):
        """center=True + audio rong khong duoc crash."""
        spec = self._run(0, "power", center=True)
        self.assertEqual(spec.shape[1], 0)

    def test_empty_audio_center_true_complex(self):
        spec = self._run(0, "complex", center=True)
        self.assertEqual(spec.dtype, np.complex64)
        self.assertEqual(spec.shape[1], 0)

    def test_exactly_one_frame(self):
        """Audio = n_fft mau -> dung 1 frame."""
        spec = self._run(self.N_FFT, "power")
        self.assertEqual(spec.shape[1], 1)

    def test_shape_consistency_all_types(self):
        """Ba output_type phai tra ve cung so frame."""
        n = 3000
        p = self._run(n, "power")
        m = self._run(n, "magnitude")
        c = self._run(n, "complex")
        self.assertEqual(p.shape[1], m.shape[1])
        self.assertEqual(p.shape[1], c.shape[1])


# ---------------------------------------------------------------------------
# T3. Power correctness: so sanh voi cong thuc tham chieu doc lap
# ---------------------------------------------------------------------------

class TestPowerCorrectness(unittest.TestCase):

    def _check(self, audio_len, n_fft, hop, win, center=False, rtol=1e-5, atol=1e-5):
        x = _make_audio(audio_len)
        got = _core_stft_optimized(x, n_fft, hop, win, center=center, output_type="power")
        ref = _ref_power(x, n_fft, hop, win, center=center)
        self.assertEqual(got.shape, ref.shape, "shape mismatch")
        np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol,
                                   err_msg="Power in-place khong bang |Z|^2 reference")

    def test_standard_config(self):
        self._check(16000, 512, 160, 400)

    def test_vad_config(self):
        self._check(16000, 256, 80, 256)

    def test_diarization_config(self):
        self._check(16000, 512, 160, 512)

    def test_center_true(self):
        self._check(8000, 512, 160, 400, center=True)

    def test_power_nonnegative(self):
        """Power spectrum khong the am."""
        x = _make_audio(5000)
        spec = _core_stft_optimized(x, 512, 160, 400, output_type="power")
        self.assertTrue(np.all(spec >= 0), "Power spectrum chua gia tri am")

    def test_magnitude_vs_power(self):
        """magnitude^2 phai bang power."""
        x = _make_audio(5000)
        power = _core_stft_optimized(x, 512, 160, 400, output_type="power")
        mag = _core_stft_optimized(x, 512, 160, 400, output_type="magnitude")
        np.testing.assert_allclose(
            mag ** 2, power, rtol=1e-4, atol=1e-4,
            err_msg="magnitude^2 != power"
        )

    def test_complex_abs_vs_magnitude(self):
        """abs(complex)^2 phai bang power."""
        x = _make_audio(5000)
        power = _core_stft_optimized(x, 512, 160, 400, output_type="power")
        comp = _core_stft_optimized(x, 512, 160, 400, output_type="complex")
        np.testing.assert_allclose(
            (np.abs(comp) ** 2).astype(np.float32), power,
            rtol=1e-4, atol=1e-4,
            err_msg="abs(complex)^2 != power"
        )


# ---------------------------------------------------------------------------
# T4. Streaming parity: chunk nho -> frame khop batch center=False
# ---------------------------------------------------------------------------

class TestStreamingParity(unittest.TestCase):

    N_FFT = 512
    HOP = 160
    WIN = 400

    def _batch(self, audio):
        return _core_stft_optimized(audio, self.N_FFT, self.HOP, self.WIN,
                                    center=False, output_type="power")

    def _stream(self, audio, chunk_sizes):
        streamer = StreamingSTFT(self.N_FFT, self.HOP, self.WIN, output_type="power")
        all_frames = []
        pos = 0
        for size in chunk_sizes:
            spec = streamer.process_chunk(audio[pos:pos + size])
            if spec.shape[1] > 0:
                all_frames.append(spec)
            pos += size
        return np.concatenate(all_frames, axis=1) if all_frames else np.empty(
            (self.N_FFT // 2 + 1, 0), dtype=np.float32)

    def _check_parity(self, audio_len, chunk_sizes, seed=99):
        audio = _make_audio(audio_len, seed)
        batch = self._batch(audio)
        stream = self._stream(audio, chunk_sizes)
        # Streaming co the co it frame hon batch neu co tail (flush chua duoc goi)
        n = min(batch.shape[1], stream.shape[1])
        self.assertEqual(stream.shape[1], n,
                         f"Streaming tao them frame thua: {stream.shape[1]} > {n}")
        np.testing.assert_allclose(
            stream[:, :n], batch[:, :n],
            rtol=1e-4, atol=1e-4,
            err_msg="Frame streaming khong khop batch"
        )

    def test_equal_chunks(self):
        self._check_parity(5000, [500] * 10)

    def test_variable_chunks(self):
        self._check_parity(5000, [300, 200, 500, 400, 300, 3300])

    def test_tiny_chunks(self):
        self._check_parity(4000, [50] * 80)

    def test_single_chunk(self):
        """1 chunk = toan bo audio -> streaming == batch."""
        audio = _make_audio(5000)
        batch = self._batch(audio)
        stream = self._stream(audio, [5000])
        np.testing.assert_allclose(stream, batch, rtol=1e-4, atol=1e-4)

    def test_chunk_larger_than_audio(self):
        """Chunk lon hon audio khong duoc crash."""
        audio = _make_audio(1000)
        stream = self._stream(audio, [5000])
        batch = self._batch(audio)
        n = min(batch.shape[1], stream.shape[1])
        np.testing.assert_allclose(stream[:, :n], batch[:, :n], rtol=1e-4, atol=1e-4)

    def test_frame_order_preserved(self):
        """Thu tu frame phai nhat quan - dung chunk nho."""
        audio = _make_audio(3200, seed=7)
        batch = self._batch(audio)
        stream = self._stream(audio, [100] * 32)
        n = min(batch.shape[1], stream.shape[1])
        np.testing.assert_allclose(stream[:, :n], batch[:, :n], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# T5. Buffer ownership: sua mang dau vao sau khi goi process_chunk
# ---------------------------------------------------------------------------

class TestBufferOwnership(unittest.TestCase):

    def test_mutate_chunk_after_call_no_effect_on_buffer(self):
        """
        Sau khi goi process_chunk(), sua doi chunk dau vao khong duoc
        lam thay doi du lieu buffer noi bo.
        """
        streamer = StreamingSTFT(n_fft=512, hop_length=160, win_length=400)
        chunk = np.ones(300, dtype=np.float32)
        streamer.process_chunk(chunk)  # chunk -> buffer (< n_fft)

        # Snapshot buffer truoc khi sua
        buf_before = streamer._buffer.copy()
        # Sua doi chunk sau khi goi
        chunk[:] = 999.0
        buf_after = streamer._buffer.copy()

        np.testing.assert_array_equal(buf_before, buf_after,
                                      err_msg="Buffer bi anh huong boi mutation cua chunk")

    def test_mutate_large_chunk_after_frames_produced(self):
        """
        Khi chunk tao ra frame, phan du luu vao buffer cung phai doc lap.
        """
        streamer = StreamingSTFT(n_fft=512, hop_length=160, win_length=400)
        # chunk du lon de tao frame
        chunk = np.arange(2000, dtype=np.float32)
        streamer.process_chunk(chunk)

        buf_before = streamer._buffer.copy()
        chunk[:] = -1.0  # sua doi chunk
        buf_after = streamer._buffer.copy()

        np.testing.assert_array_equal(buf_before, buf_after,
                                      err_msg="Buffer phan du bi corrupt sau khi sua chunk lon")

    def test_small_chunk_then_mutate(self):
        """Chunk nho (chua du 1 frame) -> buffer phai doc lap."""
        streamer = StreamingSTFT(n_fft=512, hop_length=160, win_length=400)
        chunk = np.linspace(0, 1, 200, dtype=np.float32)
        streamer.process_chunk(chunk)

        buf_before = streamer._buffer.copy()
        chunk[:] = 0.0
        buf_after = streamer._buffer.copy()

        np.testing.assert_array_equal(buf_before, buf_after,
                                      err_msg="Buffer chua du 1 frame bi corrupt")

    def test_buffer_does_not_retain_large_array(self):
        """
        Buffer noi bo chi giu phan du, khong giu tham chieu vao chunk lon.
        """
        import gc
        streamer = StreamingSTFT(n_fft=512, hop_length=160, win_length=400)
        big_chunk = np.zeros(100_000, dtype=np.float32)
        streamer.process_chunk(big_chunk)
        del big_chunk
        gc.collect()
        # Buffer phai nho hon n_fft mau (khong giu base cua chunk lon)
        self.assertLess(len(streamer._buffer), streamer.n_fft,
                        "Buffer giu qua nhieu mau — co the dang giu view vao chunk lon")
        if streamer._buffer.base is not None:
            # Neu co base, base khong duoc la array rat lon
            base_size = streamer._buffer.base.nbytes
            self.assertLess(base_size, streamer.n_fft * 4 * 2,
                            f"Buffer co base {base_size} bytes — nghi ngo view vao chunk lon")


# ---------------------------------------------------------------------------
# T6. flush(): tin hieu ngan, ket thuc dung ranh gioi frame, co phan du
# ---------------------------------------------------------------------------

class TestFlush(unittest.TestCase):

    N_FFT = 512
    HOP = 160
    WIN = 400

    def _make_streamer(self, output_type="power"):
        return StreamingSTFT(self.N_FFT, self.HOP, self.WIN, output_type=output_type)

    def test_flush_empty_stream(self):
        """Flush khi chua co du lieu -> mang rong."""
        streamer = self._make_streamer()
        result = streamer.flush()
        self.assertEqual(result.shape, (self.N_FFT // 2 + 1, 0))
        self.assertEqual(result.dtype, np.float32)

    def test_flush_empty_stream_complex(self):
        streamer = self._make_streamer("complex")
        result = streamer.flush()
        self.assertEqual(result.dtype, np.complex64)
        self.assertEqual(result.shape[1], 0)

    def test_flush_short_signal(self):
        """Tin hieu ngan hon n_fft -> flush phai tao dung 1 frame."""
        streamer = self._make_streamer()
        audio = _make_audio(200)  # < n_fft=512
        streamer.process_chunk(audio)  # tao 0 frame thuong
        result = streamer.flush()
        self.assertEqual(result.shape, (self.N_FFT // 2 + 1, 1),
                         f"Tin hieu ngan: flush phai tao 1 frame, got shape={result.shape}")
        self.assertEqual(result.dtype, np.float32)

    def test_flush_signal_at_frame_boundary(self):
        """Tin hieu ket thuc dung tai ranh gioi frame -> flush khong tao them frame."""
        # Xay dung audio khop chinh xac 5 frame (center=False)
        # len(x) = n_fft + (n_frames - 1) * hop_length = 512 + 4*160 = 1152
        audio = _make_audio(1152)
        streamer = self._make_streamer()
        spec = streamer.process_chunk(audio)
        self.assertEqual(spec.shape[1], 5, f"Phai co 5 frame thuong, got {spec.shape[1]}")
        result = streamer.flush()
        self.assertEqual(result.shape[1], 0,
                         "Tin hieu ket thuc dung ranh gioi: flush khong duoc tao frame du")

    def test_flush_signal_with_tail(self):
        """Tin hieu co phan du sau frame cuoi -> flush tao them 1 frame."""
        # 5 frame day du + 50 mau du
        audio = _make_audio(1152 + 50)  # 1202 mau
        streamer = self._make_streamer()
        spec = streamer.process_chunk(audio)
        self.assertEqual(spec.shape[1], 5)
        result = streamer.flush()
        self.assertEqual(result.shape[1], 1,
                         f"Co phan du: flush phai tao 1 frame, got {result.shape[1]}")

    def test_flush_idempotent(self):
        """Flush lan 2 phai tra ve rong, khong phat lai frame cu."""
        audio = _make_audio(700)  # co phan du sau 1 frame
        streamer = self._make_streamer()
        streamer.process_chunk(audio)
        first = streamer.flush()
        second = streamer.flush()
        self.assertGreaterEqual(first.shape[1], 0)  # lan dau co the co frame
        self.assertEqual(second.shape[1], 0,
                         "Flush lan 2 khong duoc phat lai frame cu")

    def test_flush_resets_state(self):
        """Sau flush, stream phai bat dau lai tu dau (session moi)."""
        streamer = self._make_streamer()
        audio1 = _make_audio(700)
        streamer.process_chunk(audio1)
        streamer.flush()

        # Stream moi: process chunk ngan
        audio2 = _make_audio(300)
        spec2 = streamer.process_chunk(audio2)  # < n_fft -> 0 frame
        self.assertEqual(spec2.shape[1], 0)
        # flush phai tao frame tu audio2, khong pha tron audio1
        result = streamer.flush()
        self.assertEqual(result.shape[1], 1)

    def test_flush_dtype_power(self):
        streamer = self._make_streamer("power")
        streamer.process_chunk(_make_audio(700))
        result = streamer.flush()
        if result.shape[1] > 0:
            self.assertEqual(result.dtype, np.float32)

    def test_flush_dtype_complex(self):
        streamer = self._make_streamer("complex")
        streamer.process_chunk(_make_audio(700))
        result = streamer.flush()
        if result.shape[1] > 0:
            self.assertEqual(result.dtype, np.complex64)

    def test_flush_values_reasonable(self):
        """Frame cuoi phai la zero-padded STFT hop le (khong NaN/Inf)."""
        audio = _make_audio(700)
        streamer = self._make_streamer()
        streamer.process_chunk(audio)
        result = streamer.flush()
        if result.shape[1] > 0:
            self.assertFalse(np.any(np.isnan(result)), "Frame flush chua NaN")
            self.assertFalse(np.any(np.isinf(result)), "Frame flush chua Inf")
            self.assertTrue(np.all(result >= 0), "Power frame flush chua gia tri am")


# ---------------------------------------------------------------------------
# T7. Regression: preset hien tai voi dau vao hop le
# ---------------------------------------------------------------------------

class TestPresets(unittest.TestCase):

    def test_vad_streamer_default(self):
        streamer = create_vad_streamer(sample_rate=16000)
        audio = _make_audio(3200)
        spec = streamer.process_chunk(audio)
        self.assertGreater(spec.shape[1], 0)
        self.assertEqual(spec.dtype, np.float32)
        self.assertEqual(spec.shape[0], streamer.n_fft // 2 + 1)

    def test_vad_streamer_flush(self):
        streamer = create_vad_streamer(sample_rate=16000)
        audio = _make_audio(3200)
        streamer.process_chunk(audio)
        result = streamer.flush()
        self.assertIn(result.shape[1], [0, 1])

    def test_vad_streamer_small_sr_raises(self):
        """sample_rate qua nho phai raise loi ro rang, khong crash ZeroDivisionError."""
        with self.assertRaises(ValueError):
            create_vad_streamer(sample_rate=100)

    def test_asr_streamer_default(self):
        streamer = create_asr_streamer(sample_rate=16000)
        audio = _make_audio(4800)
        spec = streamer.process_chunk(audio)
        self.assertGreater(spec.shape[1], 0)
        self.assertEqual(spec.dtype, np.float32)

    def test_asr_streamer_flush(self):
        streamer = create_asr_streamer(sample_rate=16000)
        audio = _make_audio(4800)
        streamer.process_chunk(audio)
        result = streamer.flush()
        self.assertIn(result.shape[1], [0, 1])

    def test_asr_streamer_small_sr_raises(self):
        with self.assertRaises(ValueError):
            create_asr_streamer(sample_rate=50)

    def test_stft_for_asr_patch(self):
        audio = _make_audio(16000)
        spec = stft_for_asr_patch(audio, sample_rate=16000)
        self.assertEqual(spec.dtype, np.float32)
        self.assertGreater(spec.shape[1], 0)

    def test_stft_for_asr_patch_empty(self):
        """center=True + audio rong khong duoc crash."""
        spec = stft_for_asr_patch(np.array([], dtype=np.float32), sample_rate=16000)
        self.assertEqual(spec.shape[1], 0)

    def test_stft_for_diarization_magnitude(self):
        audio = _make_audio(16000)
        spec = stft_for_diarization(audio, sample_rate=16000, return_magnitude=True)
        self.assertEqual(spec.dtype, np.float32)
        self.assertGreater(spec.shape[1], 0)

    def test_stft_for_diarization_power(self):
        audio = _make_audio(16000)
        spec = stft_for_diarization(audio, sample_rate=16000, return_magnitude=False)
        self.assertEqual(spec.dtype, np.float32)
        self.assertTrue(np.all(spec >= 0))

    def test_stft_for_diarization_empty(self):
        spec = stft_for_diarization(np.array([], dtype=np.float32), sample_rate=16000)
        self.assertEqual(spec.shape[1], 0)

    def test_preset_shapes_consistent(self):
        """Tat ca preset phai tra ve (n_bins, num_frames) khong co shape la."""
        audio = _make_audio(16000)
        s1 = stft_for_asr_patch(audio)
        s2 = stft_for_diarization(audio)
        self.assertEqual(s1.ndim, 2)
        self.assertEqual(s2.ndim, 2)

    def test_reset_clears_state(self):
        """reset() phai xoa buffer va flag lich su."""
        streamer = create_vad_streamer()
        streamer.process_chunk(_make_audio(3200))
        streamer.reset()
        self.assertEqual(len(streamer._buffer), 0)
        self.assertFalse(streamer._has_produced_frames)
        # Flush sau reset phai tra rong
        result = streamer.flush()
        self.assertEqual(result.shape[1], 0)


# ---------------------------------------------------------------------------
# T8. Kiem tra _empty_spec helper
# ---------------------------------------------------------------------------

class TestEmptySpec(unittest.TestCase):

    def test_power_dtype(self):
        spec = _empty_spec(512, "power")
        self.assertEqual(spec.dtype, np.float32)
        self.assertEqual(spec.shape, (257, 0))

    def test_magnitude_dtype(self):
        spec = _empty_spec(512, "magnitude")
        self.assertEqual(spec.dtype, np.float32)

    def test_complex_dtype(self):
        spec = _empty_spec(512, "complex")
        self.assertEqual(spec.dtype, np.complex64)
        self.assertEqual(spec.shape, (257, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
