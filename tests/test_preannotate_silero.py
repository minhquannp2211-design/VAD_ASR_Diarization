import csv
import os
import sys
import wave
from importlib import util
from unittest import mock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_DIR = os.path.join(ROOT, "VAD", "report", "tầng 4", "human_annotation_pilot")


def _load_module(name: str, filename: str):
    path = os.path.join(PILOT_DIR, filename)
    spec = util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


preannotate_silero = _load_module("preannotate_silero", "preannotate_silero.py")
pilot_tools = _load_module("pilot_tools", "pilot_tools.py")


def _write_test_wav(path: str, sr: int, samples: np.ndarray, n_channels: int = 1) -> None:
    pcm16 = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def test_interval_conversion_leading_and_trailing_non_speech():
    """
    Kiểm tra chuyển đổi các đoạn SPEECH phát hiện được thành chuỗi khoảng nửa mở liên tiếp
    có khoảng NON_SPEECH ở đầu, ở giữa và ở cuối, phủ kín [0.000, duration_s).
    """
    rows = preannotate_silero.speech_segments_to_contiguous_intervals(
        excerpt_id="E01_stat20_16k",
        speech_segments_s=[(1.200, 2.450), (3.250, 5.430)],
        duration_s=6.200,
    )
    assert len(rows) == 5
    expected = [
        ("NON_SPEECH", 0.000, 1.200),
        ("SPEECH", 1.200, 2.450),
        ("NON_SPEECH", 2.450, 3.250),
        ("SPEECH", 3.250, 5.430),
        ("NON_SPEECH", 5.430, 6.200),
    ]
    for row, (exp_label, exp_s, exp_e) in zip(rows, expected):
        assert row.excerpt_id == "E01_stat20_16k"
        assert row.label == exp_label
        assert row.start_time_s == pytest.approx(exp_s, abs=1e-9)
        assert row.end_time_s == pytest.approx(exp_e, abs=1e-9)
        assert row.annotator_id == "silero_vad_preannotation"
        assert "NOT human ground truth" in row.notes


def test_speech_touching_start_and_end_boundaries():
    """
    Kiểm tra khi tiếng nói bắt đầu ngay tại 0.000s hoặc kết thúc đúng tại duration_s:
    không được sinh các khoảng NON_SPEECH độ dài 0 ở hai đầu.
    """
    rows = preannotate_silero.speech_segments_to_contiguous_intervals(
        excerpt_id="E10_start_16k",
        speech_segments_s=[(0.000, 1.250), (2.050, 5.500)],
        duration_s=5.500,
    )
    assert len(rows) == 3
    assert [(r.label, r.start_time_s, r.end_time_s) for r in rows] == [
        ("SPEECH", 0.000, 1.250),
        ("NON_SPEECH", 1.250, 2.050),
        ("SPEECH", 2.050, 5.500),
    ]


def test_adjacent_and_overlapping_speech_segments_merged():
    """
    Kiểm tra các đoạn SPEECH chạm nhau (adjacent), chồng lấn (overlapping), hoặc cách nhau
    dưới 0.5 ms (làm tròn về cùng mốc ms) phải được gộp thành một khoảng SPEECH liên thông,
    tuyệt đối không sinh khoảng NON_SPEECH độ dài 0 hoặc hai dòng SPEECH liền kề.
    """
    rows = preannotate_silero.speech_segments_to_contiguous_intervals(
        excerpt_id="E02_stat20_8k",
        speech_segments_s=[
            (1.000, 2.000),
            (2.000, 3.100),        # chạm biên chính xác tại 2.000s
            (3.1002, 4.000),       # chạm biên sau khi làm tròn 3 chữ số thập phân (3.100s)
            (3.800, 4.500),        # chồng lấn với đoạn trước
        ],
        duration_s=6.000,
    )
    assert len(rows) == 3
    assert [(r.label, r.start_time_s, r.end_time_s) for r in rows] == [
        ("NON_SPEECH", 0.000, 1.000),
        ("SPEECH", 1.000, 4.500),
        ("NON_SPEECH", 4.500, 6.000),
    ]


def test_empty_detections_returns_full_duration_non_speech():
    """
    Kiểm tra khi không phát hiện được đoạn SPEECH nào (danh sách rỗng hoặc chỉ có đoạn suy biến),
    hàm trả về đúng 1 khoảng NON_SPEECH phủ trọn [0.000, duration_s).
    """
    rows_empty = preannotate_silero.speech_segments_to_contiguous_intervals(
        excerpt_id="E06_stat10_8k",
        speech_segments_s=[],
        duration_s=7.300,
    )
    assert len(rows_empty) == 1
    assert rows_empty[0].label == "NON_SPEECH"
    assert rows_empty[0].start_time_s == 0.000
    assert rows_empty[0].end_time_s == pytest.approx(7.300, abs=1e-9)

    # Đoạn quá ngắn (< 0.5 ms) bị triệt tiêu sau làm tròn cũng phải trả về 1 khoảng NON_SPEECH đầy đủ
    rows_subms = preannotate_silero.speech_segments_to_contiguous_intervals(
        excerpt_id="E06_stat10_8k",
        speech_segments_s=[(2.0001, 2.0003)],
        duration_s=7.300,
    )
    assert len(rows_subms) == 1
    assert rows_subms[0].label == "NON_SPEECH"
    assert rows_subms[0].start_time_s == 0.000
    assert rows_subms[0].end_time_s == pytest.approx(7.300, abs=1e-9)


def test_full_duration_coverage_and_csv_serialization_contiguity(tmp_path):
    """
    Kiểm tra các mốc lẻ (sub-millisecond từ phép chia sample / sample_rate) sau khi ghi ra CSV
    và đọc lại qua `pilot_tools.validate_annotations` không tạo khe hở (gap) hay chồng lấn (overlap)
    và khớp chính xác với duration_s của từng excerpt trong manifest.
    """
    manifest = pilot_tools.load_manifest()
    all_rows = []
    for i, (eid, spec) in enumerate(manifest.items()):
        # Các mốc sample không chia hết cho 1000 để thử thách bước làm tròn .3f
        sr = spec.sample_rate
        s1 = (int(0.733 * sr) + 1) / float(sr)
        e1 = (int(2.117 * sr) + 3) / float(sr)
        s2 = (int(3.041 * sr) + 7) / float(sr)
        e2 = (int(4.889 * sr) + 2) / float(sr)
        segs = [] if (i % 4 == 0) else [(s1, e1), (s2, e2)]
        rows = preannotate_silero.speech_segments_to_contiguous_intervals(
            excerpt_id=eid,
            speech_segments_s=segs,
            duration_s=spec.duration_s,
        )
        all_rows.extend(rows)

    out_csv = str(tmp_path / "draft_preannotation.csv")
    preannotate_silero.write_preannotation_csv(all_rows, out_csv)

    # Kiểm tra header CSV chính xác
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == [
            "excerpt_id",
            "start_time_s",
            "end_time_s",
            "label",
            "annotator_id",
            "notes",
        ]

    # Đọc lại bằng pilot_tools và xác nhận vượt qua toàn bộ kiểm tra tính hợp lệ
    loaded_ivs = pilot_tools.load_annotations_csv(out_csv)
    pilot_tools.validate_annotations(loaded_ivs, manifest)


def test_error_handling_missing_empty_unsupported_sr_and_protected_files(tmp_path):
    """
    Kiểm tra thông báo lỗi rõ ràng cho:
    - File WAV không tồn tại;
    - File WAV rỗng (0 byte hoặc 0 sample);
    - Sample rate không thuộc {8000, 16000} Hz (ví dụ 22050 Hz);
    - File WAV stereo (2 kênh);
    - Cố tình ghi đè lên `annotation_template.csv` hoặc `pilot_manifest.csv`;
    - Thiếu thư viện `silero_vad`.
    """
    # 1. File không tồn tại
    with pytest.raises(FileNotFoundError, match="Không tìm thấy file WAV"):
        preannotate_silero.read_pcm_wav_mono(str(tmp_path / "missing.wav"))

    # 2. File 0 byte
    empty_file = str(tmp_path / "zero_bytes.wav")
    with open(empty_file, "wb"):
        pass
    with pytest.raises(ValueError, match="File WAV rỗng"):
        preannotate_silero.read_pcm_wav_mono(empty_file)

    # 3. File WAV có header hợp lệ nhưng 0 sample
    zero_frames_wav = str(tmp_path / "zero_frames.wav")
    _write_test_wav(zero_frames_wav, 16000, np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="0 frames"):
        preannotate_silero.read_pcm_wav_mono(zero_frames_wav)

    # 4. Sample rate không hỗ trợ (22050 Hz)
    bad_sr_wav = str(tmp_path / "sr_22050.wav")
    _write_test_wav(bad_sr_wav, 22050, np.zeros(2205, dtype=np.float32))
    with pytest.raises(ValueError, match="Tần số lấy mẫu không được hỗ trợ"):
        preannotate_silero.read_pcm_wav_mono(bad_sr_wav)

    # 5. WAV 2 kênh (stereo)
    stereo_wav = str(tmp_path / "stereo.wav")
    _write_test_wav(stereo_wav, 16000, np.zeros(1600, dtype=np.float32), n_channels=2)
    with pytest.raises(ValueError, match="WAV mono"):
        preannotate_silero.read_pcm_wav_mono(stereo_wav)

    # 6. Đọc thành công ở cả 8 kHz và 16 kHz không resample
    for sr in (8000, 16000):
        ok_wav = str(tmp_path / f"ok_{sr}.wav")
        _write_test_wav(ok_wav, sr, np.sin(np.linspace(0, 10, sr, dtype=np.float32)) * 0.5)
        audio, read_sr, dur = preannotate_silero.read_pcm_wav_mono(ok_wav)
        assert read_sr == sr
        assert dur == pytest.approx(1.0, abs=1e-6)
        assert audio.shape == (sr,)

    # 7. Từ chối ghi đè lên annotation_template.csv
    with pytest.raises(ValueError, match="Từ chối ghi đè"):
        preannotate_silero.check_safe_output_path(
            os.path.join(PILOT_DIR, "annotation_template.csv")
        )

    # 8. Báo lỗi rõ ràng khi thiếu gói silero_vad
    with mock.patch.dict(sys.modules, {"silero_vad": None}):
        with pytest.raises(RuntimeError, match="silero-vad"):
            preannotate_silero._import_silero_dependencies()


def test_cli_arguments_and_generated_preannotation_csv_validation():
    """
    Kiểm tra bộ phân tích tham số dòng lệnh (--threshold, --min-speech-ms, --min-silence-ms, --speech-pad-ms)
    và xác nhận file `preannotation_silero.csv` thực tế vượt qua bộ kiểm tra `pilot_tools.validate_annotations`
    trên đủ 11 đoạn trích của gói pilot.
    """
    parser = preannotate_silero.build_arg_parser()
    args = parser.parse_args(
        [
            "--threshold",
            "0.45",
            "--min-speech-ms",
            "200",
            "--min-silence-ms",
            "120",
            "--speech-pad-ms",
            "40",
        ]
    )
    assert args.threshold == pytest.approx(0.45)
    assert args.min_speech_ms == 200
    assert args.min_silence_ms == 120
    assert args.speech_pad_ms == 40

    pre_csv = os.path.join(PILOT_DIR, "preannotation_silero.csv")
    if os.path.exists(pre_csv):
        manifest = pilot_tools.load_manifest()
        ivs = pilot_tools.load_annotations_csv(pre_csv)
        pilot_tools.validate_annotations(ivs, manifest)
        covered_excerpts = {iv.excerpt_id for iv in ivs}
        assert covered_excerpts == set(manifest.keys())
        for iv in ivs:
            assert iv.label in {"SPEECH", "NON_SPEECH"}
            assert iv.annotator_id == "silero_vad_preannotation"


def test_multiband_vs_silero_benchmark_outputs_and_tolerances():
    """
    Kiểm tra hàm tính dung sai biên (±10ms, ±20ms, ±50ms) và tính nhất quán số học,
    schema của các file đầu ra từ `benchmark_multiband_vs_silero.py`.
    """
    bench_mod = _load_module("benchmark_multiband_vs_silero", "benchmark_multiband_vs_silero.py")

    # 1. Kiểm tra compute_boundary_tolerances tính tay:
    # Cặp 1: onset_err = -7ms, offset_err = +18ms -> |on|<=10,20,50; |off|<=20,50 -> both<=20,50
    # Cặp 2: onset_err = +28ms, offset_err = -45ms -> |on|<=50; |off|<=50 -> both<=50
    # Cặp 3: onset_err = +80ms, offset_err = +5ms  -> |on| không đạt; |off|<=10,20,50 -> both không đạt
    tol = bench_mod.compute_boundary_tolerances(
        onset_errs_ms=[-7.0, 28.0, 80.0],
        offset_errs_ms=[18.0, -45.0, 5.0],
        tolerances_ms=(10.0, 20.0, 50.0),
    )
    assert tol[10.0] == (1, 1, 0)
    assert tol[20.0] == (1, 2, 1)
    assert tol[50.0] == (2, 3, 2)

    # 2. Kiểm tra schema và tính nhất quán của 4 file output đã sinh
    manifest = pilot_tools.load_manifest()
    pred_csv = os.path.join(PILOT_DIR, "multiband_vad_predictions.csv")
    per_file_csv = os.path.join(PILOT_DIR, "multiband_vs_silero_per_file.csv")
    summary_md = os.path.join(PILOT_DIR, "multiband_vs_silero_summary.md")
    run_log = os.path.join(PILOT_DIR, "multiband_vad_pilot_run.log")

    assert os.path.exists(pred_csv)
    assert os.path.exists(per_file_csv)
    assert os.path.exists(summary_md)
    assert os.path.exists(run_log)

    # Kiểm tra multiband_vad_predictions.csv: mọi đoạn nằm trong [0.0, duration_s] và không chồng lấn
    preds_by_eid = {}
    with open(pred_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["excerpt_id"]
            assert eid in manifest
            s_t = float(row["start_time_s"])
            e_t = float(row["end_time_s"])
            dur_t = float(row["duration_s"])
            assert 0.0 <= s_t < e_t <= manifest[eid].duration_s + 1e-6
            assert dur_t == pytest.approx(e_t - s_t, abs=1e-3)
            preds_by_eid.setdefault(eid, []).append((s_t, e_t))

    for eid, seg_list in preds_by_eid.items():
        for k in range(len(seg_list) - 1):
            assert seg_list[k][1] <= seg_list[k + 1][0] + 1e-6

    # Kiểm tra multiband_vs_silero_per_file.csv: đủ 11 dòng, TP + FP + FN + TN == expected_frames_5ms
    with open(per_file_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 11
    for r in rows:
        eid = r["excerpt_id"]
        spec = manifest[eid]
        tp = int(r["tp"])
        fp = int(r["fp"])
        fn = int(r["fn"])
        tn = int(r["tn"])
        unc = int(r["uncertain_frames"])
        assert tp + fp + fn + tn + unc == spec.expected_frames_5ms
        assert r["chunk_invariance_ok"] == "True"
        assert r["timestamps_within_duration"] == "True"
        assert int(r["n_pred_segs"]) == len(preds_by_eid.get(eid, []))


