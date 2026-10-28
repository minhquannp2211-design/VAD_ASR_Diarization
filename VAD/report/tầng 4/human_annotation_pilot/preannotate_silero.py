"""
VAD/report/tầng 4/human_annotation_pilot/preannotate_silero.py

Công cụ tiền gán nhãn độc lập (Independent VAD Pre-Annotation Utility) sử dụng gói `silero-vad`
chính thức để tạo bản nháp phân đoạn `[start_time_s, end_time_s)` trực tiếp từ 11 file WAV
trong gói Human-Annotation Pilot (`VAD/report/tầng 4/human_annotation_pilot/wav/`).

LƯU Ý QUAN TRỌNG VỀ PHƯƠNG PHÁP LUẬN:
- Đầu ra `preannotation_silero.csv` CHỈ LÀ BẢN NHÁP TIỀN GÁN NHÃN TỰ ĐỘNG (Automated Pre-Annotation)
  nhằm hỗ trợ người gán nhãn tiết kiệm thời gian căn chỉnh biên ban đầu.
- Đây TUYỆT ĐỐI KHÔNG PHẢI nhãn chuẩn của con người (NOT human ground truth) và không được dùng
  thay thế cho bước nghe kiểm tra độc lập của con người.
- Công cụ này chỉ gán 2 nhãn `SPEECH` và `NON_SPEECH` từ sóng âm WAV (không sinh nhãn `UNCERTAIN`
  và không lấy từ transcript hay pseudo-label của hệ thống). Người gán nhãn cần nghe lại file WAV,
  điều chỉnh biên nếu cần, bổ sung nhãn `UNCERTAIN` ở các vùng biên mơ hồ và đổi `annotator_id`
  sang mã người gán nhãn thật sau khi rà soát xong.
- Tuyệt đối không ghi đè lên `annotation_template.csv` hay bất kỳ kết quả benchmark nào.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import wave
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WAV_DIR = os.path.join(PILOT_DIR, "wav")
DEFAULT_MANIFEST = os.path.join(PILOT_DIR, "pilot_manifest.csv")
DEFAULT_OUTPUT_CSV = os.path.join(PILOT_DIR, "preannotation_silero.csv")

SUPPORTED_SAMPLE_RATES = (8000, 16000)
PREANNOTATION_ANNOTATOR_ID = "silero_vad_preannotation"
PREANNOTATION_NOTES = (
    "AUTO_PREANNOTATION (silero-vad): draft label from waveform only, "
    "NOT human ground truth; requires human listener review"
)
PROTECTED_FILENAMES = {
    "annotation_template.csv",
    "pilot_manifest.csv",
    "bao_cao_benchmark_end_to_end_stage_c2.txt",
    "bao_cao_audit_benchmark_stage_d1.txt",
}
CSV_HEADER = ["excerpt_id", "start_time_s", "end_time_s", "label", "annotator_id", "notes"]


@dataclass(frozen=True)
class PreannotationRow:
    excerpt_id: str
    start_time_s: float
    end_time_s: float
    label: str
    annotator_id: str = PREANNOTATION_ANNOTATOR_ID
    notes: str = PREANNOTATION_NOTES

    def to_csv_row(self) -> List[str]:
        return [
            self.excerpt_id,
            f"{self.start_time_s:.3f}",
            f"{self.end_time_s:.3f}",
            self.label,
            self.annotator_id,
            self.notes,
        ]


@dataclass(frozen=True)
class FileSummary:
    excerpt_id: str
    wav_path: str
    sample_rate: int
    duration_s: float
    speech_segment_count: int
    speech_duration_s: float
    total_intervals: int


def _import_silero_dependencies() -> Tuple[Any, Callable[..., Any], Callable[..., Any]]:
    """
    Nhập `torch` và `silero_vad` khi cần chạy mô hình. Nếu thiếu gói, báo lỗi rõ ràng kèm lệnh cài đặt.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện 'torch' cần thiết cho silero-vad. "
            "Hãy cài đặt bằng lệnh: py -m pip install torch silero-vad"
        ) from exc

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu gói 'silero-vad' chính thức. "
            "Hãy cài đặt bằng lệnh: py -m pip install silero-vad"
        ) from exc

    return torch, load_silero_vad, get_speech_timestamps


def read_pcm_wav_mono(wav_path: str) -> Tuple[np.ndarray, int, float]:
    """
    Đọc file WAV PCM mono ở 8 kHz hoặc 16 kHz mà KHÔNG resample.
    Trả về `(waveform_float32, sample_rate, duration_s)`.

    Ném lỗi tường minh nếu:
    - File không tồn tại hoặc rỗng (0 byte / 0 sample);
    - File không phải mono (1 kênh) hoặc không phải PCM không nén;
    - Tần số lấy mẫu không thuộc `{8000, 16000}`.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(
            f"Không tìm thấy file WAV: '{wav_path}'. "
            "Nếu chưa xuất WAV cho gói pilot, hãy chạy: "
            "py \"VAD/report/tầng 4/human_annotation_pilot/pilot_tools.py\" export-wavs"
        )
    if os.path.getsize(wav_path) == 0:
        raise ValueError(f"File WAV rỗng (0 bytes): '{wav_path}'")

    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            comptype = wf.getcomptype()
            raw_bytes = wf.readframes(n_frames)
    except wave.Error as exc:
        raise ValueError(f"File WAV không hợp lệ hoặc bị hỏng '{wav_path}': {exc}") from exc

    if comptype != "NONE":
        raise ValueError(
            f"Chỉ hỗ trợ WAV PCM không nén (comptype='NONE'), nhận comptype='{comptype}' tại '{wav_path}'"
        )
    if n_channels != 1:
        raise ValueError(
            f"Yêu cầu WAV mono (1 kênh), nhưng '{wav_path}' có {n_channels} kênh."
        )
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"Tần số lấy mẫu không được hỗ trợ ({sample_rate} Hz) tại '{wav_path}'. "
            f"Silero VAD yêu cầu đúng {SUPPORTED_SAMPLE_RATES} Hz (không tự ý resample)."
        )
    if n_frames <= 0 or len(raw_bytes) == 0:
        raise ValueError(f"File WAV không chứa mẫu âm thanh nào (0 frames): '{wav_path}'")

    if sampwidth == 2:
        pcm = np.frombuffer(raw_bytes, dtype="<i2")
        audio = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    elif sampwidth == 1:
        pcm = np.frombuffer(raw_bytes, dtype=np.uint8)
        audio = ((pcm.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    elif sampwidth == 4:
        pcm = np.frombuffer(raw_bytes, dtype="<i4")
        audio = (pcm.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    else:
        raise ValueError(
            f"Độ rộng mẫu PCM không được hỗ trợ (sampwidth={sampwidth} bytes) tại '{wav_path}'."
        )

    if audio.size == 0:
        raise ValueError(f"File WAV rỗng sau khi giải mã PCM: '{wav_path}'")

    duration_s = float(audio.size) / float(sample_rate)
    return audio, sample_rate, duration_s


def detect_speech_segments_silero(
    audio: np.ndarray,
    sample_rate: int,
    model: Any,
    get_speech_timestamps_fn: Callable[..., Any],
    torch_module: Any,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> List[Tuple[float, float]]:
    """
    Chạy `get_speech_timestamps` của `silero-vad` trực tiếp trên waveform ở đúng `sample_rate`
    (`8000` hoặc `16000` Hz) với `return_seconds=False` để lấy chỉ số sample chính xác, sau đó
    quy đổi sang giây `(start_sample / sample_rate, end_sample / sample_rate)`.
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"sample_rate phải thuộc {SUPPORTED_SAMPLE_RATES}, nhận {sample_rate}"
        )
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("audio đầu vào phải là mảng 1 chiều không rỗng")

    tensor = torch_module.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    raw_ts = get_speech_timestamps_fn(
        tensor,
        model,
        threshold=float(threshold),
        sampling_rate=int(sample_rate),
        min_speech_duration_ms=int(min_speech_duration_ms),
        min_silence_duration_ms=int(min_silence_duration_ms),
        speech_pad_ms=int(speech_pad_ms),
        return_seconds=False,
    )

    n_samples = int(audio.size)
    segments_s: List[Tuple[float, float]] = []
    for item in raw_ts:
        s_idx = max(0, min(n_samples, int(item["start"])))
        e_idx = max(0, min(n_samples, int(item["end"])))
        if e_idx > s_idx:
            segments_s.append((s_idx / float(sample_rate), e_idx / float(sample_rate)))
    return segments_s


def speech_segments_to_contiguous_intervals(
    excerpt_id: str,
    speech_segments_s: Sequence[Tuple[float, float]],
    duration_s: float,
    annotator_id: str = PREANNOTATION_ANNOTATOR_ID,
    notes: str = PREANNOTATION_NOTES,
) -> List[PreannotationRow]:
    """
    Chuyển danh sách các đoạn `SPEECH` phát hiện được `[(start_s, end_s), ...]` thành chuỗi
    khoảng nửa mở `[start_time_s, end_time_s)` liên tục, không chồng lấn, phủ kín `[0.000, duration_s)`.

    Quy tắc bảo toàn độ liền mạch sau khi ghi CSV (3 chữ số thập phân, 1 ms):
    1. Lượng tử hóa mọi mốc biên sang lưới mili-giây `round(t, 3)` TRƯỚC khi hợp nhất và lấp khoảng trống,
       nhằm đảm bảo giá trị số học trong bộ nhớ trùng khớp tuyệt đối với chuỗi `f"{t:.3f}"` khi ghi ra CSV
       (không bao giờ tạo hở hay chồng lấn do làm tròn lúc format chuỗi).
    2. Các đoạn `SPEECH` chồng lấn hoặc chạm nhau (`next_start <= cur_end`) sau khi lượng tử hóa
       được hợp nhất thành một đoạn `SPEECH` liên thông duy nhất.
    3. Mọi khoảng trống ở đầu (`0.000 < first_start`), ở giữa hai đoạn `SPEECH`, hoặc ở cuối
       (`last_end < duration_s`) đều được lấp bằng nhãn `NON_SPEECH`.
    4. Nếu không phát hiện được đoạn `SPEECH` nào (`speech_segments_s` rỗng), trả về duy nhất
       1 dòng `NON_SPEECH` phủ trọn `[0.000, duration_s)`.
    """
    if not excerpt_id or not excerpt_id.strip():
        raise ValueError("excerpt_id không được rỗng")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"duration_s phải hữu hạn và > 0, nhận {duration_s}")

    dur_q = round(float(duration_s), 3)
    if dur_q <= 0.0:
        raise ValueError(f"duration_s quá ngắn sau khi làm tròn ms: {duration_s}")

    cleaned: List[Tuple[float, float]] = []
    for raw_s, raw_e in speech_segments_s:
        if not (np.isfinite(raw_s) and np.isfinite(raw_e)):
            raise ValueError(f"Mốc thời gian không hữu hạn trong {excerpt_id}: ({raw_s}, {raw_e})")
        if raw_e < raw_s:
            raise ValueError(
                f"Đoạn phát hiện không hợp lệ trong {excerpt_id}: end ({raw_e}) < start ({raw_s})"
            )
        s_q = max(0.0, min(dur_q, round(float(raw_s), 3)))
        e_q = max(0.0, min(dur_q, round(float(raw_e), 3)))
        if e_q > s_q:
            cleaned.append((s_q, e_q))

    cleaned.sort(key=lambda x: (x[0], x[1]))

    # Hợp nhất các đoạn SPEECH chồng lấn hoặc chạm biên (adjacent: next_s <= cur_e)
    merged: List[Tuple[float, float]] = []
    for s_q, e_q in cleaned:
        if not merged:
            merged.append((s_q, e_q))
        else:
            prev_s, prev_e = merged[-1]
            if s_q <= prev_e + 1e-9:
                merged[-1] = (prev_s, max(prev_e, e_q))
            else:
                merged.append((s_q, e_q))

    rows: List[PreannotationRow] = []
    cursor = 0.0
    for s_q, e_q in merged:
        if s_q > cursor + 1e-9:
            rows.append(
                PreannotationRow(
                    excerpt_id=excerpt_id,
                    start_time_s=cursor,
                    end_time_s=s_q,
                    label="NON_SPEECH",
                    annotator_id=annotator_id,
                    notes=notes,
                )
            )
        rows.append(
            PreannotationRow(
                excerpt_id=excerpt_id,
                start_time_s=s_q,
                end_time_s=e_q,
                label="SPEECH",
                annotator_id=annotator_id,
                notes=notes,
            )
        )
        cursor = e_q

    if cursor < dur_q - 1e-9 or not rows:
        rows.append(
            PreannotationRow(
                excerpt_id=excerpt_id,
                start_time_s=cursor,
                end_time_s=dur_q,
                label="NON_SPEECH",
                annotator_id=annotator_id,
                notes=notes,
            )
        )

    return rows


def resolve_target_wav_list(
    wav_dir: str = DEFAULT_WAV_DIR,
    manifest_path: Optional[str] = DEFAULT_MANIFEST,
) -> List[Tuple[str, str, Optional[int], Optional[float]]]:
    """
    Xác định danh sách `(excerpt_id, wav_path, expected_sr, expected_duration_s)` cần xử lý.
    - Nếu `manifest_path` tồn tại: lấy đúng thứ tự và đường dẫn từ `pilot_manifest.csv`.
    - Nếu `manifest_path` là `None`: quét toàn bộ các file `.wav` trong `wav_dir` theo thứ tự tên file.
    """
    if manifest_path is not None and os.path.exists(manifest_path):
        items: List[Tuple[str, str, Optional[int], Optional[float]]] = []
        base_dir = os.path.dirname(os.path.abspath(manifest_path))
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row["excerpt_id"].strip()
                exp_sr = int(row["sample_rate"])
                exp_dur = float(row["duration_s"])
                rel_wav = row["wav_rel_path"].strip().replace("/", os.sep)
                candidate = os.path.join(base_dir, rel_wav)
                if not os.path.exists(candidate):
                    candidate = os.path.join(wav_dir, os.path.basename(rel_wav))
                items.append((eid, candidate, exp_sr, exp_dur))
        if not items:
            raise ValueError(f"Manifest rỗng (không có excerpt nào): '{manifest_path}'")
        return items

    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục WAV: '{wav_dir}'")

    wav_files = sorted(
        f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")
    )
    if not wav_files:
        raise FileNotFoundError(f"Không có file .wav nào trong thư mục: '{wav_dir}'")

    return [
        (os.path.splitext(fn)[0], os.path.join(wav_dir, fn), None, None)
        for fn in wav_files
    ]


def check_safe_output_path(output_csv: str) -> str:
    """
    Đảm bảo không bao giờ ghi đè lên `annotation_template.csv`, `pilot_manifest.csv`
    hoặc các file báo cáo benchmark hiện có.
    """
    abs_out = os.path.abspath(output_csv)
    base_name = os.path.basename(abs_out).lower()
    if base_name in {name.lower() for name in PROTECTED_FILENAMES}:
        raise ValueError(
            f"Từ chối ghi đè lên file được bảo vệ '{os.path.basename(abs_out)}'. "
            "Hãy xuất ra file mới (mặc định: preannotation_silero.csv)."
        )
    return abs_out


def write_preannotation_csv(rows: Sequence[PreannotationRow], output_csv: str) -> str:
    """
    Ghi danh sách `PreannotationRow` ra file CSV với đúng header quy định.
    """
    safe_path = check_safe_output_path(output_csv)
    os.makedirs(os.path.dirname(safe_path) or ".", exist_ok=True)
    with open(safe_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow(r.to_csv_row())
    return safe_path


def run_preannotation(
    wav_dir: str = DEFAULT_WAV_DIR,
    manifest_path: Optional[str] = DEFAULT_MANIFEST,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    quiet: bool = False,
) -> Tuple[List[PreannotationRow], List[FileSummary]]:
    """
    Chạy toàn bộ quy trình tiền gán nhãn bằng Silero VAD trên danh sách file WAV và lưu ra CSV.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold phải nằm trong khoảng (0, 1), nhận {threshold}")
    if min_speech_duration_ms < 0:
        raise ValueError(f"min_speech_duration_ms phải >= 0, nhận {min_speech_duration_ms}")
    if min_silence_duration_ms < 0:
        raise ValueError(f"min_silence_duration_ms phải >= 0, nhận {min_silence_duration_ms}")
    if speech_pad_ms < 0:
        raise ValueError(f"speech_pad_ms phải >= 0, nhận {speech_pad_ms}")

    safe_out = check_safe_output_path(output_csv)
    targets = resolve_target_wav_list(wav_dir=wav_dir, manifest_path=manifest_path)

    # Kiểm tra trước tất cả các file WAV trước khi nạp model
    loaded_wavs: List[Tuple[str, str, np.ndarray, int, float]] = []
    for eid, wpath, exp_sr, exp_dur in targets:
        audio, sr, dur_s = read_pcm_wav_mono(wpath)
        if exp_sr is not None and sr != exp_sr:
            raise ValueError(
                f"{eid}: sample_rate của WAV ({sr} Hz) không khớp với manifest ({exp_sr} Hz)"
            )
        if exp_dur is not None and abs(dur_s - exp_dur) > 1e-3:
            raise ValueError(
                f"{eid}: duration của WAV ({dur_s:.3f}s) không khớp với manifest ({exp_dur:.3f}s)"
            )
        loaded_wavs.append((eid, wpath, audio, sr, dur_s))

    torch_mod, load_silero_vad_fn, get_speech_timestamps_fn = _import_silero_dependencies()
    model = load_silero_vad_fn(onnx=False)

    all_rows: List[PreannotationRow] = []
    summaries: List[FileSummary] = []

    if not quiet:
        print("=" * 92)
        print("SILERO VAD PRE-ANNOTATION UTILITY (DRAFT LABELS ONLY — NOT HUMAN GROUND TRUTH)")
        print(
            f"Config: threshold={threshold:.2f}, min_speech_ms={min_speech_duration_ms}, "
            f"min_silence_ms={min_silence_duration_ms}, speech_pad_ms={speech_pad_ms}"
        )
        print("-" * 92)
        print(
            f"{'Excerpt ID':18s} | {'SR (Hz)':>7s} | {'Dur (s)':>7s} | "
            f"{'Speech Segs':>11s} | {'Speech Dur (s)':>14s} | {'Total Rows':>10s}"
        )
        print("-" * 92)

    for eid, wpath, audio, sr, dur_s in loaded_wavs:
        raw_segs = detect_speech_segments_silero(
            audio=audio,
            sample_rate=sr,
            model=model,
            get_speech_timestamps_fn=get_speech_timestamps_fn,
            torch_module=torch_mod,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        rows = speech_segments_to_contiguous_intervals(
            excerpt_id=eid,
            speech_segments_s=raw_segs,
            duration_s=dur_s,
        )
        all_rows.extend(rows)

        sp_rows = [r for r in rows if r.label == "SPEECH"]
        sp_dur = sum(r.end_time_s - r.start_time_s for r in sp_rows)
        summary = FileSummary(
            excerpt_id=eid,
            wav_path=wpath,
            sample_rate=sr,
            duration_s=round(dur_s, 3),
            speech_segment_count=len(sp_rows),
            speech_duration_s=round(sp_dur, 3),
            total_intervals=len(rows),
        )
        summaries.append(summary)

        if not quiet:
            print(
                f"{summary.excerpt_id:18s} | {summary.sample_rate:7d} | {summary.duration_s:7.3f} | "
                f"{summary.speech_segment_count:11d} | {summary.speech_duration_s:14.3f} | "
                f"{summary.total_intervals:10d}"
            )

    write_preannotation_csv(all_rows, safe_out)

    if not quiet:
        tot_dur = sum(s.duration_s for s in summaries)
        tot_sp_segs = sum(s.speech_segment_count for s in summaries)
        tot_sp_dur = sum(s.speech_duration_s for s in summaries)
        print("-" * 92)
        print(
            f"{'TOTAL (11 files)':18s} | {'8k/16k':>7s} | {tot_dur:7.3f} | "
            f"{tot_sp_segs:11d} | {tot_sp_dur:14.3f} | {len(all_rows):10d}"
        )
        print("=" * 92)
        print(f"Đã ghi bản nháp tiền gán nhãn (pre-annotation) vào: {safe_out}")
        print(
            "LƯU Ý: File này là nhãn nháp tự động từ Silero VAD, KHÔNG PHẢI nhãn chuẩn của con người."
        )

    return all_rows, summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo file nháp tiền gán nhãn (preannotation_silero.csv) từ các file WAV của gói "
            "Human-Annotation Pilot bằng gói silero-vad chính thức. "
            "Đầu ra KHÔNG phải nhãn chuẩn của con người và cần được người nghe rà soát lại."
        )
    )
    parser.add_argument(
        "--wav-dir",
        default=DEFAULT_WAV_DIR,
        help="Thư mục chứa các file WAV đầu vào (mặc định: thư mục wav/ của gói pilot).",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Đường dẫn tới pilot_manifest.csv (truyền '' nếu muốn quét trực tiếp mọi file .wav trong --wav-dir).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help="Đường dẫn file CSV đầu ra (mặc định: preannotation_silero.csv, không ghi đè annotation_template.csv).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng xác suất phát hiện tiếng nói của Silero VAD trong (0, 1) (mặc định: 0.5).",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Thời lượng tiếng nói tối thiểu tính bằng ms (mặc định: 250).",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        default=100,
        help="Thời lượng khoảng lặng tối thiểu để tách 2 đoạn tiếng nói tính bằng ms (mặc định: 100).",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=30,
        help="Đệm thời gian hai đầu mỗi đoạn tiếng nói tính bằng ms (mặc định: 30).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Không in bảng tóm tắt ra màn hình.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest_arg = args.manifest.strip() if args.manifest else None
    run_preannotation(
        wav_dir=args.wav_dir,
        manifest_path=manifest_arg,
        output_csv=args.output,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
