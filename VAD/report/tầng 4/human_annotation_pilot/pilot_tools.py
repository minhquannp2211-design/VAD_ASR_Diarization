"""
VAD/report/tầng 4/human_annotation_pilot/pilot_tools.py

Công cụ hỗ trợ gói gán nhãn thủ công (Human-Annotation Pilot) cho Stage D1:
1. Xuất 11 đoạn âm thanh trộn thực tế (mixed audio mà VAD thực sự nhận) ra file WAV 16-bit PCM
   bằng thư viện chuẩn `wave` của Python (không thêm dependency mới).
2. Kiểm tra tính hợp lệ (validate) của file CSV nhãn do con người gán (`SPEECH`, `NON_SPEECH`, `UNCERTAIN`).
3. Quy đổi các khoảng nửa mở `[start_time_s, end_time_s)` đã gán nhãn sang mặt nạ cấp frame (hop = 5 ms),
   trong đó tách riêng `speech_mask`, `non_speech_mask`, `uncertain_mask` và `valid_eval_mask = ~uncertain_mask`.

Tuyệt đối KHÔNG tự động chép pseudo-label vào file nhãn của con người và KHÔNG tự bịa nhãn.
"""
from __future__ import annotations

import csv
import os
import sys
import wave
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmark.run_vad_benchmark import (  # noqa: E402
    build_benchmark_stream,
    collect_benchmark_materials,
)

VALID_LABELS = {"SPEECH", "NON_SPEECH", "UNCERTAIN"}
DEFAULT_MANIFEST = os.path.join(PILOT_DIR, "pilot_manifest.csv")


@dataclass(frozen=True)
class ExcerptSpec:
    excerpt_id: str
    scenario: str
    sample_rate: int
    seed: int
    split: str
    source_group_id: str
    stream_start_s: float
    stream_end_s: float
    duration_s: float
    expected_frames_5ms: int
    source_utterances: str
    target_phenomenon: str
    wav_rel_path: str


@dataclass(frozen=True)
class AnnotationInterval:
    excerpt_id: str
    start_time_s: float
    end_time_s: float
    label: str
    annotator_id: str = ""
    notes: str = ""


@dataclass(frozen=True)
class FrameAnnotationLabels:
    """
    Nhãn cấp frame (hop = 5 ms) chuyển đổi từ các khoảng gán nhãn thủ công:
    - speech_mask     : (T,) bool — True tại các frame thuộc khoảng SPEECH
    - non_speech_mask : (T,) bool — True tại các frame thuộc khoảng NON_SPEECH
    - uncertain_mask  : (T,) bool — True tại các frame thuộc khoảng UNCERTAIN
    - valid_eval_mask : (T,) bool — True tại các frame được đưa vào tính Miss/FA chính (= ~uncertain_mask)
    """
    excerpt_id: str
    hop_seconds: float
    speech_mask: np.ndarray
    non_speech_mask: np.ndarray
    uncertain_mask: np.ndarray
    valid_eval_mask: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.speech_mask.shape[0])

    @property
    def uncertain_fraction(self) -> float:
        if self.num_frames == 0:
            return 0.0
        return float(self.uncertain_mask.mean())


def validate_manifest_split_integrity(specs: Dict[str, ExcerptSpec]) -> None:
    """
    Kiểm tra tính toàn vẹn của phân chia tập `dev` / `eval` trong `pilot_manifest.csv`:
    1. Mọi `split` phải thuộc `{"dev", "eval"}` và có `source_group_id` khác rỗng.
    2. Mọi excerpt có cùng `source_group_id` phải thuộc cùng một `split`.
    3. Không có bất kỳ file nguồn phát ngôn (`source_utterances`) nào xuất hiện ở cả `dev` và `eval`
       (đảm bảo cả hai chiều: các phiên bản 8 kHz / 16 kHz của cùng nguồn phát ngôn và các lần lặp
       của cùng câu nói qua nhiều kịch bản nhiễu không bao giờ rò rỉ giữa `dev` và `eval`).
    """
    group_to_split: Dict[str, str] = {}
    utterance_to_split: Dict[str, Tuple[str, str]] = {}

    for eid, spec in specs.items():
        if spec.split not in {"dev", "eval"}:
            raise ValueError(f"{eid}: split không hợp lệ '{spec.split}' (phải là 'dev' hoặc 'eval')")
        if not spec.source_group_id:
            raise ValueError(f"{eid}: thiếu source_group_id")

        if spec.source_group_id in group_to_split:
            prev_split = group_to_split[spec.source_group_id]
            if prev_split != spec.split:
                raise ValueError(
                    f"Vi phạm tính toàn vẹn split theo nhóm nguồn '{spec.source_group_id}': "
                    f"xuất hiện ở cả '{prev_split}' và '{spec.split}' ({eid})"
                )
        else:
            group_to_split[spec.source_group_id] = spec.split

        utts = [u.strip() for u in spec.source_utterances.split("|") if u.strip()]
        if not utts:
            raise ValueError(f"{eid}: thiếu danh sách source_utterances")
        for u in utts:
            if u in utterance_to_split:
                prev_split, prev_eid = utterance_to_split[u]
                if prev_split != spec.split:
                    raise ValueError(
                        f"Rò rỉ câu nói nguồn '{u}' giữa hai tập: "
                        f"{prev_eid} ({prev_split}) vs {eid} ({spec.split})"
                    )
            else:
                utterance_to_split[u] = (spec.split, eid)


def load_manifest(manifest_path: str = DEFAULT_MANIFEST) -> Dict[str, ExcerptSpec]:
    specs: Dict[str, ExcerptSpec] = {}
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["excerpt_id"].strip()
            specs[eid] = ExcerptSpec(
                excerpt_id=eid,
                scenario=row["scenario"].strip(),
                sample_rate=int(row["sample_rate"]),
                seed=int(row["seed"]),
                split=row["split"].strip(),
                source_group_id=row.get("source_group_id", "").strip(),
                stream_start_s=float(row["stream_start_s"]),
                stream_end_s=float(row["stream_end_s"]),
                duration_s=float(row["duration_s"]),
                expected_frames_5ms=int(row["expected_frames_5ms"]),
                source_utterances=row["source_utterances"].strip(),
                target_phenomenon=row["target_phenomenon"].strip(),
                wav_rel_path=row["wav_rel_path"].strip(),
            )
    validate_manifest_split_integrity(specs)
    return specs


def export_pilot_wavs(
    manifest_path: str = DEFAULT_MANIFEST, out_base_dir: str = PILOT_DIR
) -> List[str]:
    """
    Tái tạo chính xác các luồng âm thanh trộn (mix_wave) từ benchmark với seed cố định
    và cắt xuất 11 đoạn excerpt ra file WAV 16-bit PCM để người gán nhãn nghe trực tiếp.
    """
    specs = load_manifest(manifest_path)
    materials_cache: Dict[int, tuple] = {}
    stream_cache: Dict[Tuple[int, str, int], np.ndarray] = {}
    written_paths: List[str] = []

    for spec in specs.values():
        sr = spec.sample_rate
        if sr not in materials_cache:
            materials_cache[sr] = collect_benchmark_materials(sr=sr)
        clean_cores, noise_pieces = materials_cache[sr]

        stream_key = (sr, spec.scenario, spec.seed)
        if stream_key not in stream_cache:
            _, _, mix_wave, _ = build_benchmark_stream(
                clean_cores, noise_pieces, sr, spec.scenario, seed=spec.seed
            )
            stream_cache[stream_key] = mix_wave
        mix_wave = stream_cache[stream_key]

        s_idx = int(round(spec.stream_start_s * sr))
        e_idx = int(round(spec.stream_end_s * sr))
        excerpt_wave = mix_wave[s_idx:e_idx]

        # Chuẩn hóa biên độ đỉnh về <= 0.95 nếu vượt ngưỡng để tránh clipping khi ghi PCM 16-bit
        peak = float(np.max(np.abs(excerpt_wave))) if excerpt_wave.size else 0.0
        scale = 0.95 / peak if peak > 0.95 else 1.0
        pcm16 = np.clip(np.round(excerpt_wave * scale * 32767.0), -32768, 32767).astype(np.int16)

        wav_path = os.path.join(out_base_dir, spec.wav_rel_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        written_paths.append(wav_path)

    return written_paths


def validate_annotations(
    intervals: Sequence[AnnotationInterval],
    manifest: Dict[str, ExcerptSpec],
    tol_s: float = 1e-3,
) -> None:
    """
    Kiểm tra tính hợp lệ của danh sách khoảng gán nhãn:
    1. `excerpt_id` phải tồn tại trong `pilot_manifest.csv`.
    2. `label` phải thuộc đúng tập `{"SPEECH", "NON_SPEECH", "UNCERTAIN"}` (báo lỗi nếu còn nhãn giữ chỗ).
    3. Mỗi khoảng phải thỏa `0.0 <= start_time_s < end_time_s <= duration_s`.
    4. Trong cùng một `excerpt_id`, các khoảng sau khi sắp xếp phải liên tiếp nhau, không chồng lấn
       và phủ kín từ `0.0` đến `duration_s` (trong dung sai `tol_s = 1 ms`).
    """
    if not intervals:
        raise ValueError("Danh sách khoảng gán nhãn rỗng")

    by_excerpt: Dict[str, List[AnnotationInterval]] = {}
    for iv in intervals:
        if iv.excerpt_id not in manifest:
            raise ValueError(f"excerpt_id không tồn tại trong manifest: {iv.excerpt_id}")
        if iv.label not in VALID_LABELS:
            raise ValueError(
                f"Nhãn không hợp lệ '{iv.label}' tại {iv.excerpt_id} "
                f"[{iv.start_time_s}, {iv.end_time_s}). Phải thuộc {sorted(VALID_LABELS)}"
            )
        if not (np.isfinite(iv.start_time_s) and np.isfinite(iv.end_time_s)):
            raise ValueError(f"Thời gian không hữu hạn tại {iv.excerpt_id}")
        if iv.end_time_s <= iv.start_time_s:
            raise ValueError(
                f"Khoảng không hợp lệ tại {iv.excerpt_id}: "
                f"end_time_s ({iv.end_time_s}) <= start_time_s ({iv.start_time_s})"
            )
        by_excerpt.setdefault(iv.excerpt_id, []).append(iv)

    for eid, iv_list in by_excerpt.items():
        spec = manifest[eid]
        iv_sorted = sorted(iv_list, key=lambda x: (x.start_time_s, x.end_time_s))
        if abs(iv_sorted[0].start_time_s - 0.0) > tol_s:
            raise ValueError(
                f"{eid}: Khoảng đầu tiên phải bắt đầu tại 0.000s (nhận {iv_sorted[0].start_time_s})"
            )
        for k in range(len(iv_sorted) - 1):
            cur_end = iv_sorted[k].end_time_s
            nxt_start = iv_sorted[k + 1].start_time_s
            if abs(nxt_start - cur_end) > tol_s:
                if nxt_start < cur_end:
                    raise ValueError(
                        f"{eid}: Phát hiện hai khoảng chồng lấn tại {cur_end:.3f}s và {nxt_start:.3f}s"
                    )
                raise ValueError(
                    f"{eid}: Phát hiện khoảng trống chưa gán nhãn giữa {cur_end:.3f}s và {nxt_start:.3f}s"
                )
        if abs(iv_sorted[-1].end_time_s - spec.duration_s) > tol_s:
            raise ValueError(
                f"{eid}: Khoảng cuối cùng phải kết thúc tại {spec.duration_s:.3f}s "
                f"(nhận {iv_sorted[-1].end_time_s:.3f}s)"
            )


def intervals_to_frame_labels(
    excerpt_id: str,
    intervals: Sequence[AnnotationInterval],
    num_frames: int,
    hop_seconds: float = 0.005,
) -> FrameAnnotationLabels:
    """
    Chuyển danh sách các khoảng nửa mở `[start_time_s, end_time_s)` của một excerpt sang nhãn cấp frame.
    Quy ước: Frame `t` (có thời điểm bắt đầu `t_frame = t * hop_seconds`) nhận nhãn của khoảng
    chứa `t_frame`, tức `start_time_s <= t * hop_seconds < end_time_s` (với dung sai 1e-9).
    Nếu một frame trùng biên hoặc bị đánh dấu bởi nhiều khoảng do làm tròn dưới 1 frame,
    thứ tự ưu tiên an toàn là: `UNCERTAIN` > `SPEECH` > `NON_SPEECH`.
    """
    if num_frames < 0:
        raise ValueError("num_frames phải >= 0")
    if hop_seconds <= 0.0:
        raise ValueError("hop_seconds phải > 0")

    sp = np.zeros(num_frames, dtype=bool)
    ns = np.zeros(num_frames, dtype=bool)
    unc = np.zeros(num_frames, dtype=bool)

    for iv in intervals:
        if iv.excerpt_id != excerpt_id:
            continue
        if iv.label not in VALID_LABELS:
            raise ValueError(f"Nhãn không hợp lệ: {iv.label}")
        s_f = max(0, int(np.ceil(iv.start_time_s / hop_seconds - 1e-9)))
        e_f = min(num_frames, int(np.ceil(iv.end_time_s / hop_seconds - 1e-9)))
        if e_f <= s_f:
            continue
        if iv.label == "UNCERTAIN":
            unc[s_f:e_f] = True
            sp[s_f:e_f] = False
            ns[s_f:e_f] = False
        elif iv.label == "SPEECH":
            target = slice(s_f, e_f)
            sp[target] = (~unc[target])
            ns[target] = False
        elif iv.label == "NON_SPEECH":
            target = slice(s_f, e_f)
            ns[target] = (~unc[target]) & (~sp[target])

    valid_eval = ~unc
    return FrameAnnotationLabels(
        excerpt_id=excerpt_id,
        hop_seconds=hop_seconds,
        speech_mask=sp,
        non_speech_mask=ns,
        uncertain_mask=unc,
        valid_eval_mask=valid_eval,
    )


def evaluate_predictions_against_human_labels(
    pred_mask: np.ndarray,
    human_labels: FrameAnnotationLabels,
) -> Dict[str, float]:
    """
    Đánh giá mặt nạ dự đoán `pred_mask` (T,) so với nhãn con người `human_labels`:
    - Các frame có `uncertain_mask == True` được LOẠI TRỪ khỏi tử số và mẫu số của Miss Rate,
      False Alarm Rate, Precision, Recall, F1 chính (`valid_eval_mask = ~uncertain_mask`).
    - Tỷ lệ frame `UNCERTAIN` và tỷ lệ dự đoán dương tính trong vùng `UNCERTAIN` được báo cáo riêng.
    """
    pred = np.asarray(pred_mask, dtype=bool)
    if pred.shape != (human_labels.num_frames,):
        raise ValueError("pred_mask không khớp số frame với human_labels")

    valid = human_labels.valid_eval_mask
    gt_sp = human_labels.speech_mask[valid]
    gt_ns = human_labels.non_speech_mask[valid]
    p_val = pred[valid]

    tp = int((p_val & gt_sp).sum())
    fn = int(((~p_val) & gt_sp).sum())
    fp = int((p_val & gt_ns).sum())
    tn = int(((~p_val) & gt_ns).sum())

    n_pos = tp + fn
    n_neg = fp + tn
    miss_pct = (fn / n_pos * 100.0) if n_pos > 0 else 0.0
    fa_pct = (fp / n_neg * 100.0) if n_neg > 0 else 0.0
    rec_pct = (tp / n_pos * 100.0) if n_pos > 0 else 0.0
    prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1_pct = (2.0 * tp / (2 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 0.0

    unc_mask = human_labels.uncertain_mask
    unc_cnt = int(unc_mask.sum())
    unc_pct = (unc_cnt / human_labels.num_frames * 100.0) if human_labels.num_frames > 0 else 0.0
    unc_pred_pos_pct = (float(pred[unc_mask].mean()) * 100.0) if unc_cnt > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "miss_pct": miss_pct,
        "fa_pct": fa_pct,
        "precision_pct": prec_pct,
        "recall_pct": rec_pct,
        "f1_pct": f1_pct,
        "uncertain_frames": unc_cnt,
        "uncertain_pct": unc_pct,
        "pred_pos_in_uncertain_pct": unc_pred_pos_pct,
    }


def load_annotations_csv(csv_path: str) -> List[AnnotationInterval]:
    out: List[AnnotationInterval] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                AnnotationInterval(
                    excerpt_id=row["excerpt_id"].strip(),
                    start_time_s=float(row["start_time_s"]),
                    end_time_s=float(row["end_time_s"]),
                    label=row["label"].strip(),
                    annotator_id=row.get("annotator_id", "").strip(),
                    notes=row.get("notes", "").strip(),
                )
            )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human-Annotation Pilot Tools (Stage D1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export-wavs", help="Xuất 11 file WAV âm thanh trộn cho pilot")
    p_exp.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p_exp.add_argument("--out-dir", default=PILOT_DIR)

    p_val = sub.add_parser("validate", help="Kiểm tra file CSV nhãn con người đã điền")
    p_val.add_argument("csv_path", help="Đường dẫn tới file CSV nhãn đã điền")
    p_val.add_argument("--manifest", default=DEFAULT_MANIFEST)

    args = parser.parse_args()
    if args.cmd == "export-wavs":
        paths = export_pilot_wavs(args.manifest, args.out_dir)
        print(f"Đã xuất {len(paths)} file WAV vào {args.out_dir}:")
        for p in paths:
            print("  -", p)
    elif args.cmd == "validate":
        manifest = load_manifest(args.manifest)
        ivs = load_annotations_csv(args.csv_path)
        validate_annotations(ivs, manifest)
        print(f"HỢP LỆ: Đã kiểm tra {len(ivs)} khoảng gán nhãn trong {args.csv_path}.")
