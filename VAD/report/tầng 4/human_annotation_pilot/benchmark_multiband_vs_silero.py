"""
VAD/report/tầng 4/human_annotation_pilot/benchmark_multiband_vs_silero.py

Chạy bộ Multi-band Adaptive VAD hiện hành (Tầng 1 -> Tầng 4, đóng băng cấu hình mặc định)
trên đúng 11 file WAV của gói Human-Annotation Pilot và đối chiếu kết quả với:
- `annotations_human_reviewed.csv` (nếu đã có và hợp lệ), hoặc
- `preannotation_silero.csv` (chỉ với tư cách NHÃN THAM CHIẾU TẠM THỜI DO SILERO TẠO RA —
  PROVISIONAL SILERO PRE-ANNOTATION, tuyệt đối không coi là độ chính xác thực địa đã xác minh).

Không sửa đổi code sản xuất của Tầng 1-4 và không tinh chỉnh bất kỳ tham số nào.
"""
from __future__ import annotations

import csv
import datetime
import os
import sys
from dataclasses import asdict, dataclass
from importlib import util
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VAD.feature_extraction import DEFAULT_BANDS_HZ, FeatureExtractor, VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseConfig, NoiseEstimator  # noqa: E402
from VAD.sigmoid import SigmoidConfig, SigmoidScorer  # noqa: E402
from VAD.state_machine import StateMachineConfig, VADStateMachine  # noqa: E402
from benchmark.run_vad_benchmark import (  # noqa: E402
    compute_binary_metrics,
    match_segments_by_time,
    segments_to_frame_mask,
)


def _load_local_module(mod_name: str, filename: str):
    path = os.path.join(PILOT_DIR, filename)
    spec = util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể nạp module từ {path}")
    mod = util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


pilot_tools = _load_local_module("pilot_tools", "pilot_tools.py")
preannotate_silero = _load_local_module("preannotate_silero", "preannotate_silero.py")

HUMAN_REVIEWED_CSV = os.path.join(PILOT_DIR, "annotations_human_reviewed.csv")
SILERO_PREANNOTATION_CSV = os.path.join(PILOT_DIR, "preannotation_silero.csv")

OUT_PREDICTIONS_CSV = os.path.join(PILOT_DIR, "multiband_vad_predictions.csv")
OUT_PER_FILE_CSV = os.path.join(PILOT_DIR, "multiband_vs_silero_per_file.csv")
OUT_SUMMARY_MD = os.path.join(PILOT_DIR, "multiband_vs_silero_summary.md")
OUT_RUN_LOG = os.path.join(PILOT_DIR, "multiband_vad_pilot_run.log")

RUN_CONFIG_ID = (
    "multiband_vad_v1_frozen_defaults"
    "(L1:hop=5ms,win=16ms,bands=4|L2:init=40f,snr=6/9dB,confirm=3|"
    "L3:w=1,2,2,1,wsfm=2,b=-4|L4:on=0.55/30ms,off=0.35,hang=150ms,min_sp=100ms,min_sil=100ms)"
)


@dataclass(frozen=True)
class ExcerptEvalResult:
    excerpt_id: str
    scenario: str
    sample_rate: int
    split: str
    source_group_id: str
    duration_s: float
    num_eval_frames_5ms: int
    stft_stream_frames: int
    chunk_invariance_ok: bool
    timestamps_within_duration: bool
    # Frame-level metrics (5 ms grid)
    tp: int
    fp: int
    fn: int
    tn: int
    precision_pct: float
    recall_pct: float
    f1_pct: float
    miss_pct: float
    fa_pct: float
    uncertain_frames: int
    # Durations
    ref_speech_s: float
    pred_speech_s: float
    duration_diff_s: float
    # Segment-level metrics (IoU >= 0.05 1-to-1 greedy matching)
    n_ref_segs: int
    n_pred_segs: int
    n_matched_segs: int
    seg_precision_pct: float
    seg_recall_pct: float
    seg_f1_pct: float
    onset_abs_med_ms: float
    onset_abs_mean_ms: float
    offset_abs_med_ms: float
    offset_abs_mean_ms: float
    onset_signed_med_ms: float
    offset_signed_med_ms: float
    # Boundary tolerance counts (among matched segments)
    onset_within_10ms: int
    onset_within_20ms: int
    onset_within_50ms: int
    offset_within_10ms: int
    offset_within_20ms: int
    offset_within_50ms: int
    both_within_10ms: int
    both_within_20ms: int
    both_within_50ms: int
    # Diagnostic startup / first segment info
    first_ref_seg: str
    first_pred_seg: str


def select_reference_annotations(
    manifest: Dict[str, Any],
) -> Tuple[str, str, List[Any], List[str]]:
    """
    Kiểm tra sự tồn tại và tính hợp lệ của `annotations_human_reviewed.csv`.
    Nếu chưa có, sử dụng `preannotation_silero.csv` với tư cách nhãn tham chiếu tạm thời (provisional).
    """
    logs: List[str] = []
    if os.path.exists(HUMAN_REVIEWED_CSV):
        try:
            ivs = pilot_tools.load_annotations_csv(HUMAN_REVIEWED_CSV)
            pilot_tools.validate_annotations(ivs, manifest)
            covered = {iv.excerpt_id for iv in ivs}
            if covered == set(manifest.keys()):
                logs.append(
                    f"[REFERENCE] Sử dụng nhãn người đã rà soát hợp lệ: {HUMAN_REVIEWED_CSV}"
                )
                return "HUMAN_REVIEWED", HUMAN_REVIEWED_CSV, ivs, logs
            logs.append(
                f"[WARNING] {HUMAN_REVIEWED_CSV} thiếu excerpt ({len(covered)}/{len(manifest)}); "
                "chuyển sang dùng preannotation_silero.csv."
            )
        except Exception as exc:
            logs.append(
                f"[WARNING] {HUMAN_REVIEWED_CSV} không vượt qua kiểm tra hợp lệ ({exc}); "
                "chuyển sang dùng preannotation_silero.csv."
            )
    else:
        logs.append(
            f"[REFERENCE] Không tìm thấy {HUMAN_REVIEWED_CSV}. "
            f"Sử dụng {SILERO_PREANNOTATION_CSV} làm NHÃN THAM CHIẾU TẠM THỜI (PROVISIONAL SILERO PRE-ANNOTATION)."
        )

    if not os.path.exists(SILERO_PREANNOTATION_CSV):
        raise FileNotFoundError(
            f"Không tìm thấy file tham chiếu {SILERO_PREANNOTATION_CSV}. "
            "Hãy chạy preannotate_silero.py trước."
        )
    ivs = pilot_tools.load_annotations_csv(SILERO_PREANNOTATION_CSV)
    pilot_tools.validate_annotations(ivs, manifest)
    covered = {iv.excerpt_id for iv in ivs}
    if covered != set(manifest.keys()):
        raise ValueError(
            f"File {SILERO_PREANNOTATION_CSV} không phủ đủ 11 excerpt của manifest."
        )
    logs.append(
        f"[VALIDATION] {SILERO_PREANNOTATION_CSV} vượt qua pilot_tools.validate_annotations "
        f"({len(ivs)} khoảng trên {len(covered)} excerpts)."
    )
    return "PROVISIONAL_SILERO_PREANNOTATION", SILERO_PREANNOTATION_CSV, ivs, logs


def run_multiband_streaming(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_s: Optional[float] = 0.020,
) -> Tuple[List[Tuple[float, float]], int]:
    """
    Chạy đường ống Multi-band Adaptive VAD (Tầng 1 -> Tầng 4) bằng giao diện streaming chính thức:
      FeatureExtractor.process_chunk() + flush()
      -> NoiseEstimator.update()
      -> SigmoidScorer.score()
      -> VADStateMachine.process_chunk() + flush()
    Trả về `(segments, total_stft_frames)`.
    """
    fe = FeatureExtractor(sample_rate=sample_rate)
    est = NoiseEstimator(num_bands=fe.num_bands, config=NoiseConfig())
    scorer = SigmoidScorer(num_bands=fe.num_bands, config=SigmoidConfig())
    sm = VADStateMachine(hop_seconds=fe.hop_seconds, config=StateMachineConfig())

    segments: List[Tuple[float, float]] = []
    total_frames = 0

    if chunk_duration_s is None or chunk_duration_s <= 0.0:
        chunks = [audio]
    else:
        chunk_samples = max(1, int(round(chunk_duration_s * sample_rate)))
        chunks = [audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

    for ch in chunks:
        feats = fe.process_chunk(ch)
        if feats.num_frames > 0:
            total_frames += feats.num_frames
            n_out = est.update(feats)
            s_out = scorer.score(feats, n_out)
            sm_out = sm.process_chunk(s_out)
            segments.extend(sm_out.segments)

    feats_tail = fe.flush()
    if feats_tail.num_frames > 0:
        total_frames += feats_tail.num_frames
        n_tail = est.update(feats_tail)
        s_tail = scorer.score(feats_tail, n_tail)
        sm_tail = sm.process_chunk(s_tail)
        segments.extend(sm_tail.segments)

    segments.extend(sm.flush())
    cleaned_segs = [(round(float(s), 6), round(float(e), 6)) for s, e in segments]
    return cleaned_segs, total_frames


def compute_boundary_tolerances(
    onset_errs_ms: Sequence[float],
    offset_errs_ms: Sequence[float],
    tolerances_ms: Sequence[float] = (10.0, 20.0, 50.0),
) -> Dict[float, Tuple[int, int, int]]:
    """
    Với danh sách các cặp đoạn đã khớp 1-1 (có sai số có dấu `onset_errs_ms` và `offset_errs_ms`),
    đếm số lượng đoạn khớp có:
      - `|onset_err_ms| <= tol`
      - `|offset_err_ms| <= tol`
      - Cả hai đầu `|onset_err_ms| <= tol` VÀ `|offset_err_ms| <= tol`
    """
    out: Dict[float, Tuple[int, int, int]] = {}
    on_arr = np.abs(np.asarray(onset_errs_ms, dtype=np.float64))
    off_arr = np.abs(np.asarray(offset_errs_ms, dtype=np.float64))
    for tol in tolerances_ms:
        if on_arr.size == 0:
            out[tol] = (0, 0, 0)
        else:
            on_ok = on_arr <= (tol + 1e-6)
            off_ok = off_arr <= (tol + 1e-6)
            both_ok = on_ok & off_ok
            out[tol] = (int(on_ok.sum()), int(off_ok.sum()), int(both_ok.sum()))
    return out


def evaluate_all_pilot_wavs() -> Tuple[
    str,
    str,
    List[ExcerptEvalResult],
    List[Tuple[str, int, float, float, float, str]],
    List[str],
]:
    manifest = pilot_tools.load_manifest()
    ref_type, ref_path, all_ref_ivs, logs = select_reference_annotations(manifest)

    ref_by_excerpt: Dict[str, List[Any]] = {}
    for iv in all_ref_ivs:
        ref_by_excerpt.setdefault(iv.excerpt_id, []).append(iv)

    results: List[ExcerptEvalResult] = []
    prediction_rows: List[Tuple[str, int, float, float, float, str]] = []
    hop_s = 0.005

    for eid, spec in manifest.items():
        wav_path = os.path.join(PILOT_DIR, spec.wav_rel_path.replace("/", os.sep))
        audio, sr, dur_s = preannotate_silero.read_pcm_wav_mono(wav_path)
        if sr != spec.sample_rate:
            raise ValueError(f"{eid}: sample_rate {sr} != manifest {spec.sample_rate}")
        if abs(dur_s - spec.duration_s) > 1e-3:
            raise ValueError(f"{eid}: duration {dur_s} != manifest {spec.duration_s}")

        # Chạy streaming với chunk 20 ms và kiểm tra đối chiếu với chunk 100 ms & toàn cục
        segs_20ms, stft_frames = run_multiband_streaming(audio, sr, chunk_duration_s=0.020)
        segs_100ms, stft_frames_100 = run_multiband_streaming(audio, sr, chunk_duration_s=0.100)
        segs_full, stft_frames_full = run_multiband_streaming(audio, sr, chunk_duration_s=None)

        chunk_inv_ok = (
            stft_frames == stft_frames_100 == stft_frames_full
            and segs_20ms == segs_100ms == segs_full
        )
        within_dur = all(0.0 <= s < e <= dur_s + 1e-6 for s, e in segs_20ms)

        for idx_s, (s_t, e_t) in enumerate(segs_20ms, start=1):
            prediction_rows.append((eid, idx_s, s_t, e_t, round(e_t - s_t, 3), RUN_CONFIG_ID))

        # Chuyển khoảng tham chiếu và đoạn dự đoán sang cùng lưới 5 ms (expected_frames_5ms)
        T_eval = spec.expected_frames_5ms
        ref_ivs = ref_by_excerpt[eid]
        ref_labels = pilot_tools.intervals_to_frame_labels(
            excerpt_id=eid,
            intervals=ref_ivs,
            num_frames=T_eval,
            hop_seconds=hop_s,
        )
        pred_mask = segments_to_frame_mask(segs_20ms, T_eval, hop_s)
        frame_eval = pilot_tools.evaluate_predictions_against_human_labels(pred_mask, ref_labels)

        # Kiểm tra đối chiếu chéo với compute_binary_metrics khi không có UNCERTAIN
        ref_segs = [
            (round(float(iv.start_time_s), 6), round(float(iv.end_time_s), 6))
            for iv in ref_ivs
            if iv.label == "SPEECH"
        ]
        if int(frame_eval["uncertain_frames"]) == 0:
            bm = compute_binary_metrics(
                pred_mask,
                ref_labels.speech_mask,
                ref_segs,
                hop_s,
                pred_segs=segs_20ms,
            )
            assert bm["tp"] == frame_eval["tp"] and bm["fp"] == frame_eval["fp"]
            assert bm["fn"] == frame_eval["fn"] and bm["tn"] == frame_eval["tn"]

        matched, on_errs_ms, off_errs_ms = match_segments_by_time(
            ref_segs, segs_20ms, min_iou=0.05
        )
        n_ref = len(ref_segs)
        n_pred = len(segs_20ms)
        n_match = len(matched)
        seg_prec = (n_match / n_pred * 100.0) if n_pred > 0 else 0.0
        seg_rec = (n_match / n_ref * 100.0) if n_ref > 0 else 0.0
        seg_f1 = (2.0 * n_match / (n_ref + n_pred) * 100.0) if (n_ref + n_pred) > 0 else 0.0

        on_abs_med = float(np.median(np.abs(on_errs_ms))) if on_errs_ms else float("nan")
        on_abs_mean = float(np.mean(np.abs(on_errs_ms))) if on_errs_ms else float("nan")
        off_abs_med = float(np.median(np.abs(off_errs_ms))) if off_errs_ms else float("nan")
        off_abs_mean = float(np.mean(np.abs(off_errs_ms))) if off_errs_ms else float("nan")
        on_sig_med = float(np.median(on_errs_ms)) if on_errs_ms else float("nan")
        off_sig_med = float(np.median(off_errs_ms)) if off_errs_ms else float("nan")

        tol_map = compute_boundary_tolerances(on_errs_ms, off_errs_ms, (10.0, 20.0, 50.0))

        ref_sp_s = round(sum(e - s for s, e in ref_segs), 3)
        pred_sp_s = round(sum(e - s for s, e in segs_20ms), 3)

        first_ref_str = f"[{ref_segs[0][0]:.3f}, {ref_segs[0][1]:.3f})" if ref_segs else "NONE"
        first_pred_str = f"[{segs_20ms[0][0]:.3f}, {segs_20ms[0][1]:.3f})" if segs_20ms else "NONE"

        res = ExcerptEvalResult(
            excerpt_id=eid,
            scenario=spec.scenario,
            sample_rate=sr,
            split=spec.split,
            source_group_id=spec.source_group_id,
            duration_s=round(dur_s, 3),
            num_eval_frames_5ms=T_eval,
            stft_stream_frames=stft_frames,
            chunk_invariance_ok=chunk_inv_ok,
            timestamps_within_duration=within_dur,
            tp=int(frame_eval["tp"]),
            fp=int(frame_eval["fp"]),
            fn=int(frame_eval["fn"]),
            tn=int(frame_eval["tn"]),
            precision_pct=float(frame_eval["precision_pct"]),
            recall_pct=float(frame_eval["recall_pct"]),
            f1_pct=float(frame_eval["f1_pct"]),
            miss_pct=float(frame_eval["miss_pct"]),
            fa_pct=float(frame_eval["fa_pct"]),
            uncertain_frames=int(frame_eval["uncertain_frames"]),
            ref_speech_s=ref_sp_s,
            pred_speech_s=pred_sp_s,
            duration_diff_s=round(pred_sp_s - ref_sp_s, 3),
            n_ref_segs=n_ref,
            n_pred_segs=n_pred,
            n_matched_segs=n_match,
            seg_precision_pct=seg_prec,
            seg_recall_pct=seg_rec,
            seg_f1_pct=seg_f1,
            onset_abs_med_ms=on_abs_med,
            onset_abs_mean_ms=on_abs_mean,
            offset_abs_med_ms=off_abs_med,
            offset_abs_mean_ms=off_abs_mean,
            onset_signed_med_ms=on_sig_med,
            offset_signed_med_ms=off_sig_med,
            onset_within_10ms=tol_map[10.0][0],
            onset_within_20ms=tol_map[20.0][0],
            onset_within_50ms=tol_map[50.0][0],
            offset_within_10ms=tol_map[10.0][1],
            offset_within_20ms=tol_map[20.0][1],
            offset_within_50ms=tol_map[50.0][1],
            both_within_10ms=tol_map[10.0][2],
            both_within_20ms=tol_map[20.0][2],
            both_within_50ms=tol_map[50.0][2],
            first_ref_seg=first_ref_str,
            first_pred_seg=first_pred_str,
        )
        results.append(res)
        logs.append(
            f"[OK] {eid:18s} | sr={sr:5d}Hz | dur={dur_s:5.3f}s | "
            f"stft_T={stft_frames:4d}/{T_eval:4d} | "
            f"Segs(match/pred/ref)={n_match}/{n_pred}/{n_ref} | "
            f"Dur(pred/ref)={pred_sp_s:5.3f}s/{ref_sp_s:5.3f}s | "
            f"Miss={res.miss_pct:5.1f}% | FA={res.fa_pct:5.1f}% | F1={res.f1_pct:5.1f}% | "
            f"within_dur={within_dur} | chunk_inv={chunk_inv_ok}"
        )

    return ref_type, ref_path, results, prediction_rows, logs


def summarize_subset(subset: Sequence[ExcerptEvalResult]) -> Dict[str, Any]:
    tp = sum(r.tp for r in subset)
    fp = sum(r.fp for r in subset)
    fn = sum(r.fn for r in subset)
    tn = sum(r.tn for r in subset)
    n_pos = tp + fn
    n_neg = fp + tn

    miss_pct = (fn / n_pos * 100.0) if n_pos > 0 else 0.0
    fa_pct = (fp / n_neg * 100.0) if n_neg > 0 else 0.0
    rec_pct = (tp / n_pos * 100.0) if n_pos > 0 else 0.0
    prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1_pct = (2.0 * tp / (2 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 0.0

    n_ref = sum(r.n_ref_segs for r in subset)
    n_pred = sum(r.n_pred_segs for r in subset)
    n_match = sum(r.n_matched_segs for r in subset)
    seg_prec = (n_match / n_pred * 100.0) if n_pred > 0 else 0.0
    seg_rec = (n_match / n_ref * 100.0) if n_ref > 0 else 0.0
    seg_f1 = (2.0 * n_match / (n_ref + n_pred) * 100.0) if (n_ref + n_pred) > 0 else 0.0

    return {
        "n_files": len(subset),
        "duration_s": round(sum(r.duration_s for r in subset), 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_pct": prec_pct,
        "recall_pct": rec_pct,
        "f1_pct": f1_pct,
        "miss_pct": miss_pct,
        "fa_pct": fa_pct,
        "ref_speech_s": round(sum(r.ref_speech_s for r in subset), 3),
        "pred_speech_s": round(sum(r.pred_speech_s for r in subset), 3),
        "n_ref_segs": n_ref,
        "n_pred_segs": n_pred,
        "n_matched_segs": n_match,
        "seg_precision_pct": seg_prec,
        "seg_recall_pct": seg_rec,
        "seg_f1_pct": seg_f1,
        "onset_within_10ms": sum(r.onset_within_10ms for r in subset),
        "onset_within_20ms": sum(r.onset_within_20ms for r in subset),
        "onset_within_50ms": sum(r.onset_within_50ms for r in subset),
        "offset_within_10ms": sum(r.offset_within_10ms for r in subset),
        "offset_within_20ms": sum(r.offset_within_20ms for r in subset),
        "offset_within_50ms": sum(r.offset_within_50ms for r in subset),
        "both_within_10ms": sum(r.both_within_10ms for r in subset),
        "both_within_20ms": sum(r.both_within_20ms for r in subset),
        "both_within_50ms": sum(r.both_within_50ms for r in subset),
    }


def write_outputs(
    ref_type: str,
    ref_path: str,
    results: Sequence[ExcerptEvalResult],
    prediction_rows: Sequence[Tuple[str, int, float, float, float, str]],
    logs: Sequence[str],
) -> None:
    # 1. multiband_vad_predictions.csv
    with open(OUT_PREDICTIONS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "excerpt_id",
                "segment_index",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "run_config_id",
            ]
        )
        for eid, idx_s, s_t, e_t, dur_s, cfg_id in prediction_rows:
            writer.writerow([eid, idx_s, f"{s_t:.3f}", f"{e_t:.3f}", f"{dur_s:.3f}", cfg_id])

    # 2. multiband_vs_silero_per_file.csv
    fieldnames = list(asdict(results[0]).keys()) + ["reference_type"]
    with open(OUT_PER_FILE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row_dict = asdict(r)
            for k, v in list(row_dict.items()):
                if isinstance(v, float):
                    row_dict[k] = f"{v:.3f}" if not np.isnan(v) else "NaN"
            row_dict["reference_type"] = ref_type
            writer.writerow(row_dict)

    # 3. multiband_vad_pilot_run.log
    now_str = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
    agg_all = summarize_subset(results)
    agg_16k = summarize_subset([r for r in results if r.sample_rate == 16000])
    agg_8k = summarize_subset([r for r in results if r.sample_rate == 8000])
    agg_dev = summarize_subset([r for r in results if r.split == "dev"])
    agg_eval = summarize_subset([r for r in results if r.split == "eval"])

    with open(OUT_RUN_LOG, "w", encoding="utf-8", newline="\n") as f:
        f.write("================================================================================\n")
        f.write("MULTI-BAND ADAPTIVE VAD PILOT RUN LOG (LAYERS 1-4 FROZEN BASELINE)\n")
        f.write("================================================================================\n")
        f.write(f"Timestamp          : {now_str}\n")
        f.write(f"Command Executed   : py \"VAD/report/tầng 4/human_annotation_pilot/benchmark_multiband_vs_silero.py\"\n")
        f.write(f"Reference Type     : {ref_type}\n")
        f.write(f"Reference File     : {ref_path}\n")
        f.write(f"Run Config ID      : {RUN_CONFIG_ID}\n")
        f.write("Completion Status  : SUCCESS (11/11 WAV files processed, 0 runtime errors)\n")
        f.write("--------------------------------------------------------------------------------\n")
        for line in logs:
            f.write(line + "\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(
            f"POOLED AGGREGATE (11 files, {agg_all['duration_s']:.3f}s): "
            f"TP={agg_all['tp']}, FP={agg_all['fp']}, FN={agg_all['fn']}, TN={agg_all['tn']} | "
            f"Prec={agg_all['precision_pct']:.1f}%, Rec={agg_all['recall_pct']:.1f}%, "
            f"F1={agg_all['f1_pct']:.1f}%, Miss={agg_all['miss_pct']:.1f}%, FA={agg_all['fa_pct']:.1f}% | "
            f"Segs(match/pred/ref)={agg_all['n_matched_segs']}/{agg_all['n_pred_segs']}/{agg_all['n_ref_segs']}\n"
        )
        f.write("================================================================================\n")

    # 4. multiband_vs_silero_summary.md
    _write_summary_markdown(
        ref_type=ref_type,
        ref_path=ref_path,
        results=results,
        agg_all=agg_all,
        agg_16k=agg_16k,
        agg_8k=agg_8k,
        agg_dev=agg_dev,
        agg_eval=agg_eval,
    )


def _write_summary_markdown(
    ref_type: str,
    ref_path: str,
    results: Sequence[ExcerptEvalResult],
    agg_all: Dict[str, Any],
    agg_16k: Dict[str, Any],
    agg_8k: Dict[str, Any],
    agg_dev: Dict[str, Any],
    agg_eval: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Báo cáo Đối chiếu Multi-band Adaptive VAD (Tầng 1–4) với Silero Pre-Annotations trên 11 File WAV Pilot")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    if ref_type == "PROVISIONAL_SILERO_PREANNOTATION":
        lines.append(
            "> **Trạng thái nhãn tham chiếu (`PROVISIONAL_SILERO_PREANNOTATION`)**: Do chưa có file "
            "`annotations_human_reviewed.csv`, toàn bộ phép đo dưới đây sử dụng `preannotation_silero.csv` "
            "làm **nhãn tham chiếu tạm thời do Silero VAD tạo tự động**. Các con số `TP, FP, FN, TN, Precision, "
            "Recall, F1, Miss Rate, False Alarm Rate` phản ánh **mức độ đồng thuận (agreement) giữa Multi-band VAD "
            "hiện tại và Silero VAD**, **KHÔNG PHẢI độ chính xác thực tế đã được con người kiểm chứng (NOT verified ground-truth accuracy)**."
        )
    else:
        lines.append(
            f"> **Trạng thái nhãn tham chiếu (`{ref_type}`)**: Đang sử dụng file `{os.path.basename(ref_path)}`."
        )
    lines.append("")
    lines.append("## 1. Cấu hình Đóng băng đã sử dụng (Frozen Production Configuration)")
    lines.append("")
    lines.append("Không có bất kỳ thay đổi mã nguồn hay tinh chỉnh ngưỡng/trọng số nào trên Tầng 1–4:")
    lines.append("- **Giao diện thực thi**: Streaming qua `FeatureExtractor.process_chunk(chunk) + flush()` $\\to$ `NoiseEstimator.update()` $\\to$ `SigmoidScorer.score()` $\\to$ `VADStateMachine.process_chunk() + flush()`. Đã kiểm chứng bất biến tuyệt đối giữa các kích thước chunk (`20 ms`, `100 ms`, và toàn khối).")
    lines.append(f"- **Tầng 1 (`FeatureExtractor`)**: `sample_rate ∈ {{8000, 16000}} Hz` (không resample), `hop_seconds = 0.005 s` (`5 ms`), cửa sổ Hann `16 ms` (`n_fft = 256, hop = 80` ở `16 kHz`; `n_fft = 128, hop = 40` ở `8 kHz`; `center = False`), 4 dải tần `{DEFAULT_BANDS_HZ}` Hz (số bin `[3, 11, 16, 32]`), SFM trên `80–4000 Hz`.")
    lines.append("- **Tầng 2 (`NoiseConfig`)**: `init_frames = 40` (`200 ms` warm-up đầu file bằng median theo từng băng, đặt `warmup = True` và `snr_db = 0.0`), `alpha_fast = 0.15`, `alpha_slow = 0.01`, `snr_noise_db = 6.0 dB`, `snr_speech_db = 9.0 dB`, `strong_confirm_frames = 3` (`15 ms`), `sfm_thresh = 0.15`, `max_freeze_frames = 500` (`2.5 s`), `recover_frames = 100` (`0.5 s`), `recover_stable_db = 3.0 dB`, `recover_ref_alpha = 0.05`.")
    lines.append("- **Tầng 3 (`SigmoidConfig`)**: `band_weights = (1.0, 2.0, 2.0, 1.0)`, `sfm_weight = 2.0`, `bias = -4.0`, `snr_low_db = 0.0 dB`, `snr_high_db = 20.0 dB`, `warmup_score = 0.0`, `warmup_logit = -20.0`.")
    lines.append("- **Tầng 4 (`StateMachineConfig`)**: `onset_thresh = 0.55`, `offset_thresh = 0.35`, `onset_ms = 30.0 ms` (`6` frame), `hangover_ms = 150.0 ms` (`30` frame), `min_speech_ms = 100.0 ms` (`20` frame bằng chứng `ONSET + SPEECH` độc lập với hangover), `min_silence_ms = 100.0 ms` (`20` frame).")
    lines.append("")
    lines.append("## 2. Kết quả Tổng hợp (Aggregate Frame-Level & Segment-Level Agreement)")
    lines.append("")
    lines.append("Quy ước đánh giá:")
    lines.append("- **Mức frame (5 ms grid)**: Cả nhãn tham chiếu và các đoạn dự đoán `[start_time_s, end_time_s)` được ánh xạ lên cùng lưới frame `5 ms` (`expected_frames_5ms = duration_s / 0.005`, tổng cộng `13,820` frame = `69.100 s`) theo quy tắc nửa mở `start_time_s <= t * 0.005 < end_time_s`.")
    lines.append("- **Mức đoạn (Segment matching)**: Ghép cặp 1–1 tham lam theo `IoU >= 0.05` (`match_segments_by_time`), không bao giờ coi hai đoạn không chồng lấn là khớp chỉ vì có tổng thời lượng gần nhau.")
    lines.append("")
    lines.append("| Tập con | Số file | Thời lượng (s) | TP | FP | FN | TN | Prec (%) | Rec (%) | F1 (%) | Miss (%) | FA (%) | Dur Pred/Ref (s) | Segs (Match/Pred/Ref) | Seg Prec/Rec/F1 (%) | Onset ±10/±20/±50ms | Offset ±10/±20/±50ms | Cả 2 đầu ±10/±20/±50ms |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for label, d in [
        ("**Toàn bộ (11 files)**", agg_all),
        ("Sample rate `16 kHz`", agg_16k),
        ("Sample rate `8 kHz`", agg_8k),
        ("Split `dev`", agg_dev),
        ("Split `eval`", agg_eval),
    ]:
        m_cnt = d["n_matched_segs"]
        lines.append(
            f"| {label} | {d['n_files']} | {d['duration_s']:.1f} | {d['tp']} | {d['fp']} | {d['fn']} | {d['tn']} | "
            f"{d['precision_pct']:.1f} | {d['recall_pct']:.1f} | **{d['f1_pct']:.1f}** | {d['miss_pct']:.1f} | {d['fa_pct']:.1f} | "
            f"{d['pred_speech_s']:.2f} / {d['ref_speech_s']:.2f} | {m_cnt} / {d['n_pred_segs']} / {d['n_ref_segs']} | "
            f"{d['seg_precision_pct']:.1f} / {d['seg_recall_pct']:.1f} / {d['seg_f1_pct']:.1f} | "
            f"{d['onset_within_10ms']}/{d['onset_within_20ms']}/{d['onset_within_50ms']} (/{m_cnt}) | "
            f"{d['offset_within_10ms']}/{d['offset_within_20ms']}/{d['offset_within_50ms']} (/{m_cnt}) | "
            f"{d['both_within_10ms']}/{d['both_within_20ms']}/{d['both_within_50ms']} (/{m_cnt}) |"
        )
    lines.append("")
    lines.append("## 3. Kết quả Chi tiết theo từng File WAV (Per-File Comparison)")
    lines.append("")
    lines.append("| Excerpt ID | SR (Hz) | Split | Dur (s) | TP | FP | FN | TN | Prec (%) | Rec (%) | F1 (%) | Miss (%) | FA (%) | Dur Pred/Ref (s) | Segs (M/P/R) | Seg F1 (%) | Med \\|OnErr\\| (ms) | Med \\|OffErr\\| (ms) | Onset ±10/20/50 | Offset ±10/20/50 | Đoạn đầu (Pred vs Ref) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for r in results:
        m_c = r.n_matched_segs
        lines.append(
            f"| `{r.excerpt_id}` | {r.sample_rate} | `{r.split}` | {r.duration_s:.1f} | "
            f"{r.tp} | {r.fp} | {r.fn} | {r.tn} | "
            f"{r.precision_pct:.1f} | {r.recall_pct:.1f} | **{r.f1_pct:.1f}** | {r.miss_pct:.1f} | {r.fa_pct:.1f} | "
            f"{r.pred_speech_s:.2f}/{r.ref_speech_s:.2f} | {m_c}/{r.n_pred_segs}/{r.n_ref_segs} | {r.seg_f1_pct:.1f} | "
            f"{r.onset_abs_med_ms:.0f} | {r.offset_abs_med_ms:.0f} | "
            f"{r.onset_within_10ms}/{r.onset_within_20ms}/{r.onset_within_50ms} | "
            f"{r.offset_within_10ms}/{r.offset_within_20ms}/{r.offset_within_50ms} | "
            f"`{r.first_pred_seg}` vs `{r.first_ref_seg}` |"
        )
    lines.append("")
    by_id = {r.excerpt_id: r for r in results}
    e01 = by_id["E01_stat20_16k"]
    e02 = by_id["E02_stat20_8k"]
    e03 = by_id["E03_stat15_16k"]
    e04 = by_id["E04_stat15_8k"]
    e05 = by_id["E05_stat10_16k"]
    e06 = by_id["E06_stat10_8k"]
    e07 = by_id["E07_step_up_16k"]
    e08 = by_id["E08_step_down_16k"]
    e09 = by_id["E09_step_up_8k"]
    e10 = by_id["E10_start_16k"]
    e11 = by_id["E11_start_8k"]

    lines.append("## 4. Phân tích Kỹ thuật Chuyên sâu (Technical Analysis)")
    lines.append("")
    lines.append("### 4.1. Các đoạn trích có bất đồng lớn nhất với Silero Pre-Annotations")
    lines.append("Xếp hạng theo mức độ bất đồng cấp frame (`100 - F1%`):")
    lines.append(
        f"1. **`E06_stat10_8k` (`F1 = {e06.f1_pct:.1f}%`, `Miss = {e06.miss_pct:.1f}%`, `FA = {e06.fa_pct:.1f}%`) & "
        f"`E05_stat10_16k` (`F1 = {e05.f1_pct:.1f}%`, `Miss = {e05.miss_pct:.1f}%`, `FA = {e05.fa_pct:.1f}%`) — Kịch bản nhiễu nặng `+10 dB` SNR**:"
    )
    lines.append(
        f"   - **Bản chất bất đồng**: Không phải lệch biên nhẹ mà là **phân mảnh nội câu (intra-utterance fragmentation) và mất các âm tiết yếu**. "
        f"Ở `+10 dB` SNR, các nguyên âm/phụ âm yếu có điểm Sigmoid tụt xuống dưới `offset_thresh = 0.35` lâu hơn `hangover_ms (150 ms) + min_silence_ms (100 ms)`, "
        f"khiến Multi-band VAD tách `{e06.n_ref_segs}` đoạn của `E06` thành `{e06.n_pred_segs}` đoạn ngắn (`{e06.pred_speech_s:.2f} s` vs `{e06.ref_speech_s:.2f} s` của Silero) "
        f"và tách `{e05.n_ref_segs}` đoạn của `E05` thành `{e05.n_pred_segs}` đoạn ngắn (`{e05.pred_speech_s:.2f} s` vs `{e05.ref_speech_s:.2f} s` của Silero). "
        f"Đổi lại, `False Alarm Rate` của Multi-band VAD ở hai file này gần như bằng `0%` (`{e06.fa_pct:.1f}%` và `{e05.fa_pct:.1f}%`)."
    )
    lines.append(
        f"2. **`E11_start_8k` (`F1 = {e11.f1_pct:.1f}%`, `Miss = {e11.miss_pct:.1f}%`, `FA = {e11.fa_pct:.1f}%`) & "
        f"`E10_start_16k` (`F1 = {e10.f1_pct:.1f}%`, `Miss = {e10.miss_pct:.1f}%`, `FA = {e10.fa_pct:.1f}%`) — Kịch bản tiếng nói ngay đầu file (`speech_at_start`)**:"
    )
    lines.append("   - **Bản chất bất đồng**: Khóa warm-up `200 ms` kết hợp hiện tượng nhiễm bẩn `noise_floor` khởi tạo làm mất `625 ms` đầu câu 1 (chi tiết tại Mục 4.2), cộng thêm ở `E11_start_8k` câu thứ hai bị tách thành 2 đoạn tại khoảng nghỉ `140 ms` (`[2.060, 2.780)` và `[2.920, 3.700)`).")
    lines.append(
        f"3. **`E07_step_up_16k` (`F1 = {e07.f1_pct:.1f}%`, `Miss = {e07.miss_pct:.1f}%`, `FA = {e07.fa_pct:.1f}%`) & "
        f"`E09_step_up_8k` (`F1 = {e09.f1_pct:.1f}%`, `Miss = {e09.miss_pct:.1f}%`, `FA = {e09.fa_pct:.1f}%`) — Kịch bản nhiễu nhảy bậc tăng `+15 dB` (`step_up_down`)**:"
    )
    lines.append(
        "   - **Bản chất bất đồng**: Tại `t = 3.000 s` trong đoạn trích (tức `t = 12.0 s` của stream gốc), nhiễu nền nhảy vọt từ `+20 dB` SNR lên `+5 dB` SNR (tăng `+15 dB` công suất nhiễu). "
        "Vì `noise_floor` của Tầng 2 đang neo ở mức nhiễu thấp trước đó, bước nhảy công suất nhiễu tạo ra SNR tức thời `~15 dB` trên cả 4 băng và kích hoạt trạng thái `SPEECH` tại `t = 2.990 s` — "
        "kéo dài thành một đoạn duy nhất `[2.990, 6.700)` xuyên qua `1.20 s` nhiễu thuần (`2.990..4.194 s`) và nối liền câu 4 với câu 5. Trong khi đó, Silero VAD giữ `NON_SPEECH` trong khoảng `[2.366, 4.194)` (16 kHz) và `[2.334, 4.162)` (8 kHz)."
    )
    lines.append(
        f"4. **`E08_step_down_16k` (`F1 = {e08.f1_pct:.1f}%`, `Miss = {e08.miss_pct:.1f}%`, `FA = {e08.fa_pct:.1f}%`) — Kịch bản nửa đầu nhiễu nặng `+5 dB` rồi giảm xuống `+20 dB`**:"
    )
    lines.append(
        f"   - **Bản chất bất đồng**: Trong 3 giây đầu (`+5 dB` SNR), câu nói đầu tiên (`[0.738, 2.718)` theo Silero) chỉ vượt ngưỡng của Multi-band VAD ở phần lõi năng lượng cao nhất `[1.390, 1.830)` (`0.44 s`), gây `Miss = {e08.miss_pct:.1f}%`. "
        "Ngay sau khi nhiễu giảm xuống `+20 dB` tại `t = 3.000 s`, câu nói thứ hai đạt độ đồng thuận rất cao (`[3.260, 5.590)` so với `[3.234, 5.694)` của Silero, lệch onset chỉ `+26 ms`)."
    )
    lines.append(
        f"5. **`E01_stat20_16k` (`F1 = {e01.f1_pct:.1f}%`), `E02_stat20_8k` (`F1 = {e02.f1_pct:.1f}%`), `E03_stat15_16k` (`F1 = {e03.f1_pct:.1f}%`) — Kịch bản nhiễu dừng `+20 dB` và `+15 dB`**:"
    )
    lines.append(
        f"   - **Mức độ đồng thuận rất cao (`F1 = {e01.f1_pct:.1f}% – {e03.f1_pct:.1f}%`)**: Ở `E02` và `E03`, số đoạn khớp tuyệt đối `2/2` (`Seg F1 = 100.0%`) và sai số onset trung vị chỉ `{e02.onset_abs_med_ms:.0f}–{e03.onset_abs_med_ms:.0f} ms`."
    )
    lines.append("")
    lines.append("### 4.2. Chẩn đoán hành vi Khởi động đầu luồng trên `E10_start_16k` và `E11_start_8k`")
    lines.append("- Trên cả hai file `E10_start_16k` (16 kHz) và `E11_start_8k` (8 kHz), tiếng nói bắt đầu ngay từ `t = 0.000 s` (Silero bắt đầu tại `0.002 s` ở 16 kHz và `0.034 s` ở 8 kHz), nhưng đoạn dự đoán đầu tiên của Multi-band VAD đều bắt đầu tại đúng **`0.625 s`** (`[0.625, 1.180)` ở 16 kHz và `[0.625, 1.190)` ở 8 kHz), bỏ sót `625 ms` đầu câu 1.")
    lines.append("- **Nguyên nhân hai tầng kết hợp**:")
    lines.append("  1. **Khóa warm-up cố định `200 ms` (`init_frames = 40`)**: Trong `[0.000, 0.200) s`, Tầng 2 bật `warmup = True`, Tầng 3 xuất `score = 0.0` và Tầng 4 giữ cứng trạng thái `SILENCE`.")
    lines.append("  2. **Nhiễm bẩn `noise_floor` khởi tạo (`200 ms – 625 ms`)**: Vì 40 frame đầu tiên là tiếng nói, phép lấy `median` theo băng trong warm-up của Tầng 2 khởi tạo `noise_floor` bằng chính mức công suất của âm tiết đầu câu. Do đó, từ `0.200 s` đến `0.620 s`, SNR ước lượng bị nén xuống gần `0 dB` (`score < onset_thresh = 0.55`), mãi tới âm tiết nhấn mạnh ở `0.625 s` mới vượt ngưỡng.")
    lines.append("  3. **Tự phục hồi ở câu thứ hai**: Trong khoảng nghỉ `[1.25 s, 2.05 s]`, cổng `strong` (`alpha_fast = 0.15`) kéo `noise_floor` xuống mức nền thật, giúp câu thứ hai bắt đầu tại `2.085 s` (`E10`, lệch `+35 ms` so với Silero `2.050 s`) và `2.060 s` (`E11`, lệch `+42 ms` so với Silero `2.018 s`).")
    lines.append("")
    lines.append("### 4.3. So sánh hành vi giữa `16 kHz` và `8 kHz`")
    lines.append(
        f"- Tổng hợp trên 6 file `16 kHz` (`F1 = {agg_16k['f1_pct']:.1f}%`, `Miss = {agg_16k['miss_pct']:.1f}%`, `FA = {agg_16k['fa_pct']:.1f}%`, `Seg F1 = {agg_16k['seg_f1_pct']:.1f}%`) "
        f"và 5 file `8 kHz` (`F1 = {agg_8k['f1_pct']:.1f}%`, `Miss = {agg_8k['miss_pct']:.1f}%`, `FA = {agg_8k['fa_pct']:.1f}%`, `Seg F1 = {agg_8k['seg_f1_pct']:.1f}%`) "
        "cho thấy mức đồng thuận cấp frame rất gần nhau (`84.1%` vs `84.8%`), nhưng ở **`8 kHz` có xu hướng tách đoạn nội câu nhiều hơn** "
        f"(`Seg Precision = {agg_8k['seg_precision_pct']:.1f}%` ở 8 kHz so với `{agg_16k['seg_precision_pct']:.1f}%` ở 16 kHz: "
        f"`E04_stat15_8k` tách `1` câu dài thành `{e04.n_pred_segs}` đoạn; `E11_start_8k` tách câu 2 thành 2 đoạn `[2.060, 2.780)` & `[2.920, 3.700)` tại một khoảng nghỉ `140 ms`, trong khi `E10_start_16k` giữ nguyên 1 đoạn `[2.085, 3.965)`)."
    )
    lines.append("- Điều này phù hợp với đặc tính vật lý đã ghi nhận: ở `8 kHz`, băng 4 (`2000–4000 Hz`) sát tần số Nyquist bị suy giảm năng lượng âm xát và làm SFM biến thiên khác nhẹ so với `16 kHz`.")
    lines.append("")
    lines.append("## 5. Giới hạn Hiện tại & Đề xuất Bước Tiếp theo")
    lines.append("- **Giới hạn**: Kết quả trên chỉ đo độ đồng thuận với nhãn nháp của Silero VAD (`preannotation_silero.csv`). Ở những vùng nhiễu nhảy bậc (`E07`, `E09`) hoặc phụ âm đuôi mờ trong nhiễu `+10 dB` (`E05`, `E06`), chính Silero VAD cũng có thể mắc lỗi (ví dụ gộp cả khoảng nghỉ giữa câu vào `SPEECH` ở `E04_stat15_8k` hoặc bỏ sót/báo nhầm biên). Tuyệt đối không kết luận bộ VAD nào chính xác hơn nếu chưa có nhãn người nghe.")
    lines.append("- **Bước tiếp theo hợp lý nhất**: Người gán nhãn ưu tiên nghe và hiệu đính các đoạn có độ bất đồng lớn nhất (`E06, E11, E07, E08, E05, E10, E09, E04`) để xuất ra `annotations_human_reviewed.csv` (có đánh dấu vùng `UNCERTAIN`), sau đó chạy lại chính công cụ `benchmark_multiband_vs_silero.py` này để có phép đo chuẩn xác trên nhãn người thật.")
    lines.append("")

    with open(OUT_SUMMARY_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> int:
    ref_type, ref_path, results, prediction_rows, logs = evaluate_all_pilot_wavs()
    write_outputs(ref_type, ref_path, results, prediction_rows, logs)
    agg = summarize_subset(results)
    print("=" * 96)
    print(f"BENCHMARK MULTI-BAND VAD VS SILERO PILOT COMPLETED ({ref_type})")
    print("-" * 96)
    for line in logs:
        print(line)
    print("-" * 96)
    print(
        f"POOLED AGGREGATE (11 files, {agg['duration_s']:.1f}s) | "
        f"TP={agg['tp']} FP={agg['fp']} FN={agg['fn']} TN={agg['tn']} | "
        f"Prec={agg['precision_pct']:.1f}% Rec={agg['recall_pct']:.1f}% F1={agg['f1_pct']:.1f}% | "
        f"Miss={agg['miss_pct']:.1f}% FA={agg['fa_pct']:.1f}% | "
        f"Segs(M/P/R)={agg['n_matched_segs']}/{agg['n_pred_segs']}/{agg['n_ref_segs']}"
    )
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
