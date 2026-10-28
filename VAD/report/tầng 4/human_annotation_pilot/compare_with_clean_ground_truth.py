"""
VAD/report/tầng 4/human_annotation_pilot/compare_with_clean_ground_truth.py

So sánh và đối chiếu cả Multi-band Adaptive VAD và Silero VAD với
BẢN ĐÁP ÁN MẪU TIẾNG NÓI SẠCH GỐC (clean_ground_truth.csv).
"""
import csv
import os
import sys

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from importlib import util
spec_pt = util.spec_from_file_location("pilot_tools", os.path.join(PILOT_DIR, "pilot_tools.py"))
pilot_tools = util.module_from_spec(spec_pt)
sys.modules["pilot_tools"] = pilot_tools
spec_pt.loader.exec_module(pilot_tools)

from benchmark.run_vad_benchmark import segments_to_frame_mask

def run_comparison():
    manifest = pilot_tools.load_manifest()

    # Load clean ground truth
    clean_csv = os.path.join(PILOT_DIR, "clean_ground_truth.csv")
    clean_ivs = pilot_tools.load_annotations_csv(clean_csv)
    clean_by_eid = {}
    for iv in clean_ivs:
        clean_by_eid.setdefault(iv.excerpt_id, []).append(iv)

    # Load silero predictions
    sil_csv = os.path.join(PILOT_DIR, "preannotation_silero.csv")
    sil_ivs = pilot_tools.load_annotations_csv(sil_csv)
    sil_by_eid = {}
    for iv in sil_ivs:
        sil_by_eid.setdefault(iv.excerpt_id, []).append(iv)

    # Load multiband predictions
    mb_csv = os.path.join(PILOT_DIR, "multiband_vad_predictions.csv")
    mb_by_eid = {}
    with open(mb_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mb_by_eid.setdefault(row["excerpt_id"], []).append(
                (float(row["start_time_s"]), float(row["end_time_s"]))
            )

    print("=" * 104)
    print("KẾT QUẢ ĐỐI CHIẾU VỚI BẢN ĐÁP ÁN MẪU TIẾNG NÓI SẠCH GỐC (CLEAN GROUND TRUTH)")
    print("=" * 104)
    print(f"{'Excerpt ID':18s} | {'Multi-band Adaptive VAD':^40s} | {'Silero VAD':^40s}")
    print(f"{'':18s} | {'F1':>6s} | {'Miss':>6s} | {'FA':>6s} | {'Dur P/Ref':>13s} | {'F1':>6s} | {'Miss':>6s} | {'FA':>6s} | {'Dur P/Ref':>13s}")
    print("-" * 104)

    agg_mb = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    agg_sil = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for eid, spec in manifest.items():
        T = spec.expected_frames_5ms
        labels = pilot_tools.intervals_to_frame_labels(eid, clean_by_eid[eid], num_frames=T, hop_seconds=0.005)
        ref_sp_s = sum(float(iv.end_time_s) - float(iv.start_time_s) for iv in clean_by_eid[eid] if iv.label == "SPEECH")

        # Multiband
        mb_mask = segments_to_frame_mask(mb_by_eid.get(eid, []), T, 0.005)
        mb_res = pilot_tools.evaluate_predictions_against_human_labels(mb_mask, labels)
        mb_sp_s = sum(e - s for s, e in mb_by_eid.get(eid, []))

        # Silero
        sil_segs = [(iv.start_time_s, iv.end_time_s) for iv in sil_by_eid[eid] if iv.label == "SPEECH"]
        sil_mask = segments_to_frame_mask(sil_segs, T, 0.005)
        sil_res = pilot_tools.evaluate_predictions_against_human_labels(sil_mask, labels)
        sil_sp_s = sum(e - s for s, e in sil_segs)

        for k in ("tp", "fp", "fn", "tn"):
            agg_mb[k] += mb_res[k]
            agg_sil[k] += sil_res[k]

        mb_dur_str = f"{mb_sp_s:.2f}/{ref_sp_s:.2f}s"
        sil_dur_str = f"{sil_sp_s:.2f}/{ref_sp_s:.2f}s"

        print(
            f"{eid:18s} | {mb_res['f1_pct']:5.1f}% | {mb_res['miss_pct']:5.1f}% | {mb_res['fa_pct']:5.1f}% | {mb_dur_str:>13s} | "
            f"{sil_res['f1_pct']:5.1f}% | {sil_res['miss_pct']:5.1f}% | {sil_res['fa_pct']:5.1f}% | {sil_dur_str:>13s}"
        )

    def calc_pooled(d):
        tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
        f1 = 2 * tp / (2 * tp + fp + fn) * 100
        prec = tp / (tp + fp) * 100
        rec = tp / (tp + fn) * 100
        miss = fn / (tp + fn) * 100
        fa = fp / (fp + tn) * 100
        return f1, prec, rec, miss, fa

    f1_m, prec_m, rec_m, miss_m, fa_m = calc_pooled(agg_mb)
    f1_s, prec_s, rec_s, miss_s, fa_s = calc_pooled(agg_sil)

    print("-" * 104)
    print(
        f"{'TỔNG HỢP (Multi-band)':24s} | F1 = {f1_m:5.1f}% | Precision = {prec_m:5.1f}% | Recall = {rec_m:5.1f}% | Miss = {miss_m:5.1f}% | FA = {fa_m:5.1f}%"
    )
    print(
        f"{'TỔNG HỢP (Silero)':24s} | F1 = {f1_s:5.1f}% | Precision = {prec_s:5.1f}% | Recall = {rec_s:5.1f}% | Miss = {miss_s:5.1f}% | FA = {fa_s:5.1f}%"
    )
    print("=" * 104)

if __name__ == "__main__":
    run_comparison()
