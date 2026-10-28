import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.run_vad_benchmark import (  # noqa: E402
    compute_binary_metrics,
    mask_to_segments,
    match_segments_by_time,
    segments_to_frame_mask,
)


def test_hand_calculated_tp_fp_fn_tn_miss_fa_prec_rec_f1():
    """
    Phase D1.2: Kiểm tra đối chiếu tính tay cho TP, FP, FN, TN, Miss Rate, False Alarm Rate,
    Precision, Recall và F1 trên mặt nạ 10 frame (hop_s = 0.01s).
    - gt   = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0] -> Tổng speech = 5 frame (indices 2,3,4,5,8), non-speech = 5 frame (0,1,6,7,9)
    - pred = [0, 1, 1, 1, 0, 1, 1, 0, 0, 0]
      + TP: indices {2, 3, 5} -> TP = 3
      + FN: indices {4, 8}    -> FN = 2
      + FP: indices {1, 6}    -> FP = 2
      + TN: indices {0, 7, 9} -> TN = 3
      + Miss Rate = 2 / 5 * 100 = 40.0%
      + False Alarm Rate = 2 / 5 * 100 = 40.0%
      + Recall = 3 / 5 * 100 = 60.0%
      + Precision = 3 / (3 + 2) * 100 = 60.0%
      + F1 = 2 * 3 / (6 + 2 + 2) * 100 = 60.0%
    """
    hop_s = 0.01
    gt = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    pred = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=bool)
    gt_segs = mask_to_segments(gt, hop_s, bridge_gap_s=0.0, min_dur_s=0.01)

    res = compute_binary_metrics(pred, gt, gt_segs, hop_s)
    assert res["tp"] == 3
    assert res["fn"] == 2
    assert res["fp"] == 2
    assert res["tn"] == 3
    assert res["miss_pct"] == pytest.approx(40.0)
    assert res["fa_pct"] == pytest.approx(40.0)
    assert res["recall_pct"] == pytest.approx(60.0)
    assert res["precision_pct"] == pytest.approx(60.0)
    assert res["f1_pct"] == pytest.approx(60.0)
    assert res["ref_speech_s"] == pytest.approx(0.05)
    assert res["pred_speech_s"] == pytest.approx(0.05)


def test_one_frame_interval_and_interval_ending_at_final_frame():
    """
    Phase D1.2: Kiểm tra biên:
    1) Đoạn đúng 1 frame tại đầu stream [0.000, 0.005) và giữa stream [0.015, 0.020).
    2) Đoạn kết thúc ngay tại frame cuối cùng T-1: [0.045, 0.050) với T = 10, hop_s = 0.005.
    3) Tính khứ hồi (round-trip) giữa segments_to_frame_mask và mask_to_segments khi tắt nối/lọc.
    """
    hop_s = 0.005
    T = 10
    segs = [(0.000, 0.005), (0.015, 0.020), (0.045, 0.050)]
    mask = segments_to_frame_mask(segs, T, hop_s)

    expected = np.zeros(T, dtype=bool)
    expected[0] = True
    expected[3] = True
    expected[9] = True
    np.testing.assert_array_equal(mask, expected)

    roundtrip_segs = mask_to_segments(mask, hop_s, bridge_gap_s=0.0, min_dur_s=0.005)
    assert len(roundtrip_segs) == 3
    for (s1, e1), (s2, e2) in zip(roundtrip_segs, segs):
        assert s1 == pytest.approx(s2, abs=1e-9)
        assert e1 == pytest.approx(e2, abs=1e-9)


def test_empty_reference_and_prediction_masks():
    """Phase D1.2: Kiểm tra mặt nạ rỗng T=0."""
    hop_s = 0.005
    empty_mask = segments_to_frame_mask([], 0, hop_s)
    assert empty_mask.shape == (0,) and empty_mask.dtype == bool
    assert mask_to_segments(empty_mask, hop_s) == []

    res = compute_binary_metrics(empty_mask, empty_mask, [], hop_s)
    assert res["tp"] == 0 and res["fp"] == 0 and res["fn"] == 0 and res["tn"] == 0
    assert res["miss_pct"] == 0.0 and res["fa_pct"] == 0.0
    assert res["precision_pct"] == 0.0 and res["recall_pct"] == 0.0 and res["f1_pct"] == 0.0


def test_perfect_prediction_completely_missed_and_all_silence_reference():
    """Phase D1.2: Kiểm tra 3 trường hợp cực trị: dự đoán hoàn hảo, bỏ sót hoàn toàn, và chuẩn toàn im lặng."""
    hop_s = 0.005
    T = 20
    gt = np.zeros(T, dtype=bool)
    gt[5:15] = True  # 10 frame speech, 10 frame silence
    gt_segs = mask_to_segments(gt, hop_s)

    # 1. Dự đoán hoàn hảo (Perfect prediction)
    res_perf = compute_binary_metrics(gt.copy(), gt, gt_segs, hop_s)
    assert res_perf["tp"] == 10 and res_perf["tn"] == 10
    assert res_perf["fp"] == 0 and res_perf["fn"] == 0
    assert res_perf["miss_pct"] == 0.0 and res_perf["fa_pct"] == 0.0
    assert res_perf["precision_pct"] == 100.0 and res_perf["recall_pct"] == 100.0
    assert res_perf["f1_pct"] == 100.0
    assert res_perf["onset_err_med_ms"] == 0.0 and res_perf["offset_err_med_ms"] == 0.0

    # 2. Bỏ sót hoàn toàn tiếng nói (Completely missed speech)
    pred_zero = np.zeros(T, dtype=bool)
    res_miss = compute_binary_metrics(pred_zero, gt, gt_segs, hop_s)
    assert res_miss["tp"] == 0 and res_miss["fn"] == 10
    assert res_miss["fp"] == 0 and res_miss["tn"] == 10
    assert res_miss["miss_pct"] == 100.0 and res_miss["fa_pct"] == 0.0
    assert res_miss["precision_pct"] == 0.0 and res_miss["recall_pct"] == 0.0
    assert res_miss["f1_pct"] == 0.0
    assert np.isnan(res_miss["onset_err_med_ms"])

    # 3. Chuẩn toàn im lặng (All-silence reference) với 4 frame báo nhầm
    gt_silence = np.zeros(T, dtype=bool)
    pred_fa = np.zeros(T, dtype=bool)
    pred_fa[2:6] = True
    res_sil = compute_binary_metrics(pred_fa, gt_silence, [], hop_s)
    assert res_sil["tp"] == 0 and res_sil["fn"] == 0
    assert res_sil["fp"] == 4 and res_sil["tn"] == 16
    assert res_sil["miss_pct"] == 0.0
    assert res_sil["fa_pct"] == pytest.approx(4 / 20 * 100.0)
    assert res_sil["precision_pct"] == 0.0 and res_sil["f1_pct"] == 0.0


def test_touching_non_overlapping_intervals_and_one_to_one_matching():
    """
    Phase D1.2: Kiểm tra:
    1) Hai đoạn nửa mở chạm nhau nhưng không chồng lấn: [0.010, 0.030) và [0.030, 0.050).
       - Giữa hai đoạn không có frame chồng lấn và cũng không bị sót frame biên 0.030.
       - Khi ghép với đoạn chuẩn, match_segments_by_time đảm bảo ánh xạ 1-1 (mỗi đoạn chuẩn
         chỉ ghép tối đa 1 đoạn dự đoán có IoU cao nhất, đoạn chạm biên không chồng lấn có ov=0 -> không ghép).
    2) Khoảng nửa mở không nằm tròn trên lưới frame (sub-frame human annotation) tuân thủ đúng
       t_start <= t * hop_s < t_end.
    """
    hop_s = 0.01  # 10 ms / frame
    T = 8
    seg_a = (0.01, 0.03)  # frames 1, 2
    seg_b = (0.03, 0.05)  # frames 3, 4

    mask_a = segments_to_frame_mask([seg_a], T, hop_s)
    mask_b = segments_to_frame_mask([seg_b], T, hop_s)
    mask_both = segments_to_frame_mask([seg_a, seg_b], T, hop_s)

    # Không chồng lấn (giao bằng rỗng) và hợp bằng đúng frames 1..4
    assert not (mask_a & mask_b).any()
    np.testing.assert_array_equal(mask_a | mask_b, mask_both)
    np.testing.assert_array_equal(np.where(mask_both)[0], np.array([1, 2, 3, 4]))

    # Kiểm tra match_segments_by_time: đoạn chạm biên [0.03, 0.05) có overlap = 0.0 với [0.01, 0.03) -> không ghép!
    matched_touch, _, _ = match_segments_by_time([seg_a], [seg_b], min_iou=0.05)
    assert matched_touch == []

    # Nếu có 1 đoạn GT [0.01, 0.05) và 2 đoạn dự đoán chạm nhau [0.01, 0.03) & [0.03, 0.06),
    # ghép 1-1 chỉ chọn đúng 1 đoạn dự đoán có IoU lớn hơn ([0.03, 0.06) có ov=0.02, un=0.05; [0.01, 0.03) có ov=0.02, un=0.04 -> IoU=0.5 > 0.4)
    matched_1to1, on_err, off_err = match_segments_by_time(
        [(0.01, 0.05)], [(0.01, 0.03), (0.03, 0.06)], min_iou=0.05
    )
    assert len(matched_1to1) == 1
    assert matched_1to1[0][0] == 0 and matched_1to1[0][1] == 0
    assert on_err[0] == pytest.approx(0.0)
    assert off_err[0] == pytest.approx(-20.0)

    # Kiểm tra khoảng nhãn con người không nằm tròn trên bội số hop_s: [0.012, 0.038) với hop_s=0.01
    # Các mốc frame t*0.01 nằm trong [0.012, 0.038) là t=2 (0.02s) và t=3 (0.03s)
    sub_mask = segments_to_frame_mask([(0.012, 0.038)], T, hop_s)
    np.testing.assert_array_equal(np.where(sub_mask)[0], np.array([2, 3]))


def test_pilot_annotation_validation_and_uncertain_exclusion():
    """
    Phase D1.3 & D1.4: Kiểm tra công cụ quy đổi nhãn con người (SPEECH, NON_SPEECH, UNCERTAIN)
    và loại trừ vùng UNCERTAIN khỏi mẫu số Miss/FA chính.
    """
    from importlib import util

    pilot_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "VAD",
        "report",
        "tầng 4",
        "human_annotation_pilot",
        "pilot_tools.py",
    )
    spec = util.spec_from_file_location("pilot_tools", pilot_path)
    assert spec is not None and spec.loader is not None
    pilot_tools = util.module_from_spec(spec)
    sys.modules["pilot_tools"] = pilot_tools
    spec.loader.exec_module(pilot_tools)

    manifest = pilot_tools.load_manifest()
    assert len(manifest) == 11
    assert "E01_stat20_16k" in manifest

    # Kiểm tra tính toàn vẹn của phân chia dev/eval (không rò rỉ câu nói nguồn, cặp 8k/16k cùng split)
    dev_utts = set()
    eval_utts = set()
    for spec_item in manifest.values():
        utts = {u.strip() for u in spec_item.source_utterances.split("|") if u.strip()}
        if spec_item.split == "dev":
            dev_utts |= utts
        else:
            eval_utts |= utts
    assert dev_utts.isdisjoint(eval_utts)
    assert manifest["E10_start_16k"].split == manifest["E11_start_8k"].split == "dev"
    assert manifest["E02_stat20_8k"].split == manifest["E03_stat15_16k"].split == "eval"
    assert manifest["E07_step_up_16k"].split == manifest["E09_step_up_8k"].split == "eval"

    # Kiểm tra validate_manifest_split_integrity từ chối manifest bị rò rỉ câu nói giữa dev và eval
    leaky_manifest = dict(manifest)
    leaky_manifest["E11_start_8k"] = pilot_tools.ExcerptSpec(
        **{**manifest["E11_start_8k"].__dict__, "split": "eval", "source_group_id": "SG_LEAK"}
    )
    with pytest.raises(ValueError, match="Rò rỉ câu nói nguồn"):
        pilot_tools.validate_manifest_split_integrity(leaky_manifest)

    # Template gốc chưa điền nhãn phải bị validate_annotations từ chối
    template_csv = os.path.join(os.path.dirname(pilot_path), "annotation_template.csv")
    raw_template = pilot_tools.load_annotations_csv(template_csv)
    with pytest.raises(ValueError, match="Nhãn không hợp lệ"):
        pilot_tools.validate_annotations(raw_template, manifest)

    # Tạo thử danh sách khoảng hợp lệ cho 10 frame (hop = 0.01s -> 0.10s):
    # [0.00, 0.04) -> NON_SPEECH (4 frames: 0,1,2,3)
    # [0.04, 0.06) -> UNCERTAIN  (2 frames: 4,5)
    # [0.06, 0.10) -> SPEECH     (4 frames: 6,7,8,9)
    ivs = [
        pilot_tools.AnnotationInterval("E_TEST", 0.00, 0.04, "NON_SPEECH"),
        pilot_tools.AnnotationInterval("E_TEST", 0.04, 0.06, "UNCERTAIN"),
        pilot_tools.AnnotationInterval("E_TEST", 0.06, 0.10, "SPEECH"),
    ]
    labels = pilot_tools.intervals_to_frame_labels("E_TEST", ivs, num_frames=10, hop_seconds=0.01)
    assert labels.uncertain_mask.sum() == 2
    assert labels.valid_eval_mask.sum() == 8

    # Dự đoán: báo True tại frame 1 (FP), frame 4 (trong vùng UNCERTAIN -> không tính FP!), và frames 6,7,8 (3 TP, 1 FN tại frame 9)
    pred = np.zeros(10, dtype=bool)
    pred[[1, 4, 6, 7, 8]] = True
    eval_res = pilot_tools.evaluate_predictions_against_human_labels(pred, labels)

    # Trên 8 frame hợp lệ (4 SPEECH, 4 NON_SPEECH): TP=3, FN=1, FP=1, TN=3
    assert eval_res["tp"] == 3 and eval_res["fn"] == 1
    assert eval_res["fp"] == 1 and eval_res["tn"] == 3
    assert eval_res["miss_pct"] == pytest.approx(25.0)
    assert eval_res["fa_pct"] == pytest.approx(25.0)
    assert eval_res["uncertain_frames"] == 2
    assert eval_res["uncertain_pct"] == pytest.approx(20.0)
    assert eval_res["pred_pos_in_uncertain_pct"] == pytest.approx(50.0)

