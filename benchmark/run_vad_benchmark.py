import os
import sys
import numpy as np
import av

# Hỗ trợ biến môi trường VAD_ROOT, fallback tự động về thư mục gốc repo
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not os.path.exists(os.path.join(DEFAULT_ROOT, "VAD")):
    DEFAULT_ROOT = r"c:\file\VAD_ASR_Diarization" if os.path.exists(r"c:\file\VAD_ASR_Diarization\VAD") else os.getcwd()

ROOT = os.environ.get("VAD_ROOT", DEFAULT_ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from VAD.feature_extraction import FeatureExtractor, VADFeatures
from VAD.noise_estimator import NoiseEstimator
from VAD.sigmoid import SigmoidScorer
from VAD.state_machine import StateMachineConfig, VADStateMachine

MP3_DIR = os.path.join(ROOT, "test_data", "mp3")
TRANSCRIPT_PATH = os.path.join(ROOT, "test_data", "transcriptAll.txt")


def decode_mp3_mono(path, target_sr=16000):
    container = av.open(path)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    chunks = []
    for frame in container.decode(audio=0):
        for of in resampler.resample(frame):
            chunks.append(of.to_ndarray()[0])
    for of in resampler.resample(None):
        chunks.append(of.to_ndarray()[0])
    container.close()
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def extract_clean_speech_core(audio, sr, hop_ms=5):
    """Tách lõi phát ngôn từ file nguồn dựa trên chênh lệch năng lượng p95/p05.
    Lưu ý: p95/p05 chỉ là độ chênh năng lượng ngắn hạn trong file để lọc nguồn có vùng lặng rõ,
    KHÔNG phải SNR thực sự của nguồn thu."""
    hop = int(sr * hop_ms / 1000)
    win = int(sr * 0.016)
    n_frames = 1 + max(0, (len(audio) - win) // hop)
    if n_frames < 20:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
    frame_e = np.mean(frames ** 2, axis=1) + 1e-12
    # Loại bỏ các frame đệm số 0 của MP3 khi ước lượng p05/p95
    non_zero_e = frame_e[frame_e > 1e-9]
    if len(non_zero_e) < 20:
        return None
    p95 = np.percentile(non_zero_e, 95)
    p05 = np.percentile(non_zero_e, 5)
    p95_p05_db = 10.0 * np.log10(p95 / p05)
    if p95_p05_db < 35.0:
        return None

    # Ngưỡng năng lượng giả lập để cắt bỏ khoảng lặng đầu/cuối trên file nguồn
    thresh = max(p05 * 31.6, p95 * 3e-4)
    active = frame_e > thresh
    idx = np.where(active)[0]
    if len(idx) < 40:
        return None
    s_frame, e_frame = int(idx[0]), int(idx[-1])
    s_sample = s_frame * hop
    e_sample = min(len(audio), e_frame * hop + win)
    core = audio[s_sample:e_sample].copy()

    # Phủ taper 10ms ở 2 đầu đoạn cắt để tránh xung nhiễu nhân tạo khi ghép
    taper = min(int(sr * 0.01), len(core) // 4)
    if taper > 0:
        ramp = np.linspace(0.0, 1.0, taper, dtype=np.float32)
        core[:taper] *= ramp
        core[-taper:] *= ramp[::-1]
    return core, p95_p05_db


def collect_benchmark_materials(sr=16000):
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    noise_files = [
        "FPTOpenSpeechData_Set001_V0.1_008210.mp3",
        "FPTOpenSpeechData_Set001_V0.1_010753.mp3",
        "FPTOpenSpeechData_Set001_V0.1_011841.mp3",
    ]
    noise_pieces = []
    for nf in noise_files:
        a = decode_mp3_mono(os.path.join(MP3_DIR, nf), target_sr=sr)
        s_trim, e_trim = int(0.25 * sr), int(0.15 * sr)
        if len(a) > s_trim + e_trim + sr // 2:
            a = a[s_trim:len(a) - e_trim]
            rms = np.sqrt(np.mean(a ** 2) + 1e-12)
            noise_pieces.append(a / rms)

    clean_cores = []
    cand_indices = np.linspace(0, len(lines) - 1, 120, dtype=int)
    for idx in cand_indices:
        fname = lines[idx].split("|")[0]
        if fname in noise_files:
            continue
        a = decode_mp3_mono(os.path.join(MP3_DIR, fname), target_sr=sr)
        res = extract_clean_speech_core(a, sr)
        if res is not None:
            core, p95_p05_db = res
            dur = len(core) / sr
            if 1.2 <= dur <= 4.0:
                rms = np.sqrt(np.mean(core ** 2) + 1e-12)
                clean_cores.append((fname, core / rms, dur, p95_p05_db))
        if len(clean_cores) >= 16:
            break

    return clean_cores, noise_pieces


def tile_real_noise(noise_pieces, total_samples, sr, rng):
    fade = int(sr * 0.02)
    out = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while pos < total_samples:
        piece = noise_pieces[int(rng.integers(0, len(noise_pieces)))]
        if pos == 0:
            take = min(len(piece), total_samples)
            out[:take] = piece[:take]
            pos += take
        else:
            overlap = min(fade, pos, len(piece))
            w = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            out[pos - overlap:pos] = out[pos - overlap:pos] * (1.0 - w) + piece[:overlap] * w
            rem = min(len(piece) - overlap, total_samples - pos)
            if rem > 0:
                out[pos:pos + rem] = piece[overlap:overlap + rem]
                pos += rem
            else:
                break
    return out


def build_benchmark_stream(clean_cores, noise_pieces, sr, scenario, seed=100):
    """Ghép stream ~36.8s. Nhãn được tạo từ tín hiệu sạch s_wave TRƯỚC khi trộn nhiễu."""
    rng = np.random.default_rng(seed)
    gaps_sec = [1.2, 0.8, 1.5, 0.6, 2.0, 1.0, 1.8, 0.7, 2.5, 1.2]
    if scenario == "speech_at_start":
        gaps_sec[0] = 0.0  # Nói ngay từ 0 ms (nằm trọn trong vùng warm-up đầu stream)

    utt_intervals = []
    cursor = 0
    s_parts = []
    speech_rms = 0.1
    for i in range(10):
        gap_samples = int(gaps_sec[i] * sr)
        if gap_samples > 0:
            s_parts.append(np.zeros(gap_samples, dtype=np.float32))
            cursor += gap_samples
        fname, core_unit, dur, _ = clean_cores[i % len(clean_cores)]
        seg_audio = (core_unit * speech_rms).astype(np.float32)
        start_t = cursor / sr
        end_t = (cursor + len(seg_audio)) / sr
        utt_intervals.append((start_t, end_t, fname))
        s_parts.append(seg_audio)
        cursor += len(seg_audio)

    tail_samples = int(2.0 * sr)
    s_parts.append(np.zeros(tail_samples, dtype=np.float32))
    cursor += tail_samples
    s_wave = np.concatenate(s_parts)
    total_samples = len(s_wave)

    raw_noise = tile_real_noise(noise_pieces, total_samples, sr, rng)
    if scenario == "stationary_20db":
        n_wave = raw_noise * (speech_rms / (10.0 ** (20.0 / 20.0)))
    elif scenario == "stationary_15db":
        n_wave = raw_noise * (speech_rms / (10.0 ** (15.0 / 20.0)))
    elif scenario == "stationary_10db":
        n_wave = raw_noise * (speech_rms / (10.0 ** (10.0 / 20.0)))
    elif scenario == "step_up_down":
        # Thống nhất đúng kịch bản: SNR danh nghĩa +20 dB (0-12s), +5 dB (12-24s), +20 dB (24s-hết)
        env = np.full(total_samples, speech_rms / (10.0 ** (20.0 / 20.0)), dtype=np.float32)
        t1, t2 = int(12.0 * sr), int(24.0 * sr)
        env[t1:t2] = speech_rms / (10.0 ** (5.0 / 20.0))
        n_wave = raw_noise * env
    elif scenario == "speech_at_start":
        n_wave = raw_noise * (speech_rms / (10.0 ** (15.0 / 20.0)))
    else:
        raise ValueError(scenario)

    mix_wave = s_wave + n_wave
    return s_wave, n_wave, mix_wave, utt_intervals


def mask_to_segments(mask, hop_s, bridge_gap_s=0.15, min_dur_s=0.03):
    """Chuyển chuỗi nhị phân theo frame thành danh sách đoạn [(start_s, end_s), ...].
    Nối các khoảng ngắt ngắn <= bridge_gap_s và lọc bỏ gai ngắn < min_dur_s.
    Lưu ý: trả về end_s = (e_f + 1) * hop_s theo quy ước nửa mở [start_s, end_s) đồng nhất với Tầng 4."""
    m = mask.copy().astype(bool)
    T = len(m)
    if T == 0 or not m.any():
        return []

    max_gap_frames = int(round(bridge_gap_s / hop_s))
    min_dur_frames = max(1, int(round(min_dur_s / hop_s)))

    # Nối các khoảng ngắt ngắn <= max_gap_frames
    idx = np.where(m)[0]
    for k in range(len(idx) - 1):
        gap = idx[k + 1] - idx[k] - 1
        if 0 < gap <= max_gap_frames:
            m[idx[k] + 1:idx[k + 1]] = True

    # Trích xuất các đoạn liên thông (exclusive end = e_f + 1)
    padded = np.concatenate([[False], m, [False]])
    starts = np.where(padded[1:] & (~padded[:-1]))[0]
    ends = np.where((~padded[1:]) & padded[:-1])[0] - 1

    segs = []
    for s_f, e_f in zip(starts, ends):
        if (e_f - s_f + 1) >= min_dur_frames:
            segs.append((s_f * hop_s, (e_f + 1) * hop_s))
    return segs


def segments_to_frame_mask(segments, num_frames, hop_s):
    """Chuyển danh sách đoạn nửa mở [(start_time, end_time), ...] về mặt nạ nhị phân shape (T,).
    Quy ước: Frame t (có mốc bắt đầu t * hop_s) thuộc đoạn nửa mở [start_time, end_time) khi và chỉ khi
    start_time <= t * hop_s < end_time (với dung sai số học 1e-9 cho bội số nguyên của hop_s)."""
    if num_frames < 0:
        raise ValueError("num_frames phải >= 0")
    if hop_s <= 0:
        raise ValueError("hop_s phải > 0")
    mask = np.zeros(num_frames, dtype=bool)
    if num_frames == 0:
        return mask
    for start_t, end_t in segments:
        if end_t < start_t:
            raise ValueError(f"Đoạn không hợp lệ: end_time ({end_t}) < start_time ({start_t})")
        if end_t == start_t:
            continue
        s_f = max(0, int(np.ceil(start_t / hop_s - 1e-9)))
        e_f = min(num_frames, int(np.ceil(end_t / hop_s - 1e-9)))
        if e_f > s_f:
            mask[s_f:e_f] = True
    return mask


def compute_binary_metrics(pred_mask, gt_speech_frame, gt_segs, hop_s, pred_segs=None):
    """
    Tính bộ chỉ số thống nhất cho một mặt nạ dự đoán `pred_mask` trên toàn bộ T frame (bao gồm cả warm-up):
      - TP = sum(pred_mask & gt_speech_frame)
      - FN = sum((~pred_mask) & gt_speech_frame)
      - FP = sum(pred_mask & (~gt_speech_frame))
      - TN = sum((~pred_mask) & (~gt_speech_frame))
      - Miss Rate (%)   = FN / (TP + FN) * 100  (mẫu số = tổng số frame có tiếng nói chuẩn gt_speech_frame.sum())
      - False Alarm (%) = FP / (FP + TN) * 100  (mẫu số = tổng số frame không có tiếng nói (~gt_speech_frame).sum())
      - Recall (%)      = TP / (TP + FN) * 100  (= 100 - Miss Rate)
      - Precision (%)   = TP / (TP + FP) * 100  (0.0 nếu TP + FP == 0)
      - F1 (%)          = 2 * TP / (2 * TP + FP + FN) * 100
    """
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_speech_frame, dtype=bool)
    tp = int((pred & gt).sum())
    fn = int(((~pred) & gt).sum())
    fp = int((pred & (~gt)).sum())
    tn = int(((~pred) & (~gt)).sum())

    n_pos = tp + fn
    n_neg = fp + tn
    miss_pct = (fn / n_pos * 100.0) if n_pos > 0 else 0.0
    fa_pct = (fp / n_neg * 100.0) if n_neg > 0 else 0.0
    recall_pct = (tp / n_pos * 100.0) if n_pos > 0 else 0.0
    prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    f1_pct = (2.0 * tp / (2 * tp + fp + fn) * 100.0) if (2 * tp + fp + fn) > 0 else 0.0

    if pred_segs is None:
        pred_segs = mask_to_segments(pred, hop_s, bridge_gap_s=0.15, min_dur_s=0.03)
    matched, onset_errs_ms, offset_errs_ms = match_segments_by_time(gt_segs, pred_segs, min_iou=0.05)

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "miss_pct": miss_pct,
        "fa_pct": fa_pct,
        "precision_pct": prec_pct,
        "recall_pct": recall_pct,
        "f1_pct": f1_pct,
        "ref_speech_s": n_pos * hop_s,
        "pred_speech_s": (tp + fp) * hop_s,
        "n_gt_segs": len(gt_segs),
        "n_pred_segs": len(pred_segs),
        "n_matched_segs": len(matched),
        "onset_err_med_ms": float(np.median(np.abs(onset_errs_ms))) if onset_errs_ms else float("nan"),
        "offset_err_med_ms": float(np.median(np.abs(offset_errs_ms))) if offset_errs_ms else float("nan"),
    }


def match_segments_by_time(gt_segs, pred_segs, min_iou=0.05):
    """Ghép 1-1 các đoạn dự đoán với đoạn chuẩn theo độ chồng lấn thời gian (IoU giảm dần).
    Trả về (matched_pairs, onset_errs_ms, offset_errs_ms)."""
    if not gt_segs or not pred_segs:
        return [], [], []

    candidates = []
    for gi, (g_on, g_off) in enumerate(gt_segs):
        for pi, (p_on, p_off) in enumerate(pred_segs):
            ov = max(0.0, min(g_off, p_off) - max(g_on, p_on))
            if ov > 0.0:
                un = max(g_off, p_off) - min(g_on, p_on)
                iou = ov / max(un, 1e-9)
                if iou >= min_iou:
                    candidates.append((iou, ov, gi, pi))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    used_g = set()
    used_p = set()
    matched = []
    onset_errs_ms = []
    offset_errs_ms = []

    for iou, ov, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            g_on, g_off = gt_segs[gi]
            p_on, p_off = pred_segs[pi]
            matched.append((gi, pi, iou))
            onset_errs_ms.append((p_on - g_on) * 1000.0)
            offset_errs_ms.append((p_off - g_off) * 1000.0)

    return matched, onset_errs_ms, offset_errs_ms


def build_clean_ground_truth(feats_s, utt_intervals, bins_per_band, hop_s):
    """Tạo nhãn giả lập theo năng lượng TRỰC TIẾP từ tín hiệu tiếng nói sạch s_wave
    trước khi trộn nhiễu (hoàn toàn độc lập với n_wave, mix_wave và VAD)."""
    P_s = feats_s.band_power  # (4, T)
    T = feats_s.num_frames
    times = np.arange(T) * hop_s

    # Công suất toàn dải của tín hiệu sạch tại mỗi frame
    total_clean_power = np.sum(P_s * bins_per_band[:, None], axis=0) / np.sum(bins_per_band)

    in_utt_seg = np.zeros(T, dtype=bool)
    gt_speech_frame = np.zeros(T, dtype=bool)
    gt_band_speech = np.zeros_like(P_s, dtype=bool)

    for st, et, _ in utt_intervals:
        u_mask = (times >= st) & (times <= et)
        in_utt_seg |= u_mask
        if u_mask.any():
            # Ngưỡng frame có tiếng nói trên nguồn sạch: trong phạm vi -25 dB so với p95 toàn dải của câu sạch đó
            p95_frame = np.percentile(total_clean_power[u_mask], 95)
            gt_speech_frame[u_mask] = total_clean_power[u_mask] >= (p95_frame * 3.162e-3)

            # Ngưỡng từng băng có năng lượng tiếng nói sạch: trong phạm vi -20 dB so với p95 của chính băng đó trên nguồn sạch
            p95_band = np.percentile(P_s[:, u_mask], 95, axis=1, keepdims=True)
            band_thresh = np.maximum(p95_band * 1e-2, 1e-7)
            gt_band_speech[:, u_mask] = (
                gt_speech_frame[u_mask][None, :] & (P_s[:, u_mask] >= band_thresh)
            )

    gt_pause_frame = in_utt_seg & (~gt_speech_frame)
    pure_noise_frame = ~in_utt_seg

    # Tạo danh sách đoạn chuẩn (ground-truth segments) từ gt_speech_frame trên nguồn sạch
    gt_segs = mask_to_segments(gt_speech_frame, hop_s, bridge_gap_s=0.15, min_dur_s=0.03)
    return {
        "in_utt_seg": in_utt_seg,
        "gt_speech_frame": gt_speech_frame,
        "gt_band_speech": gt_band_speech,
        "gt_pause_frame": gt_pause_frame,
        "pure_noise_frame": pure_noise_frame,
        "gt_segs": gt_segs,
    }


def evaluate_stream(s_wave, n_wave, mix_wave, utt_intervals, sr):
    fe_s = FeatureExtractor(sample_rate=sr)
    fe_n = FeatureExtractor(sample_rate=sr)
    fe_m = FeatureExtractor(sample_rate=sr)
    est = NoiseEstimator(num_bands=fe_m.num_bands)
    scorer = SigmoidScorer(num_bands=fe_m.num_bands)
    sm = VADStateMachine(hop_seconds=fe_m.hop_seconds, config=StateMachineConfig())

    feats_s = fe_s.process_batch(s_wave)
    feats_n = fe_n.process_batch(n_wave)
    feats_m = fe_m.process_batch(mix_wave)
    out = est.update(feats_m)
    sig_out = scorer.score(feats_m, out)
    sm_out = sm.process_chunk(sig_out)
    l4_segs = sm_out.segments + sm.flush()

    T = feats_m.num_frames
    hop_s = fe_m.hop_seconds
    assert out.num_frames == T and sig_out.num_frames == T and sm_out.num_frames == T

    # 1. Ground-Truth từ nguồn sạch (trước khi trộn nhiễu)
    bins_per_band = (fe_s.band_weights > 0).sum(axis=1)
    gt = build_clean_ground_truth(feats_s, utt_intervals, bins_per_band, hop_s)
    in_utt_seg = gt["in_utt_seg"]
    gt_speech_frame = gt["gt_speech_frame"]
    gt_band_speech = gt["gt_band_speech"]
    gt_pause_frame = gt["gt_pause_frame"]
    pure_noise_frame = gt["pure_noise_frame"]
    gt_segs = gt["gt_segs"]

    # 2. Giải phẫu các lần update_mask == True (sau warm-up)
    valid = ~out.warmup
    upd = out.update_mask[:, valid]
    in_utt_v = in_utt_seg[valid]
    pause_v = gt_pause_frame[valid]
    band_sp_v = gt_band_speech[:, valid]
    sp_frame_v = gt_speech_frame[valid]
    noise_v = pure_noise_frame[valid]

    utt_upd = upd[:, in_utt_v]
    utt_upd_total = np.maximum(utt_upd.sum(axis=1), 1)
    utt_frames_cnt = max(1, int(in_utt_v.sum()))

    grp_pause_cnt = (utt_upd & pause_v[in_utt_v][None, :]).sum(axis=1)
    grp_band_silent_cnt = (utt_upd & sp_frame_v[in_utt_v][None, :] & (~band_sp_v[:, in_utt_v])).sum(axis=1)
    grp_leak_cnt = (utt_upd & band_sp_v[:, in_utt_v]).sum(axis=1)

    # 3. Sai số ước lượng Noise Floor (so với công suất nhiễu thực P_n làm mượt nhẹ)
    P_n = feats_n.band_power
    kernel = np.ones(40) / 40.0
    P_n_smooth = np.vstack([np.convolve(P_n[b], kernel, mode="same") for b in range(4)])
    floor_err_db = 10.0 * np.log10(
        np.maximum(out.noise_floor[:, valid], 1e-12) / np.maximum(P_n_smooth[:, valid], 1e-12)
    )
    med_abs_floor_err_db = np.median(np.abs(floor_err_db), axis=1)
    bias_floor_in_speech_db = np.median(floor_err_db[:, sp_frame_v], axis=1)

    # 4. Ba nhóm bộ phát hiện (Baselines):
    # (A) Layer 2 Raw SNR detector: (~warmup) & (max_band SNR >= 9.0 dB)
    l2_mask = (~out.warmup) & (out.snr_db.max(axis=0) >= est.config.snr_speech_db)
    l2_metrics = compute_binary_metrics(l2_mask, gt_speech_frame, gt_segs, hop_s)

    # (B) Layer 3 Raw Sigmoid threshold detectors (0.3, 0.5, 0.7) không qua State Machine
    l3_metrics = {}
    for th in (0.3, 0.5, 0.7):
        m_th = (~sig_out.warmup) & (sig_out.score >= th)
        l3_metrics[th] = compute_binary_metrics(m_th, gt_speech_frame, gt_segs, hop_s)

    # (C) Layer 4 Final Output: từ danh sách đoạn hoàn chỉnh l4_segs (process_chunk + flush)
    l4_seg_mask = segments_to_frame_mask(l4_segs, T, hop_s)
    l4_metrics = compute_binary_metrics(
        l4_seg_mask, gt_speech_frame, gt_segs, hop_s, pred_segs=l4_segs
    )

    # (D) Layer 4 Immediate speech_mask (trạng thái tức thời SPEECH | HANGOVER, chưa gồm ONSET và chưa lọc min_speech)
    l4_imm_metrics = compute_binary_metrics(
        sm_out.speech_mask, gt_speech_frame, gt_segs, hop_s
    )

    # 5. Kiểm tra bất biến streaming (chunk-invariance) xuyên Tầng 2 -> Tầng 3 -> Tầng 4 trên luồng thực
    for chunk_f in (1, 13, 80, 500):
        est_c = NoiseEstimator(num_bands=fe_m.num_bands)
        sc_c = SigmoidScorer(num_bands=fe_m.num_bands)
        sm_c = VADStateMachine(hop_seconds=hop_s, config=StateMachineConfig())
        segs_c = []
        for idx_f in range(0, T, chunk_f):
            fc = VADFeatures(
                band_power=feats_m.band_power[:, idx_f:idx_f + chunk_f],
                sfm=feats_m.sfm[idx_f:idx_f + chunk_f],
            )
            nc = est_c.update(fc)
            sc = sc_c.score(fc, nc)
            smc = sm_c.process_chunk(sc)
            segs_c.extend(smc.segments)
        segs_c.extend(sm_c.flush())
        assert len(segs_c) == len(l4_segs), f"Chunk {chunk_f} lệch số đoạn: {len(segs_c)} vs {len(l4_segs)}"
        for (a0, a1), (b0, b1) in zip(segs_c, l4_segs):
            assert abs(a0 - b0) < 1e-9 and abs(a1 - b1) < 1e-9

    # 6. Chẩn đoán riêng cho câu 1 và câu 2 (đặc biệt cho kịch bản speech_at_start)
    times = np.arange(T) * hop_s
    u0_s, u0_e, _ = utt_intervals[0]
    u1_s, u1_e, _ = utt_intervals[1]
    u0_gt = gt_speech_frame & (times >= u0_s) & (times <= u0_e)
    u1_gt = gt_speech_frame & (times >= u1_s) & (times <= u1_e)
    u0_post_warm = (times >= max(u0_s, 40 * hop_s)) & (times <= u0_e)

    startup_diag = {
        "u0_interval": (u0_s, u0_e),
        "u1_interval": (u1_s, u1_e),
        "u0_post_warm_mean_snr_db": float(out.snr_db[:, u0_post_warm].mean()) if u0_post_warm.any() else 0.0,
        "u0_post_warm_mean_score": float(sig_out.score[u0_post_warm].mean()) if u0_post_warm.any() else 0.0,
        "u0_l4_recall_pct": float(l4_seg_mask[u0_gt].mean() * 100.0) if u0_gt.any() else 0.0,
        "u1_l4_recall_pct": float(l4_seg_mask[u1_gt].mean() * 100.0) if u1_gt.any() else 0.0,
        "first_l4_seg": l4_segs[0] if l4_segs else None,
    }

    return {
        "T": T,
        "dur_s": T * hop_s,
        "utt_upd_rate_pct": (utt_upd.mean(axis=1) * 100.0),
        "noise_upd_rate_pct": (upd[:, noise_v].mean(axis=1) * 100.0),
        "share_pause_pct": (grp_pause_cnt / utt_upd_total * 100.0),
        "share_band_silent_pct": (grp_band_silent_cnt / utt_upd_total * 100.0),
        "share_leak_pct": (grp_leak_cnt / utt_upd_total * 100.0),
        "leak_over_utt_frames_pct": (grp_leak_cnt / utt_frames_cnt * 100.0),
        "med_abs_floor_err_db": med_abs_floor_err_db,
        "bias_floor_in_speech_db": bias_floor_in_speech_db,
        "raw_miss_pct": l2_metrics["miss_pct"],
        "raw_fa_pct": l2_metrics["fa_pct"],
        "n_gt_segs": l2_metrics["n_gt_segs"],
        "n_pred_segs": l2_metrics["n_pred_segs"],
        "n_matched_segs": l2_metrics["n_matched_segs"],
        "onset_err_med_ms": l2_metrics["onset_err_med_ms"],
        "offset_err_med_ms": l2_metrics["offset_err_med_ms"],
        "l2": l2_metrics,
        "l3": l3_metrics,
        "l4": l4_metrics,
        "l4_imm": l4_imm_metrics,
        "startup_diag": startup_diag,
    }


def _fmt_row(label, m):
    return (
        f"    {label:20s} | Miss={m['miss_pct']:5.1f}% | FA={m['fa_pct']:5.1f}% | "
        f"Prec={m['precision_pct']:5.1f}% | Rec={m['recall_pct']:5.1f}% | F1={m['f1_pct']:5.1f}% | "
        f"Dur(pred/ref)={m['pred_speech_s']:4.1f}s/{m['ref_speech_s']:4.1f}s | "
        f"Segs(match/pred/gt)={m['n_matched_segs']:2d}/{m['n_pred_segs']:2d}/{m['n_gt_segs']:2d} | "
        f"|OnErr|={m['onset_err_med_ms']:3.0f}ms | |OffErr|={m['offset_err_med_ms']:3.0f}ms"
    )


if __name__ == "__main__":
    scenarios = ["stationary_20db", "stationary_15db", "stationary_10db", "step_up_down", "speech_at_start"]
    for sr in (16000, 8000):
        print(f"\n==================================================================")
        print(f"BENCHMARK END-TO-END (TẦNG 1 -> TẦNG 4) — SAMPLE RATE: {sr} Hz")
        print(f"==================================================================")
        clean_cores, noise_pieces = collect_benchmark_materials(sr=sr)
        print(
            f"Đã chọn {len(clean_cores)} đoạn phát ngôn (độ chênh năng lượng p95/p05 >= 35 dB) "
            f"& {len(noise_pieces)} nguồn nhiễu thực (-N)."
        )

        agg = {
            "l2": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
            "l3_0.3": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
            "l3_0.5": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
            "l3_0.7": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
            "l4": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
            "l4_imm": {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "pred_segs": 0, "gt_segs": 0, "matched": 0},
        }

        for scen in scenarios:
            s_w, n_w, m_w, utt_ivs = build_benchmark_stream(clean_cores, noise_pieces, sr, scen)
            res = evaluate_stream(s_w, n_w, m_w, utt_ivs, sr)
            print(f"\n--- Kịch bản: {scen} (thời lượng {res['dur_s']:.1f}s, {res['T']} frames, {len(utt_ivs)} câu ghép) ---")
            print(f"  1) Sai số tuyệt đối trung vị của noise_floor so với nền thật (dB): {np.round(res['med_abs_floor_err_db'], 2)}")
            print(f"  2) Độ lệch noise_floor trong lúc đang phát âm (Speech Bias dB)  : {np.round(res['bias_floor_in_speech_db'], 2)}")
            print(f"  3) So sánh 3 Baseline phát hiện (trên toàn bộ {res['T']} frame, bao gồm warm-up):")
            print(_fmt_row("L2 Raw (SNR>=9dB)", res["l2"]))
            print(_fmt_row("L3 Raw (p>=0.30)", res["l3"][0.3]))
            print(_fmt_row("L3 Raw (p>=0.50)", res["l3"][0.5]))
            print(_fmt_row("L3 Raw (p>=0.70)", res["l3"][0.7]))
            print(_fmt_row("L4 Final Segments", res["l4"]))
            print(_fmt_row("L4 Imm speech_mask", res["l4_imm"]))
            if scen == "speech_at_start":
                sd = res["startup_diag"]
                print(
                    f"  4) Chẩn đoán Startup (speech_at_start): "
                    f"Utt1={sd['u0_interval'][0]:.2f}..{sd['u0_interval'][1]:.2f}s "
                    f"(mean SNR sau warm-up={sd['u0_post_warm_mean_snr_db']:.2f}dB, mean p={sd['u0_post_warm_mean_score']:.2f}, "
                    f"L4 Recall Utt1={sd['u0_l4_recall_pct']:.1f}%) | "
                    f"Utt2={sd['u1_interval'][0]:.2f}..{sd['u1_interval'][1]:.2f}s "
                    f"(L4 Recall Utt2={sd['u1_l4_recall_pct']:.1f}%) | "
                    f"Đoạn L4 đầu tiên={sd['first_l4_seg']}"
                )

            for key, m_dict in [
                ("l2", res["l2"]),
                ("l3_0.3", res["l3"][0.3]),
                ("l3_0.5", res["l3"][0.5]),
                ("l3_0.7", res["l3"][0.7]),
                ("l4", res["l4"]),
                ("l4_imm", res["l4_imm"]),
            ]:
                for k_cnt in ("tp", "fn", "fp", "tn"):
                    agg[key][k_cnt] += m_dict[k_cnt]
                agg[key]["pred_segs"] += m_dict["n_pred_segs"]
                agg[key]["gt_segs"] += m_dict["n_gt_segs"]
                agg[key]["matched"] += m_dict["n_matched_segs"]

        print(f"\n=== TỔNG HỢP 5 KỊCH BẢN (AGGREGATE POOLED FRAMES) — {sr} Hz ===")
        hop_s = 0.005
        for key, label in [
            ("l2", "L2 Raw (SNR>=9dB)"),
            ("l3_0.3", "L3 Raw (p>=0.30)"),
            ("l3_0.5", "L3 Raw (p>=0.50)"),
            ("l3_0.7", "L3 Raw (p>=0.70)"),
            ("l4", "L4 Final Segments"),
            ("l4_imm", "L4 Imm speech_mask"),
        ]:
            d = agg[key]
            tp, fn, fp, tn = d["tp"], d["fn"], d["fp"], d["tn"]
            miss = fn / (tp + fn) * 100.0
            fa = fp / (fp + tn) * 100.0
            rec = tp / (tp + fn) * 100.0
            prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
            f1 = 2.0 * tp / (2 * tp + fp + fn) * 100.0
            print(
                f"  {label:20s} | Miss={miss:5.1f}% | FA={fa:5.1f}% | Prec={prec:5.1f}% | "
                f"Rec={rec:5.1f}% | F1={f1:5.1f}% | Dur(pred/ref)={(tp+fp)*hop_s:5.1f}s/{(tp+fn)*hop_s:5.1f}s | "
                f"Segs(match/pred/gt)={d['matched']:2d}/{d['pred_segs']:3d}/{d['gt_segs']:2d}"
            )

