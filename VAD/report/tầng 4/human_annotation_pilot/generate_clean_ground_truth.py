"""
VAD/report/tầng 4/human_annotation_pilot/generate_clean_ground_truth.py

Trích xuất tín hiệu tiếng nói sạch gốc (Clean Speech Signals - s_wave) tương ứng
với 11 đoạn trích của gói Pilot, xuất ra 11 file WAV sạch và sinh file đáp án chuẩn mẫu
`clean_ground_truth.csv` trực tiếp từ năng lượng tiếng nói sạch (không phụ thuộc vào Silero).
"""
from __future__ import annotations

import csv
import os
import sys
import wave
from typing import Dict, List, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PILOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PILOT_DIR, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from VAD.feature_extraction import FeatureExtractor  # noqa: E402
from benchmark.run_vad_benchmark import (  # noqa: E402
    build_benchmark_stream,
    build_clean_ground_truth,
    collect_benchmark_materials,
    mask_to_segments,
    segments_to_frame_mask,
)

CLEAN_WAV_DIR = os.path.join(PILOT_DIR, "wav_clean")
OUT_CLEAN_CSV = os.path.join(PILOT_DIR, "clean_ground_truth.csv")
MANIFEST_PATH = os.path.join(PILOT_DIR, "pilot_manifest.csv")


def export_clean_speech_wavs_and_ground_truth(
    manifest_path: str = MANIFEST_PATH,
    clean_wav_dir: str = CLEAN_WAV_DIR,
    out_csv: str = OUT_CLEAN_CSV,
    bridge_gap_s: float = 0.15,
    min_dur_s: float = 0.03,
) -> Tuple[List[str], str]:
    """
    1. Tái tạo luồng tiếng nói sạch `s_wave` (trước khi trộn nhiễu) từ vật liệu FPT gốc.
    2. Cắt và ghi 11 file WAV sạch vào `wav_clean/`.
    3. Trích xuất các đoạn tiếng nói chuẩn `[start_time_s, end_time_s)` từ `build_clean_ground_truth()`
       và tạo chuỗi khoảng nửa mở liên tiếp không chồng lấn phủ kín `[0.000, duration_s)`.
    4. Ghi ra `clean_ground_truth.csv`.
    """
    from importlib import util
    spec_pt = util.spec_from_file_location("pilot_tools", os.path.join(PILOT_DIR, "pilot_tools.py"))
    pilot_tools = util.module_from_spec(spec_pt)
    sys.modules["pilot_tools"] = pilot_tools
    spec_pt.loader.exec_module(pilot_tools)

    manifest = pilot_tools.load_manifest(manifest_path)
    os.makedirs(clean_wav_dir, exist_ok=True)

    materials_cache: Dict[int, tuple] = {}
    stream_cache: Dict[Tuple[int, str, int], Tuple[np.ndarray, list, Any]] = {}

    csv_rows = []
    exported_wavs = []

    print("=" * 96)
    print("XUẤT BẢN THU TIẾNG NÓI SẠCH GỐC & SINH ĐÁP ÁN CHUẨN MẪU (CLEAN GROUND TRUTH)")
    print("=" * 96)
    print(f"{'Excerpt ID':18s} | {'SR (Hz)':>7s} | {'Dur (s)':>7s} | {'Clean Speech Segs':>18s} | {'Clean Speech Dur':>16s}")
    print("-" * 96)

    for eid, spec in manifest.items():
        sr = spec.sample_rate
        if sr not in materials_cache:
            materials_cache[sr] = collect_benchmark_materials(sr=sr)
        clean_cores, noise_pieces = materials_cache[sr]

        stream_key = (sr, spec.scenario, spec.seed)
        if stream_key not in stream_cache:
            s_wave, _, _, utt_intervals = build_benchmark_stream(
                clean_cores, noise_pieces, sr, spec.scenario, seed=spec.seed
            )
            fe = FeatureExtractor(sample_rate=sr)
            feats_s = fe.process_batch(s_wave)
            bins_per_band = (fe.band_weights > 0).sum(axis=1)
            gt_info = build_clean_ground_truth(feats_s, utt_intervals, bins_per_band, fe.hop_seconds)
            stream_cache[stream_key] = (s_wave, utt_intervals, gt_info)

        s_wave, utt_intervals, gt_info = stream_cache[stream_key]

        # Cắt đúng đoạn trích âm thanh sạch tương ứng
        s_idx = int(round(spec.stream_start_s * sr))
        e_idx = int(round(spec.stream_end_s * sr))
        clean_audio = s_wave[s_idx:e_idx]

        # Ghi file WAV PCM 16-bit sạch
        peak = float(np.max(np.abs(clean_audio))) if clean_audio.size else 0.0
        scale = 0.95 / peak if peak > 0.95 else 1.0
        pcm16 = np.clip(np.round(clean_audio * scale * 32767.0), -32768, 32767).astype(np.int16)

        clean_wav_name = f"{eid}_clean.wav"
        clean_wav_path = os.path.join(clean_wav_dir, clean_wav_name)
        with wave.open(clean_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        exported_wavs.append(clean_wav_path)

        # Trích xuất các đoạn SPEECH chuẩn từ gt_info['gt_speech_frame'] trong phạm vi [stream_start_s, stream_end_s)
        # Để đảm bảo tính chính xác tuyệt đối ở cấp độ frame của đoạn trích:
        T_stream = gt_info['gt_speech_frame'].shape[0]
        hop_s = 0.005
        f_start = int(round(spec.stream_start_s / hop_s))
        f_end = int(round(spec.stream_end_s / hop_s))
        f_end = min(T_stream, f_end)
        
        excerpt_clean_mask = gt_info['gt_speech_frame'][f_start:f_end]
        
        # Nối khoảng nghỉ ngắn <= bridge_gap_s (150ms) và lọc gai < min_dur_s (30ms) như chuẩn VAD
        raw_segs = mask_to_segments(excerpt_clean_mask, hop_s, bridge_gap_s=bridge_gap_s, min_dur_s=min_dur_s)
        
        # Chuyển sang danh sách khoảng phủ kín [0.000, duration_s)
        dur_q = round(spec.duration_s, 3)
        cursor = 0.0
        excerpt_rows = []
        for s_t, e_t in raw_segs:
            s_q = max(0.0, min(dur_q, round(float(s_t), 3)))
            e_q = max(0.0, min(dur_q, round(float(e_t), 3)))
            if e_q <= s_q:
                continue
            if s_q > cursor + 1e-9:
                excerpt_rows.append([eid, f"{cursor:.3f}", f"{s_q:.3f}", "NON_SPEECH", "clean_speech_energy_ground_truth", "Pure silence/background gap in clean source"])
            excerpt_rows.append([eid, f"{s_q:.3f}", f"{e_q:.3f}", "SPEECH", "clean_speech_energy_ground_truth", "Active human speech confirmed on clean source signal"])
            cursor = e_q
        if cursor < dur_q - 1e-9 or not excerpt_rows:
            excerpt_rows.append([eid, f"{cursor:.3f}", f"{dur_q:.3f}", "NON_SPEECH", "clean_speech_energy_ground_truth", "Pure silence/background gap in clean source"])

        csv_rows.extend(excerpt_rows)

        sp_dur = sum(float(r[2]) - float(r[1]) for r in excerpt_rows if r[3] == "SPEECH")
        sp_cnt = sum(1 for r in excerpt_rows if r[3] == "SPEECH")
        print(f"{eid:18s} | {sr:7d} | {dur_q:7.3f} | {sp_cnt:18d} | {sp_dur:15.3f}s")

    # Ghi file CSV đáp án mẫu
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["excerpt_id", "start_time_s", "end_time_s", "label", "annotator_id", "notes"])
        for r in csv_rows:
            writer.writerow(r)

    print("-" * 96)
    print(f"Đã xuất {len(exported_wavs)} file WAV tiếng nói sạch vào: {clean_wav_dir}")
    print(f"Đã ghi đáp án mẫu chuẩn (Clean Ground Truth) vào: {out_csv}")
    print("=" * 96)

    # Kiểm tra hợp lệ qua pilot_tools
    loaded_ivs = pilot_tools.load_annotations_csv(out_csv)
    pilot_tools.validate_annotations(loaded_ivs, manifest)
    print(f"XÁC THỰC THÀNH CÔNG: Đã kiểm tra {len(loaded_ivs)} khoảng gán nhãn trong {out_csv} đạt chuẩn.")

    return exported_wavs, out_csv


if __name__ == "__main__":
    export_clean_speech_wavs_and_ground_truth()
