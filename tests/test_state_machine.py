import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VAD.feature_extraction import FeatureExtractor, VADFeatures  # noqa: E402
from VAD.noise_estimator import NoiseEstimator                    # noqa: E402
from VAD.sigmoid import SigmoidOutput, SigmoidScorer              # noqa: E402
from VAD.state_machine import (                                   # noqa: E402
    StateMachineConfig,
    StateMachineOutput,
    VADState,
    VADStateMachine,
)


def make_sig_out(score: np.ndarray, warmup: np.ndarray | None = None) -> SigmoidOutput:
    s = np.asarray(score, dtype=np.float32)
    T = s.shape[0]
    w = np.zeros(T, dtype=bool) if warmup is None else np.asarray(warmup, dtype=bool)
    return SigmoidOutput(
        score=s,
        logit=np.zeros(T, dtype=np.float32),
        snr_norm=np.zeros((4, T), dtype=np.float32),
        flatness_complement=np.zeros(T, dtype=np.float32),
        warmup=w,
    )


def test_state_transitions_and_hysteresis():
    """Kiểm tra đúng chu trình SILENCE -> ONSET -> SPEECH -> HANGOVER -> SILENCE và hysteresis."""
    # hop = 10 ms (0.01s) để nhẩm frame dễ dàng:
    # onset_ms=30ms -> 3 frame; hangover_ms=30ms -> 3 frame; min_speech=20ms -> 2 frame; min_silence=0ms -> 0 frame
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=30.0,
        hangover_ms=30.0,
        min_speech_ms=20.0,
        min_silence_ms=0.0,
    )
    sm = VADStateMachine(hop_seconds=0.01, config=cfg)

    # Frame 0..1: 0.2 (< onset_thresh) -> SILENCE
    # Frame 2..3: 0.7 (>= onset_thresh, 2 frame đầu) -> ONSET
    # Frame 4   : 0.7 (đủ 3 frame liên tiếp) -> SPEECH (đoạn bắt đầu từ frame 2)
    # Frame 5..6: 0.5 (nằm giữa offset_thresh=0.4 và onset_thresh=0.6 -> nhờ hysteresis vẫn giữ SPEECH)
    # Frame 7..9: 0.2 (< offset_thresh, đúng 3 frame hangover_frames=3) -> HANGOVER
    # Frame 10  : 0.2 (sau khi đã hết 3 frame hangover ở 7,8,9 -> hết hạn tại frame 10, đóng đoạn tại end=10, về SILENCE)
    # Frame 11  : 0.1 -> SILENCE
    scores = np.array(
        [0.2, 0.2, 0.7, 0.7, 0.7, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.1], dtype=np.float32
    )
    out = sm.process_chunk(make_sig_out(scores))
    tail = sm.flush()
    all_segs = out.segments + tail

    expected_states = np.array(
        [
            VADState.SILENCE,   # 0
            VADState.SILENCE,   # 1
            VADState.ONSET,     # 2 (onset_count=1)
            VADState.ONSET,     # 3 (onset_count=2)
            VADState.SPEECH,    # 4 (onset_count=3 -> SPEECH, start=2)
            VADState.SPEECH,    # 5 (hysteresis: 0.5 >= 0.4)
            VADState.SPEECH,    # 6 (hysteresis: 0.5 >= 0.4)
            VADState.HANGOVER,  # 7 (hangover frame 1/3)
            VADState.HANGOVER,  # 8 (hangover frame 2/3)
            VADState.HANGOVER,  # 9 (hangover frame 3/3)
            VADState.SILENCE,   # 10 (hangover hết hạn -> đóng đoạn tại exclusive end=10, về SILENCE)
            VADState.SILENCE,   # 11
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(out.states, expected_states)
    assert len(all_segs) == 1
    assert all_segs[0] == pytest.approx((0.02, 0.10))


@pytest.mark.parametrize("hangover_frames", [0, 1, 3])
def test_hangover_boundary_exact_frame_counts(hangover_frames):
    """
    Phase 1 BẮT BUỘC: Kiểm tra chính xác biên hangover cho:
      - hangover_frames = 0
      - hangover_frames = 1
      - hangover_frames > 1 (ví dụ 3)
    Với 3 frame tiếng nói tại [0, 1, 2] (first below-offset frame tại index 3):
      - Số frame mang trạng thái HANGOVER phải bằng đúng `hangover_frames` (tại 3 .. 3 + hangover_frames - 1).
      - Ngay tại frame `3 + hangover_frames`, trạng thái phải chuyển về SILENCE.
      - Chỉ số kết thúc đoạn (exclusive end_frame) phải bằng đúng `3 + hangover_frames`.
    """
    hop_s = 0.01  # 10 ms / frame
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=20.0,                        # 2 frame
        hangover_ms=hangover_frames * 10.0,   # 0, 1, hoặc 3 frame
        min_speech_ms=20.0,                   # 2 frame
        min_silence_ms=0.0,
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)
    assert sm.hangover_frames == hangover_frames

    # Frame 0..2: 0.8 (3 frame speech evidence: frame 0=ONSET, frame 1=SPEECH, frame 2=SPEECH)
    # Frame 3..8: 0.1 (6 frame dưới offset_thresh; frame 3 là frame dưới ngưỡng đầu tiên)
    scores = np.array([0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    # Kiểm tra số lượng và vị trí các frame HANGOVER
    hangover_indices = np.where(out.states == int(VADState.HANGOVER))[0]
    expected_hangover_indices = np.arange(3, 3 + hangover_frames, dtype=int)
    np.testing.assert_array_equal(hangover_indices, expected_hangover_indices)

    # Ngay tại frame 3 + hangover_frames trở đi phải là SILENCE
    assert (out.states[3 + hangover_frames:] == int(VADState.SILENCE)).all()

    # Exclusive end_frame phải bằng đúng 3 + hangover_frames
    expected_end_frame = 3 + hangover_frames
    assert len(segs) == 1
    assert segs[0] == pytest.approx((0.0, expected_end_frame * hop_s))


@pytest.mark.parametrize("hangover_frames", [1, 3])
def test_score_rising_at_hangover_expiry_boundary_starts_new_onset(hangover_frames):
    """
    Stage C1.2 (Mục 1 & 2): Kiểm tra hồi quy tại đúng biên hết hạn hangover (g = g0 + H):
    - Một đoạn tiếng nói hợp lệ được xác lập tại frame 0..2 (3 frame >= min_speech=3).
    - Có đúng H frame liên tiếp (g0 .. g0 + H - 1, với g0 = 3) có điểm dưới offset_thresh -> HANGOVER.
    - Ngay tại frame g0 + H (biên hết hạn hangover), điểm vọt lên trên onset_thresh (0.85),
      sau đó tụt về 0.1 (chỉ có 1 frame cao < onset_frames=3):
        + Frame g0 + H KHÔNG ĐƯỢC khôi phục (revive) trực tiếp đoạn cũ về SPEECH.
        + Với onset_frames = 3 > 1, frame g0 + H bắt buộc phải mở một ứng viên ONSET mới
          (out.states[g0 + H] == VADState.ONSET).
        + Vì ứng viên ONSET mới chỉ dài 1 frame (< 3 frame) rồi tắt, nó bị hủy, và đoạn cũ
          đã đóng chính xác tại exclusive end_frame = g0 + H (tức (0.0, (3 + H) * hop_s)).
    - Ngoài ra, nếu từ g0 + H có đủ 3 frame >= onset_thresh để xác nhận đoạn mới và min_silence_ms > 0,
      hậu xử lý pending-segment và min_silence vẫn hoạt động bình thường.
    """
    hop_s = 0.01  # 10 ms / frame
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=30.0,                        # 3 frame (> 1)
        hangover_ms=hangover_frames * 10.0,   # H = 1 hoặc 3 frame
        min_speech_ms=30.0,                   # 3 frame
        min_silence_ms=20.0,                  # 2 frame
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)
    g0 = 3
    expiry_idx = g0 + hangover_frames

    # Trường hợp A: Tại đúng biên expiry_idx (g0 + H), chỉ có 1 frame vọt lên 0.85 rồi tắt về 0.1
    scores = np.full(expiry_idx + 6, 0.1, dtype=np.float32)
    scores[0:g0] = 0.85          # Frame 0..2: xác nhận SPEECH (start=0, 3 frame bằng chứng)
    scores[expiry_idx] = 0.85    # Frame g0 + H: vọt lên > onset_thresh ngay sau khi H frame hangover vừa hết

    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    # H frame từ g0 .. g0 + H - 1 là HANGOVER
    np.testing.assert_array_equal(
        out.states[g0:expiry_idx], np.full(hangover_frames, VADState.HANGOVER, dtype=np.int8)
    )
    # Tại đúng frame g0 + H: bắt buộc là ONSET mới (KHÔNG phải SPEECH!)
    assert out.states[expiry_idx] == int(VADState.ONSET)
    # Frame kế tiếp tụt về 0.1 -> hủy ONSET mới về SILENCE
    assert out.states[expiry_idx + 1] == int(VADState.SILENCE)
    # Đoạn cũ kết thúc chính xác tại exclusive end_frame = g0 + H = expiry_idx
    assert len(segs) == 1
    assert segs[0] == pytest.approx((0.0, expiry_idx * hop_s))

    # Trường hợp B: Từ expiry_idx có đủ 3 frame >= onset_thresh để thành đoạn thứ hai
    # -> Vì đoạn 1 đóng tại expiry_idx và đoạn 2 bắt đầu tại expiry_idx (gap = 0 <= min_silence_frames = 2),
    # cơ chế _pending_seg và min_silence_ms nối liền bình thường thành [0, expiry_idx + 3 + H)
    sm.reset()
    scores_merge = np.full(expiry_idx + 3 + hangover_frames + 4, 0.1, dtype=np.float32)
    scores_merge[0:g0] = 0.85
    scores_merge[expiry_idx:expiry_idx + 3] = 0.85
    out_merge = sm.process_scores(scores_merge)
    segs_merge = out_merge.segments + sm.flush()

    # Tại expiry_idx và expiry_idx + 1 vẫn phải đi qua ONSET trước khi vào SPEECH tại expiry_idx + 2
    assert out_merge.states[expiry_idx] == int(VADState.ONSET)
    assert out_merge.states[expiry_idx + 1] == int(VADState.ONSET)
    assert out_merge.states[expiry_idx + 2] == int(VADState.SPEECH)
    assert len(segs_merge) == 1
    assert segs_merge[0] == pytest.approx((0.0, (expiry_idx + 3 + hangover_frames) * hop_s))


def test_score_rising_on_final_hangover_frame_before_expiry_resumes_speech():
    """
    Stage C1.2 (Mục 3): Kiểm tra đối ứng khi điểm tăng trở lại (>= offset_thresh) ngay tại frame
    HANGOVER cuối cùng (g = g0 + H - 1, tức TRƯỚC khi hết hạn H frame):
    - Đoạn ứng viên đang mở được phép khôi phục trực tiếp về SPEECH mà không cần qua ONSET.
    """
    hop_s = 0.01  # 10 ms / frame
    H = 3
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=30.0,       # 3 frame
        hangover_ms=30.0,    # H = 3 frame (g0, g0+1, g0+2)
        min_speech_ms=30.0,  # 3 frame
        min_silence_ms=0.0,
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)

    # Frame 0..2 : 0.85 -> ONSET (0, 1), SPEECH (2)
    # Frame 3..4 : 0.10 -> 2 frame HANGOVER đầu (g0=3, g0+1=4)
    # Frame 5    : 0.50 (>= offset_thresh=0.4, dù < onset_thresh=0.6) ngay tại frame HANGOVER cuối (g0+H-1=5)
    #              -> khôi phục trực tiếp về SPEECH tại frame 5!
    # Frame 6..8 : 0.10 -> 3 frame HANGOVER mới (6, 7, 8), đóng đoạn tại exclusive end_frame = 9
    # Frame 9    : 0.10 -> SILENCE
    scores = np.array([0.85, 0.85, 0.85, 0.10, 0.10, 0.50, 0.10, 0.10, 0.10, 0.10], dtype=np.float32)
    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    assert out.states[3] == int(VADState.HANGOVER)
    assert out.states[4] == int(VADState.HANGOVER)
    assert out.states[5] == int(VADState.SPEECH)  # Khôi phục trực tiếp về SPEECH trước khi hết hạn!
    assert (out.states[6:9] == int(VADState.HANGOVER)).all()
    assert out.states[9] == int(VADState.SILENCE)
    assert len(segs) == 1
    assert segs[0] == pytest.approx((0.0, 9 * hop_s))


def test_min_speech_ms_measures_qualifying_speech_evidence_not_hangover():
    """
    Phase 2 BẮT BUỘC: Đánh giá chuỗi điểm vượt ngưỡng đúng bằng thời lượng onset tối thiểu
    rồi lập tức tụt xuống dưới offset_thresh.
    - Cấu hình: hop=10ms, onset_ms=30ms (3 frame), hangover_ms=150ms (15 frame), min_speech_ms=100ms (10 frame).
    - Nếu một xung chỉ đạt ngưỡng trong đúng 3 frame (30 ms < min_speech_ms=100 ms) rồi tắt,
      dù máy trạng thái ngân thêm 15 frame hangover (tổng cửa sổ 18 frame = 180 ms > 100 ms),
      bằng chứng tiếng nói thực sự chỉ có 3 frame (30 ms < 100 ms) nên BẮT BUỘC phải bị loại bỏ.
    - Ngược lại, khi bằng chứng tiếng nói thực sự đạt >= 10 frame (100 ms), đoạn mới được giữ lại
      và phát ra kèm đuôi hangover đầy đủ.
    """
    hop_s = 0.01  # 10 ms / frame
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=30.0,       # 3 frame
        hangover_ms=150.0,   # 15 frame
        min_speech_ms=100.0, # 10 frame
        min_silence_ms=20.0, # 2 frame
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)

    # Kịch bản 1: Chỉ có đúng 3 frame đạt ngưỡng (frame 0..2), sau đó 25 frame = 0.1
    scores_short = np.zeros(28, dtype=np.float32)
    scores_short[0:3] = 0.85
    out_short = sm.process_scores(scores_short)
    segs_short = out_short.segments + sm.flush()
    # Trạng thái vẫn đi qua ONSET (0,1) -> SPEECH (2) -> HANGOVER (3..17) -> SILENCE (18..27),
    # nhưng đoạn bị loại bỏ vì speech_evidence_frames = 3 < min_speech_frames = 10
    assert out_short.states[2] == int(VADState.SPEECH)
    assert (out_short.states[3:18] == int(VADState.HANGOVER)).all()
    assert out_short.states[18] == int(VADState.SILENCE)
    assert segs_short == []

    # Kịch bản 2: Có đủ 10 frame bằng chứng tiếng nói (frame 0..9), sau đó 20 frame = 0.1
    sm.reset()
    scores_valid = np.zeros(30, dtype=np.float32)
    scores_valid[0:10] = 0.85
    out_valid = sm.process_scores(scores_valid)
    segs_valid = out_valid.segments + sm.flush()
    assert len(segs_valid) == 1
    # Đoạn hợp lệ bắt đầu từ frame 0 và kết thúc tại 10 + 15 = frame 25 (tức 0.00s -> 0.25s)
    assert segs_valid[0] == pytest.approx((0.0, 25 * hop_s))


def test_short_score_dips_during_speech_held_by_hangover():
    """Hố điểm ngắn trong lúc nói (< hangover_frames) phải quay lại SPEECH, không làm đứt đoạn."""
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=20.0,     # 4 frame @ 5 ms
        hangover_ms=30.0,  # 6 frame @ 5 ms
        min_speech_ms=20.0,
        min_silence_ms=0.0,
    )
    sm = VADStateMachine(hop_seconds=0.005, config=cfg)

    scores = np.concatenate(
        [
            np.full(10, 0.8, dtype=np.float32),
            np.full(4, 0.1, dtype=np.float32),
            np.full(6, 0.8, dtype=np.float32),
        ]
    )
    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    assert (out.states[10:14] == int(VADState.HANGOVER)).all()
    assert (out.states[14:] == int(VADState.SPEECH)).all()
    assert len(segs) == 1
    assert segs[0] == pytest.approx((0.0, 20 * 0.005))


def test_short_noise_bursts_during_silence_rejected():
    """Gai nhiễu ngắn (< onset_frames) chỉ lên ONSET rồi hủy về SILENCE, không tạo đoạn."""
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=25.0,     # 5 frame @ 5 ms
        hangover_ms=50.0,
        min_speech_ms=25.0,
        min_silence_ms=20.0,
    )
    sm = VADStateMachine(hop_seconds=0.005, config=cfg)

    scores = np.array(
        [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.1], dtype=np.float32
    )
    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    assert not (out.states == int(VADState.SPEECH)).any()
    assert out.states[6] == int(VADState.SILENCE)
    assert out.states[12] == int(VADState.SILENCE)
    assert segs == []


def test_speech_at_stream_start_warmup_contract_and_end_to_end_constant_power():
    """
    Phase 3 & Phase 4 BẮT BUỘC:
    1) Kiểm tra hợp đồng warm-up độc lập trên التầng 4: khi warmup[:40]=True và tiếng nói tiếp tục sau frame 40,
       đoạn phát hiện bị trễ đúng 40 frame (200 ms), bắt đầu tại t = 0.200 s.
    2) Kiểm tra tích hợp End-to-End (Tầng 1 -> Tầng 2 -> Tầng 3 -> Tầng 4) khi âm thanh đầu vào chứa
       tín hiệu kiểu tiếng nói có MỨC CÔNG SUẤT KHÔNG ĐỔI ngay từ t = 0 s:
       - Quan sát trực tiếp SNR của Tầng 2, score của Tầng 3 và state của Tầng 4 sau warm-up.
       - Vì 40 frame warm-up đầu chứa chính tín hiệu công suất không đổi đó, Tầng 2 khởi tạo `noise_floor`
         bằng median của mức công suất tiếng nói -> sau frame 40, `snr_db` xoay quanh ~0 dB, `score` của Tầng 3
         dưới ngưỡng `onset_thresh` (0.55) -> Tầng 4 giữ nguyên ở `SILENCE` và BỎ SÓT (MISS) hoàn toàn đoạn nói
         đầu tiên đó cho đến khi có khoảng lặng để `noise_floor` hạ xuống nền nhiễu thật.
    """
    hop_s = 0.005
    cfg = StateMachineConfig(
        onset_thresh=0.55,
        offset_thresh=0.35,
        onset_ms=30.0,      # 6 frame @ 5 ms
        hangover_ms=100.0,  # 20 frame @ 5 ms
        min_speech_ms=50.0,
        min_silence_ms=50.0,
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)

    # Phần A: Tầng 4 với score cao sau warm-up -> phát hiện bị trễ 200 ms (mất 40 frame đầu)
    T = 300
    scores = np.zeros(T, dtype=np.float32)
    scores[:200] = 0.95
    warmup = np.zeros(T, dtype=bool)
    warmup[:40] = True

    out = sm.process_scores(scores, warmup=warmup)
    segs = out.segments + sm.flush()

    assert (out.states[:40] == int(VADState.SILENCE)).all()
    assert not out.speech_mask[:40].any()
    assert (out.states[40:45] == int(VADState.ONSET)).all()
    assert out.states[45] == int(VADState.SPEECH)
    # Với hangover_frames = 20 và first below-offset frame tại index 200 -> exclusive end = 200 + 20 = 220
    assert len(segs) == 1
    assert segs[0] == pytest.approx((40 * hop_s, 220 * hop_s))

    # Phần B (Phase 4): Tích hợp End-to-End Tầng 1 -> Tầng 2 -> Tầng 3 -> Tầng 4 từ waveform
    # Câu nói 1 (0.0s .. 1.0s, mức công suất không đổi ngay từ đầu stream) -> bị MISS do nhiễm warm-up!
    # Khoảng lặng (1.0s .. 2.0s, chỉ có nhiễu nền nhỏ) -> Tầng 2 hạ noise_floor về mức nhiễu thật.
    # Câu nói 2 (2.0s .. 3.0s, cùng mức công suất như câu 1) -> được PHÁT HIỆN đầy đủ!
    sr = 16000
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(sr * 3) * 0.005).astype(np.float32)
    t_1s = np.arange(sr) / sr
    harmonic_speech = (
        0.15 * np.sin(2 * np.pi * 200 * t_1s)
        + 0.20 * np.sin(2 * np.pi * 600 * t_1s)
        + 0.15 * np.sin(2 * np.pi * 1500 * t_1s)
        + 0.10 * np.sin(2 * np.pi * 2500 * t_1s)
    ).astype(np.float32)
    audio[0:sr] += harmonic_speech          # Câu 1 tại [0.0s, 1.0s] (bắt đầu ngay tại t=0)
    audio[2 * sr:3 * sr] += harmonic_speech # Câu 2 tại [2.0s, 3.0s] (sau khi nền nhiễu đã hồi phục ở [1.0s, 2.0s])

    fe = FeatureExtractor(sample_rate=sr)
    est = NoiseEstimator(num_bands=fe.num_bands)
    scorer = SigmoidScorer(num_bands=fe.num_bands)
    sm_e2e = VADStateMachine(hop_seconds=fe.hop_seconds, config=cfg)

    feats = fe.process_batch(audio)
    nout = est.update(feats)
    sout = scorer.score(feats, nout)
    sm_out = sm_e2e.process_chunk(sout)
    e2e_segs = sm_out.segments + sm_e2e.flush()

    # Kiểm chứng dữ liệu thực tế ở Câu 1 sau warm-up (frames 40..190, tức 0.20s .. 0.95s):
    # 1) Do noise_floor khởi tạo bằng median của chính Câu 1, SNR trung bình sau warm-up xấp xỉ 0 dB (< 1 dB)
    utt1_post_warmup_snr = nout.snr_db[:, 40:190]
    assert np.abs(utt1_post_warmup_snr.mean()) < 1.0
    assert utt1_post_warmup_snr.max() < 3.0
    # 2) Dù SFM rất thấp (< 0.1), điểm Sigmoid ở Câu 1 chỉ đạt ~0.10..0.15 (< onset_thresh=0.55)
    utt1_post_warmup_score = sout.score[40:190]
    assert utt1_post_warmup_score.max() < 0.25
    # 3) Do đó trạng thái Tầng 4 trong suốt Câu 1 (frames 0..195) hoàn toàn ở SILENCE (Câu 1 bị MISS)
    assert (sm_out.states[:195] == int(VADState.SILENCE)).all()

    # Ngược lại, sau khoảng lặng [1.0s, 2.0s], ở Câu 2 (frames 410..580, tức 2.05s .. 2.90s):
    # SNR đạt > 25 dB, score đạt > 0.95, và Câu 2 được Tầng 4 phát hiện chính xác quanh [2.0s, 3.0s]
    assert nout.snr_db[:, 410:580].mean() > 25.0
    assert sout.score[410:580].mean() > 0.95
    assert len(e2e_segs) == 1
    assert 1.98 <= e2e_segs[0][0] <= 2.05
    assert 2.95 <= e2e_segs[0][1] <= 3.05


def test_min_speech_and_min_silence_post_processing():
    """Kiểm tra nối khoảng lặng ngắn (<= min_silence_ms) và loại bỏ đoạn quá ngắn (< min_speech_ms)."""
    hop_s = 0.01  # 10 ms / frame
    # onset=20ms (2f), hangover=20ms (2f), min_speech=50ms (5f speech evidence), min_silence=30ms (3f)
    cfg = StateMachineConfig(
        onset_thresh=0.6,
        offset_thresh=0.4,
        onset_ms=20.0,
        hangover_ms=20.0,
        min_speech_ms=50.0,
        min_silence_ms=30.0,
    )
    sm = VADStateMachine(hop_seconds=hop_s, config=cfg)

    # 1) Đoạn 1 (quá ngắn): frame 0..2 đạt 0.8 (3f speech evidence < min_speech=5f) + hangover frame 3..4 (2f) -> [0, 5)
    #    Khoảng lặng dài: frame 5..9 (5 frame > min_silence=3f) -> đoạn [0, 5) bị loại bỏ!
    # 2) Đoạn 2a: frame 10..12 đạt 0.8 (3f speech evidence) + hangover frame 13..14 (2f) -> ứng viên [10, 15)
    #    Khoảng lặng ngắn sau hangover: frame 15..17 (3 frame <= min_silence=3f)
    # 3) Đoạn 2b: frame 18..20 đạt 0.8 (3f speech evidence) + hangover frame 21..22 (2f) -> ứng viên [18, 23)
    #    Do khoảng cách 18 - 15 = 3 <= min_silence_frames(3), hai đoạn [10, 15) và [18, 23) nối thành [10, 23)
    #    với tổng bằng chứng tiếng nói = 3 + 3 = 6 frame >= min_speech(5f)!
    scores = np.zeros(30, dtype=np.float32)
    scores[0:3] = 0.8
    scores[10:13] = 0.8
    scores[18:21] = 0.8

    out = sm.process_scores(scores)
    segs = out.segments + sm.flush()

    assert len(segs) == 1
    assert segs[0] == pytest.approx((10 * hop_s, 23 * hop_s))


def test_empty_chunks_flush_and_reset():
    """Kiểm tra chunk rỗng T=0, chốt đoạn đang mở bằng flush(), và tính lặp lại sau reset()."""
    hop_s = 0.005
    sm = VADStateMachine(hop_seconds=hop_s)

    empty_out = sm.process_scores(np.zeros(0, dtype=np.float32))
    assert empty_out.num_frames == 0
    assert empty_out.segments == []
    assert empty_out.states.shape == (0,) and empty_out.states.dtype == np.int8
    assert empty_out.speech_mask.shape == (0,) and empty_out.speech_mask.dtype == bool
    assert sm.flush() == []

    scores = np.full(30, 0.9, dtype=np.float32)
    out1 = sm.process_scores(scores)
    assert out1.segments == []
    flushed1 = sm.flush()
    assert len(flushed1) == 1
    assert flushed1[0] == pytest.approx((0.0, 30 * hop_s))
    assert sm.state == VADState.SILENCE

    sm.reset()
    assert sm.frame_index == 0
    out2 = sm.process_scores(scores)
    flushed2 = sm.flush()
    assert flushed1 == flushed2
    np.testing.assert_array_equal(out1.states, out2.states)


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 13, 40, 111])
def test_chunk_invariance_across_onset_hangover_and_min_silence(chunk):
    """
    Kiểm tra tính bất biến theo kích thước chunk (chunk-invariance) khi biên chunk cắt ngang
    ONSET, SPEECH, HANGOVER, warm-up và khoảng lặng ngắn đang chờ nối (min_silence).
    """
    rng = np.random.default_rng(2026)
    T = 500
    hop_s = 0.005
    cfg = StateMachineConfig(
        onset_thresh=0.55,
        offset_thresh=0.35,
        onset_ms=30.0,      # 6 frame
        hangover_ms=75.0,   # 15 frame
        min_speech_ms=80.0, # 16 frame
        min_silence_ms=60.0,# 12 frame
    )

    scores = rng.uniform(0.05, 0.25, size=T).astype(np.float32)
    scores[50:54] = 0.85
    scores[70:110] = 0.80
    scores[90:98] = 0.20
    scores[132:165] = 0.85
    scores[220:227] = 0.75
    scores[280:360] = 0.90
    scores[430:495] = 0.75

    warmup = np.zeros(T, dtype=bool)
    warmup[:40] = True

    sm_batch = VADStateMachine(hop_seconds=hop_s, config=cfg)
    out_batch = sm_batch.process_scores(scores, warmup=warmup)
    segs_batch = out_batch.segments + sm_batch.flush()

    sm_stream = VADStateMachine(hop_seconds=hop_s, config=cfg)
    segs_stream = []
    states_parts = []
    mask_parts = []

    for i in range(0, T, chunk):
        if i % (chunk * 2) == 0:
            empty_res = sm_stream.process_scores(
                np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
            )
            segs_stream.extend(empty_res.segments)
        part = sm_stream.process_scores(scores[i:i + chunk], warmup=warmup[i:i + chunk])
        segs_stream.extend(part.segments)
        states_parts.append(part.states)
        mask_parts.append(part.speech_mask)

    segs_stream.extend(sm_stream.flush())

    np.testing.assert_array_equal(np.concatenate(states_parts), out_batch.states)
    np.testing.assert_array_equal(np.concatenate(mask_parts), out_batch.speech_mask)
    assert len(segs_stream) == len(segs_batch)
    for (s1, e1), (s2, e2) in zip(segs_stream, segs_batch):
        assert s1 == pytest.approx(s2, abs=1e-9)
        assert e1 == pytest.approx(e2, abs=1e-9)


def test_non_finite_scores_nan_pos_inf_neg_inf_both_debug_modes():
    """
    Phase 3 BẮT BUỘC: Kiểm tra NaN, +Inf, -Inf trong process_scores() với cả debug=False và debug=True.
    - Khi debug=False: mọi giá trị không hữu hạn (NaN, +Inf, -Inf) đều được coi là 0.0.
      + Tuyệt đối không để +Inf kích hoạt ONSET hay SPEECH từ SILENCE.
      + Khi đang ở SPEECH mà gặp chuỗi NaN / +Inf / -Inf, máy trạng thái phải đi vào HANGOVER và thoát về SILENCE.
    - Khi debug=True: bất kỳ NaN, +Inf hay -Inf nào trong score cũng phải lập tức ném ValueError.
    """
    cfg = StateMachineConfig(
        onset_thresh=0.55,
        offset_thresh=0.35,
        onset_ms=15.0,     # 3 frame @ 5 ms
        hangover_ms=15.0,  # 3 frame @ 5 ms
        min_speech_ms=15.0,
        min_silence_ms=10.0,
    )
    sm_prod = VADStateMachine(hop_seconds=0.005, config=cfg, debug=False)

    # 1) Đang ở SILENCE, đưa liên tiếp +Inf, -Inf, NaN -> tất cả coi là 0.0, giữ nguyên SILENCE, không tạo đoạn
    non_finite_burst = np.array(
        [np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, np.nan, np.nan], dtype=np.float32
    )
    out_burst = sm_prod.process_scores(non_finite_burst)
    assert (out_burst.states == int(VADState.SILENCE)).all()
    assert not out_burst.speech_mask.any()
    assert sm_prod.flush() == []

    # 2) Đang ở SPEECH (5 frame 0.8), sau đó gặp chuỗi [+Inf, np.nan, -Inf, +Inf] -> coi như 4 frame 0.0
    #    (3 frame HANGOVER + frame thứ 4 thoát về SILENCE)
    sm_prod.reset()
    speech_then_non_finite = np.array(
        [0.8, 0.8, 0.8, 0.8, 0.8, np.inf, np.nan, -np.inf, np.inf], dtype=np.float32
    )
    out_exit = sm_prod.process_scores(speech_then_non_finite)
    assert out_exit.states[4] == int(VADState.SPEECH)
    assert (out_exit.states[5:8] == int(VADState.HANGOVER)).all()
    assert out_exit.states[8] == int(VADState.SILENCE)

    # 3) Khi debug=True: từng loại NaN, +Inf, -Inf đều phải ném ValueError
    sm_dbg = VADStateMachine(hop_seconds=0.005, config=cfg, debug=True)
    for bad_val in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="NaN hoặc Inf"):
            sm_dbg.process_scores(np.array([0.2, bad_val, 0.2], dtype=np.float32))


def test_invalid_config_and_inputs():
    """Kiểm tra bắt lỗi cấu hình và shape/type đầu vào."""
    with pytest.raises(ValueError):
        VADStateMachine(hop_seconds=0.0)
    with pytest.raises(ValueError):
        VADStateMachine(hop_seconds=-0.005)
    with pytest.raises(ValueError):
        VADStateMachine(config=StateMachineConfig(onset_thresh=0.3, offset_thresh=0.6))
    with pytest.raises(ValueError):
        VADStateMachine(config=StateMachineConfig(onset_thresh=1.0, offset_thresh=0.5))
    with pytest.raises(ValueError):
        VADStateMachine(config=StateMachineConfig(onset_ms=0.0))
    with pytest.raises(ValueError):
        VADStateMachine(config=StateMachineConfig(hangover_ms=-10.0))

    sm_prod = VADStateMachine(debug=False)
    with pytest.raises(TypeError):
        sm_prod.process_chunk("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sm_prod.process_scores(np.zeros((2, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        sm_prod.process_scores(np.zeros(5, dtype=np.complex64))
    with pytest.raises(ValueError):
        sm_prod.process_scores(np.zeros(5, dtype=np.float32), warmup=np.zeros(3, dtype=bool))

    sm_dbg = VADStateMachine(debug=True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sm_dbg.process_scores(np.array([0.1, 1.2, 0.5], dtype=np.float32))
