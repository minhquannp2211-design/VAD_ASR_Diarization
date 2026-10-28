"""
VAD/state_machine.py - TẦNG 4: MÁY TRẠNG THÁI VAD & HẬU XỬ LÝ (VAD STATE MACHINE & POST-PROCESSING)

Đầu vào :
    - SigmoidOutput (score (T,) float32, warmup (T,) bool) từ Tầng 3
      hoặc mảng score (T,) và warmup (T,) trực tiếp qua process_scores().
    - hop_seconds: float (khoảng cách thời gian giữa 2 frame liên tiếp, mặc định 0.005 s = 5 ms).

Đầu ra  :
    - StateMachineOutput (segments, states, speech_mask) từ process_chunk() / process_scores()
    - List[Tuple[float, float]] từ flush(): danh sách các đoạn tiếng nói còn đang giữ trong bộ đệm
      khi kết thúc luồng âm thanh.

CHUỖI CHUYỂN TRẠNG THÁI (HYSTERESIS STATE MACHINE)
    SILENCE (0) -> ONSET (1) -> SPEECH (2) -> HANGOVER (3) -> SILENCE (0)

    Tại mỗi frame t (với chỉ số frame toàn cục g = frame_offset + t):
    1. Hợp đồng Warm-up (BẮT BUỘC):
       Nếu warmup[t] == True:
         - Nếu đang có đoạn tiếng nói mở (hiếm gặp, chỉ khi warmup bật lại giữa chừng), đóng đoạn đó.
         - Buộc state = SILENCE, onset_count = 0, hangover_count = 0.
         - KHÔNG chuyển sang ONSET hay SPEECH.
    2. Chính sách đầu vào không hữu hạn (Non-finite Input Policy):
       - Khi debug=True : nếu score hoặc warmup chứa NaN / +Inf / -Inf, lập tức ném ValueError.
       - Khi debug=False: mọi giá trị không hữu hạn (NaN, +Inf, -Inf) tại frame t đều được ép về
         p = 0.0 (coi như không có bằng chứng tiếng nói, tuyệt đối không để +Inf kích hoạt ONSET/SPEECH
         hay làm kẹt ở SPEECH).
    3. Khi warmup[t] == False, xét điểm phát hiện p = score[t]:
       - Đang ở SILENCE:
           + Nếu p >= onset_thresh:
               * Nếu onset_frames <= 1: vào thẳng SPEECH, mở đoạn ứng viên tại start_frame = g
                 với active_speech_frames = 1.
               * Nếu onset_frames > 1 : chuyển sang ONSET, đặt onset_count = 1, cand_start = g.
           + Ngược lại (p < onset_thresh): giữ nguyên SILENCE.
       - Đang ở ONSET:
           + Nếu p >= onset_thresh:
               * Tăng onset_count += 1.
               * Nếu onset_count >= onset_frames: xác nhận SPEECH! Đoạn ứng viên bắt đầu từ
                 cand_start (tức bao gồm cả các frame ONSET đã tích lũy), đặt
                 active_speech_frames = onset_count, rồi đặt onset_count = 0.
           + Ngược lại (p < onset_thresh): gai nhiễu ngắn chưa đủ onset_frames -> hủy ứng viên,
             quay về SILENCE, onset_count = 0.
       - Đang ở SPEECH:
           + Nếu p >= offset_thresh: tiếp tục giữ SPEECH, tăng active_speech_frames += 1.
           + Ngược lại (p < offset_thresh):
               * Nếu hangover_frames == 0: kết thúc đoạn ứng viên ngay tại end_frame = g (exclusive),
                 quay về SILENCE ngay tại frame g.
               * Nếu hangover_frames == 1: frame g là frame HANGOVER duy nhất (out_states[t] = HANGOVER);
                 hết frame g thì hangover hết hạn ngay, kết thúc đoạn ứng viên tại end_frame = g + 1
                 (exclusive) và đặt trạng thái nội bộ về SILENCE để frame g + 1 phải qua ONSET mới.
               * Nếu hangover_frames > 1 : chuyển sang HANGOVER tại frame g (frame 1/H),
                 đặt hangover_count = 1.
       - Đang ở HANGOVER (chỉ khi hangover_count < hangover_frames):
           + Nếu p >= offset_thresh (ngay tại một trong các frame g0 + 1 .. g0 + H - 1 trước khi hết hạn):
             vết lõm ngắn trong câu nói đã qua -> quay lại SPEECH, tăng active_speech_frames += 1,
             đặt hangover_count = 0 (giữ nguyên đoạn đang mở).
           + Ngược lại (p < offset_thresh):
               * Tăng hangover_count += 1; frame g vẫn mang trạng thái HANGOVER.
               * Nếu sau khi tăng, hangover_count >= hangover_frames (tức g = g0 + H - 1 là frame
                 HANGOVER cuối cùng): hết frame g này thì hangover hết hạn -> kết thúc đoạn ứng viên
                 tại end_frame = g + 1 (= g0 + H, exclusive) và đặt trạng thái nội bộ về SILENCE,
                 hangover_count = 0 (để tại frame g0 + H tiếp theo, dù điểm cao cũng không thể hồi sinh
                 trực tiếp đoạn cũ về SPEECH mà phải bắt đầu chu trình SILENCE -> ONSET mới).

HẬU XỬ LÝ ĐOẠN (POST-PROCESSING: MIN_SILENCE & MIN_SPEECH) & BẤT BIẾN STREAMING
    - Quy ước thời gian đoạn (start_time, end_time):
      + start_time = start_frame * hop_seconds (inclusive - thời điểm bắt đầu frame đầu của đoạn)
      + end_time   = end_frame * hop_seconds   (exclusive - thời điểm kết thúc frame cuối của đoạn,
        bao gồm phần đệm hangover_frames ở cuối đoạn nếu đoạn đóng tự nhiên qua HANGOVER).
    - Nối khoảng lặng ngắn (`min_silence_ms` -> `min_silence_frames`):
      + Nếu khoảng cách giữa đoạn ứng viên trước [s1, e1) (có n_sp1 frame bằng chứng tiếng nói) và
        đoạn ứng viên mới [s2, e2) (có n_sp2 frame bằng chứng tiếng nói) thỏa mãn
        (s2 - e1) <= min_silence_frames, hai đoạn được nối liền thành [s1, e2) với tổng số frame
        bằng chứng tiếng nói = n_sp1 + n_sp2.
      + Để đảm bảo TÍNH BẤT BIẾN QUA BIÊN CHUNK (chunk-invariance), một đoạn vừa khép lại ở trạng thái
        SILENCE chưa được phát ra ngay lập tức mà được giữ tạm (`_pending_seg`) cho đến khi số frame
        SILENCE sau nó vượt quá `min_silence_frames`, hoặc khi gọi `flush()`.
    - Ngữ nghĩa `min_speech_ms` (`min_speech_frames`) — ĐO BẰNG CHỨNG TIẾNG NÓI ĐỘC LẬP VỚI HANGOVER:
      + `min_speech_frames` kiểm tra số lượng frame thực sự đạt ngưỡng tiếng nói của đoạn ứng viên
        (`speech_evidence_frames`, gồm các frame ONSET đã xác nhận + các frame ở trạng thái SPEECH,
        KHÔNG tính các frame đệm HANGOVER ở cuối đoạn).
      + Lý do: Thông thường `hangover_ms` (150 ms) >= `min_speech_ms` (100 ms). Nếu cộng cả phần đệm
        `hangover_frames` vào độ dài xét duyệt `min_speech_frames`, bất kỳ xung nhiễu ngắn nào vừa đủ
        lọt qua `onset_ms` (30 ms) rồi tắt ngay lập tức cũng sẽ được cộng thêm 150 ms hangover thành
        180 ms >= 100 ms, làm vô hiệu hóa hoàn toàn bộ lọc `min_speech_ms`.
      + Khi đoạn ứng viên vượt qua kiểm tra `speech_evidence_frames >= min_speech_frames`, đoạn phát ra
        vẫn giữ nguyên mốc thời gian đầy đủ `(start_frame * hop_seconds, end_frame * hop_seconds)`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.sigmoid import SigmoidOutput  # noqa: E402


class VADState(IntEnum):
    """Mã trạng thái của máy trạng thái VAD tại từng frame."""
    SILENCE = 0
    ONSET = 1
    SPEECH = 2
    HANGOVER = 3


@dataclass(frozen=True)
class StateMachineConfig:
    """
    Cấu hình cho máy trạng thái VAD & hậu xử lý (Tầng 4).
    Các tham số thời gian được cấu hình bằng mili-giây (ms) và quy đổi sang số frame
    thông qua `hop_seconds` (không hardcode số frame tách rời `hop_seconds`).
    """
    onset_thresh: float = 0.55       # Ngưỡng điểm bắt đầu kích hoạt ONSET / SPEECH
    offset_thresh: float = 0.35      # Ngưỡng điểm duy trì SPEECH / thoát HANGOVER (<= onset_thresh)
    onset_ms: float = 30.0           # Thời gian xác nhận ONSET liên tiếp (30 ms = 6 frame @ 5 ms)
    hangover_ms: float = 150.0       # Thời gian ngân HANGOVER trước khi tắt (150 ms = 30 frame @ 5 ms)
    min_speech_ms: float = 100.0     # Thời lượng bằng chứng tiếng nói tối thiểu (100 ms = 20 frame @ 5 ms, không tính hangover)
    min_silence_ms: float = 100.0    # Khoảng lặng ngắn tối đa giữa 2 đoạn được nối liền (100 ms = 20 frame @ 5 ms)

    def validate(self) -> None:
        if not (np.isfinite(self.onset_thresh) and np.isfinite(self.offset_thresh)):
            raise ValueError("onset_thresh và offset_thresh phải hữu hạn")
        if not (0.0 < self.offset_thresh <= self.onset_thresh < 1.0):
            raise ValueError(
                "Yêu cầu 0.0 < offset_thresh <= onset_thresh < 1.0 "
                f"(nhận offset_thresh={self.offset_thresh}, onset_thresh={self.onset_thresh})"
            )
        for name, val in (
            ("onset_ms", self.onset_ms),
            ("hangover_ms", self.hangover_ms),
            ("min_speech_ms", self.min_speech_ms),
            ("min_silence_ms", self.min_silence_ms),
        ):
            if not np.isfinite(val):
                raise ValueError(f"{name} phải hữu hạn")
        if self.onset_ms <= 0.0:
            raise ValueError("onset_ms phải > 0")
        if self.hangover_ms < 0.0:
            raise ValueError("hangover_ms phải >= 0")
        if self.min_speech_ms < 0.0:
            raise ValueError("min_speech_ms phải >= 0")
        if self.min_silence_ms < 0.0:
            raise ValueError("min_silence_ms phải >= 0")

    def to_frames(self, hop_seconds: float) -> Tuple[int, int, int, int]:
        """
        Quy đổi (onset_ms, hangover_ms, min_speech_ms, min_silence_ms) sang số frame nguyên
        dựa trên `hop_seconds`.
        """
        if not np.isfinite(hop_seconds) or hop_seconds <= 0.0:
            raise ValueError(f"hop_seconds phải hữu hạn và > 0, nhận {hop_seconds}")
        hop_ms = hop_seconds * 1000.0
        onset_f = max(1, int(round(self.onset_ms / hop_ms)))
        hangover_f = max(0, int(round(self.hangover_ms / hop_ms)))
        min_speech_f = max(1, int(round(self.min_speech_ms / hop_ms)))
        min_silence_f = max(0, int(round(self.min_silence_ms / hop_ms)))
        return onset_f, hangover_f, min_speech_f, min_silence_f


@dataclass(frozen=True)
class StateMachineOutput:
    """
    Kết quả xử lý của Tầng 4 cho một chunk T frame.
    - segments   : Danh sách các đoạn `(start_time, end_time)` (tính bằng giây, end_time là exclusive)
                   đã hoàn tất và vượt qua bước lọc `min_silence` / `min_speech` trong chunk này.
    - states     : Mảng `(T,) int8` chứa mã trạng thái `VADState` (0=SILENCE, 1=ONSET, 2=SPEECH, 3=HANGOVER)
                   tại từng frame.
    - speech_mask: Mảng `(T,) bool` đánh dấu trạng thái hoạt động tức thời của máy trạng thái
                   (`True` khi `states` là `SPEECH` hoặc `HANGOVER`).
    """
    segments: List[Tuple[float, float]]
    states: np.ndarray        # (T,) int8
    speech_mask: np.ndarray   # (T,) bool

    @property
    def num_frames(self) -> int:
        return int(self.states.shape[0])


class VADStateMachine:
    """
    Máy trạng thái VAD & Hậu xử lý (Tầng 4).
    Duy trì trạng thái xuyên suốt các chunk streaming và đảm bảo tính bất biến theo kích thước chunk
    (chunk-invariance) cho danh sách đoạn tiếng nói đầu ra sau khi kết thúc bằng `flush()`.
    """

    def __init__(
        self,
        hop_seconds: float = 0.005,
        config: Optional[StateMachineConfig] = None,
        debug: bool = False,
    ) -> None:
        if not np.isfinite(hop_seconds) or hop_seconds <= 0.0:
            raise ValueError(f"hop_seconds phải hữu hạn và > 0, nhận {hop_seconds}")
        self.hop_seconds = float(hop_seconds)
        self.config = config or StateMachineConfig()
        self.config.validate()
        (
            self.onset_frames,
            self.hangover_frames,
            self.min_speech_frames,
            self.min_silence_frames,
        ) = self.config.to_frames(self.hop_seconds)
        self.debug = bool(debug)
        self.reset()

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái nội bộ để bắt đầu phiên âm thanh mới."""
        self._state: VADState = VADState.SILENCE
        self._frame_idx: int = 0
        self._onset_count: int = 0
        self._hangover_count: int = 0
        self._cand_start: Optional[int] = None
        self._active_start: Optional[int] = None
        self._active_speech_frames: int = 0
        # _pending_seg lưu (start_frame, end_frame, speech_evidence_frames)
        self._pending_seg: Optional[Tuple[int, int, int]] = None
        return None

    @property
    def state(self) -> VADState:
        """Trạng thái hiện tại của máy trạng thái (`VADState`)."""
        return self._state

    @property
    def frame_index(self) -> int:
        """Tổng số frame đã đi qua máy trạng thái kể từ lần `reset()` gần nhất."""
        return self._frame_idx

    def _close_candidate_segment(
        self,
        start_f: int,
        end_f: int,
        speech_f: int,
        emitted: List[Tuple[float, float]],
    ) -> None:
        """
        Đưa một đoạn ứng viên vừa kết thúc [start_f, end_f) (với `speech_f` frame bằng chứng tiếng nói)
        vào bộ đệm nối khoảng lặng `_pending_seg`.
        Nếu khoảng cách từ `_pending_seg` đến `start_f` <= `min_silence_frames`, gộp hai đoạn lại
        và cộng dồn số frame bằng chứng tiếng nói.
        Ngược lại, chốt `_pending_seg` cũ (nếu `p_speech >= min_speech_frames`) rồi lưu đoạn mới.
        """
        if self._pending_seg is None:
            self._pending_seg = (start_f, end_f, speech_f)
            return

        p_start, p_end, p_speech = self._pending_seg
        gap = start_f - p_end
        if gap <= self.min_silence_frames:
            self._pending_seg = (p_start, end_f, p_speech + speech_f)
        else:
            if p_speech >= self.min_speech_frames:
                emitted.append((p_start * self.hop_seconds, p_end * self.hop_seconds))
            self._pending_seg = (start_f, end_f, speech_f)

    def _maybe_release_pending_at_frame(
        self, current_frame_idx: int, emitted: List[Tuple[float, float]]
    ) -> None:
        """
        Kiểm tra tại frame `current_frame_idx` (khi không ở trong SPEECH/HANGOVER):
        Nếu đã có một đoạn ứng viên `_pending_seg = (p_start, p_end, p_speech)` và khoảng lặng tính đến
        đầu đoạn tiếng nói có thể xuất hiện sớm nhất tiếp theo chắc chắn > `min_silence_frames`,
        thì phát đoạn `_pending_seg` ngay (nếu `p_speech >= min_speech_frames`) để giảm độ trễ streaming
        mà vẫn bảo toàn chunk-invariance.
        """
        if self._pending_seg is None:
            return
        p_start, p_end, p_speech = self._pending_seg
        earliest_next_start = (
            self._cand_start if self._state == VADState.ONSET else (current_frame_idx + 1)
        )
        assert earliest_next_start is not None
        if (earliest_next_start - p_end) > self.min_silence_frames:
            if p_speech >= self.min_speech_frames:
                emitted.append((p_start * self.hop_seconds, p_end * self.hop_seconds))
            self._pending_seg = None

    def process_scores(
        self,
        score: np.ndarray,
        warmup: Optional[np.ndarray] = None,
    ) -> StateMachineOutput:
        """
        Xử lý trực tiếp mảng `score` shape `(T,)` và `warmup` shape `(T,)`.
        Cho phép kiểm thử độc lập Tầng 4 với các chuỗi điểm xác định.
        """
        score_arr = np.asarray(score)
        if np.iscomplexobj(score_arr):
            raise ValueError("score phải là mảng số thực")
        if score_arr.ndim != 1:
            raise ValueError(f"score phải là mảng 1 chiều shape (T,), nhận {score_arr.shape}")
        T = int(score_arr.shape[0])

        if warmup is None:
            warm_arr = np.zeros(T, dtype=bool)
        else:
            warm_raw = np.asarray(warmup)
            if np.iscomplexobj(warm_raw):
                raise ValueError("warmup không được là mảng số phức")
            if warm_raw.ndim != 1 or warm_raw.shape[0] != T:
                raise ValueError(
                    f"warmup phải có shape ({T},) khớp với score, nhận {warm_raw.shape}"
                )
            if self.debug and T > 0 and not np.isfinite(warm_raw).all():
                raise ValueError("warmup chứa NaN hoặc Inf")
            warm_arr = warm_raw.astype(bool, copy=False)

        if self.debug and T > 0:
            if not np.isfinite(score_arr).all():
                raise ValueError("score chứa NaN hoặc Inf")
            if (score_arr < 0.0).any() or (score_arr > 1.0).any():
                raise ValueError("score chứa giá trị ngoài đoạn [0, 1]")

        if T == 0:
            return StateMachineOutput(
                segments=[],
                states=np.empty(0, dtype=np.int8),
                speech_mask=np.empty(0, dtype=bool),
            )

        scores_f64 = score_arr.astype(np.float64, copy=False)
        out_states = np.empty(T, dtype=np.int8)
        emitted: List[Tuple[float, float]] = []

        onset_th = self.config.onset_thresh
        offset_th = self.config.offset_thresh
        onset_f = self.onset_frames
        hangover_f = self.hangover_frames

        for t in range(T):
            g = self._frame_idx
            self._frame_idx += 1

            # 1. Hợp đồng Warm-up: khóa chặt ở SILENCE, không tăng onset_count
            if warm_arr[t]:
                if self._state in (VADState.SPEECH, VADState.HANGOVER):
                    assert self._active_start is not None
                    self._close_candidate_segment(
                        self._active_start, g, self._active_speech_frames, emitted
                    )
                    self._active_start = None
                    self._active_speech_frames = 0
                self._state = VADState.SILENCE
                self._onset_count = 0
                self._hangover_count = 0
                self._cand_start = None
                out_states[t] = int(VADState.SILENCE)
                self._maybe_release_pending_at_frame(g, emitted)
                continue

            p = scores_f64[t]
            # Chính sách nhất quán khi debug=False: mọi giá trị không hữu hạn (NaN, +Inf, -Inf)
            # đều được ép về p = 0.0 để +Inf không kích hoạt nhầm SPEECH và NaN không làm kẹt SPEECH.
            if not np.isfinite(p):
                p = 0.0

            if self._state == VADState.SILENCE:
                if p >= onset_th:
                    if onset_f <= 1:
                        self._state = VADState.SPEECH
                        self._active_start = g
                        self._active_speech_frames = 1
                        self._onset_count = 0
                        self._cand_start = None
                    else:
                        self._state = VADState.ONSET
                        self._onset_count = 1
                        self._cand_start = g
                        self._maybe_release_pending_at_frame(g, emitted)
                else:
                    self._maybe_release_pending_at_frame(g, emitted)

            elif self._state == VADState.ONSET:
                if p >= onset_th:
                    self._onset_count += 1
                    if self._onset_count >= onset_f:
                        self._state = VADState.SPEECH
                        self._active_start = self._cand_start
                        self._active_speech_frames = self._onset_count
                        self._onset_count = 0
                        self._cand_start = None
                    else:
                        self._maybe_release_pending_at_frame(g, emitted)
                else:
                    # Chưa đủ onset_frames mà điểm đã tụt -> hủy ONSET, quay về SILENCE
                    self._state = VADState.SILENCE
                    self._onset_count = 0
                    self._cand_start = None
                    self._maybe_release_pending_at_frame(g, emitted)

            elif self._state == VADState.SPEECH:
                if p >= offset_th:
                    self._active_speech_frames += 1
                    out_states[t] = int(VADState.SPEECH)
                else:
                    if hangover_f <= 0:
                        assert self._active_start is not None
                        self._close_candidate_segment(
                            self._active_start, g, self._active_speech_frames, emitted
                        )
                        self._active_start = None
                        self._active_speech_frames = 0
                        self._state = VADState.SILENCE
                        out_states[t] = int(VADState.SILENCE)
                        self._maybe_release_pending_at_frame(g, emitted)
                    else:
                        # Frame g là frame HANGOVER thứ 1
                        out_states[t] = int(VADState.HANGOVER)
                        if hangover_f == 1:
                            # Hết ngay 1 frame hangover tại hết frame g (exclusive end = g + 1)
                            assert self._active_start is not None
                            self._close_candidate_segment(
                                self._active_start, g + 1, self._active_speech_frames, emitted
                            )
                            self._active_start = None
                            self._active_speech_frames = 0
                            self._state = VADState.SILENCE
                            self._hangover_count = 0
                            self._maybe_release_pending_at_frame(g, emitted)
                        else:
                            self._state = VADState.HANGOVER
                            self._hangover_count = 1
                continue

            elif self._state == VADState.HANGOVER:
                if p >= offset_th:
                    # Khôi phục lại SPEECH khi điểm vượt lại offset_thresh tại frame hangover chưa hết hạn
                    self._state = VADState.SPEECH
                    self._active_speech_frames += 1
                    self._hangover_count = 0
                    out_states[t] = int(VADState.SPEECH)
                else:
                    # Frame g vẫn nằm trong cửa sổ HANGOVER (frame thứ _hangover_count + 1)
                    out_states[t] = int(VADState.HANGOVER)
                    self._hangover_count += 1
                    if self._hangover_count >= hangover_f:
                        # Đã hoàn tất đúng hangover_f frame HANGOVER tại hết frame g (exclusive end = g + 1)
                        # Đóng đoạn ứng viên và đưa trạng thái nội bộ về SILENCE để frame g + 1 bắt đầu mới
                        assert self._active_start is not None
                        self._close_candidate_segment(
                            self._active_start, g + 1, self._active_speech_frames, emitted
                        )
                        self._active_start = None
                        self._active_speech_frames = 0
                        self._state = VADState.SILENCE
                        self._hangover_count = 0
                        self._maybe_release_pending_at_frame(g, emitted)
                continue

            out_states[t] = int(self._state)

        speech_mask = (out_states == int(VADState.SPEECH)) | (
            out_states == int(VADState.HANGOVER)
        )
        return StateMachineOutput(
            segments=emitted,
            states=out_states,
            speech_mask=speech_mask,
        )

    def process_chunk(self, sig_out: SigmoidOutput) -> StateMachineOutput:
        """
        Nhận đối tượng `SigmoidOutput` từ Tầng 3 và cập nhật máy trạng thái.
        """
        if not isinstance(sig_out, SigmoidOutput):
            raise TypeError("sig_out phải là đối tượng SigmoidOutput")
        return self.process_scores(score=sig_out.score, warmup=sig_out.warmup)

    def flush(self) -> List[Tuple[float, float]]:
        """
        Chốt phiên streaming khi kết thúc luồng âm thanh:
        - Nếu đang ở trong `SPEECH` hoặc `HANGOVER`, đóng đoạn ứng viên tại frame cuối cùng đã nhận
          (`end_frame = self._frame_idx`).
        - Nếu đang ở `ONSET` (chưa đủ `onset_frames`), hủy ứng viên `ONSET`.
        - Kiểm tra đoạn đang chờ trong `_pending_seg`: nếu `speech_evidence_frames >= min_speech_frames`
          thì phát ra.
        - Đưa trạng thái máy về `SILENCE` (giữ `frame_index` cho đến khi gọi `reset()`).
        """
        emitted: List[Tuple[float, float]] = []
        if self._state in (VADState.SPEECH, VADState.HANGOVER):
            assert self._active_start is not None
            self._close_candidate_segment(
                self._active_start, self._frame_idx, self._active_speech_frames, emitted
            )
            self._active_start = None
            self._active_speech_frames = 0

        self._state = VADState.SILENCE
        self._onset_count = 0
        self._hangover_count = 0
        self._cand_start = None

        if self._pending_seg is not None:
            p_start, p_end, p_speech = self._pending_seg
            if p_speech >= self.min_speech_frames:
                emitted.append((p_start * self.hop_seconds, p_end * self.hop_seconds))
            self._pending_seg = None

        return emitted

    def __repr__(self) -> str:
        return (
            f"VADStateMachine(hop_seconds={self.hop_seconds}, state={self._state.name}, "
            f"frame_index={self._frame_idx}, config={self.config})"
        )
