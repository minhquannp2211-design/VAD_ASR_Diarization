# PHASE 6A — Audio Engine Integration Contract (VAD ↔ shared streaming pipeline)

Status: **proposal for review, not integrated.** Scope: the boundary between the audio engine (VAD, ASR, diarization, alignment evidence) and the streaming infrastructure owned by a teammate (Ring Buffer, Sliding Window, FSM, serializer). Builds on `PHASE_5A_VAD_ASR_INTEGRATION_CONTRACT.md` (segment/crop/ASR data contracts) and does not repeat it. Nothing in the teammate's repository was touched.

Labels: **[verified]** = code or test run in this repo; **[synthetic]** = verified only on synthetic audio; **[proposed]** = design, not implemented; **[unverified]** = no evidence yet.

## 1. Existing APIs reused (unchanged)

| API | Reused for |
|---|---|
| `VAD.pipeline.streaming.VADStreamingSession` (`process_chunk`, `flush`, `reset`, `total_samples`, `is_flushed`) | All VAD processing; the adapter wraps it, never re-implements Layers 1–4 |
| `VAD.pipeline.contracts.validate_sample_rate`, `SUPPORTED_SAMPLE_RATES = (8000, 16000)`, `VADSegment`, `VADResult` | Sample-rate validation, segment/result types |
| `VADStateMachine` / `StateMachineConfig` (`onset_ms=30`, `hangover_ms=150`, `min_speech_ms=100`, hop 5 ms) | Onset/hangover/flush semantics (unmodified) |
| Phase 5A: `SpeechSegmentAudio`, `ASRSegmentResult`, crop/pre-roll formula `t_abs = crop_start_time_s + t_rel` | ASR hand-off and timestamp projection |

New (additive only): `VAD/pipeline/ingest.py` (`VADIngestAdapter`, `SpeechSpan`, `SampleDiscontinuityError`) and `tests/test_vad_ingest_adapter.py`. No existing file was modified.

## 2. Findings from inspecting the existing VAD API [verified by reading source + existing tests]

- **Input:** 1-D real mono array; `float32` recommended; any numeric dtype is accepted and cast with `astype(float32)` **without scaling**. An `int16` array would therefore reach the model as ±32768-scale values. → normalisation must happen at the boundary (done in the adapter).
- **NaN/Inf** are rejected only when `debug=True`. → the adapter always rejects them.
- **Sample rates:** 8000 and 16000 Hz only; `TypeError` for non-int/bool, `ValueError` otherwise.
- **State:** fully stateful across `process_chunk` calls (STFT overlap buffer, noise floor, scorer, state machine). Chunk-boundary invariance is covered by existing tests and re-checked here [synthetic]. After `flush()`, `process_chunk()` raises `RuntimeError`; `flush()` is idempotent; `reset()` starts a clean stream.
- **Timestamp convention:** `VADSegment` seconds are **relative to the first sample given to the session**, half-open `[start, end)`, on the 5 ms hop grid (80 samples @16 kHz, 40 @8 kHz). Number of frames = `1 + (N − n_fft) // hop`.
- **Warm-up:** first `init_frames = 40` frames (200 ms) are warm-up; no speech can be emitted during warm-up. Speech present from the first sample of a session is learned as noise and missed [synthetic, measured earlier: detected `[]`].
- **Latency:** a segment is emitted only after hangover expires: ≈ 0.135 s after its end in the synthetic run; segment **start** is known only retrospectively.
- **Flush:** pushes the zero-padded STFT tail through Layers 2–4, then closes pending speech; `duration_s == 0.0` by contract.
- **Existing tests:** chunk-size parity, flush idempotency/duration, sample-rate validation, timestamps within duration, offline/streaming parity, 8 k/16 k scenarios (`tests/test_vad_streaming.py`, `test_vad_offline.py`, `test_cross_pipeline_validation.py`). Parity tests depend on `test_data/` and were not re-run for this task.

## 3. The contract

### 3.1 Audio input [verified]

- Mono, 1-D, at the session sample rate (8000 or 16000 Hz). Stereo or other rates must be converted **by the caller** (the adapter does not resample; Phase 5A assigns resampling to the ASR-side adapter).
- Accepted dtypes: `float32/float64` in `[-1, 1]` (cast to float32) and `int16` PCM (divided by 32768). Other integer dtypes, complex, `None`, `ndim != 1`, NaN/Inf → explicit `TypeError`/`ValueError`, raised **before** any state change.

### 3.2 Absolute, monotonic sample timeline [verified]

- The caller owns the timeline: every `feed(samples, first_sample_index)` states the absolute index (in samples since stream start) of `samples[0]`. The adapter requires `first_sample_index == next_sample_index`.
- Suggested mapping on the infrastructure side: `first_sample_index` = the ring buffer's monotonic write counter **before** the write. The adapter does not import or depend on the ring buffer.
- `first_sample_index` ≠ expected raises `SampleDiscontinuityError` with `kind="gap"` (audio dropped upstream) or `kind="overlap"` (replayed audio) and the exact `expected`/`received` values. A rejected feed leaves state untouched.
- **VAD must receive each new sample exactly once, in order. Overlapping ASR windows must never be fed to the VAD** — the adapter rejects them rather than double-counting [verified: with overlapping replay the unmodified session produced wrong, split, shifted segments and 13 s of session time for 10 s of audio, measured in the earlier audit].

### 3.3 Reporting dropped audio [verified]

1. Infrastructure detects loss (e.g. ring overflow) and knows the absolute sample where audio resumes.
2. It calls `resync(first_sample_index_after_gap)`. The adapter finalizes pending speech of the old epoch (returned), resets the VAD state, sets the base to the new index, and increments `epoch`. It never moves the timeline backwards (`ValueError`).
3. Consequence: noise warm-up restarts, so speech beginning within ~200 ms after the resync point can be missed [synthetic characterization test]. Segments of different epochs are independent.
4. The adapter cannot detect loss by itself; it can only reject a feed whose index disagrees. Detection and the absolute resume index come from the infrastructure (this depends on the audit finding that the ring/slicer currently lose the absolute offset on overflow — teammate's fix).

### 3.4 VAD output and timestamp semantics [verified, synthetic]

`SpeechSpan(start_sample, end_sample, epoch)`: **integer absolute samples**, half-open `[start, end)`, multiples of the hop (80 samples @16 kHz), `start < end`, `end <= samples fed` (flush never extends beyond captured audio). Seconds = `sample / sample_rate`; milliseconds for the shared contract = `round(sample * 1000 / sample_rate)`. Spans are emitted **after** hangover, i.e. late and only when finalized. Spans are hints about where speech was detected; **VAD does not delete or gate audio** in this phase.

### 3.5 ASR results and timestamp projection [proposed; formula from Phase 5A]

- ASR receives a waveform plus its absolute start sample (`crop_start_sample`). Word times from the engine are **relative** to that waveform; absolute = `crop_start_time_s + t_rel` (never apply an engine-side offset and add it again).
- For the sliding-window infrastructure, the equivalent is the teammate's existing projection `slice.start_ms + word_rel_ms`; the engine must return **window-relative integer milliseconds** there, clamped to `[0, window_ms]` and **not beyond real audio** (zero-padded tails must not yield words past the captured duration — see audit F-09).
- Where a backend gives no word timestamps, assign the text to the VAD span/window interval; do not interpolate.

### 3.6 Diarization output and unknown speaker [proposed]

- Result: speaker label (`str`) **or `None`** when unknown/uncertain, with optional confidence. `None` is passed through unchanged (it maps to `speaker_id: null` in the shared transcript contract); the engine never invents a placeholder label.
- Labels must be stable within a meeting. Overlapping speech: report uncertainty (`None` or multiple intervals) rather than forcing one speaker.
- Per-window vs per-word speaker: **open design decision** with the infrastructure owner (the current port returns one speaker per window).

### 3.7 End of stream and ordering [proposed, VAD part verified]

1. `adapter.flush()` first — finalizes pending speech, no samples added, `next_sample_index` unchanged, idempotent.
2. Infrastructure drains and processes its remaining audio tail with ASR/diarization (windows are clipped to the real end; no padding-driven timestamps).
3. Infrastructure finalizes open transcript segments and emits final records.
`feed()` after `flush()` raises `RuntimeError`; `reset()` starts a new stream at index 0.

## 4. Proposed minimal interface

```python
class SpeechSpan:                     # frozen; absolute samples, half-open
    start_sample: int; end_sample: int; epoch: int

class VADIngestAdapter:
    def __init__(self, sample_rate: int = 16000, session: VADStreamingSession | None = None, **session_kwargs)
    next_sample_index: int            # next required first_sample_index
    samples_processed: int            # samples seen by the session in this epoch (each exactly once)
    epoch: int; is_flushed: bool
    def feed(self, samples, first_sample_index: int) -> list[SpeechSpan]   # SampleDiscontinuityError on gap/overlap
    def flush(self) -> list[SpeechSpan]
    def resync(self, first_sample_index: int) -> list[SpeechSpan]          # acknowledge dropped audio
    def reset(self) -> None
```

Boundary rule: engine code never imports infrastructure classes; infrastructure code only sees `SpeechSpan` integers. Wiring into the sliding window (tap point at ingestion, windows annotated with intersecting spans) is **not done** and needs approval.

## 5. Validation results

Command: `py -m pytest tests/test_vad_ingest_adapter.py tests/test_vad_streaming.py tests/test_pipeline_contracts.py -k "not parity" -p no:cacheprovider -q`.
Recorded result (2026-10-29, Windows, CPython via `py`), run twice on identical file contents: **77 passed, 33 deselected** (16.03 s and 26.84 s; runtime varies by machine load). The 33 deselected are the existing `*parity*` tests that need `test_data/`; they were not run. The adapter tests alone (39) were first run as 38 passed / 1 failed; the failure was a test-setup error (resync placed exactly at a burst onset, which hits the warm-up limitation in §6), fixed by resyncing in silence and adding a characterization test. All adapter tests use deterministic synthetic audio (harmonic bursts over low noise, fixed seeds).

| Requirement | Test |
|---|---|
| Same waveform, different chunk sizes (1 … 80 000 samples) → identical spans | `test_chunk_size_invariance` |
| State continuity equals driving the session directly | `test_state_continuity_matches_underlying_session` |
| Timestamps across a chunk boundary inside speech; 5 ms grid | `test_timestamps_across_chunk_boundary` |
| No sample processed twice; overlap replay rejected, state untouched | `test_no_sample_processed_twice`, `test_exactly_once_total_over_many_chunks` |
| Flush does not invent duration; idempotent; feed after flush rejected | `test_flush_does_not_invent_duration` |
| Invalid sample rates (value/type) | `test_invalid_sample_rate_value/_type` |
| Gaps explicit; resync keeps absolute timeline; no backwards resync | `test_gap_is_explicit_and_resync_*`, `test_resync_cannot_move_backwards` |
| Invalid index/audio rejected without state change; int16 normalisation | `test_invalid_*`, `test_int16_pcm_is_normalised_like_float` |

## 6. Unverified assumptions / limits

- **No accuracy claim.** Synthetic bursts are not speech; onset/hangover behaviour on real speech, weak speech near onsets, noisy audio and 8 kHz telephony are unevaluated.
- Warm-up blindness after `resync()` and at stream start is a property of the existing noise estimator; mitigation (e.g. warm-start from the previous epoch's floor) would change VAD internals and needs evidence first.
- ASR/diarization portions (§3.5–3.6) are design only; no ASR/diarization backend was run.
- CPU cost of VAD on the infrastructure's event loop was not measured.
- Compatibility with the teammate's fixes to ring buffer / slicer cannot be tested until those land.
- `SpeechSpan` ms conversion for the shared contract (`round(sample·1000/sr)`) is proposed, not agreed.
