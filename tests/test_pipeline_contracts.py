"""
tests/test_pipeline_contracts.py

Unit tests for VAD shared data contracts:
- VADSegment (half-open interval semantics, duration, validation)
- VADDiagnostics (dimension validation, optional fields, num_frames)
- VADResult (result aggregation semantics, duration properties, validation)
- PipelineMode (mode identifiers, string compatibility)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from VAD.pipeline.contracts import (
    PipelineMode,
    SUPPORTED_SAMPLE_RATES,
    VADDiagnostics,
    VADResult,
    VADSegment,
    validate_sample_rate,
)


class TestVADSegment:
    def test_valid_construction_and_properties(self) -> None:
        seg = VADSegment(0.5, 1.75)
        assert seg.start_time_s == 0.5
        assert seg.end_time_s == 1.75
        assert seg.start_time == 0.5
        assert seg.end_time == 1.75
        assert math.isclose(seg.duration_s, 1.25)
        assert math.isclose(seg.duration, 1.25)
        assert seg.to_tuple() == (0.5, 1.75)

    def test_unpacking_and_indexing(self) -> None:
        seg = VADSegment(1.2, 3.4)
        s, e = seg
        assert s == 1.2
        assert e == 3.4
        assert seg[0] == 1.2
        assert seg[1] == 3.4

    def test_integer_inputs_coerced_to_float(self) -> None:
        seg = VADSegment(0, 2)
        assert isinstance(seg.start_time_s, float)
        assert isinstance(seg.end_time_s, float)
        assert seg.start_time_s == 0.0
        assert seg.end_time_s == 2.0
        assert seg.duration_s == 2.0

    def test_invalid_negative_start(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            VADSegment(-0.1, 1.0)

    def test_invalid_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            VADSegment(1.0, 1.0)

    def test_invalid_inverted_boundaries(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            VADSegment(2.5, 1.0)

    def test_invalid_nan_inf(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            VADSegment(float("nan"), 1.0)
        with pytest.raises(ValueError, match="finite"):
            VADSegment(0.0, float("nan"))
        with pytest.raises(ValueError, match="finite"):
            VADSegment(0.0, float("inf"))
        with pytest.raises(ValueError, match="finite"):
            VADSegment(float("-inf"), 1.0)

    def test_invalid_types(self) -> None:
        with pytest.raises(TypeError, match="real number"):
            VADSegment("0.0", 1.0)  # type: ignore
        with pytest.raises(TypeError, match="real number"):
            VADSegment(0.0, "1.0")  # type: ignore
        with pytest.raises(TypeError, match="real number"):
            VADSegment(True, 1.0)  # type: ignore
        with pytest.raises(TypeError, match="real number"):
            VADSegment(0.0, False)  # type: ignore


class TestVADDiagnostics:
    def test_valid_construction_minimal(self) -> None:
        diag = VADDiagnostics(
            frame_scores=np.array([0.1, 0.6, 0.8], dtype=np.float32),
            speech_mask=np.array([False, True, True], dtype=bool),
            states=np.array([0, 1, 2], dtype=np.int8),
            warmup_mask=np.array([False, False, False], dtype=bool),
        )
        assert diag.num_frames == 3
        assert diag.noise_floor is None
        assert diag.snr_db is None

    def test_valid_construction_with_intermediates(self) -> None:
        diag = VADDiagnostics(
            frame_scores=np.array([0.1, 0.9], dtype=np.float32),
            speech_mask=np.array([False, True], dtype=bool),
            states=np.array([0, 2], dtype=np.int8),
            warmup_mask=np.array([False, False], dtype=bool),
            noise_floor=np.zeros((4, 2), dtype=np.float32),
            snr_db=np.ones((4, 2), dtype=np.float32),
        )
        assert diag.num_frames == 2
        assert diag.noise_floor.shape == (4, 2)
        assert diag.snr_db.shape == (4, 2)

    def test_mismatched_frame_dimensions(self) -> None:
        with pytest.raises(ValueError, match="speech_mask"):
            VADDiagnostics(
                frame_scores=np.zeros(3, dtype=np.float32),
                speech_mask=np.zeros(2, dtype=bool),  # Mismatch: 2 != 3
                states=np.zeros(3, dtype=np.int8),
                warmup_mask=np.zeros(3, dtype=bool),
            )

    def test_mismatched_intermediate_dimensions(self) -> None:
        with pytest.raises(ValueError, match="noise_floor"):
            VADDiagnostics(
                frame_scores=np.zeros(3, dtype=np.float32),
                speech_mask=np.zeros(3, dtype=bool),
                states=np.zeros(3, dtype=np.int8),
                warmup_mask=np.zeros(3, dtype=bool),
                noise_floor=np.zeros((4, 2), dtype=np.float32),  # 2 != 3
            )


class TestVADResult:
    def test_valid_result_construction(self) -> None:
        seg1 = VADSegment(0.5, 1.2)
        seg2 = VADSegment(2.0, 3.5)
        res = VADResult(
            segments=[seg1, seg2],
            num_frames=100,
            duration_s=0.500,
        )
        assert len(res.segments) == 2
        assert res.num_frames == 100
        assert res.duration_s == 0.500
        assert math.isclose(res.speech_duration_s, (1.2 - 0.5) + (3.5 - 2.0))
        assert res.diagnostics is None

    def test_empty_result(self) -> None:
        res = VADResult(
            segments=[],
            num_frames=0,
            duration_s=0.0,
        )
        assert res.segments == []
        assert res.num_frames == 0
        assert res.duration_s == 0.0
        assert res.speech_duration_s == 0.0

    def test_result_aggregation_semantics(self) -> None:
        # Simulate caller receiving results from 2 chunks and a flush
        res1 = VADResult(segments=[], num_frames=20, duration_s=0.100)
        res2 = VADResult(segments=[VADSegment(0.05, 0.08)], num_frames=20, duration_s=0.100)
        res_flush = VADResult(segments=[VADSegment(0.12, 0.18)], num_frames=1, duration_s=0.005)

        # Successive aggregation should contain exactly the non-duplicate list of segments
        aggregated: list[VADSegment] = []
        for r in (res1, res2, res_flush):
            aggregated.extend(r.segments)

        assert len(aggregated) == 2
        assert aggregated[0] == VADSegment(0.05, 0.08)
        assert aggregated[1] == VADSegment(0.12, 0.18)

    def test_invalid_segment_elements(self) -> None:
        with pytest.raises(TypeError, match="VADSegment"):
            VADResult(
                segments=[(0.5, 1.2)],  # Raw tuple instead of VADSegment
                num_frames=10,
                duration_s=0.050,
            )

    def test_invalid_num_frames(self) -> None:
        with pytest.raises(ValueError, match="num_frames"):
            VADResult(segments=[], num_frames=-1, duration_s=0.0)

    def test_invalid_duration_s(self) -> None:
        with pytest.raises(ValueError, match="duration_s"):
            VADResult(segments=[], num_frames=0, duration_s=-0.5)
        with pytest.raises(ValueError, match="duration_s"):
            VADResult(segments=[], num_frames=0, duration_s=float("nan"))


class TestPipelineMode:
    def test_mode_values(self) -> None:
        assert PipelineMode.STREAMING == "streaming"
        assert PipelineMode.OFFLINE == "offline"
        assert isinstance(PipelineMode.STREAMING, str)
        assert isinstance(PipelineMode.OFFLINE, str)

    def test_mode_lookup(self) -> None:
        assert PipelineMode("streaming") is PipelineMode.STREAMING
        assert PipelineMode("offline") is PipelineMode.OFFLINE
        with pytest.raises(ValueError):
            PipelineMode("invalid_mode")


class TestValidateSampleRate:
    def test_supported_constants(self) -> None:
        assert SUPPORTED_SAMPLE_RATES == (8000, 16000)

    def test_valid_integer_rates(self) -> None:
        assert validate_sample_rate(8000) == 8000
        assert validate_sample_rate(16000) == 16000
        assert isinstance(validate_sample_rate(16000), int)
        assert validate_sample_rate(np.int64(16000)) == 16000
        assert validate_sample_rate(np.int32(8000)) == 8000

    def test_invalid_types_raise_type_error(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate(16000.0)  # float
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate(8000.0)
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate("16000")  # str
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate(True)  # bool
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate(False)
        with pytest.raises(TypeError, match="integer"):
            validate_sample_rate(None)  # None

    def test_unsupported_rates_raise_value_error(self) -> None:
        for rate in (0, -1, 44100, 22050, 48000, 32000):
            with pytest.raises(ValueError, match="Unsupported sample_rate"):
                validate_sample_rate(rate)
