"""
VAD/pipeline/__init__.py

Public package interface for VAD pipelines.
Exposes shared data contracts, concrete execution wrappers, and VADPipelineRouter factory.
"""
from __future__ import annotations

from VAD.pipeline.contracts import (
    PipelineMode,
    SUPPORTED_SAMPLE_RATES,
    VADDiagnostics,
    VADResult,
    VADSegment,
    validate_sample_rate,
)
from VAD.pipeline.offline import VADOfflineRunner
from VAD.pipeline.router import VADPipelineRouter
from VAD.pipeline.streaming import VADStreamingSession

__all__ = [
    "PipelineMode",
    "SUPPORTED_SAMPLE_RATES",
    "validate_sample_rate",
    "VADSegment",
    "VADDiagnostics",
    "VADResult",
    "VADStreamingSession",
    "VADOfflineRunner",
    "VADPipelineRouter",
]
