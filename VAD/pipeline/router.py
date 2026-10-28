"""
VAD/pipeline/router.py

Deterministic Factory Router (VADPipelineRouter) for Multi-Band Adaptive VAD.
Selects and constructs one of the two concrete VAD execution paths:
- PipelineMode.STREAMING -> VADStreamingSession
- PipelineMode.OFFLINE   -> VADOfflineRunner

The router is a stateless, pure factory:
- It constructs only the requested implementation.
- It does not hold state, duplicate pipeline logic, or manage intermediate buffers.
- Once created, the caller interacts directly with the concrete returned object.
"""
from __future__ import annotations

import os
import sys
from typing import Literal, Optional, Sequence, Tuple, Union, overload

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from VAD.feature_extraction import DEFAULT_BANDS_HZ  # noqa: E402
from VAD.noise_estimator import NoiseConfig  # noqa: E402
from VAD.pipeline.contracts import (  # noqa: E402
    PipelineMode,
    validate_sample_rate,
)
from VAD.pipeline.offline import VADOfflineRunner  # noqa: E402
from VAD.pipeline.streaming import VADStreamingSession  # noqa: E402
from VAD.sigmoid import SigmoidConfig  # noqa: E402
from VAD.state_machine import StateMachineConfig  # noqa: E402


class VADPipelineRouter:
    """
    Factory router for VAD pipeline mode selection.

    Constructs either a VADStreamingSession (for causal streaming)
    or a VADOfflineRunner (for whole-file offline processing).

    Supports construction via:
        `router = VADPipelineRouter.create(PipelineMode.STREAMING, ...)`
    or direct instantiation:
        `session = VADPipelineRouter(PipelineMode.STREAMING, ...)`
    Both forms are fully typed via identical overloads.
    """

    @overload
    def __new__(
        cls,
        mode: Literal[PipelineMode.STREAMING, "streaming"],
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> VADStreamingSession:
        ...

    @overload
    def __new__(
        cls,
        mode: Literal[PipelineMode.OFFLINE, "offline"],
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> VADOfflineRunner:
        ...

    @overload
    def __new__(
        cls,
        mode: PipelineMode | str,
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> Union[VADStreamingSession, VADOfflineRunner]:
        ...

    def __new__(
        cls,
        mode: PipelineMode | str,
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> Union[VADStreamingSession, VADOfflineRunner]:
        """
        Construct a concrete pipeline instance by delegating to `VADPipelineRouter.create(...)`.
        """
        return cls.create(
            mode=mode,
            sample_rate=sample_rate,
            noise_config=noise_config,
            sigmoid_config=sigmoid_config,
            state_machine_config=state_machine_config,
            debug=debug,
            return_diagnostics=return_diagnostics,
            bands_hz=bands_hz,
        )

    @overload
    @classmethod
    def create(
        cls,
        mode: Literal[PipelineMode.STREAMING, "streaming"],
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> VADStreamingSession:
        ...

    @overload
    @classmethod
    def create(
        cls,
        mode: Literal[PipelineMode.OFFLINE, "offline"],
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> VADOfflineRunner:
        ...

    @overload
    @classmethod
    def create(
        cls,
        mode: PipelineMode | str,
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> Union[VADStreamingSession, VADOfflineRunner]:
        ...

    @classmethod
    def create(
        cls,
        mode: PipelineMode | str,
        sample_rate: int = 16000,
        noise_config: Optional[NoiseConfig] = None,
        sigmoid_config: Optional[SigmoidConfig] = None,
        state_machine_config: Optional[StateMachineConfig] = None,
        debug: bool = False,
        return_diagnostics: bool = False,
        bands_hz: Sequence[Tuple[float, float]] = DEFAULT_BANDS_HZ,
    ) -> Union[VADStreamingSession, VADOfflineRunner]:
        """
        Factory method to construct a VAD execution pipeline for the given mode.

        Args:
            mode: PipelineMode.STREAMING ("streaming") or PipelineMode.OFFLINE ("offline").
            sample_rate: Audio sample rate in Hz (8000 or 16000).
            noise_config: Optional custom NoiseConfig for Layer 2.
            sigmoid_config: Optional custom SigmoidConfig for Layer 3.
            state_machine_config: Optional custom StateMachineConfig for Layer 4.
            debug: Whether to enable assertions and input checks.
            return_diagnostics: Whether to attach VADDiagnostics to results.
            bands_hz: Frequency sub-band boundaries.

        Returns:
            VADStreamingSession if mode is STREAMING, or VADOfflineRunner if mode is OFFLINE.

        Raises:
            ValueError: If mode is unknown or sample_rate is not supported.
            TypeError: If mode is not a string or PipelineMode, or sample_rate is not an int.
        """
        # Validate mode
        if isinstance(mode, PipelineMode):
            parsed_mode = mode
        elif isinstance(mode, str):
            try:
                parsed_mode = PipelineMode(mode.lower())
            except ValueError:
                supported_modes = [m.value for m in PipelineMode]
                raise ValueError(
                    f"Invalid pipeline mode '{mode}'. Supported modes are: {supported_modes}"
                )
        else:
            raise TypeError(
                f"mode must be a PipelineMode or str, got {type(mode).__name__}"
            )

        # Validate sample rate
        sr = validate_sample_rate(sample_rate)

        if parsed_mode == PipelineMode.STREAMING:
            return VADStreamingSession(
                sample_rate=sr,
                noise_config=noise_config,
                sigmoid_config=sigmoid_config,
                state_machine_config=state_machine_config,
                debug=debug,
                return_diagnostics=return_diagnostics,
                bands_hz=bands_hz,
            )
        elif parsed_mode == PipelineMode.OFFLINE:
            return VADOfflineRunner(
                sample_rate=sr,
                noise_config=noise_config,
                sigmoid_config=sigmoid_config,
                state_machine_config=state_machine_config,
                debug=debug,
                return_diagnostics=return_diagnostics,
                bands_hz=bands_hz,
            )
        else:
            raise ValueError(f"Unhandled mode: {parsed_mode}")
