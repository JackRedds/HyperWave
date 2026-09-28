"""Optional bridge helpers for ml4gw/Torch acceleration."""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Any


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ml4gw_available() -> bool:
    """Return ``True`` when both ``ml4gw`` and ``torch`` are importable."""
    return _module_available("ml4gw") and _module_available("torch")


def torch_cuda_available() -> bool:
    """Return ``True`` when PyTorch can see a CUDA device."""
    if not _module_available("torch"):
        return False

    try:
        import torch
    except Exception:  # pragma: no cover - depends on optional runtime
        return False

    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@dataclass(frozen=True)
class ML4GWModules:
    """Lazy-imported ml4gw/Torch symbols used by HyperWave."""

    torch: Any
    SpectralDensity: Any
    Whiten: Any
    WaveformProjector: Any
    TimeDomainCBCWaveformGenerator: Any
    IMRPhenomD: Any
    IMRPhenomPv2: Any
    TaylorF2: Any
    bilby_spins_to_lalsim: Any
    SineGaussian: Any
    MultiWaveform: Any
    WhiteNoiseBurst: Any
    CosmicString: Any
    Gaussian: Any


def require_ml4gw_modules() -> ML4GWModules:
    """
    Import the ml4gw/Torch symbols HyperWave uses.

    Raises:
        ImportError: When ``ml4gw`` or ``torch`` is unavailable.
    """
    try:
        import torch
        from ml4gw.transforms import SpectralDensity, WaveformProjector, Whiten
        from ml4gw.waveforms.cbc import IMRPhenomD, IMRPhenomPv2, TaylorF2
        from ml4gw.waveforms.conversion import bilby_spins_to_lalsim
        from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
        from ml4gw.waveforms import SineGaussian, MultiWaveform, WhiteNoiseBurst, CosmicString, Gaussian
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "ml4gw acceleration requires the optional 'ml4gw' dependency and "
            "a compatible PyTorch install. Install with `pip install .[ml4gw]` "
            "and, for GPU use, a CUDA-enabled PyTorch wheel."
        ) from exc

    return ML4GWModules(
        torch=torch,
        SpectralDensity=SpectralDensity,
        Whiten=Whiten,
        WaveformProjector=WaveformProjector,
        TimeDomainCBCWaveformGenerator=TimeDomainCBCWaveformGenerator,
        IMRPhenomD=IMRPhenomD,
        IMRPhenomPv2=IMRPhenomPv2,
        TaylorF2=TaylorF2,
        bilby_spins_to_lalsim=bilby_spins_to_lalsim,
        SineGaussian=SineGaussian,
        MultiWaveform=MultiWaveform,
        WhiteNoiseBurst=WhiteNoiseBurst,
        CosmicString=CosmicString,
        Gaussian=Gaussian,
    )


def resolve_torch_device(
    *,
    gpu: bool = False,
    device: str | None = None,
    warn_fallback: bool = True,
):
    """
    Resolve the torch device for ml4gw execution.

    When ``gpu=True`` but CUDA is unavailable, the device falls back to CPU and
    emits a warning instead of failing at import time.
    """
    modules = require_ml4gw_modules()
    torch = modules.torch

    if device is not None:
        return torch.device(device)

    if gpu:
        if torch_cuda_available():
            return torch.device("cuda")

        if warn_fallback:
            warnings.warn(
                "GPU acceleration requested for ml4gw, but CUDA is unavailable. "
                "Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )

    return torch.device("cpu")


__all__ = [
    "ML4GWModules",
    "ml4gw_available",
    "require_ml4gw_modules",
    "resolve_torch_device",
    "torch_cuda_available",
]
