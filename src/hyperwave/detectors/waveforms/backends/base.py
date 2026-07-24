"""Waveform-backend interface.

A waveform backend turns a *batch* of intrinsic CBC parameters into plus/cross
frequency-domain polarisations on a fixed analysis frequency grid.

Intrinsic parameters are given in **bilby/lalsimulation convention** as a mapping
of equal-length 1-D arrays (one entry per source in the batch):

    mass_1, mass_2          solar masses
    luminosity_distance     Mpc
    theta_jn, phase         radians
    a_1, a_2                dimensionless spin magnitudes
    tilt_1, tilt_2          radians
    phi_12, phi_jl          radians
    lambda_1, lambda_2      dimensionless tidal deformabilities (default 0)
    eccentricity            default 0

``polarizations`` returns ``(h_plus, h_cross)``, each a complex array of shape
``(N, n_freq)`` aligned with ``frequency_array``. Mapping HyperWave's sampling
parameters (chirp_mass, mass_ratio, chi_1, cos_tilt_1, ...) onto this convention
is the job of the :class:`~hyperwave.detectors.waveforms.template.Template`
adapter, not the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

#: canonical intrinsic parameter names, in batch dicts
INTRINSIC_PARAMETERS_CBC = (
    "mass_1",
    "mass_2",
    "luminosity_distance",
    "theta_jn",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "lambda_1",
    "lambda_2",
    "eccentricity",
)

INTRINSIC_PARAMETERS_SG = {
    "quality",
    "frequency",
    "hrss",
    "phase",
    "eccentricity",
    "shifts",
}

INTRINSIC_PARAMETERS_CS = {
    "power",
    "amplitude",
    "f_high",
}

INTRINSIC_PARAMETERS_WNB = {
    "frequency",
    "bandwidth",
    "eccentricity",
    "phase",
    "int_hdot_squared",
    "duration",
}

INTRINSIC_PARAMETERS_GAUSSIAN = {
    "hrss",
    "polarization",
    "eccentricity",
    "duration",
}

EXTRINSIC_DEFAULTS = {
    "psi",
    "dec",
    "ra",
    "geocent_time"
}

#: defaults for any intrinsic parameter omitted from a batch
INTRINSIC_DEFAULTS = {
    "a_1": 0.0,
    "a_2": 0.0,
    "tilt_1": 0.0,
    "tilt_2": 0.0,
    "phi_12": 0.0,
    "phi_jl": 0.0,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
    "eccentricity": 0.0,
}


def normalize_intrinsic_batch(params, n, intrinsic_params):
    """Broadcast a (possibly partial) intrinsic-parameter dict to length ``n``.

    Returns a new dict with every key in :data:`INTRINSIC_PARAMETERS` present as
    a float array of shape ``(n,)``.
    """
    out = {}
    # Need to fix this
    for key in intrinsic_params:
        if key in params:
            value = np.asarray(params[key], dtype=float)
            out[key] = np.broadcast_to(value, (n,)).astype(float, copy=False)
        elif key in INTRINSIC_DEFAULTS:
            out[key] = np.full(n, INTRINSIC_DEFAULTS[key], dtype=float)
        else:
            raise KeyError(f"Missing required intrinsic parameter {key!r}.")
    return out


class WaveformBackend(ABC):
    """Abstract frequency-domain CBC waveform generator (batched)."""

    #: analysis frequency grid (set by concrete backends)
    frequency_array: np.ndarray

    @abstractmethod
    def polarizations(self, params):
        """Return ``(h_plus, h_cross)`` of shape ``(N, n_freq)`` each."""
        raise NotImplementedError

    @property
    def name(self):  # pragma: no cover - cosmetic
        return type(self).__name__


__all__ = [
    "WaveformBackend",
    "INTRINSIC_PARAMETERS_CBC",
    "INTRINSIC_PARAMETERS_SG",
    "INTRINSIC_PARAMETERS_WNB",
    "INTRINSIC_PARAMETERS_CS",
    "INTRINSIC_PARAMETERS_GAUSSIAN",
    "INTRINSIC_DEFAULTS",
    "normalize_intrinsic_batch",
    "EXTRINSIC_DEFAULTS"
]
