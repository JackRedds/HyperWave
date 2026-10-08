"""Batched GW template: parameter adapter + detector projection.

:class:`Template` is what the likelihood talks to. It

1. maps HyperWave's sampling parameters (``chirp_mass``, ``mass_ratio``,
   ``chi_1``, ``cos_tilt_1``, ``cos_theta_jn`` ...) onto the bilby/lalsimulation
   intrinsic convention used by the waveform backends,
2. asks the backend for a batch of plus/cross polarisations, and
3. projects them onto each detector with vectorised antenna patterns and a
   continuous-phase geocentric time delay,

returning ``(N, n_ifo, n_freq)`` on the masked analysis grid in one call. The
per-parameter spin/inclination interpretation matches HyperWave's existing
ml4gw path (``a_1 = |chi_1|``, ``tilt_1 = arccos(cos_tilt_1)`` flipped for
``chi_1 < 0``; ``theta_jn = arccos(cos_theta_jn)``) so the two backends agree.
"""

from __future__ import annotations

import numpy as np

from ..geometry import get_detector
from .backends.lal_backend import LALCBCWaveform
from .backends.ml4gw_backend import ML4GWCBCWaveform, ML4GWBurstWaveform

# Default sampled parameters (geocent_time is usually supplied via
# ``static_parameters``, matching HyperWave's existing examples).
DEFAULT_BBH_PARAMETERS = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
    "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1", "cos_tilt_2",
    "phi_12", "phi_jl",
]

BACKENDS = {
    "IMRPhenomD": ML4GWCBCWaveform,
    "IMRPhenomPv2": ML4GWCBCWaveform,
    "TaylorF2": ML4GWCBCWaveform,
    "SineGaussian": ML4GWBurstWaveform,
    "MultiSineGaussian": ML4GWBurstWaveform,
    "WhiteNoiseBurst": ML4GWBurstWaveform,
    "Gaussian": ML4GWBurstWaveform,
    "CosmicString": ML4GWBurstWaveform,
}


class Template:
    def __init__(
        self,
        detectors,
        frequency_array,
        sampling_rate,
        duration,
        start_time,
        minimum_frequency=20.0,
        maximum_frequency=None,
        reference_frequency=50.0,
        approximant="IMRPhenomPv2",
        parameters=None,
        static_parameters=None,
        backend="lal",
        trigger_time=None,
        n_jobs=1,
        gpu=False,
        torch_device=None,
        sequence=False,
        generator_kwargs=None,
    ):
        # sequence=True (lal backend only): evaluate exactly at frequency_array,
        # which may be sparse/non-uniform — used by the heterodyne likelihood.
        self.sequence = bool(sequence)
        # extra constructor kwargs for the ml4gw burst generator
        # (e.g. {"polarized": True} for WhiteNoiseBurst)
        self.generator_kwargs = dict(generator_kwargs or {})
        self.detector_names = [str(d) for d in detectors]
        self.detectors = [get_detector(name) for name in self.detector_names]
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.sampling_rate = float(sampling_rate)
        self.duration = float(duration)
        self.start_time = float(start_time)
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = (
            float(self.frequency_array[-1]) if maximum_frequency is None else float(maximum_frequency)
        )
        self.reference_frequency = float(reference_frequency)
        self.approximant = str(approximant).strip("'\"")
        self.parameters = list(parameters) if parameters is not None else list(DEFAULT_BBH_PARAMETERS)
        self.static_parameters = dict(static_parameters or {})
        self.trigger_time = trigger_time if trigger_time is not None else self.start_time
        self.n_jobs = int(n_jobs)

        self.mask = (self.frequency_array >= self.minimum_frequency) & (
            self.frequency_array <= self.maximum_frequency
        )
        self._f_masked = self.frequency_array[self.mask]

        self.backend_name = backend
        self.backend = self._build_backend(backend, gpu, torch_device)

    # -- backend ----------------------------------------------------------
    def _build_backend(self, backend, gpu, torch_device):
        backend = str(backend).lower()
        if backend == "lal":
            if self.generator_kwargs:
                raise ValueError("generator_kwargs is only supported by the 'ml4gw' backend.")
            return LALCBCWaveform(
                self.frequency_array,
                approximant=self.approximant,
                reference_frequency=self.reference_frequency,
                minimum_frequency=self.minimum_frequency,
                maximum_frequency=self.maximum_frequency,
                n_jobs=self.n_jobs,
                sequence=self.sequence,
            )
        if backend == "ml4gw":
            if self.sequence:
                raise ValueError("sequence=True is only supported by the 'lal' backend.")
            backend_cls = BACKENDS[self.approximant]
            right_pad = float(self.start_time + self.duration - self.trigger_time)
            kw = {}
            if self.generator_kwargs:
                if backend_cls is not ML4GWBurstWaveform:
                    raise ValueError("generator_kwargs is only supported for ml4gw burst waveforms.")
                kw["generator_kwargs"] = self.generator_kwargs
            return backend_cls(
                self.frequency_array,
                approximant=self.approximant,
                reference_frequency=self.reference_frequency,
                minimum_frequency=self.minimum_frequency,
                duration=self.duration,
                sampling_rate=self.sampling_rate,
                right_pad=max(0.0, min(right_pad, self.duration)),
                gpu=gpu,
                torch_device=torch_device,
                **kw,
            )
        raise ValueError(f"Unknown waveform backend {backend!r}. Expected 'lal' or 'ml4gw'.")

    def _named_from_theta(self, thetas):
        """Turn a ``(N, ndim)`` sampling array into a dict of ``(N,)`` arrays."""
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        named = {name: thetas[:, i] for i, name in enumerate(self.parameters)}
        n = thetas.shape[0]
        for key, value in self.static_parameters.items():
            named[key] = np.full(n, float(value))
        return named, n

    # -- projection -------------------------------------------------------
    def _project(self, hp, hc, named, masked=True):
        """Project ``(N, n_freq)`` polarisations onto detectors.

        Returns ``(N, n_ifo, n_freq)`` on the masked analysis grid (``masked=True``)
        or the full frequency grid (``masked=False``, used for injections).
        """
        if masked:
            hp_m = hp[:, self.mask]
            hc_m = hc[:, self.mask]
            f = self._f_masked
        else:
            hp_m = hp
            hc_m = hc
            f = self.frequency_array
        ra = np.asarray(named["ra"], float)
        dec = np.asarray(named["dec"], float)
        psi = np.asarray(named["psi"], float)
        if "geocent_time" in named:
            gps = np.asarray(named["geocent_time"], float)
        else:
            gps = np.full(hp.shape[0], float(self.trigger_time))

        n = hp.shape[0]
        out = np.zeros((n, len(self.detectors), len(f)), dtype=complex)
        for j, det in enumerate(self.detectors):
            fp, fc = det.antenna_response(ra, dec, psi, gps)          # (N,)
            dt = (gps - self.start_time) + det.time_delay_from_geocenter(ra, dec, gps)
            signal = fp[:, None] * hp_m + fc[:, None] * hc_m          # (N, n_freq)
            signal *= np.exp(-2j * np.pi * f[None, :] * dt[:, None])
            out[:, j, :] = signal
        return out

    # -- public API -------------------------------------------------------
    def make_injections_to_ifo_batch(self, thetas, masked=True):
        """Batched projected waveforms, shape ``(N, n_ifo, n_freq[_masked])``."""
        named, _ = self._named_from_theta(thetas)
        hp, hc = self.backend.polarizations(named)
        return self._project(hp, hc, named, masked=masked)

    def make_injections_to_ifo(self, gw_params):
        """Legacy single-vector path: returns ``{ifo_name: masked complex array}``."""
        signals = self.make_injections_to_ifo_batch(np.atleast_2d(gw_params))
        return {name: signals[0, j, :] for j, name in enumerate(self.detector_names)}

    def frequency_array_masked(self):
        return self._f_masked


__all__ = ["Template", "DEFAULT_BBH_PARAMETERS"]
