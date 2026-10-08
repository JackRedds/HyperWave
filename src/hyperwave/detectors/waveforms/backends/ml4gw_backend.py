"""ml4gw / Torch waveform backend (batched, optional).

``ML4GWWaveform`` generates a whole batch of CBC polarisations in a single Torch
call, which is the natural fast path on GPU. ml4gw implements only
``IMRPhenomD``, ``IMRPhenomPv2`` and ``TaylorF2`` and exposes a *time-domain*
generator, so this backend produces time-domain plus/cross and FFTs them to the
analysis grid using HyperWave's :func:`~hyperwave.detectors.strain.nfft`
convention. Detector projection is then done by
:class:`~hyperwave.detectors.waveforms.template.Template` exactly as for the LAL
backend, so both backends share one (continuous-phase) time-delay convention.

.. warning::

   This backend is **experimental**. ml4gw cannot be installed on Python 3.13
   (it requires ``python < 3.13``), so it is untested in such environments, and
   the merger time/phase reference relative to the bit-exact LAL backend must be
   validated (``tests/test_waveform_backends.py``) before production use. The LAL
   backend is the default.
"""

from __future__ import annotations

import numpy as np

from ....ml4gw import require_ml4gw_modules, resolve_torch_device
from .base import (WaveformBackend, normalize_intrinsic_batch, INTRINSIC_PARAMETERS_SG,
                   INTRINSIC_PARAMETERS_CS, INTRINSIC_PARAMETERS_GAUSSIAN, INTRINSIC_PARAMETERS_WNB,
                   INTRINSIC_PARAMETERS_CBC)

from ..parameters import HyperwaveToCBC

ML4GW_APPROXIMANTS = ("IMRPhenomD", "IMRPhenomPv2", "TaylorF2", "SineGaussian", "MultiWaveform", "WhiteNoiseBurst", "CosmicString", "Gaussian")


def _chirp_mass_mass_ratio(mass_1, mass_2):
    total = mass_1 + mass_2
    chirp_mass = (mass_1 * mass_2) ** 0.6 / total**0.2
    mass_ratio = mass_2 / mass_1
    return chirp_mass, mass_ratio


class ML4GWWaveform(WaveformBackend):
    def __init__(
        self,
        frequency_array,
        approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=20.0,
        *,
        duration,
        sampling_rate,
        right_pad=0.0,
        gpu=False,
        torch_device=None,
    ):
        approximant = str(approximant).strip("'\"")
        if approximant not in ML4GW_APPROXIMANTS:
            raise ValueError(
                f"ml4gw backend supports {ML4GW_APPROXIMANTS}, not {approximant!r}. "
                "Use the LAL backend for this approximant."
            )
        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.approximant_name = approximant
        self.reference_frequency = float(reference_frequency)
        self.minimum_frequency = float(minimum_frequency)
        self.duration = float(duration)
        self.sampling_rate = float(sampling_rate)
        self.right_pad = float(right_pad)

        self._modules = require_ml4gw_modules()
        self._device = resolve_torch_device(gpu=gpu, device=torch_device)
        approximant_map = {
            "IMRPhenomD": self._modules.IMRPhenomD,
            "IMRPhenomPv2": self._modules.IMRPhenomPv2,
            "TaylorF2": self._modules.TaylorF2,
            "SineGaussian": self._modules.SineGaussian,
            "MultiWaveform": self._modules.MultiWaveform,
            "WhiteNoiseBurst": self._modules.WhiteNoiseBurst,
            "CosmicString": self._modules.CosmicString,
            "Gaussian": self._modules.Gaussian
        }
        parameter_map = {
            "IMRPhenomD": INTRINSIC_PARAMETERS_CBC,
            "IMRPhenomPv2": INTRINSIC_PARAMETERS_CBC,
            "TaylorF2": INTRINSIC_PARAMETERS_CBC,
            "SineGaussian": INTRINSIC_PARAMETERS_SG,
            "MultiWaveform": INTRINSIC_PARAMETERS_SG,
            "WhiteNoiseBurst": INTRINSIC_PARAMETERS_WNB,
            "CosmicString": INTRINSIC_PARAMETERS_CS,
            "Gaussian": INTRINSIC_PARAMETERS_GAUSSIAN
        }
        self._intrinsic_parameters = parameter_map[approximant]
        self._approximant = approximant_map[approximant]

    def _tensor(self, values):
        torch = self._modules.torch
        return torch.as_tensor(np.asarray(values, dtype=float), dtype=torch.float64, device=self._device)

    def parameter_adapter(self, batch):
        return NotImplementedError

    def _phase_correction(self, hp_fd, hc_fd, batch):
        """Override in subclasses if additional phase corrections are needed"""
        return hp_fd, hc_fd

    def _fft_correction(self, hp_fd, hc_fd, hp):
        torch = self._modules.torch
        # ml4gw's TimeDomainCBCWaveformGenerator places the coalescence at
        # (duration - right_pad) within the window. The LAL backend and the
        # template's geocent_time projection both expect the coalescence at t=0,
        # so reference it back: H(f) -> H(f) * exp(+2j pi f t_c). Without this the
        # geocent placement is applied twice and the waveform is time-shifted.
        t_c = float(self.duration - self.right_pad)
        freqs = torch.fft.rfftfreq(hp.shape[-1], d=1.0 / self.sampling_rate).to(hp_fd.device)
        ang = (2.0 * np.pi * t_c) * freqs
        phase = torch.complex(torch.cos(ang), torch.sin(ang))
        hp_fd = hp_fd * phase
        hc_fd = hc_fd * phase
        return hp_fd, hc_fd

    def polarizations(self, params):
        torch = self._modules.torch
        keys = list(params)
        n = max((np.asarray(params[k]).size for k in keys), default=1) if keys else 1
        batch = normalize_intrinsic_batch(params, n, self._intrinsic_parameters)

        waveform_params = self.parameter_adapter(batch)
        hc, hp = self._generator(**waveform_params)  # (N, T) each

        # FFT to single-sided / fs convention on the analysis grid
        hp_fd = torch.fft.rfft(hp, dim=-1) / self.sampling_rate
        hc_fd = torch.fft.rfft(hc, dim=-1) / self.sampling_rate

        hp_fd, hc_fd = self._fft_correction(hp_fd, hc_fd, hp)

        hp_fd = (hp_fd).detach().cpu().numpy()
        hc_fd = (hc_fd).detach().cpu().numpy()

        hp_fd, hc_fd = self._phase_correction(hp_fd, hc_fd, batch)
        n_freq = len(self.frequency_array)
        hp_out = np.zeros((n, n_freq), dtype=complex)
        hc_out = np.zeros((n, n_freq), dtype=complex)
        m = min(n_freq, hp_fd.shape[-1])
        hp_out[:, :m] = hp_fd[:, :m]
        hc_out[:, :m] = hc_fd[:, :m]
        return hp_out, hc_out


class ML4GWCBCWaveform(ML4GWWaveform):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._approximant = self._approximant().to(self._device)

        self._generator = self._modules.TimeDomainCBCWaveformGenerator(
            approximant=self._approximant,
            sample_rate=self.sampling_rate,
            duration=self.duration,
            f_min=self.minimum_frequency,
            f_ref=self.reference_frequency,
            right_pad=self.right_pad,
        ).to(self._device)

    def parameter_adapter(self, batch):
        mass_1 = batch["mass_1"]
        mass_2 = batch["mass_2"]
        chirp_mass, mass_ratio = _chirp_mass_mass_ratio(mass_1, mass_2)

        params = {
            "chirp_mass": self._tensor(chirp_mass),
            "mass_ratio": self._tensor(mass_ratio),
            "mass_1": self._tensor(mass_1),
            "mass_2": self._tensor(mass_2),
            "distance": self._tensor(batch["luminosity_distance"]),
            "phic": self._tensor(batch["phase"]),
        }

        if self.approximant_name == "IMRPhenomPv2":
            incl, s1x, s1y, s1z, s2x, s2y, s2z = self._modules.bilby_spins_to_lalsim(
                theta_jn=self._tensor(batch["theta_jn"]),
                phi_jl=self._tensor(batch["phi_jl"]),
                tilt_1=self._tensor(batch["tilt_1"]),
                tilt_2=self._tensor(batch["tilt_2"]),
                phi_12=self._tensor(batch["phi_12"]),
                a_1=self._tensor(batch["a_1"]),
                a_2=self._tensor(batch["a_2"]),
                mass_1=self._tensor(mass_1),
                mass_2=self._tensor(mass_2),
                f_ref=self.reference_frequency,
                phi_ref=self._tensor(batch["phase"]),
            )
            params.update(
                inclination=incl, s1x=s1x, s1y=s1y, s1z=s1z, s2x=s2x, s2y=s2y, s2z=s2z
            )
        else:  # aligned-spin approximants
            chi1z = batch["a_1"] * np.cos(batch["tilt_1"])
            chi2z = batch["a_2"] * np.cos(batch["tilt_2"])
            params.update(
                inclination=self._tensor(batch["theta_jn"]),
                chi1=self._tensor(chi1z),
                chi2=self._tensor(chi2z),
                s1z=self._tensor(chi1z),
                s2z=self._tensor(chi2z),
            )
        return params
    
    def _phase_correction(self, hp_fd, hc_fd, batch):
        # ml4gw applies the coalescence phase with the opposite sign and a pi
        # offset relative to LAL: empirically arg<h_LAL | h_ml4gw> = pi - 2*phase
        # (independent of inclination/sky), so rotate each waveform by
        # exp(+i(2*phase - pi)) to match the LAL phase convention.
        ph = batch["phase"]
        ph = ph.detach().cpu().numpy() if hasattr(ph, "detach") else np.asarray(ph, dtype=float)
        corr = np.exp(1j * (2.0 * ph.reshape(-1).astype(float) - np.pi))[:, None]
        hp_fd = hp_fd * corr
        hc_fd = hc_fd * corr

        return hp_fd, hc_fd
        
    def polarizations(self, params):
        intrinsic = HyperwaveToCBC.convert(params)
        print(intrinsic)
        return super().polarizations(intrinsic)


class ML4GWBurstWaveform(ML4GWWaveform):
    def __init__(
        self,
        *args,
        generator_kwargs=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # forwarded to the ml4gw generator's constructor, e.g. {"polarized": True}
        # for WhiteNoiseBurst
        self.generator_kwargs = dict(generator_kwargs or {})
        self._generator = self._approximant(
            self.sampling_rate, self.duration, **self.generator_kwargs
        ).to(self._device)

    def parameter_adapter(self, batch):
        params = {
            k: self._tensor(v) for k, v in batch.items() if k in self._intrinsic_parameters
        }
        return params
    
    def polarizations(self, params):
        return super().polarizations(params)


__all__ = ["ML4GWWaveform", "ML4GWCBCWWaveform", "ML4GWBurstWaveform", 
           "ML4GW_APPROXIMANTS", "component_masses"]

