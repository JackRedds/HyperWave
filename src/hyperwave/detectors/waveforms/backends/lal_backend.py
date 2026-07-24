"""Direct-lalsimulation waveform backend.

``LALWaveform`` generates frequency-domain CBC polarisations by calling
``lalsimulation`` directly, reproducing ``bilby.gw.source.lal_binary_black_hole``
*bit-for-bit* (verified to ``max abs diff = 0``). It is the default backend
because it covers every lalsimulation approximant.

lalsimulation's FD entry points generate one source per C call, so a batch is
produced by a parallel (``joblib``) loop over the intrinsic generations; the
expensive part is the intrinsic waveform, and the cheap extrinsic
projection/time-shift is vectorised separately in
:class:`~hyperwave.detectors.waveforms.template.Template`.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .base import WaveformBackend, normalize_intrinsic_batch, INTRINSIC_PARAMETERS_CBC
from ..parameters import HyperwaveToCBC

# bilby/lal constants (matched exactly for numerical agreement)
PARSEC = 3.085677581491367e16
SOLAR_MASS = 1.988409870698050731911960804878414216e30


def _spins_to_lalsim(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
                     mass_1_si, mass_2_si, reference_frequency, phase):
    """Port of ``bilby.gw.conversion.bilby_to_lalsimulation_spins`` (scalar)."""
    import lalsimulation as lalsim

    if (a_1 == 0 or tilt_1 in (0, np.pi)) and (a_2 == 0 or tilt_2 in (0, np.pi)):
        return (
            theta_jn,
            0.0, 0.0, a_1 * np.cos(tilt_1),
            0.0, 0.0, a_2 * np.cos(tilt_2),
        )
    return lalsim.SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
        mass_1_si, mass_2_si, reference_frequency, phase,
    )


class LALCBCWaveform(WaveformBackend):
    def __init__(
        self,
        frequency_array,
        approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=20.0,
        maximum_frequency=None,
        *,
        catch_waveform_errors=False,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
        n_jobs=1,
        sequence=False,
    ):
        import lalsimulation as lalsim

        self.frequency_array = np.asarray(frequency_array, dtype=float)
        self.approximant_name = str(approximant).strip("'\"")
        self.approximant = lalsim.GetApproximantFromString(self.approximant_name)
        self.reference_frequency = float(reference_frequency)
        self.minimum_frequency = float(minimum_frequency)
        self.maximum_frequency = (
            float(self.frequency_array[-1]) if maximum_frequency is None else float(maximum_frequency)
        )
        self.catch_waveform_errors = bool(catch_waveform_errors)
        self.pn_spin_order = int(pn_spin_order)
        self.pn_tidal_order = int(pn_tidal_order)
        self.pn_phase_order = int(pn_phase_order)
        self.pn_amplitude_order = int(pn_amplitude_order)
        self.n_jobs = int(n_jobs)
        # Sequence mode evaluates the waveform exactly at self.frequency_array
        # via SimInspiralChooseFDWaveformSequence — the grid may be sparse and
        # NON-uniform (relative-binning edge grids). Only FD approximants
        # support it, and the sequence API has no eccentricity argument.
        self.sequence = bool(sequence)

        self._delta_frequency = self.frequency_array[1] - self.frequency_array[0]
        self._bounds = (self.frequency_array >= self.minimum_frequency) & (
            self.frequency_array <= self.maximum_frequency
        )
        self._is_fd = bool(lalsim.SimInspiralImplementedFDApproximants(self.approximant))
        if self.sequence and not self._is_fd:
            raise ValueError(
                f"sequence=True needs a frequency-domain approximant; {self.approximant_name} is TD-only."
            )

    # -- internal ---------------------------------------------------------
    def _waveform_dictionary(self, lambda_1, lambda_2):
        import lal
        import lalsimulation as lalsim

        d = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(d, float(lambda_1))
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(d, float(lambda_2))
        lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(d, self.pn_spin_order)
        lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(d, self.pn_tidal_order)
        lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(d, self.pn_phase_order)
        lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(d, self.pn_amplitude_order)
        return d

    def _single(self, p):
        import lalsimulation as lalsim

        farr = self.frequency_array
        n = len(farr)
        mass_1_si = p["mass_1"] * SOLAR_MASS
        mass_2_si = p["mass_2"] * SOLAR_MASS
        distance_si = p["luminosity_distance"] * 1e6 * PARSEC
        phase = p["phase"]

        iota, s1x, s1y, s1z, s2x, s2y, s2z = _spins_to_lalsim(
            p["theta_jn"], p["phi_jl"], p["tilt_1"], p["tilt_2"], p["phi_12"],
            p["a_1"], p["a_2"], mass_1_si, mass_2_si, self.reference_frequency, phase,
        )

        wf_dict = self._waveform_dictionary(p["lambda_1"], p["lambda_2"])

        if self.sequence:
            import lal

            f_eval = farr[self._bounds]
            freqs = lal.CreateREAL8Sequence(len(f_eval))
            freqs.data = f_eval
            try:
                hplus, hcross = lalsim.SimInspiralChooseFDWaveformSequence(
                    phase, mass_1_si, mass_2_si, s1x, s1y, s1z, s2x, s2y, s2z,
                    self.reference_frequency, distance_si, iota,
                    wf_dict, self.approximant, freqs,
                )
            except Exception as exc:  # pragma: no cover - depends on params
                if not self.catch_waveform_errors:
                    raise
                if exc.args and exc.args[0] == "Internal function call failed: Input domain error":
                    return None
                raise
            hp = np.zeros(n, dtype=complex)
            hc = np.zeros(n, dtype=complex)
            hp[self._bounds] = hplus.data.data
            hc[self._bounds] = hcross.data.data
            return hp, hc

        if self.pn_amplitude_order != 0:
            start_frequency = lalsim.SimInspiralfLow2fStart(
                float(self.minimum_frequency), int(self.pn_amplitude_order), self.approximant
            )
        else:
            start_frequency = self.minimum_frequency

        wf_func = lalsim.SimInspiralChooseFDWaveform if self._is_fd else lalsim.SimInspiralFD
        try:
            hplus, hcross = wf_func(
                mass_1_si, mass_2_si, s1x, s1y, s1z, s2x, s2y, s2z,
                distance_si, iota, phase, 0.0, p["eccentricity"], 0.0,
                self._delta_frequency, start_frequency, self.maximum_frequency,
                self.reference_frequency, wf_dict, self.approximant,
            )
        except Exception as exc:  # pragma: no cover - depends on params
            if not self.catch_waveform_errors:
                raise
            edom = exc.args[0] == "Internal function call failed: Input domain error"
            if edom:
                return None
            raise

        data_plus = hplus.data.data
        data_cross = hcross.data.data
        if len(data_plus) > n:
            hp = np.array(data_plus[:n], dtype=complex)
            hc = np.array(data_cross[:n], dtype=complex)
        else:
            hp = np.zeros(n, dtype=complex)
            hc = np.zeros(n, dtype=complex)
            hp[: len(data_plus)] = data_plus
            hc[: len(data_cross)] = data_cross

        hp *= self._bounds
        hc *= self._bounds

        if not self._is_fd:
            dt = 1.0 / hplus.deltaF + (
                hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds * 1e-9
            )
            shift = np.exp(-1j * 2 * np.pi * dt * farr[self._bounds])
            hp[self._bounds] *= shift
            hc[self._bounds] *= shift

        return hp, hc
    
    # -- public -----------------------------------------------------------
    def polarizations(self, params):
        intrinsic = HyperwaveToCBC.convert(params)
        keys = list(intrinsic)
        n = max((np.asarray(intrinsic[k]).size for k in keys), default=1) if keys else 1
        batch = normalize_intrinsic_batch(intrinsic, n, INTRINSIC_PARAMETERS_CBC)
        rows = [{k: float(batch[k][i]) for k in batch} for i in range(n)]

        if self.n_jobs > 1 and n > 1:
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._single)(r) for r in rows)
        else:
            results = [self._single(r) for r in rows]

        n_freq = len(self.frequency_array)
        hp = np.zeros((n, n_freq), dtype=complex)
        hc = np.zeros((n, n_freq), dtype=complex)
        for i, res in enumerate(results):
            if res is None:
                continue
            hp[i], hc[i] = res
        return hp, hc


__all__ = ["LALCBCWaveform", "PARSEC", "SOLAR_MASS"]
