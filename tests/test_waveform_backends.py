"""Waveform backends vs bilby reference + batch consistency."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_bilby, requires_lal, requires_ml4gw


def _bilby_polarizations(frequency_array, approximant="IMRPhenomPv2"):
    import bilby

    fixed = dict(
        waveform_approximant=approximant, reference_frequency=50.0, minimum_frequency=20.0
    )
    wfg = bilby.gw.WaveformGenerator(
        duration=4.0,
        sampling_frequency=4096,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=fixed,
    )
    return wfg


@requires_lal
@requires_bilby
@pytest.mark.parametrize("approximant", ["IMRPhenomPv2", "IMRPhenomD", "IMRPhenomXPHM"])
def test_lal_backend_bit_exact_vs_bilby(segment, approximant):

    from hyperwave.detectors.waveforms.backends.lal_backend import LALWaveform

    farr = segment["frequency_array"]
    # Non-precessing intrinsic params (bilby convention).
    intrinsic = dict(
        mass_1=[36.0], mass_2=[29.0], luminosity_distance=[1000.0],
        theta_jn=[0.4], phase=[0.641716], a_1=[0.0], a_2=[0.0],
        tilt_1=[0.0], tilt_2=[0.0], phi_12=[0.0], phi_jl=[0.0],
    )
    wf = LALWaveform(farr, approximant=approximant, reference_frequency=50.0,
                     minimum_frequency=20.0, maximum_frequency=None)
    hp, hc = wf.polarizations(intrinsic)

    pol = _bilby_polarizations(farr, approximant).frequency_domain_strain(
        dict(mass_1=36.0, mass_2=29.0, luminosity_distance=1000.0, theta_jn=0.4,
             phase=0.641716, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
             phi_12=0.0, phi_jl=0.0)
    )
    np.testing.assert_array_equal(hp[0], pol["plus"])
    np.testing.assert_array_equal(hc[0], pol["cross"])


@requires_lal
@requires_bilby
def test_template_projection_matches_bilby(segment, bbh_theta, bbh_parameter_names):
    import bilby

    from hyperwave.detectors.waveforms import Template

    tmpl = Template(
        detectors=segment["detectors"], frequency_array=segment["frequency_array"],
        sampling_rate=segment["sampling_rate"], duration=segment["duration"],
        start_time=segment["start_time"], minimum_frequency=segment["minimum_frequency"],
        maximum_frequency=segment["maximum_frequency"], reference_frequency=50.0,
        approximant="IMRPhenomPv2", parameters=bbh_parameter_names,
        static_parameters={"geocent_time": segment["trigger_time"]},
    )
    batch = tmpl.make_injections_to_ifo_batch(np.array([bbh_theta]))

    ifos = bilby.gw.detector.InterferometerList(segment["detectors"])
    for ifo in ifos:
        ifo.set_strain_data_from_zero_noise(
            sampling_frequency=segment["sampling_rate"], duration=segment["duration"],
            start_time=segment["start_time"],
        )
        ifo.minimum_frequency = segment["minimum_frequency"]
        ifo.maximum_frequency = segment["maximum_frequency"]
    pol = _bilby_polarizations(segment["frequency_array"]).frequency_domain_strain(
        dict(zip(bbh_parameter_names, bbh_theta), geocent_time=segment["trigger_time"])
    )
    mask = ifos[0].frequency_mask
    for j, ifo in enumerate(ifos):
        ref = ifo.get_detector_response(pol, dict(zip(bbh_parameter_names, bbh_theta),
                                                  geocent_time=segment["trigger_time"]))[mask]
        # Waveform is bit-exact; residual is the float32 antenna tensor (~1e-7 rel).
        np.testing.assert_allclose(batch[0, j], ref, rtol=1e-6, atol=1e-30)


@requires_lal
def test_batch_rows_are_independent(segment, bbh_theta, bbh_parameter_names):
    from hyperwave.detectors.waveforms import Template

    tmpl = Template(
        detectors=segment["detectors"], frequency_array=segment["frequency_array"],
        sampling_rate=segment["sampling_rate"], duration=segment["duration"],
        start_time=segment["start_time"], minimum_frequency=segment["minimum_frequency"],
        maximum_frequency=segment["maximum_frequency"], approximant="IMRPhenomPv2",
        parameters=bbh_parameter_names, static_parameters={"geocent_time": segment["trigger_time"]},
    )
    other = list(bbh_theta)
    other[0] *= 1.05
    batch = tmpl.make_injections_to_ifo_batch(np.array([bbh_theta, other, bbh_theta]))
    assert batch.shape[0] == 3
    # Identical params -> identical rows.
    np.testing.assert_array_equal(batch[0], batch[2])
    # Different chirp mass -> meaningfully different waveform (strain ~ 1e-23,
    # so compare relative to the signal scale rather than np.allclose defaults).
    scale = np.max(np.abs(batch[0]))
    assert np.max(np.abs(batch[0] - batch[1])) / scale > 1e-3


@requires_ml4gw
@requires_lal
def test_ml4gw_close_to_lal_at_residual_level(segment, bbh_theta, bbh_parameter_names):
    """ml4gw uses a different time-delay discretisation, so only require
    agreement at the whitened-residual / amplitude level, not bin-exact."""
    from hyperwave.detectors.waveforms import Template

    common = dict(
        detectors=segment["detectors"], frequency_array=segment["frequency_array"],
        sampling_rate=segment["sampling_rate"], duration=segment["duration"],
        start_time=segment["start_time"], minimum_frequency=segment["minimum_frequency"],
        maximum_frequency=segment["maximum_frequency"], approximant="IMRPhenomPv2",
        parameters=bbh_parameter_names, static_parameters={"geocent_time": segment["trigger_time"]},
    )
    lal_batch = Template(backend="lal", **common).make_injections_to_ifo_batch(np.array([bbh_theta]))
    ml_batch = Template(backend="ml4gw", **common).make_injections_to_ifo_batch(np.array([bbh_theta]))
    # Compare amplitude spectra (phase/time conventions differ by construction).
    lal_amp = np.abs(lal_batch[0])
    ml_amp = np.abs(ml_batch[0])
    rel = np.linalg.norm(lal_amp - ml_amp) / np.linalg.norm(lal_amp)
    assert rel < 0.1
