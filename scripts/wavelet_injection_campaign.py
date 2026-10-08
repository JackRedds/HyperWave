"""Injection campaign: waveform-agnostic wavelet reconstruction of many signals.

Where ``wavelet_reconstruction.py --waveform <family>`` injects *one* hand-picked signal,
this script draws the source parameters of ``N`` signals from a prior
distribution and reconstructs each one:

1. draw the parameters of injection ``i`` from the family's prior (bilby
   ``PriorDict``; defaults below, override any of them with ``--prior-file``),
   optionally rescaling the amplitude to a network SNR drawn from ``--snr-range``,
2. generate an independent detector-noise realisation and inject the signal,
3. reconstruct it with a variable number of Morlet-Gabor wavelets (Eryn RJ-MCMC,
   same sampler/proposal options as the single-injection scripts), and
4. save ``inj_XXXX.npz`` per injection plus a campaign summary
   (``campaign_summary.json`` and an overlap-vs-SNR plot) in ``--outdir``.

Injection ``i`` uses its own child seed stream (``SeedSequence(seed).spawn(N)[i]``)
for the parameter draw, the noise and torch, so any subset of indices can be run
independently (e.g. one per SLURM array task) and gives identical results.
Finished injections are skipped on re-runs (``--no-resume`` to redo them).

Run::

    # look at the drawn parameters/SNRs without sampling
    python scripts/wavelet_injection_campaign.py --waveform sg --n-injections 20 --dry-run
    # full campaign on the GPU
    python scripts/wavelet_injection_campaign.py --waveform sg --n-injections 20 \\
        --device gpu --proposal flowfisher --outdir results/sg_campaign
    # one injection per SLURM array task, then aggregate
    python scripts/wavelet_injection_campaign.py ... --indices $SLURM_ARRAY_TASK_ID
    python scripts/wavelet_injection_campaign.py ... --summarize-only
"""

from __future__ import annotations

import argparse
import copy
import gc
import glob
import json
import os
import time
import torch

import numpy as np

# Eryn (current releases) still call np.in1d, removed in NumPy 2.0.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from bilby.core.prior import Cosine, DeltaFunction, LogUniform, PowerLaw, PriorDict, Uniform  # noqa: E402
from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.moves import (DistributionGenerateRJ, GaussianMove, MTDistGenMoveRJ)  # noqa: E402
from eryn.state import State  # noqa: E402

from hyperwave.detectors.lvk import DetectorNoise, GW  # noqa: E402
from hyperwave.detectors.waveforms import WaveletTemplate, network_optimal_snr  # noqa: E402
from hyperwave.inference import (  # noqa: E402
    FlowTrainingCallback,
    WaveletConvergenceStopping,
    WaveletFisherMove,
    WaveletHalfCycleMove,
    WaveletSkyRingMove,
    build_flow_proposal,
    build_guided_birth,
    build_mf_birth,
    build_wavelet_priors,
    guided_initial_wavelets,
    make_flow_distribution_move,
    make_flow_rj_move,
)
from hyperwave.likelihoods import WaveletLikelihood  # noqa: E402
from hyperwave.ml4gw import torch_cuda_available  # noqa: E402
from hyperwave.plots import wavelet_reconstruction as wr  # noqa: E402

DETECTORS = ["H1", "L1"]
TRIGGER_TIME = 1268189526.951953


def _sky_priors():
    return dict(
        ra=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        dec=Cosine(),
        psi=Uniform(0.0, np.pi, boundary="periodic"),
    )


def _bbh_priors():
    return PriorDict(dict(
        chirp_mass=Uniform(20.0, 40.0),          # >=20 Msun keeps the chirp inside 4 s from 20 Hz
        mass_ratio=Uniform(0.5, 1.0),
        luminosity_distance=PowerLaw(alpha=2, minimum=300.0, maximum=1500.0),
        phase=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        chi_1=Uniform(-0.8, 0.8),                # signed spin + cos_tilt (hyperwave convention)
        chi_2=Uniform(-0.8, 0.8),
        cos_theta_jn=Uniform(-1.0, 1.0),
        cos_tilt_1=Uniform(-1.0, 1.0),
        cos_tilt_2=Uniform(-1.0, 1.0),
        phi_12=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        phi_jl=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        **_sky_priors(),
    ))


def _sg_priors():
    return PriorDict(dict(
        quality=Uniform(3.0, 30.0),
        frequency=Uniform(40.0, 400.0),
        hrss=LogUniform(2.5e-23, 2.5e-22),
        phase=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        eccentricity=DeltaFunction(0.0),         # linearly polarised, as in wavelet_reconstruction.py --waveform sg
        shifts=DeltaFunction(0.0),
        **_sky_priors(),
    ))


def _cs_priors():
    return PriorDict(dict(
        power=DeltaFunction(-4.0 / 3.0),         # cusps
        amplitude=LogUniform(1e-21, 1e-20),
        f_high=Uniform(100.0, 1000.0),
        **_sky_priors(),
    ))


def _gaussian_priors():
    return PriorDict(dict(
        hrss=LogUniform(1.5e-19, 1e-18),
        polarization=DeltaFunction(0.0),
        eccentricity=DeltaFunction(0.0),
        duration=LogUniform(5e-4, 3e-3),
        **_sky_priors(),
    ))


def _wnb_priors():
    return PriorDict(dict(
        frequency=Uniform(100.0, 400.0),         # centre frequency
        bandwidth=Uniform(50.0, 200.0),
        eccentricity=DeltaFunction(0.0),
        phase=Uniform(0.0, 2 * np.pi, boundary="periodic"),
        int_hdot_squared=LogUniform(3e-39, 1e-37),
        duration=Uniform(0.02, 0.1),
        **_sky_priors(),
    ))


def _cbc_ellipticity(p):
    # CBC cross-polarisation convention (lalsimulation): h_cross = -eps * i * h_plus,
    # eps = 2 cos(iota) / (1 + cos^2 iota). Only used on the --fixed-sky path.
    c = p["cos_theta_jn"]
    return -2.0 * c / (1.0 + c**2)


# Per-family injection setup. Parameter orders match the single-injection
# scripts. ``amplitude`` is the parameter rescaled by --snr-range, with the
# network SNR scaling as amplitude**snr_power.
FAMILIES = {
    "bbh": dict(
        names=["chirp_mass", "mass_ratio", "luminosity_distance", "psi", "phase",
               "ra", "dec", "chi_1", "chi_2", "cos_theta_jn", "cos_tilt_1",
               "cos_tilt_2", "phi_12", "phi_jl"],
        approximant="IMRPhenomPv2", waveform_backend=None, priors=_bbh_priors,
        amplitude="luminosity_distance", snr_power=-1.0, ellipticity=_cbc_ellipticity),
    "sg": dict(
        names=["quality", "frequency", "hrss", "phase", "eccentricity", "shifts", "psi",
               "ra", "dec"],
        approximant="SineGaussian", waveform_backend="ml4gw", priors=_sg_priors,
        amplitude="hrss", snr_power=1.0, ellipticity=lambda p: 0.0),
    "cs": dict(
        names=["power", "amplitude", "f_high", "psi", "ra", "dec"],
        approximant="CosmicString", waveform_backend="ml4gw", priors=_cs_priors,
        amplitude="amplitude", snr_power=1.0, ellipticity=lambda p: 0.0),
    "gaussian": dict(
        names=["hrss", "polarization", "eccentricity", "duration", "psi", "ra", "dec"],
        approximant="Gaussian", waveform_backend="ml4gw", priors=_gaussian_priors,
        amplitude="hrss", snr_power=1.0, ellipticity=lambda p: 0.0),
    "wnb": dict(
        names=["frequency", "bandwidth", "eccentricity", "phase", "int_hdot_squared",
               "psi", "ra", "dec", "duration"],
        approximant="WhiteNoiseBurst", waveform_backend="ml4gw", priors=_wnb_priors,
        amplitude="int_hdot_squared", snr_power=0.5, ellipticity=lambda p: 0.0),
}


def build_prior(args):
    """Default prior for ``args.waveform``, with entries from ``--prior-file`` on top."""
    fam = FAMILIES[args.waveform]
    prior = fam["priors"]()
    if args.prior_file:
        override = PriorDict(filename=args.prior_file)
        unknown = set(override) - set(fam["names"])
        if unknown:
            raise SystemExit(f"--prior-file has keys {sorted(unknown)} that are not "
                             f"{args.waveform} parameters {fam['names']}")
        prior.update(override)
    missing = set(fam["names"]) - set(prior)
    if missing:
        raise SystemExit(f"no prior for {sorted(missing)}")
    return prior


def draw_parameters(prior, names, rng):
    """One draw from ``prior`` using ``rng`` (inverse-CDF, no bilby global RNG)."""
    return {k: float(prior[k].rescale(rng.uniform())) for k in names}


def make_noise_and_injector(args, fam, noise_seed, device=None):
    noise = DetectorNoise(args.duration, args.fs, TRIGGER_TIME, DETECTORS,
                          minimum_frequency=args.fmin, maximum_frequency=args.fmax)
    noise.generate_noise(real_noise=False, seed=noise_seed)
    kw = {} if fam["waveform_backend"] is None else {"waveform_backend": fam["waveform_backend"]}
    if args.polarized:
        kw["generator_kwargs"] = {"polarized": True}
    injector = GW(noise, approximant=fam["approximant"], reference_frequency=50.0,
                  parameters=fam["names"], static_parameters={"geocent_time": TRIGGER_TIME},
                  **kw)
    return noise, injector


def pure_signal(injector, theta, torch_seed):
    # ml4gw's WhiteNoiseBurst draws its noise with an unseeded torch.randn, so seed
    # torch before every generation: the injected signal, the "pure signal" used for
    # the SNR, and the CPU gpu-check twin then all see the SAME waveform.
    torch.manual_seed(torch_seed)
    return injector.make_injections_to_ifo_batch(np.array([theta]))[0]


def build_problem(args, params, seed, device):
    """Noise + injection of ``params`` + wavelet likelihood (cf. the per-family scripts)."""
    fam = FAMILIES[args.waveform]
    noise, injector = make_noise_and_injector(args, fam, seed)
    theta = [params[k] for k in fam["names"]]

    torch.manual_seed(seed)
    injector.make_injections_to_ifo(theta)

    f, asd0 = injector.detector_asd_masked(0)
    asd1 = injector.detector_asd_masked(1)[1]
    psd = np.array([asd0**2, asd1**2])
    data = np.array([injector.detector_data_fd(0), injector.detector_data_fd(1)])
    df = f[1] - f[0]

    signal = pure_signal(injector, theta, seed)
    inj_snr = network_optimal_snr(signal, psd, df)

    template = WaveletTemplate(
        DETECTORS, injector.frequency_array_unmasked(), args.duration,
        start_time=noise.start_time, minimum_frequency=args.fmin, maximum_frequency=args.fmax,
        reference_time=TRIGGER_TIME, psd=psd, amplitude_param="snr", gpu=device == "gpu",
    )
    likelihood = WaveletLikelihood(data=data, psd=psd, template=template)
    return dict(template=template, likelihood=likelihood, params=params,
                ellipticity=fam["ellipticity"](params), inj_snr=inj_snr, df=df, freqs=f,
                psd=psd, true_signal=signal, data=data)


def draw_injection(args, prior, index, seed_seq):
    """Parameters + integer seed of injection ``index`` (SNR-rescaled if requested)."""
    fam = FAMILIES[args.waveform]
    rng = np.random.default_rng(seed_seq)
    params = draw_parameters(prior, fam["names"], rng)
    seed = int(rng.integers(0, 2**31 - 1))
    target_snr = None
    if args.snr_range is not None:
        target_snr = float(rng.uniform(*args.snr_range))
        # the SNR does not depend on the noise, so any seed's PSD works here
        _, injector = make_noise_and_injector(args, fam, seed)
        f, asd0 = injector.detector_asd_masked(0)
        psd = np.array([asd0**2, injector.detector_asd_masked(1)[1] ** 2])
        snr0 = network_optimal_snr(pure_signal(injector, [params[k] for k in fam["names"]], seed),
                                   psd, f[1] - f[0])
        params[fam["amplitude"]] *= (target_snr / snr0) ** (1.0 / fam["snr_power"])
    return params, seed, target_snr


def reconstruction_summary(sampler, template, true_signal, psd, *, sample_sky,
                           data=None, fixed_sky=None, n_draws=800, discard_frac=0.3, chunk=200, seed=0,
                           time_domain=True):
    """Posterior reconstruction vs injection (see wavelet_reconstruction.py).

    Network overlap ``<h_rec|h_inj> / sqrt(<h_rec|h_rec><h_inj|h_inj>)`` and
    recovered SNR ``sqrt(<h_rec|h_rec>)`` per posterior draw, plus the median and
    90% band of ``|h(f)|`` (and of the whitened time series) per detector.
    """
    df = float(template.df)
    wav, msk, sky = wr.select_wavelet_draws(
        sampler, sample_sky=sample_sky, discard_frac=discard_frac,
        n_draws=n_draws, seed=seed)
    hrec = wr.reconstruct_fd(template, wav, msk, sky=sky, fixed_sky=fixed_sky, chunk=chunk)

    inv = 1.0 / psd[None]
    tt = float(4 * df * np.sum((true_signal.conj() * true_signal * inv[0]).real))
    rr = 4 * df * np.sum((hrec.conj() * hrec * inv).real, axis=(1, 2))
    tr = 4 * df * np.sum((hrec.conj() * true_signal[None] * inv).real, axis=(1, 2))
    good = rr > 0
    overlap = np.where(good, tr / np.sqrt(np.where(good, rr, 1.0) * tt), 0.0)
    rec_snr = np.sqrt(rr)

    # Residual chi^2 per degree of freedom, <d-h|d-h> / dof per draw. Each complex
    # bin carries 2 real dof (pure noise -> 1 +- sqrt(2/dof)); the network value
    # also subtracts the draw's parameter count (5 per active wavelet, + 4 sky).
    # chi2_dof_noise is the same statistic for d - h_inj (the noise realisation).
    chi2 = {}
    if data is not None:
        data = np.asarray(data)
        nbin = (np.isfinite(inv[0]) & (inv[0] > 0)).sum(axis=-1)            # (n_ifo,)
        resid = data[None] - hrec
        c_ifo = 4 * df * np.sum((resid.conj() * resid * inv).real, axis=2)  # (D, n_ifo)
        ndof = 2 * nbin.sum() - (5 * msk.sum(axis=1) + (4 if sample_sky else 0))
        noise = data - true_signal
        c_noise = 4 * df * np.sum((noise.conj() * noise * inv[0]).real)
        chi2 = dict(chi2_dof=(c_ifo.sum(axis=1) / ndof)[good], chi2_ndof=ndof[good],
                    chi2_dof_ifo=(c_ifo / (2 * nbin))[good],
                    chi2_dof_noise=float(c_noise / (2 * nbin.sum())))

    amp = np.abs(hrec)
    out = dict(overlap=overlap[good], rec_snr=rec_snr[good], inj_snr=float(np.sqrt(tt)),
               median=np.median(amp, axis=0),
               band_lo=np.percentile(amp, 5, axis=0),
               band_hi=np.percentile(amp, 95, axis=0))

    out.update(chi2)

    if time_domain:
        asd = np.sqrt(psd)
        _, h_t = wr.whiten_to_td(hrec, template.frequency_array, template.mask, asd=asd)
        t, inj_t = wr.whiten_to_td(true_signal, template.frequency_array,
                                   template.mask, asd=asd)
        out.update(
            t=t,
            median_t=np.median(h_t, axis=0),
            band_lo_t=np.percentile(h_t, 5, axis=0),
            band_hi_t=np.percentile(h_t, 95, axis=0),
            inj_t=inj_t,
        )
    return out


def amortized_initial_state(ckpt_path, data_noisy, psd, df, nt, nw, lmax, rng):
    """Initial (coords, inds) for the signal branch from the amortized flow."""
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _sys.path.insert(0, os.path.join(_here, "wavelet"))
    from train_amortized_flow import AmortizedWaveletModel, P_LO, P_HI
    from eval_amortized_instant import flow_draws_to_params

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigma = np.sqrt(np.asarray(psd) / (4.0 * float(df)))
    xw = np.asarray(data_noisy) / sigma
    if xw.shape[-1] != ck["n_freq"]:
        raise ValueError(f"ckpt n_freq={ck['n_freq']} != data {xw.shape[-1]} "
                         "(train band/duration must match the analysis)")
    x = np.stack([xw.real, xw.imag], axis=-1)
    xt = torch.from_numpy(
        np.transpose(x, (0, 2, 1)).reshape(1, 4, -1).astype(np.float32))
    model = AmortizedWaveletModel(ck["n_freq"], pool=ck.get("pool", "avg"))
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        emb = model.embed_data(xt)
        pD = torch.softmax(model.d_logits(emb), -1).numpy()[0]
        n_states = nt * nw
        Ds = np.clip(rng.choice(pD.size, size=n_states, p=pD), 1, lmax)
        feats = model.flow(emb.expand(int(Ds.sum()), -1)).sample().numpy()
    wav = flow_draws_to_params(feats)
    eps = 1e-4 * (P_HI - P_LO)
    wav[:, :4] = np.clip(wav[:, :4], (P_LO + eps)[:4], (P_HI - eps)[:4])
    wav[:, 3] = np.maximum(wav[:, 3], 0.05)
    coords = wav[rng.integers(0, wav.shape[0], nt * nw * lmax)].reshape(nt, nw, lmax, 5)
    inds = np.zeros((nt, nw, lmax), bool)
    off = 0
    for s in range(n_states):
        it, iw = divmod(s, nw)
        k = int(Ds[s])
        coords[it, iw, :k] = wav[off:off + k]
        inds[it, iw, :k] = True
        off += k
    return coords, inds


def gpu_check(args, prob, params, seed):
    """GPU likelihood must match a CPU twin built from the same injection."""
    l_cpu = build_problem(args, params, seed, "cpu")["likelihood"]
    likelihood = prob["likelihood"]
    ra, dec, psi, ell = params["ra"], params["dec"], params["psi"], prob["ellipticity"]
    rng_c = np.random.default_rng(123)
    k = [3, 1, 5, 2, 4]
    M = sum(k)
    w = np.column_stack([
        rng_c.uniform(0.5, 3.5, M), rng_c.uniform(args.fmin + 5, args.fmax - 5, M),
        rng_c.uniform(2, 20, M), rng_c.uniform(5, 20, M), rng_c.uniform(0, 2 * np.pi, M)])
    wg = np.concatenate([np.full(n, i) for i, n in enumerate(k)])
    if args.sample_sky:
        ex = np.column_stack([rng_c.uniform(0, 2 * np.pi, len(k)), rng_c.uniform(-1, 1, len(k)),
                              rng_c.uniform(0, np.pi, len(k)), rng_c.uniform(-1, 1, len(k))])
        eg = np.arange(len(k))
        a = likelihood.grouped_log_like_sky([w, ex], [wg, eg], n_groups=len(k))
        b = l_cpu.grouped_log_like_sky([w, ex], [wg, eg], n_groups=len(k))
    else:
        a = likelihood.grouped_log_like(w, wg, ra, dec, psi, ell)
        b = l_cpu.grouped_log_like(w, wg, ra, dec, psi, ell)
    rel = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)
    print(f"[gpu-check] max|GPU-CPU| rel log-like diff = {rel:.2e}")
    assert rel < 1e-6, "GPU and CPU likelihoods disagree!"


def run_inference(args, prob, seed, backend_file=None):
    """Eryn RJ-MCMC wavelet reconstruction of one injection (cf. wavelet_reconstruction.py)."""
    template, likelihood = prob["template"], prob["likelihood"]
    data_noisy, psd, df = prob["data"], prob["psd"], prob["df"]
    params, ellipticity = prob["params"], prob["ellipticity"]
    ra, dec, psi = params["ra"], params["dec"], params["psi"]

    spec = build_wavelet_priors(duration=args.duration, minimum_frequency=args.fmin,
                                maximum_frequency=args.fmax, nleaves_max=args.nleaves_max,
                                amplitude_param="snr", rho_star=args.rho_star)
    rng = np.random.default_rng(seed)
    nt, nw, lmax = args.ntemps, args.nwalkers, args.nleaves_max
    n_walkers_total = nt * nw

    if args.sample_sky:
        # Two branches: variable wavelets (RJ) + a fixed-dim sky leaf per walker,
        # with separate (Gibbs) moves for the wavelets and the sky.
        branch_names = ["signal", "extrinsic"]
        priors = {b: spec["priors"][b] for b in branch_names}
        ndims = {"signal": 5, "extrinsic": 4}
        nmax = {"signal": lmax, "extrinsic": 1}
        nmin = {"signal": 0, "extrinsic": 1}
        moves = [
            (GaussianMove({"signal": np.diag(np.array([0.05, 5.0, 1.0, 1.0, 0.3]) ** 2)},
                          gibbs_sampling_setup="signal"), 0.5),
            (GaussianMove({"extrinsic": np.diag((args.sky_step * np.array([1.0, 1.0, 1.0, 0.5])) ** 2)},
                          gibbs_sampling_setup="extrinsic"), 0.5),
        ]

        def log_like_fn(params, groups):
            return likelihood.grouped_log_like_sky(params, groups)

        coords = {
            "signal": spec["priors"]["signal"].rvs(size=nt * nw * lmax).reshape(nt, nw, lmax, 5),
            "extrinsic": spec["priors"]["extrinsic"].rvs(size=nt * nw).reshape(nt, nw, 1, 4),
        }
        inds = {"signal": np.zeros((nt, nw, lmax), bool),
                "extrinsic": np.ones((nt, nw, 1), bool)}

        def warm():
            wf = coords["signal"][inds["signal"]]
            wg = np.repeat(np.arange(n_walkers_total),
                           inds["signal"].reshape(n_walkers_total, lmax).sum(1))
            ex = coords["extrinsic"][inds["extrinsic"]]
            eg = np.arange(n_walkers_total)
            return likelihood.grouped_log_like_sky([wf, ex], [wg, eg], n_groups=n_walkers_total)
    else:
        # Single branch, sky fixed at the injected values (fast path).
        branch_names = ["signal"]
        priors = {"signal": spec["priors"]["signal"]}
        ndims = {"signal": 5}
        nmax = {"signal": lmax}
        nmin = {"signal": 0}
        moves = [(GaussianMove({"signal": np.diag(np.array([0.05, 5.0, 1.0, 1.0, 0.3]) ** 2)}), 1.0)]

        def log_like_fn(params, groups):
            return likelihood.grouped_log_like(params, groups, ra, dec, psi, ellipticity)

        coords = {"signal": spec["priors"]["signal"].rvs(
            size=nt * nw * lmax).reshape(nt, nw, lmax, 5)}
        inds = {"signal": np.zeros((nt, nw, lmax), bool)}

        def warm():
            wf = coords["signal"][inds["signal"]]
            wg = np.repeat(np.arange(n_walkers_total),
                           inds["signal"].reshape(n_walkers_total, lmax).sum(1))
            return likelihood.grouped_log_like(wf, wg, ra, dec, psi, ellipticity)

    for it in range(nt):
        for iw in range(nw):
            inds["signal"][it, iw, : int(rng.integers(1, 6))] = True

    stopper = None
    stop_kwargs = {}
    if args.converge:
        stopper = WaveletConvergenceStopping(
            nleaves_branch="signal", nleaves_max=lmax,
            sky_branch="extrinsic" if args.sample_sky else None,
            autocorr_mult=args.autocorr_mult, target_ess=args.target_ess,
            pd_tol=args.pd_tol, verbose=True,
        )
        stop_kwargs = dict(stopping_fn=stopper, stopping_iterations=args.check_every)

    # Warm-start wavelets near the signal for data-informed / flow proposals.
    if args.proposal in ("guided", "flow", "fisher", "flowfisher", "mffisher", "mlflow"):
        coords["signal"] = guided_initial_wavelets(
            template, data_noisy, psd, spec, (nt, nw, lmax), rng)
    if args.init == "amortized":
        c_sig, i_sig = amortized_initial_state(
            args.init_ckpt, data_noisy, psd, df, nt, nw, lmax, rng)
        coords["signal"] = c_sig
        inds["signal"] = i_sig

    # BayesWave-style in-model cascade (Fisher dominant + half-cycle + sky-ring).
    if args.sample_sky:
        fisher_inmodel = [
            (WaveletFisherMove(branch_name="signal"), 0.7),
            (WaveletHalfCycleMove(branch_name="signal"), 0.1),
            (WaveletSkyRingMove(template.detector_names, template.reference_time,
                                branch_name="extrinsic"), 0.1),
        ]
    else:
        fisher_inmodel = [
            (WaveletFisherMove(branch_name="signal"), 0.8),
            (WaveletHalfCycleMove(branch_name="signal"), 0.1),
        ]

    if args.stretch:
        from hyperwave.inference.wavelet_proposals import WaveletGroupStretchMove
        stretch = WaveletGroupStretchMove(
            nfriends=args.stretch_friends, n_iter_update=args.stretch_update,
            gibbs_sampling_setup="signal", a=args.stretch_a)
        scale = 1.0 - args.stretch
        fisher_inmodel = [(m, w * scale) for (m, w) in fisher_inmodel]
        fisher_inmodel.append((stretch, args.stretch))

    device = "cuda" if (args.device == "gpu" and torch_cuda_available()) else "cpu"
    update_kwargs = {}
    if args.proposal in ("guided", "fisher", "mffisher"):
        if args.proposal == "mffisher":
            guided = build_mf_birth(template, data_noisy, psd, spec)
        else:
            guided = build_guided_birth(template, data_noisy, psd, spec)
        gen = {b: guided[b] for b in branch_names}
        if args.proposal in ("fisher", "mffisher") and args.num_try > 1:
            # MTDistGenMoveRJ changes one model at a time: signal branch only
            # (see wavelet_reconstruction.py for the details).
            gen_signal = {"signal": guided["signal"]}
            mt_kw = dict(num_try=args.num_try, nleaves_max=nmax, nleaves_min=nmin)
            if "extrinsic" in branch_names:
                mt_kw["gibbs_sampling_setup"] = "signal"
            rj_moves = MTDistGenMoveRJ(gen_signal, **mt_kw)
        else:
            rj_moves = DistributionGenerateRJ(gen, nleaves_max=nmax, nleaves_min=nmin)
        if args.proposal in ("fisher", "mffisher"):
            moves = fisher_inmodel + [(m, 0.1 * w) for (m, w) in moves]
    elif args.proposal == "fisherflow":
        flow = build_flow_proposal(args.duration, minimum_frequency=args.fmin,
                                   maximum_frequency=args.fmax, device=device,
                                   train_config={"epochs": args.flow_epochs})
        flow_branch = {"signal": flow}
        guided = build_guided_birth(template, data_noisy, psd, spec)
        gen = {b: guided[b] for b in branch_names}
        rj_moves = DistributionGenerateRJ(gen, nleaves_max=nmax, nleaves_min=nmin)
        callback = FlowTrainingCallback(
            flow_branch, every=args.flow_train_every, verbose=True,
            temperature_indices=(0,) if args.flow_train_temps == "cold" else None)
        update_kwargs = dict(update_fn=callback, update_iterations=args.flow_train_every)
        moves = ([(make_flow_distribution_move(flow_branch, gibbs_sampling_setup="signal"), 0.3)]
                 + fisher_inmodel + [(m, 0.1 * w) for (m, w) in moves])
    elif args.proposal == "flowfisher":
        flow = build_flow_proposal(args.duration, minimum_frequency=args.fmin,
                                   maximum_frequency=args.fmax, device=device,
                                   train_config={"epochs": args.flow_epochs})
        flow_branch = {"signal": flow}
        rj_gen = dict(flow_branch)
        if "extrinsic" in branch_names:
            rj_gen["extrinsic"] = spec["extrinsic"]
        rj_moves = make_flow_rj_move(rj_gen, nleaves_max=nmax, nleaves_min=nmin)
        callback = FlowTrainingCallback(
            flow_branch, every=args.flow_train_every, verbose=True,
            temperature_indices=(0,) if args.flow_train_temps == "cold" else None)
        update_kwargs = dict(update_fn=callback, update_iterations=args.flow_train_every)
        moves = fisher_inmodel + [(m, 0.1 * w) for (m, w) in moves]
    elif args.proposal == "mlflow":
        if not args.mlflow_ckpt:
            raise SystemExit("--proposal mlflow requires --mlflow-ckpt PATH "
                             "(train one with examples/ml/train_wavelet_flow.py)")
        from hyperwave.ml.proposals import WaveletFlowDistribution, load_flow_model
        from hyperwave.detectors.waveforms.wavelets import morlet_gabor_fd, amplitude_from_snr
        flow_model = load_flow_model(args.mlflow_ckpt, device=device)
        flow_birth = WaveletFlowDistribution(flow_model, device=device)
        flow_birth.set_residual(data_noisy)
        rj_gen = {"signal": flow_birth}
        if "extrinsic" in branch_names:
            rj_gen["extrinsic"] = spec["extrinsic"]
        rj_moves = DistributionGenerateRJ(rj_gen, nleaves_max=nmax, nleaves_min=nmin)
        _fvec_masked = template.frequency_array_masked()

        def _refresh_residual(iter_idx, last_sample, sampler_):
            # residual = data minus the cold-walker-0 wavelet model
            try:
                signal_leaves = last_sample.branches["signal"].coords
                live_inds = last_sample.branches["signal"].inds
                live = signal_leaves[0, 0][live_inds[0, 0]]
                if live.shape[0] == 0:
                    flow_birth.set_residual(data_noisy)
                    return
                s_ref = np.interp(live[:, 1], _fvec_masked, psd[0])
                amps = amplitude_from_snr(live[:, 3], live[:, 1], live[:, 2], s_ref)
                h_p = np.sum(morlet_gabor_fd(_fvec_masked, live[:, 0], live[:, 1],
                                             live[:, 2], amps, live[:, 4]), axis=0)
                flow_birth.set_residual(data_noisy - np.broadcast_to(h_p, data_noisy.shape))
            except Exception as exc:  # pragma: no cover - keep sampler running
                print(f"[mlflow] residual refresh skipped: {exc}")
        update_kwargs = dict(update_fn=_refresh_residual,
                             update_iterations=args.flow_train_every)
        moves = fisher_inmodel + [(m, 0.1 * w) for (m, w) in moves]
    elif args.proposal == "flow":
        flow = build_flow_proposal(args.duration, minimum_frequency=args.fmin,
                                   maximum_frequency=args.fmax, device=device,
                                   train_config={"epochs": args.flow_epochs})
        flow_branch = {"signal": flow}
        moves = [(make_flow_distribution_move(flow_branch, gibbs_sampling_setup="signal"), 0.5)] + moves
        rj_gen = dict(flow_branch)
        if "extrinsic" in branch_names:
            rj_gen["extrinsic"] = spec["extrinsic"]
        rj_moves = make_flow_rj_move(rj_gen, nleaves_max=nmax, nleaves_min=nmin)
        callback = FlowTrainingCallback(
            flow_branch, every=args.flow_train_every, verbose=True,
            temperature_indices=(0,) if args.flow_train_temps == "cold" else None)
        update_kwargs = dict(update_fn=callback, update_iterations=args.flow_train_every)
    else:
        rj_moves = True

    temp_kwargs = dict(ntemps=nt)
    if args.proposal in ("fisher", "flowfisher"):
        temp_kwargs.update(Tmax=np.inf, adaptive=True)

    if backend_file and os.path.exists(backend_file):
        os.remove(backend_file)  # always a fresh chain for a (re-)run injection

    sampler = EnsembleSampler(
        nw, ndims, log_like_fn, priors,
        tempering_kwargs=temp_kwargs,
        nbranches=len(branch_names), branch_names=branch_names,
        nleaves_max=nmax, nleaves_min=nmin,
        provide_groups=True, vectorize=True,
        moves=moves, rj_moves=rj_moves,
        fill_zero_leaves_val=likelihood.empty_log_likelihood,
        periodic=spec["periodic"],
        backend=backend_file,
        **update_kwargs,
        **stop_kwargs,
    )
    state = State(coords, inds=inds)
    burn = args.burn if args.burn is not None else args.nsteps // 2
    _ = warm()  # JIT/transfer one batch off the clock

    t1 = time.perf_counter()
    sampler.run_mcmc(state, args.nsteps, burn=burn, progress=args.progress, thin_by=args.thin)
    sample_t = time.perf_counter() - t1

    sampled_steps = int(np.asarray(sampler.get_log_like()).shape[0])
    nleaves = sampler.get_nleaves()["signal"][:, 0].astype(int)  # cold chain
    try:
        im_acc = float(np.mean(np.asarray(sampler.acceptance_fraction)[0]))
        rj_acc = float(np.mean(np.asarray(sampler.rj_acceptance_fraction)[0]))
    except Exception:
        im_acc = rj_acc = float("nan")
    converged = bool(stopper and stopper.last.get("consecutive", 0) >= stopper.n_consecutive)

    fixed_sky = None if args.sample_sky else (ra, dec, psi, ellipticity)
    summ = reconstruction_summary(sampler, template, prob["true_signal"], psd,
                                  data=prob["data"], sample_sky=args.sample_sky, fixed_sky=fixed_sky,
                                  n_draws=args.draws, seed=seed)
    chain = sampler.get_chain()
    extrinsic = (chain["extrinsic"][:, 0].reshape(-1, 4).astype(np.float32)
                 if "extrinsic" in chain else np.zeros((0, 4), np.float32))
    return dict(summ=summ, nleaves=nleaves, extrinsic=extrinsic, converged=converged,
                sampled_steps=sampled_steps, burn=burn, sample_seconds=sample_t,
                im_acc=im_acc, rj_acc=rj_acc, stopper=(stopper.last if stopper else {}))


def save_injection(path, args, index, seed, params, target_snr, prob, res):
    summ = res["summ"]
    td = {k: summ[k] for k in ("t", "median_t", "band_lo_t", "band_hi_t", "inj_t") if k in summ}
    stop_info = {f"stop_{k}": v for k, v in res["stopper"].items()}
    np.savez(path, index=index, seed=seed, waveform=args.waveform, polarized=args.polarized,
             injection=json.dumps(params), target_snr=np.nan if target_snr is None else target_snr,
             inj_snr=prob["inj_snr"], converged=res["converged"],
             nleaves=res["nleaves"], extrinsic=res["extrinsic"],
             inj_sky=np.array([params["ra"], params["dec"], params["psi"]]),
             sampled_steps=res["sampled_steps"], burn=res["burn"],
             sample_seconds=res["sample_seconds"], im_acc=res["im_acc"], rj_acc=res["rj_acc"],
             device=prob["template"].backend_name, overlap=summ["overlap"],
             rec_snr=summ["rec_snr"], recon_median=summ["median"],
             recon_band_lo=summ["band_lo"], recon_band_hi=summ["band_hi"],
             freqs=prob["freqs"], true_signal=prob["true_signal"],
             **{k: v for k, v in summ.items() if k.startswith("chi2")},
             **td, **stop_info)


def save_recon_plots(base, prob, summ):
    asd = np.sqrt(prob["psd"])
    for j, ifo in enumerate(DETECTORS):
        fd = dict(median=summ["median"][j], lower=summ["band_lo"][j], upper=summ["band_hi"][j])
        wr.plot_wavelet_fd(prob["freqs"], fd, ifo, signal_fd=prob["true_signal"][j], asd=asd[j],
                           df=prob["df"], outpath=f"{base}_{ifo}_fd.png", also_pdf=False)
        if "t" in summ:
            td = dict(median=summ["median_t"][j], lower=summ["band_lo_t"][j],
                      upper=summ["band_hi_t"][j])
            wr.plot_wavelet_td(summ["t"], td, ifo, signal_td=summ["inj_t"][j],
                               outpath=f"{base}_{ifo}_td.png", also_pdf=False)


def summarize(outdir):
    """Aggregate every ``inj_*.npz`` in ``outdir`` into a JSON table + plot."""
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, "inj_*.npz"))):
        r = np.load(path, allow_pickle=False)
        ov, rs = r["overlap"], r["rec_snr"]
        vals, counts = np.unique(r["nleaves"], return_counts=True)
        q = lambda a: [float(np.percentile(a, p)) for p in (5, 50, 95)] if a.size else [np.nan] * 3
        rows.append(dict(
            index=int(r["index"]), seed=int(r["seed"]), waveform=str(r["waveform"]),
            injection=json.loads(str(r["injection"])), inj_snr=float(r["inj_snr"]),
            overlap_5_50_95=q(ov), rec_snr_5_50_95=q(rs),
            nwavelets_mode=int(vals[np.argmax(counts)]), converged=bool(r["converged"]),
            sampled_steps=int(r["sampled_steps"]), sample_seconds=float(r["sample_seconds"]),
            im_acc=float(r["im_acc"]), rj_acc=float(r["rj_acc"]), file=os.path.basename(path)))
    if not rows:
        print(f"[summary] no inj_*.npz in {outdir}")
        return rows
    with open(os.path.join(outdir, "campaign_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)

    snr = np.array([r["inj_snr"] for r in rows])
    ov = np.array([r["overlap_5_50_95"] for r in rows])
    print(f"\n========== CAMPAIGN ({len(rows)} injections) ==========")
    print(f"{'idx':>5} {'inj SNR':>8} {'overlap med [90%]':>24} {'rec SNR':>8} {'D':>3} "
          f"{'conv':>5} {'min':>6}")
    for r in rows:
        o = r["overlap_5_50_95"]
        print(f"{r['index']:5d} {r['inj_snr']:8.1f} {o[1]:8.3f} [{o[0]:.3f}, {o[2]:.3f}] "
              f"{r['rec_snr_5_50_95'][1]:8.1f} {r['nwavelets_mode']:3d} "
              f"{'yes' if r['converged'] else 'no':>5} {r['sample_seconds'] / 60:6.1f}")
    print(f"median overlap over injections: {np.nanmedian(ov[:, 1]):.3f}   "
          f"converged: {sum(r['converged'] for r in rows)}/{len(rows)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(snr, ov[:, 1], yerr=[ov[:, 1] - ov[:, 0], ov[:, 2] - ov[:, 1]],
                fmt="o", ms=4, capsize=2)
    ax.set_xlabel("injected network SNR")
    ax.set_ylabel("network overlap (median, 90% CI)")
    ax.set_ylim(min(0.0, float(np.nanmin(ov[:, 0]))), 1.02)
    ax.set_title(f"{rows[0]['waveform']} wavelet reconstruction, {len(rows)} injections")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "campaign_overlap_vs_snr.png"), dpi=150)
    plt.close(fig)
    print(f"summary -> {os.path.join(outdir, 'campaign_summary.json')} "
          f"(+ campaign_overlap_vs_snr.png)")
    return rows


def free_device_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # campaign
    p.add_argument("--waveform", choices=sorted(FAMILIES), required=True,
                   help="source family to inject")
    p.add_argument("--n-injections", type=int, default=10)
    p.add_argument("--indices", type=int, nargs="+", default=None,
                   help="run only these injection indices (e.g. $SLURM_ARRAY_TASK_ID); "
                        "default: all of range(n-injections)")
    p.add_argument("--prior-file", type=str, default=None,
                   help="bilby .prior file; its entries replace the family defaults")
    p.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="rescale each injection's amplitude parameter so its network "
                        "optimal SNR is uniform in [LO, HI] (default: use the drawn amplitude)")
    p.add_argument("--outdir", type=str, default="results/injection_campaign")
    p.add_argument("--polarized", action="store_true",
                   help="wnb only: build h_cross from the same noise as h_plus (ml4gw WhiteNoiseBurst(polarized=True)), so the burst is elliptically polarized like the wavelet model instead of unpolarized")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="re-run injections whose inj_XXXX.npz already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="draw and inject only: print the parameters and SNRs, no sampling")
    p.add_argument("--summarize-only", action="store_true",
                   help="only aggregate existing inj_*.npz in --outdir")
    p.add_argument("--recon-plots", action="store_true",
                   help="save per-injection FD/TD reconstruction plots")
    p.add_argument("--hdf-backend", action="store_true",
                   help="stream each chain to <outdir>/inj_XXXX_chain.h5 (Eryn HDFBackend) "
                        "instead of RAM -- use for long, high-nleaves runs")
    p.add_argument("--progress", action="store_true", help="show Eryn's progress bar")
    # data / sampler (as in the single-injection scripts)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fs", type=float, default=2048.0)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=512.0)
    p.add_argument("--nwalkers", type=int, default=50)
    p.add_argument("--ntemps", type=int, default=10)
    p.add_argument("--nsteps", type=int, default=2000)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--burn", type=int, default=None)
    p.add_argument("--nleaves-max", type=int, default=20)
    p.add_argument("--rho-star", type=float, default=5.0)
    p.add_argument("--sky-step", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0, help="campaign base seed")
    p.add_argument("--sample-sky", dest="sample_sky", action="store_true", default=True)
    p.add_argument("--fixed-sky", dest="sample_sky", action="store_false")
    p.add_argument("--converge", dest="converge", action="store_true", default=True)
    p.add_argument("--no-converge", dest="converge", action="store_false")
    p.add_argument("--check-every", type=int, default=200)
    p.add_argument("--autocorr-mult", type=float, default=50.0)
    p.add_argument("--target-ess", type=float, default=2000.0)
    p.add_argument("--pd-tol", type=float, default=0.02)
    p.add_argument("--proposal", choices=["standard", "guided", "flow", "fisher", "flowfisher",
                                          "fisherflow", "mffisher", "mlflow"],
                   default="standard", help="see wavelet_reconstruction.py --help")
    p.add_argument("--mlflow-ckpt", type=str, default=None)
    p.add_argument("--num-try", type=int, default=1)
    p.add_argument("--stretch", type=float, default=0.0)
    p.add_argument("--stretch-a", type=float, default=2.0)
    p.add_argument("--stretch-friends", type=int, default=40)
    p.add_argument("--stretch-update", type=int, default=50)
    p.add_argument("--flow-train-temps", choices=["cold", "all"], default="cold")
    p.add_argument("--flow-train-every", type=int, default=50)
    p.add_argument("--flow-epochs", type=int, default=200)
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--init", choices=["default", "amortized"], default="default")
    p.add_argument("--init-ckpt", type=str, default="results/amortized_flow.pt")
    args = p.parse_args()
    if args.polarized and args.waveform != "wnb":
        p.error("--polarized only applies to --waveform wnb")

    os.makedirs(args.outdir, exist_ok=True)
    if args.summarize_only:
        summarize(args.outdir)
        return

    prior = build_prior(args)
    print(f"[prior] {args.waveform}:")
    for k in FAMILIES[args.waveform]["names"]:
        print(f"    {k:>20s}: {prior[k]}")
    if args.snr_range is not None:
        print(f"    amplitude '{FAMILIES[args.waveform]['amplitude']}' rescaled to "
              f"SNR ~ U{tuple(args.snr_range)}")

    seed_seqs = np.random.SeedSequence(args.seed).spawn(args.n_injections)
    indices = args.indices if args.indices is not None else range(args.n_injections)
    checked_gpu = False
    for i in indices:
        if not 0 <= i < args.n_injections:
            raise SystemExit(f"injection index {i} outside range({args.n_injections})")
        path = os.path.join(args.outdir, f"inj_{i:04d}.npz")
        if args.resume and os.path.exists(path) and not args.dry_run:
            print(f"[inj {i}] exists, skipping ({path})")
            continue

        params, seed, target_snr = draw_injection(args, prior, i, seed_seqs[i])
        t0 = time.perf_counter()
        prob = build_problem(args, params, seed, args.device)
        pstr = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
        print(f"\n[inj {i}] seed={seed} injected_network_SNR={prob['inj_snr']:.1f}"
              + (f" (target {target_snr:.1f})" if target_snr is not None else "")
              + f" ({time.perf_counter() - t0:.2f}s)\n    {pstr}")
        if args.dry_run:
            continue

        if prob["template"].backend_name != "numpy" and not checked_gpu:
            gpu_check(args, prob, params, seed)
            checked_gpu = True

        backend_file = (os.path.join(args.outdir, f"inj_{i:04d}_chain.h5")
                        if args.hdf_backend else None)
        res = run_inference(args, prob, seed, backend_file=backend_file)
        ov, rs = res["summ"]["overlap"], res["summ"]["rec_snr"]
        print(f"[inj {i}] {'CONVERGED' if res['converged'] else 'hit step cap'} after "
              f"{res['sampled_steps']} steps, {res['sample_seconds'] / 60:.1f} min | "
              f"acc in-model {res['im_acc']:.3f} RJ {res['rj_acc']:.3f} | "
              f"overlap {np.median(ov):.3f} [{np.percentile(ov, 5):.3f}, "
              f"{np.percentile(ov, 95):.3f}] | rec SNR {np.median(rs):.1f} "
              f"(inj {prob['inj_snr']:.1f})")
        save_injection(path, args, i, seed, params, target_snr, prob, res)
        if args.recon_plots:
            save_recon_plots(os.path.join(args.outdir, f"inj_{i:04d}"), prob, res["summ"])
        del res, prob
        free_device_memory()

    if not args.dry_run:
        summarize(args.outdir)


if __name__ == "__main__":
    main()
