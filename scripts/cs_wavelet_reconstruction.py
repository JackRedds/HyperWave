"""Timed BBH injection + waveform-agnostic wavelet reconstruction (CPU/GPU).

End-to-end demo:

1. generate detector noise (analytic design PSD) for a network,
2. inject a CBC (IMRPhenomPv2 via the LAL backend) into the data,
3. reconstruct it with a *variable* number of Morlet-Gabor wavelets using Eryn
   reversible-jump MCMC, with the induced-SNR prior, and
4. report wall-clock timing (and project it against a reference runtime).

The reconstruction uses the GPU-friendly batched path: a single Eryn branch with
``vectorize=True, provide_groups=True`` hands *all* active wavelets across *all*
walkers to :meth:`WaveletLikelihood.grouped_log_like` in one call, which
generates them in one batched ``exp`` and does one batched inner product on the
selected device (NumPy or CuPy).

Run::

    python examples/bbh_wavelet_reconstruction.py --device gpu --nsteps 2000
    python examples/bbh_wavelet_reconstruction.py --device cpu --nsteps 500

The sky location and ellipticity are held fixed at the injected values (the
fast, GPU-batched regime); sampling them jointly is a documented extension.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
import torch

import numpy as np

# Eryn (current releases) still call np.in1d, removed in NumPy 2.0.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from eryn.ensemble import EnsembleSampler  # noqa: E402
from eryn.moves import GaussianMove  # noqa: E402
from eryn.state import State  # noqa: E402

from hyperwave.detectors.lvk import DetectorNoise, GW  # noqa: E402
from hyperwave.detectors.waveforms import WaveletTemplate, network_optimal_snr  # noqa: E402
from eryn.moves import (DistributionGenerateRJ, GaussianMove, MTDistGenMoveRJ,  # noqa: E402,F811
                        GroupStretchMove)
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

CS_PARAMETER_NAMES = [
    "power", "amplitude", "f_high", "psi",
    "ra", "dec",
]


RA_INJ, DEC_INJ, PSI_INJ = 1.375, -0.2108, 1.1


def make_cs(seed):
    power = -4.0 / 3.0
    amplitude = 6.0e-21
    f_high = 1000.0
    params = dict(
        power=power, amplitude=amplitude, f_high=f_high, psi=PSI_INJ,
        ra=RA_INJ, dec=DEC_INJ,
    )
    theta = [params[k] for k in CS_PARAMETER_NAMES]
    # CBC cross-polarisation convention (lalsimulation): h_cross = -eps * i * h_plus,
    # with eps = 2 cos(iota) / (1 + cos^2 iota). The sign matters for the fixed-sky
    # path; when the sky is sampled, ellipticity is free over [-1, 1].
    ellipticity = 0.0
    return params, theta, ellipticity


def build_problem(args):
    fmin, fmax = args.fmin, args.fmax
    detectors = ["H1", "L1"]
    trigger_time = 1268189526.951953

    noise = DetectorNoise(args.duration, args.fs, trigger_time, detectors,
                          minimum_frequency=fmin, maximum_frequency=fmax)
    noise.generate_noise(real_noise=False, seed=args.seed)

    params, theta, ellipticity = make_cs(args.seed)
    injector = GW(noise, approximant="CosmicString", reference_frequency=50.0,
                  parameters=CS_PARAMETER_NAMES,
                  static_parameters={"geocent_time": trigger_time},
                  waveform_backend='ml4gw')

    torch.manual_seed(args.seed)
    injector.make_injections_to_ifo(theta)  # add the WNB to the data

    f, asd0 = injector.detector_asd_masked(0)
    asd1 = injector.detector_asd_masked(1)[1]
    psd = np.array([asd0**2, asd1**2])
    data = np.array([injector.detector_data_fd(0), injector.detector_data_fd(1)])
    df = f[1] - f[0]

    # pure injected signal (no data mutation) for the network optimal SNR
    torch.manual_seed(args.seed)
    signal = injector.make_injections_to_ifo_batch(np.array([theta]))[0]
    inj_snr = network_optimal_snr(signal, psd, df)

    gpu = args.device == "gpu"
    template = WaveletTemplate(
        detectors, injector.frequency_array_unmasked(), args.duration,
        start_time=noise.start_time, minimum_frequency=fmin, maximum_frequency=fmax,
        reference_time=trigger_time, psd=psd, amplitude_param="snr", gpu=gpu,
    )
    likelihood = WaveletLikelihood(data=data, psd=psd, template=template)
    return template, likelihood, params, ellipticity, inj_snr, df, f, psd, signal, data


def reconstruction_summary(sampler, template, true_signal, psd, *, sample_sky,
                           data=None, fixed_sky=None, n_draws=800, discard_frac=0.3, chunk=200, seed=0,
                           time_domain=True):
    """Compare the posterior wavelet reconstruction with the injected signal.

    Returns network overlap (match) and recovered SNR per posterior draw plus the
    median reconstruction and a 90% amplitude band per detector. The overlap is
    the noise-weighted, normalised inner product
    ``O = <h_rec|h_inj> / sqrt(<h_rec|h_rec><h_inj|h_inj>)`` summed over detectors
    (1.0 = perfect shape match); the recovered SNR is ``sqrt(<h_rec|h_rec>)``.

    All resolution-dependent quantities (``df`` and the time grid) come from
    ``template`` -- the actual data, not a fixed 4 s / 2048 Hz segment.
    Frequency-domain bands and the median are summarised on the **amplitude**
    ``|h(f)|`` (not the complex parts) so the median is consistent with the band;
    ``time_domain`` adds a whitened time-domain reconstruction.
    """
    df = float(template.df)                              # actual frequency resolution
    # Reuse the library's wavelet reconstruction (chain -> per-draw waveforms);
    # the example only adds the overlap/SNR metrics on top.
    wav, msk, sky = wr.select_wavelet_draws(
        sampler, sample_sky=sample_sky, discard_frac=discard_frac,
        n_draws=n_draws, seed=seed)
    hrec = wr.reconstruct_fd(template, wav, msk, sky=sky, fixed_sky=fixed_sky, chunk=chunk)

    inv = 1.0 / psd[None]                                # (1, n_ifo, n_freq)
    tt = float(4 * df * np.sum((true_signal.conj() * true_signal * inv[0]).real))
    rr = 4 * df * np.sum((hrec.conj() * hrec * inv).real, axis=(1, 2))     # (D,)
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

    # Frequency-domain amplitude summary: percentiles of |h(f)| (consistent
    # median/band; see hyperwave.plots.wavelet_reconstruction for why).
    amp = np.abs(hrec)
    out = dict(overlap=overlap[good], rec_snr=rec_snr[good], inj_snr=float(np.sqrt(tt)),
               median=np.median(amp, axis=0),
               band_lo=np.percentile(amp, 5, axis=0),
               band_hi=np.percentile(amp, 95, axis=0))

    # Whitened time-domain reconstruction (real strain -> pointwise percentiles).
    # The time grid is derived from the template's own frequency array.
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
    """Initial (coords, inds) for the signal branch from the amortized flow.

    Embeds the whitened data once, then per walker draws D ~ p(D|d) and D
    leaves from the conditional flow. The flow nails the TF support (envelope
    overlap ~0.57) but not coherent phases — exactly what an MCMC initial
    state needs: right region, sampler supplies the precision.
    """
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _sys.path.insert(0, os.path.join(_here, "wavelet"))
    import torch
    from train_amortized_flow import AmortizedWaveletModel, P_LO, P_HI
    from eval_amortized_instant import flow_draws_to_params

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sigma = np.sqrt(np.asarray(psd) / (4.0 * float(df)))
    xw = np.asarray(data_noisy) / sigma
    if xw.shape[-1] != ck["n_freq"]:
        raise ValueError(f"ckpt n_freq={ck['n_freq']} != data {xw.shape[-1]} "
                         "(train band/duration must match the analysis)")
    x = np.stack([xw.real, xw.imag], axis=-1)                    # (n_ifo, nf, 2)
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
    # keep strictly inside the prior support (edges can be -inf under eryn priors)
    eps = 1e-4 * (P_HI - P_LO)
    wav[:, :4] = np.clip(wav[:, :4], (P_LO + eps)[:4], (P_HI - eps)[:4])
    wav[:, 3] = np.maximum(wav[:, 3], 0.05)                      # SNR floor
    coords = wav[rng.integers(0, wav.shape[0], nt * nw * lmax)].reshape(nt, nw, lmax, 5)
    inds = np.zeros((nt, nw, lmax), bool)
    off = 0
    for s in range(n_states):
        it, iw = divmod(s, nw)
        k = int(Ds[s])
        coords[it, iw, :k] = wav[off:off + k]
        inds[it, iw, :k] = True
        off += k
    print(f"[init] amortized: E[D|d]={float((np.arange(pD.size)*pD).sum()):.1f}  "
          f"walker D range [{Ds.min()},{Ds.max()}]")
    return coords, inds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fs", type=float, default=2048.0)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=512.0)
    p.add_argument("--nwalkers", type=int, default=50)
    p.add_argument("--ntemps", type=int, default=10)
    p.add_argument("--nsteps", type=int, default=2000)
    p.add_argument("--thin", type=int, default=1,
                   help="store every Nth sample (thin_by) -- caps the in-memory "
                        "chain so long, high-nleaves runs do not OOM")
    p.add_argument("--burn", type=int, default=None)
    p.add_argument("--nleaves-max", type=int, default=20)
    p.add_argument("--rho-star", type=float, default=5.0,
                   help="peak of the per-wavelet induced-SNR prior (rho_*); raise "
                        "(e.g. 8-10) to allow higher per-wavelet SNR / more recovered power")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inj-ra", type=float, default=None, help="override injected RA [rad]")
    p.add_argument("--inj-dec", type=float, default=None, help="override injected Dec [rad]")
    p.add_argument("--inj-psi", type=float, default=None, help="override injected psi [rad]")
    p.add_argument("--sample-sky", dest="sample_sky", action="store_true", default=True,
                   help="sample sky position ra/dec/psi/ellipticity (default)")
    p.add_argument("--fixed-sky", dest="sample_sky", action="store_false",
                   help="hold the sky fixed at the injected values (fast single-branch path)")
    p.add_argument("--outfile", type=str, default=None)
    p.add_argument("--production-steps", type=int, default=5000,
                   help="reference production length for the timing projection")
    p.add_argument("--reference-hours", type=float, default=4.0,
                   help="reference pipeline runtime (hours) for the timing projection")
    # Convergence-based stopping (nsteps becomes the hard maximum cap).
    p.add_argument("--converge", dest="converge", action="store_true", default=True,
                   help="stop on autocorrelation + model-order convergence (default; nsteps is the cap)")
    p.add_argument("--no-converge", dest="converge", action="store_false",
                   help="disable convergence stopping and run exactly nsteps")
    p.add_argument("--check-every", type=int, default=200,
                   help="convergence check cadence (sampler iterations)")
    p.add_argument("--autocorr-mult", type=float, default=50.0,
                   help="require chain length > autocorr_mult * tau")
    p.add_argument("--target-ess", type=float, default=2000.0,
                   help="require effective sample size above this")
    p.add_argument("--pd-tol", type=float, default=0.02,
                   help="split-half total-variation tolerance on p(n_wavelets)")
    p.add_argument("--proposal", choices=["standard", "guided", "flow", "fisher", "flowfisher", "fisherflow", "mffisher", "mlflow"], default="standard",
                   help="'standard' = prior-draw RJ births; 'guided' = data-informed "
                        "time-frequency births; 'fisher' = guided births + BayesWave-style "
                        "local Fisher + half-cycle in-model moves (best mixing); 'flow' = learned "
                        "normalizing-flow proposals for births and parameters (+guided warm start); "
                        "'mffisher' = matched-filter births (data-fitted SNR+phase) + Fisher in-model; "
                        "'mlflow' = pretrained source-agnostic wavelet flow (Path A) loaded from "
                        "--mlflow-ckpt + Fisher in-model")
    p.add_argument("--mlflow-ckpt", type=str, default=None,
                   help="path to a flow_birth.pt checkpoint (proposal=mlflow); required when "
                        "--proposal mlflow is selected")
    p.add_argument("--num-try", type=int, default=1,
                   help="multiple-try births (proposal=fisher): propose N candidate "
                        "wavelets per birth and keep the best by likelihood (N>1)")
    p.add_argument("--stretch", type=float, default=0.0,
                   help="Tier-1 mixing lever: weight (0-1) of an affine-invariant "
                        "Stretch ensemble move on the signal block (BayesWave-DE analogue). "
                        "0 disables; 0.3 is a good starting share.")
    p.add_argument("--stretch-a", type=float, default=2.0,
                   help="Stretch move scale parameter a (eryn default 2.0)")
    p.add_argument("--stretch-friends", type=int, default=40,
                   help="GroupStretchMove: size of the reference 'friends' group")
    p.add_argument("--stretch-update", type=int, default=50,
                   help="GroupStretchMove: refresh the friends group every N iters")
    p.add_argument("--flow-train-temps", choices=["cold", "all"], default="cold",
                   help="chains that train the flow: cold (T=0, paper default) or all "
                        "(hot chains broaden the proposal; may aid tempering exchange)")
    p.add_argument("--flow-train-every", type=int, default=50,
                   help="retrain the flow every N sampler iterations (proposal=flow)")
    p.add_argument("--flow-epochs", type=int, default=200,
                   help="max epochs per flow retrain (proposal=flow)")
    p.add_argument("--draws", type=int, default=800,
                   help="posterior draws used for the reconstruction-vs-injection comparison")
    p.add_argument("--recon-plot", type=str, default=None,
                   help="path to save the frequency-domain reconstruction plot")
    p.add_argument("--init", choices=["default", "amortized"], default="default",
                   help="'amortized' = initialise walker states (D and leaves) from "
                        "the trained amortized flow p(D, theta | whitened data) "
                        "instead of prior/guided draws")
    p.add_argument("--init-ckpt", type=str, default="results/amortized_flow.pt",
                   help="amortized-flow checkpoint (--init amortized)")
    args = p.parse_args()

    # apply injection-sky overrides BEFORE any build_problem call, so the GPU
    # problem and its CPU gpu-check twin are built from the SAME injection
    global RA_INJ, DEC_INJ, PSI_INJ
    if args.inj_ra is not None: RA_INJ = args.inj_ra
    if args.inj_dec is not None: DEC_INJ = args.inj_dec
    if args.inj_psi is not None: PSI_INJ = args.inj_psi

    t0 = time.perf_counter()
    (template, likelihood, params, ellipticity, inj_snr, df, freqs, psd, true_signal,
     data_noisy) = build_problem(args)
    ra, dec, psi = params["ra"], params["dec"], params["psi"]
    setup_t = time.perf_counter() - t0
    print(f"[setup] device={template.backend_name} band=[{args.fmin},{args.fmax}]Hz "
          f"n_freq={template.frequency_array_masked().size} injected_network_SNR={inj_snr:.1f} "
          f"({setup_t:.2f}s)")

    spec = build_wavelet_priors(duration=args.duration, minimum_frequency=args.fmin,
                                maximum_frequency=args.fmax, nleaves_max=args.nleaves_max,
                                amplitude_param="snr", rho_star=args.rho_star)
    rng = np.random.default_rng(args.seed)
    nt, nw, lmax = args.ntemps, args.nwalkers, args.nleaves_max
    n_walkers_total = nt * nw

    if args.sample_sky:
        # Two branches: variable wavelets (RJ) + a fixed-dim sky leaf per walker.
        branch_names = ["signal", "extrinsic"]
        priors = {b: spec["priors"][b] for b in branch_names}
        ndims = {"signal": 5, "extrinsic": 4}
        nmax = {"signal": lmax, "extrinsic": 1}
        nmin = {"signal": 0, "extrinsic": 1}
        moves = [(GaussianMove({
            "signal": np.diag(np.array([0.05, 5.0, 1.0, 1.0, 0.3]) ** 2),
            "extrinsic": np.diag(np.array([0.1, 0.1, 0.1, 0.05]) ** 2),
        }), 1.0)]

        def log_like_fn(params, groups):
            return likelihood.grouped_log_like_sky(params, groups)

        coords = {
            "signal": spec["priors"]["signal"].rvs(size=nt * nw * lmax).reshape(nt, nw, lmax, 5),
            "extrinsic": spec["priors"]["extrinsic"].rvs(size=nt * nw).reshape(nt, nw, 1, 4),
        }
        inds = {"signal": np.zeros((nt, nw, lmax), bool),
                "extrinsic": np.ones((nt, nw, 1), bool)}
        for it in range(nt):
            for iw in range(nw):
                inds["signal"][it, iw, : int(rng.integers(1, 6))] = True

        def warm():  # JIT/transfer one batch off the clock
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
        for it in range(nt):
            for iw in range(nw):
                inds["signal"][it, iw, : int(rng.integers(1, 6))] = True

        def warm():
            wf = coords["signal"][inds["signal"]]
            wg = np.repeat(np.arange(n_walkers_total),
                           inds["signal"].reshape(n_walkers_total, lmax).sum(1))
            return likelihood.grouped_log_like(wf, wg, ra, dec, psi, ellipticity)

    # GPU correctness check: GPU result must match a CPU twin on the same inputs.
    if template.backend_name != "numpy":
        cpu_args = copy.copy(args)
        cpu_args.device = "cpu"
        _, l_cpu, _, ell_cpu, _, _, _, _, _, _ = build_problem(cpu_args)
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
            a = likelihood.grouped_log_like(w, wg, ra, dec, psi, ellipticity)
            b = l_cpu.grouped_log_like(w, wg, ra, dec, psi, ell_cpu)
        rel = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)
        print(f"[gpu-check] max|GPU-CPU| rel log-like diff = {rel:.2e}")
        assert rel < 1e-6, "GPU and CPU likelihoods disagree!"

    stopper = None
    stop_kwargs = {}
    if args.converge:
        stopper = WaveletConvergenceStopping(
            nleaves_branch="signal", nleaves_max=lmax,
            autocorr_mult=args.autocorr_mult, target_ess=args.target_ess,
            pd_tol=args.pd_tol, verbose=True,
        )
        stop_kwargs = dict(stopping_fn=stopper, stopping_iterations=args.check_every)

    # Warm-start wavelets near the signal for data-informed / flow proposals.
    if args.proposal in ("guided", "flow", "fisher", "flowfisher", "mffisher", "mlflow"):
        coords["signal"] = guided_initial_wavelets(
            template, data_noisy, psd, spec, (nt, nw, lmax), rng)
    # Amortized init overrides: D and leaves straight from p(D, theta | data).
    if args.init == "amortized":
        c_sig, i_sig = amortized_initial_state(
            args.init_ckpt, data_noisy, psd, df, nt, nw, lmax, rng)
        coords["signal"] = c_sig
        inds["signal"] = i_sig

    # BayesWave-style in-model cascade (Fisher dominant + half-cycle + sky-ring),
    # shared by the 'fisher' and 'flowfisher' proposals.
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

    # Tier-1 mixing lever: affine-invariant ensemble (Stretch) move on the signal
    # block. This is the eryn analogue of BayesWave's Differential-Evolution
    # proposal -- it jumps along the line between two walkers, so it captures the
    # amp/phase/Q/t0 correlations that the per-leaf Fisher move (local Gaussian)
    # misses. Gibbs-restricted to "signal" so it never touches the extrinsic leaf
    # or the variable-D structure. Re-weight the cascade so the moves still sum to
    # ~1 with Stretch taking a meaningful share.
    if args.stretch:
        # The proper BayesWave-DE analogue for the RJ wavelet leaf model is a
        # GroupStretchMove (a plain red-blue StretchMove would need
        # 2*nleaves_max*5 ~ 400 walkers). WaveletGroupStretchMove implements the
        # abstract find_friends/setup_friends over the variable-D leaf pool.
        from hyperwave.inference.wavelet_proposals import WaveletGroupStretchMove
        stretch = WaveletGroupStretchMove(
            nfriends=args.stretch_friends, n_iter_update=args.stretch_update,
            gibbs_sampling_setup="signal", a=args.stretch_a)
        scale = 1.0 - args.stretch
        fisher_inmodel = [(m, w * scale) for (m, w) in fisher_inmodel]
        fisher_inmodel.append((stretch, args.stretch))
        print(f"[stretch] wavelet group-stretch on signal "
              f"(weight={args.stretch}, a={args.stretch_a}, nfriends={args.stretch_friends})")

    update_kwargs = {}
    if args.proposal in ("guided", "fisher", "mffisher"):
        if args.proposal == "mffisher":
            guided = build_mf_birth(template, data_noisy, psd, spec)
            print("[rj] matched-filter births: data-fitted SNR + phase")
        else:
            guided = build_guided_birth(template, data_noisy, psd, spec)
        gen = {b: guided[b] for b in branch_names}
        if args.proposal in ("fisher", "mffisher") and args.num_try > 1:
            # multiple-try births: propose num_try candidate wavelets per birth and
            # keep one weighted by likelihood -> high birth acceptance (efficient RJ).
            # MTDistGenMoveRJ can only change ONE model at a time, so restrict it to
            # the signal branch (the extrinsic leaf is fixed at nmin=nmax=1 anyway);
            # passing both branches raises "Can only propose change to one model".
            # gen has only the signal branch (MT changes one model at a time), but
            # nleaves bounds must cover every branch eryn iterates (incl. the fixed
            # extrinsic leaf) or it KeyErrors looking up nleaves_max['extrinsic'].
            # eryn hands *all* state branches to the MT proposal at runtime, so we
            # also Gibbs-restrict it to "signal" -- otherwise it sees both branches
            # and raises "Can only propose change to one model at a time with MT".
            gen_signal = {"signal": guided["signal"]}
            mt_kw = dict(num_try=args.num_try, nleaves_max=nmax, nleaves_min=nmin)
            if "extrinsic" in branch_names:
                mt_kw["gibbs_sampling_setup"] = "signal"
            rj_moves = MTDistGenMoveRJ(gen_signal, **mt_kw)
            print(f"[rj] multiple-try births: num_try={args.num_try} (signal branch)")
        else:
            rj_moves = DistributionGenerateRJ(gen, nleaves_max=nmax, nleaves_min=nmin)
        if args.proposal in ("fisher", "mffisher"):
            moves = fisher_inmodel + [(m, 0.1) for (m, _w) in moves]
    elif args.proposal == "fisherflow":
        # ISOLATION CONFIG: data-informed (guided) births exactly as in 'fisher',
        # plus an IN-MODEL flow independence move -- so any difference vs 'fisher'
        # is attributable to the in-model flow, and vs 'flowfisher' to the births.
        device = "cuda" if (args.device == "gpu" and torch_cuda_available()) else "cpu"
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
                 + fisher_inmodel + [(m, 0.1) for (m, _w) in moves])
        print(f"[fisherflow] device={device} guided births + Fisher cascade + IN-MODEL flow "
              f"(w=0.3), train_every={args.flow_train_every}")
    elif args.proposal == "flowfisher":
        # learned-flow births (efficient RJ) + the proven Fisher in-model cascade.
        device = "cuda" if (args.device == "gpu" and torch_cuda_available()) else "cpu"
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
        moves = fisher_inmodel + [(m, 0.1) for (m, _w) in moves]
        print(f"[flowfisher] device={device} flow births + Fisher in-model, "
              f"train_every={args.flow_train_every}")
    elif args.proposal == "mlflow":
        # Pretrained source-agnostic flow loaded from disk (Path A).
        # Trained once on random Morlet-Gabor mixtures, then deployed as an
        # exact-MH birth proposal for any source (BBH, BNS, burst, glitch).
        if not args.mlflow_ckpt:
            raise SystemExit("--proposal mlflow requires --mlflow-ckpt PATH "
                             "(train one with examples/ml/train_wavelet_flow.py)")
        from hyperwave.ml.proposals import WaveletFlowDistribution, load_flow_model
        device = "cuda" if (args.device == "gpu" and torch_cuda_available()) else "cpu"
        flow_model = load_flow_model(args.mlflow_ckpt, device=device)
        flow_birth = WaveletFlowDistribution(flow_model, device=device)
        flow_birth.set_residual(data_noisy)  # initial: residual == data
        rj_gen = {"signal": flow_birth}
        if "extrinsic" in branch_names:
            rj_gen["extrinsic"] = spec["extrinsic"]
        rj_moves = DistributionGenerateRJ(rj_gen, nleaves_max=nmax, nleaves_min=nmin)
        # Refresh the residual context periodically: data minus the cold-chain
        # mean wavelet model. The callback runs every flow_train_every steps.
        from hyperwave.detectors.waveforms.wavelets import morlet_gabor_fd, amplitude_from_snr
        # Cache the masked frequency grid once. The unmasked template.frequency_array
        # is longer than psd[0] (which is band-limited to [fmin, fmax]); using it in
        # np.interp throws "fp and xp are not of the same length" -- the entire
        # residual refresh was silently skipped on the first mlflow run (11569648)
        # and the flow ran without any conditioning -> birth acceptance 0.006.
        _fvec_masked = template.frequency_array_masked()
        def _refresh_residual(iter_idx, last_sample, sampler_):
            try:
                signal_leaves = last_sample.branches["signal"].coords  # (nt, nw, lmax, 5)
                inds = last_sample.branches["signal"].inds            # (nt, nw, lmax) bool
                live = signal_leaves[0, 0][inds[0, 0]]                # cold-walker-0 live wavelets
                if live.shape[0] == 0:
                    flow_birth.set_residual(data_noisy)
                    return
                s_ref = np.interp(live[:, 1], _fvec_masked, psd[0])
                amps = amplitude_from_snr(live[:, 3], live[:, 1], live[:, 2], s_ref)
                h_p = np.sum(morlet_gabor_fd(_fvec_masked, live[:, 0], live[:, 1],
                                             live[:, 2], amps, live[:, 4]), axis=0)
                model_fd = np.broadcast_to(h_p, data_noisy.shape)
                flow_birth.set_residual(data_noisy - model_fd)
            except Exception as exc:  # pragma: no cover - keep sampler running
                print(f"[mlflow] residual refresh skipped: {exc}")
        update_kwargs = dict(update_fn=_refresh_residual,
                             update_iterations=args.flow_train_every)
        moves = fisher_inmodel + [(m, 0.1) for (m, _w) in moves]
        print(f"[mlflow] device={device} pretrained flow={args.mlflow_ckpt}")
    elif args.proposal == "flow":
        device = "cuda" if (args.device == "gpu" and torch_cuda_available()) else "cpu"
        flow = build_flow_proposal(args.duration, minimum_frequency=args.fmin,
                                   maximum_frequency=args.fmax, device=device,
                                   train_config={"epochs": args.flow_epochs})
        flow_branch = {"signal": flow}
        # in-model: learned flow over wavelet params (signal only, via gibbs so it
        # does not touch the extrinsic branch) + local Gaussian (both branches)
        moves = [(make_flow_distribution_move(flow_branch, gibbs_sampling_setup="signal"), 0.5)] + moves
        # births/deaths drawn from the learned flow; extrinsic stays fixed
        rj_gen = dict(flow_branch)
        if "extrinsic" in branch_names:
            rj_gen["extrinsic"] = spec["extrinsic"]
        rj_moves = make_flow_rj_move(rj_gen, nleaves_max=nmax, nleaves_min=nmin)
        callback = FlowTrainingCallback(
            flow_branch, every=args.flow_train_every, verbose=True,
            temperature_indices=(0,) if args.flow_train_temps == "cold" else None)
        update_kwargs = dict(update_fn=callback, update_iterations=args.flow_train_every)
        print(f"[flow] device={device} train_every={args.flow_train_every}")
    else:
        rj_moves = True

    # BayesWave-style hot, adaptive temperature ladder for the fisher proposal
    # (one chain reaching the prior, Tmax=inf), default geometric otherwise.
    temp_kwargs = dict(ntemps=nt)
    if args.proposal in ("fisher", "flowfisher"):
        temp_kwargs.update(Tmax=np.inf, adaptive=True)

    sampler = EnsembleSampler(
        nw, ndims, log_like_fn, priors,
        tempering_kwargs=temp_kwargs,
        nbranches=len(branch_names), branch_names=branch_names,
        nleaves_max=nmax, nleaves_min=nmin,
        provide_groups=True, vectorize=True,
        moves=moves, rj_moves=rj_moves,
        fill_zero_leaves_val=likelihood.empty_log_likelihood,
        periodic=spec["periodic"],
        **update_kwargs,
        **stop_kwargs,
    )
    state = State(coords, inds=inds)
    burn = args.burn if args.burn is not None else args.nsteps // 2
    _ = warm()  # excluded from timing

    # Direct likelihood micro-benchmark (isolates device cost from the sampler;
    # Eryn's no-op per-step floor is ~18 ms, so this dominates the step).
    nb = 15
    tb = time.perf_counter()
    for _ in range(nb):
        warm()
    like_ms = (time.perf_counter() - tb) / nb * 1e3
    print(f"[bench] one full-batch likelihood call: {like_ms:.1f} ms on {template.backend_name}")

    mode = "converge (cap %d)" % args.nsteps if args.converge else "fixed %d" % args.nsteps
    print(f"[run] sample_sky={args.sample_sky} nwalkers={nw} ntemps={nt} steps={mode} "
          f"burn={burn} nleaves_max={lmax} -> {n_walkers_total} walkers/step")
    t1 = time.perf_counter()
    sampler.run_mcmc(state, args.nsteps, burn=burn, progress=True, thin_by=args.thin)
    sample_t = time.perf_counter() - t1

    # Convergence stopping ends early, so use the actually-stored sampling steps.
    sampled_steps = int(np.asarray(sampler.get_log_like()).shape[0])
    total_steps = sampled_steps + burn
    per_step = sample_t / max(total_steps, 1)
    nleaves = sampler.get_nleaves()["signal"][:, 0].astype(int)  # cold chain
    vals, counts = np.unique(nleaves, return_counts=True)

    # in-model + RJ (birth/death) acceptance on the cold chain -- the proposal
    # efficiency metric (high RJ acceptance => efficient trans-dimensional moves).
    try:
        im_acc = float(np.mean(np.asarray(sampler.acceptance_fraction)[0]))
        rj_acc = float(np.mean(np.asarray(sampler.rj_acceptance_fraction)[0]))
    except Exception:
        im_acc = rj_acc = float("nan")

    converged = bool(stopper and stopper.last.get("consecutive", 0) >= stopper.n_consecutive)
    ref_seconds = args.reference_hours * 3600.0
    print("\n========== TIMING ==========")
    print(f"device               : {template.backend_name}")
    print(f"acceptance           : in-model {im_acc:.3f}  |  RJ births/deaths {rj_acc:.3f}")
    if args.converge:
        status = "CONVERGED" if converged else "hit max-steps cap (not yet converged)"
        info = stopper.last if stopper else {}
        print(f"stopping             : {status} at {sampled_steps} sampling steps "
              f"(tau={info.get('tau', float('nan')):.1f}, ESS={info.get('ess', float('nan')):.0f}, "
              f"pD_tv={info.get('pD_tv', float('nan')):.3f}, Rhat={info.get('rhat', float('nan')):.3f})")
    print(f"this run wall-clock  : {sample_t:.1f} s ({sample_t/60:.1f} min) for {total_steps} "
          f"ensemble steps (burn {burn} + {sampled_steps}), {n_walkers_total} walkers/step")
    print(f"per step             : {per_step*1e3:.1f} ms "
          f"({per_step/n_walkers_total*1e6:.1f} us / walker)")
    like_evals = total_steps * n_walkers_total
    print(f"likelihood evals     : ~{like_evals:,} ({like_evals/sample_t:,.0f} /s)")
    print(f"vs reference (~{args.reference_hours:.0f} h): this run is x{ref_seconds/max(sample_t,1e-9):.0f} faster")
    print("\n====== RECONSTRUCTION ======")
    print(f"injected network SNR: {inj_snr:.1f}")
    print("posterior on number of wavelets (cold chain):")
    for v, c in zip(vals, counts):
        print(f"    D={v:2d}: {c/counts.sum():.3f}")

    # signal-level comparison: overlap (match) and recovered SNR vs the injection
    fixed_sky = None if args.sample_sky else (ra, dec, psi, ellipticity)
    summ = reconstruction_summary(sampler, template, true_signal, psd,
                                  data=data_noisy, sample_sky=args.sample_sky, fixed_sky=fixed_sky,
                                  n_draws=args.draws, seed=args.seed)
    ov, rs = summ["overlap"], summ["rec_snr"]
    q = lambda a: (np.percentile(a, 5), np.median(a), np.percentile(a, 95))
    o5, o50, o95 = q(ov)
    s5, s50, s95 = q(rs)
    print("\n--- true vs reconstructed signal ---")
    print(f"network overlap (match): median {o50:.3f}  90%CI [{o5:.3f}, {o95:.3f}]")
    print(f"recovered network SNR  : median {s50:.1f}   90%CI [{s5:.1f}, {s95:.1f}]   "
          f"(injected {inj_snr:.1f})")
    if "chi2_dof" in summ:
        c5, c50, c95 = q(summ["chi2_dof"])
        print(f"residual chi2/dof      : median {c50:.3f}  90%CI [{c5:.3f}, {c95:.3f}]   "
              f"(noise only {summ['chi2_dof_noise']:.3f}, dof {int(np.median(summ['chi2_ndof']))})")

    if args.recon_plot:
        # per-detector FD + whitened-TD plots from the shared
        # hyperwave.plots.wavelet_reconstruction module (consistent colours/style).
        detectors = ["H1", "L1"]
        asd = np.sqrt(psd)
        base, ext = os.path.splitext(args.recon_plot)
        for j, ifo in enumerate(detectors):
            fd = dict(median=summ["median"][j], lower=summ["band_lo"][j], upper=summ["band_hi"][j])
            wr.plot_wavelet_fd(freqs, fd, ifo, signal_fd=true_signal[j], asd=asd[j], df=df,
                               outpath=f"{base}_{ifo}_fd{ext}", also_pdf=False)
            if "t" in summ:
                td = dict(median=summ["median_t"][j], lower=summ["band_lo_t"][j],
                          upper=summ["band_hi_t"][j])
                wr.plot_wavelet_td(summ["t"], td, ifo, signal_td=summ["inj_t"][j],
                                   outpath=f"{base}_{ifo}_td{ext}", also_pdf=False)
        print(f"reconstruction plots -> {base}_<IFO>_fd{ext} (+ _td{ext})")

    if args.outfile:
        td = {k: summ[k] for k in ("t", "median_t", "band_lo_t", "band_hi_t", "inj_t")
              if k in summ}
        np.savez(args.outfile, nleaves=nleaves, inj_snr=inj_snr, converged=converged,
                 extrinsic=(sampler.get_chain()["extrinsic"][:, 0].reshape(-1, 4).astype(np.float32)
                            if "extrinsic" in sampler.get_chain() else np.zeros((0, 4))),
                 inj_sky=np.array([RA_INJ, DEC_INJ, PSI_INJ]),
                 sampled_steps=sampled_steps, sample_seconds=sample_t, per_step=per_step,
                 device=template.backend_name, overlap=ov, rec_snr=rs,
                 recon_median=summ["median"], recon_band_lo=summ["band_lo"],
                 recon_band_hi=summ["band_hi"], freqs=freqs, true_signal=true_signal,
                 **{k: v for k, v in summ.items() if k.startswith("chi2")},
                 **td, **({} if not stopper else stopper.last))
        print(f"\nsaved -> {args.outfile}")


if __name__ == "__main__":
    main()
