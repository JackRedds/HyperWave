# Changes to the wavelet reconstruction method and `bbh_wavelet_reconstruction.py`

This note records the changes made to HyperWave's wavelet (BayesWave-style RJMCMC)
reconstruction since the "original" baseline, and what each change does.

- **Original library code:** `src/hyperwave/` as of `8af6f64` (the last upstream commit
  before these changes).
- **Original script:** `scripts/bbh_wavelet_reconstruction.py` as first added in `370e562`.
- **Changes covered:** commits `2bdd62c`, `9d81a69`, `3908c17`, `b20f73c`, `58e9feb`
  (2026-09-27 to 2026-10-06).

To see the full diff: `git diff 370e562 58e9feb -- scripts/bbh_wavelet_reconstruction.py src/hyperwave`

---

## 1. Changes to `scripts/bbh_wavelet_reconstruction.py`

### 1.1 Separate (Gibbs) in-model moves for the wavelets and the sky (`3908c17`)

**Before:** with `--sample-sky`, the base in-model move was one joint `GaussianMove`
that perturbed the wavelet parameters `[t0, f0, Q, amp, phi0]` and the extrinsic
parameters `[ra, dec, psi, ellipticity]` in the same proposal (sky widths 0.1 rad).

**After:** two separate Gaussian moves, each Gibbs-restricted to one branch and each
picked half the time:

```python
GaussianMove({"signal": diag([0.05, 5.0, 1.0, 1.0, 0.3])**2}, gibbs_sampling_setup="signal")     # w=0.5
GaussianMove({"extrinsic": diag(sky_step*[1, 1, 1, 0.5])**2}, gibbs_sampling_setup="extrinsic")  # w=0.5
```

**What it does:** a joint move has to land well in both the wavelet and the sky
parameters at once, and a ~6° sky jump on its own is rejected at these SNRs. The
cold chain therefore almost never moved its sky position (in-model acceptance
around 0.6%). Its sky samples were a small pool of states swapped down from the
hotter chains, so the sky maps came out spiky. Splitting the moves lets the
wavelets and the sky each get accepted on their own.

### 1.2 New `--sky-step` option (`3908c17`)

Sets the standard deviation (rad) of the sky-only Gaussian move on ra/dec/psi; the
ellipticity uses half this value. The default of **0.01** replaces the old
hard-coded 0.1, which was much too large to be accepted.

### 1.3 Move weights now multiply instead of being overwritten (`3908c17`)

For the `fisher`, `fisherflow`, `flowfisher` and `mlflow` proposals, the base moves
were added to the cascade as `[(m, 0.1) for (m, _w) in moves]`, so every base move
got weight 0.1 whatever its own weight was. This is now `(m, 0.1 * w)`.

**What it does:** now that there are two base moves of weight 0.5 each (1.1), they
keep their relative split and together still take 0.1 of the cascade, instead of
each getting 0.1 (0.2 in total).

### 1.4 Sky convergence check enabled (`3908c17`)

`WaveletConvergenceStopping` is now given `sky_branch="extrinsic"` when the sky is
sampled, so `--converge` also checks that the sky position has mixed (see 2.2).
The final "stopping" line also prints the split-R-hat (`Rhat=...`).

### 1.5 Residual χ²/dof diagnostic (`b20f73c`)

`reconstruction_summary()` takes a new `data=` argument (the script passes
`data_noisy`). When it is given, the summary computes, for each posterior draw:

- `chi2_dof`: network `<d−h|d−h>` divided by the degrees of freedom, where
  dof = 2 × (number of valid frequency bins) − (5 × active wavelets + 4 if the sky is sampled);
- `chi2_dof_ifo`: the same value for each detector (dof = 2 × bins, with no parameter subtraction);
- `chi2_ndof`: the dof used for each draw;
- `chi2_dof_noise`: the same statistic for `d − h_injected`, i.e. the actual noise
  realisation, as a reference value.

**What it does:** this is a goodness-of-fit check. For pure Gaussian noise the
value should be 1 ± √(2/dof). A value well above the noise-only reference means
power is left unmodelled; a value well below it means the wavelets are fitting
noise. The median and 90% CI are printed, and all `chi2*` keys are saved to the
output `.npz`.

### 1.6 Disk-backed chain storage: `--backend-file` (`b20f73c`)

When this is given, the path is passed to Eryn's `EnsembleSampler(backend=...)`, so
the chain is streamed to an HDF5 file (HDFBackend) instead of being kept in RAM. Any
existing file at that path is deleted first, because the script always starts a
fresh run and does not resume.

**What it does:** the default in-memory backend holds the full
`(nsteps, ntemps, nwalkers, nleaves_max, ndim)` array in memory, which runs out of
memory on long runs with a high `nleaves_max`.

### 1.7 Seeded injection (`2bdd62c`)

`torch.manual_seed(args.seed)` is called before both `make_injections_to_ifo` and
`make_injections_to_ifo_batch`, and `torch` is now imported at the top of the script.

**What it does:** makes the injection reproducible, and makes sure the
injection added to the data and the "pure signal" used for the optimal SNR are
identical when the ml4gw/torch backend draws random numbers (needed for the WNB
script this was shared with). For the deterministic IMRPhenomPv2/LAL BBH injection
it has no effect.

> Side effects to be aware of: the script now needs `torch` installed even for
> CPU/LAL-only runs, and the comment on the injection line was copied from the WNB
> script and now wrongly says "add the WNB to the data".

---

## 2. Changes to the wavelet reconstruction method (library)

### 2.1 `WaveletSkyRingMove` rewritten so it carries the polarisation and wavelets along (`58e9feb`)

File: `src/hyperwave/inference/wavelet_proposals.py`

**Before:** the sky-ring move rotated only `(ra, dec)` about the axis joining the
first two detectors (the ring of constant arrival-time difference), with a
`cos(dec)` Hastings factor. Changing the sky alone changes the antenna-pattern ratio
between the two detectors, which costs about SNR² in log-likelihood, so at high SNR
the move was almost never accepted.

**After:** the move is a deterministic bijection, given a random rotation angle ω
and a random "branch" bit:

1. The line of sight is rotated by ω about the detector baseline (the same ring as before).
2. A new `(psi', ellipticity')` and a complex factor `c` are solved for, so that
   the projected signal in **both** detectors is unchanged. This is a 2×2 complex
   linear solve in the elliptical-polarisation representation
   `r = A z + B z̄`. `psi'` is only defined mod π/2, and the branch bit picks
   between the two solutions.
3. Every active wavelet is updated with `amplitude *= |c|`, `phi0 += arg(c)` and
   `t0 -= Δt`, where Δt is the change in the geocentre-to-detector delay.
4. The Hastings factor is the full log-Jacobian of the map (extrinsic part +
   `n_active · log|c|`). Proposals at polarisation-degenerate skies are rejected
   (`-inf`).

The core maths is exposed as the new function `sky_ring_polarization_map(...)`
(vectorised over all walkers), and is tested in `tests/test_sky_ring_move.py`.

Other changes in this rewrite:
- Detector geometry and GMST now come from `hyperwave.detectors.geometry`
  (`get_detector`, `greenwich_mean_sidereal_time`) instead of calling `lal` directly.
- The move must propose the `extrinsic` and `signal` branches together. It raises a
  `KeyError` if the `signal` branch is missing (i.e. if it is Gibbs-split), and
  expects exactly one extrinsic leaf.

**What it does:** the likelihood is now (almost) unchanged along the ring, apart
from the small `f + f0` wavelet term. Jumps along the ring are therefore decided by
the prior and get accepted, so the sampler can actually explore the ring-shaped sky
degeneracy instead of staying in one spot on it.

### 2.2 Convergence criterion overhauled (`3908c17`, `b20f73c`)

File: `src/hyperwave/inference/convergence.py` (`WaveletConvergenceStopping`).
Tests: `tests/test_convergence.py`, `tests/test_convergence_sky.py`.

| Aspect | Original | New | Why |
|---|---|---|---|
| τ estimator | Eryn `get_integrated_act` (fixed 50-lag window) | Walker-averaged ACF via FFT plus Sokal's automated window (smallest M ≥ 5τ(M)), as in emcee | The fixed window caps τ at about 100, so it underestimates τ for exactly the slow-mixing chains the check is meant to catch |
| Monitored scalars | cold-chain log L and nleaves | + the **sky position** as the 3 Cartesian components of the line-of-sight unit vector (when `sky_branch` is set) | log L and nleaves can decorrelate quickly while the sky barely moves. Using Cartesian components avoids problems at the ra = 0/2π wrap |
| Between-walker check | none | **Rank-normalised, folded split-R-hat** (Vehtari et al. 2021) on every monitored scalar, required `< rhat_tol` (1.01) | Catches walkers stuck in different modes (e.g. some at D=3, some at D=5), which a pooled split-half test misses |
| p(D) stationarity | split-half total variation `< pd_tol` | TV compared with its Monte-Carlo noise at the effective sample size: noise must be `< pd_tol` **and** TV `≤ pd_nsigma` (3) × noise | Separates "p(D) is known precisely enough" from "p(D) is not drifting", and does not pass or fail just because of sampling noise |
| Railing | not checked | Fails if `p(D = nleaves_max) > rail_tol` (0.01) | A posterior piled up at the cap can look perfectly stationary |
| Burn-in | fixed `discard_frac` (0.3) | `max(discard_frac·n, burn_mult·τ_prev)`, capped at n/2 (`burn_mult=20`) | Throws away enough of the start when τ is long |
| Diagnostics | τ, ESS, pD_tv | adds τ for each scalar, R-hat for each scalar, pD noise, rail mass, discard | Saved in `stopper.last` and in the output `.npz` |

New constructor arguments: `sky_branch`, `rhat_tol`, `pd_nsigma`, `rail_tol`, `burn_mult`.
The verbose line now also prints the sky τ, R-hat, the TV noise level, and a `RAILING` flag.

### 2.3 Sky-map post-processing added (`2bdd62c`, `3908c17`)

New modules: `src/hyperwave/skymap.py` and `src/hyperwave/plots/skymap.py`
(`plot_skymap`), exported from `hyperwave` and `hyperwave.plots`. Tests:
`tests/test_skymap.py`.

- `load_sky_samples`, `searched_area`, `searched_probability`, `credible_area`,
  `sky_localization_summary`: the standard sky-localisation statistics computed from the RA/Dec samples
  in a reconstruction `.npz`.
- Two density estimates: an equal-area (ra, sin dec) histogram (the default, with no
  extra dependencies) and `method="healpix"`, a re-implementation of the
  adaptive multi-order HEALPix histogram used by AMPLFI / `ligo.skymap`
  (`healpix_skymap`, `rasterize_healpix`, `crossmatch_sky`), so the numbers can be
  compared directly with AMPLFI. It needs `healpy`, which is the new `skymap` extra in
  `pyproject.toml`.
- `sky_autocorr_time` / `thin_sky_samples` (and `thin=True` on loading): thin
  the MCMC chain by its autocorrelation time before building a histogram, because
  repeated states (from rejected proposals) otherwise show up as isolated spikes.

### 2.4 Smaller compatibility fixes (`2bdd62c`, `9d81a69`)

- `np.trapz` → `np.trapezoid` in `DataInformedMarginal` (`np.trapz` is
  deprecated/removed in NumPy 2).
- `matplotlib.cm.get_cmap` → `matplotlib.colormaps[...]` in `plots/hyper.py`
  (`get_cmap` was removed in Matplotlib 3.9); the matplotlib pin was relaxed from
  `<3.7` to `>=3.5`.
- ml4gw backend: `MultiSineGaussian` renamed to `MultiWaveform` to match the
  ml4gw API (`ml4gw.py`, `backends/ml4gw_backend.py`).

---

## 3. Related additions (not changes to the original files)

- Sibling scripts with the same structure for other injected morphologies:
  `wnb_`, `sg_`, `cs_` and `gaussian_wavelet_reconstruction.py`. They received the same
  edits as 1.1–1.6.
- `scripts/wavelet_injection_campaign.py` + `injection_campaign_submit.slurm`
  (injection campaigns), and `wavelet_reconstruction_submit.slurm` / `wavelet_test.sh`
  (cluster submission and smoke tests).
