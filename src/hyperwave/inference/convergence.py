"""Convergence-based stopping for Eryn runs.

Instead of a fixed number of steps, stop when the chain has produced *enough
independent samples* and the science marginal of interest (here the posterior on
the number of wavelets) has stopped moving. This is the Goodman-Weare /
Foreman-Mackey autocorrelation criterion, combined with rank-normalised split-R-hat
across walkers and a noise-aware split-half stationarity check on the model-order
posterior, and is always used under a hard maximum-steps cap (the ``nsteps``
passed to ``run_mcmc``).

The criterion plugs into Eryn via::

    sampler = EnsembleSampler(..., stopping_fn=stopper, stopping_iterations=K)

and is checked every ``K`` sampler iterations during the sampling phase.

Why these signals (and not evidence):

* Autocorrelation time tau on the cold-chain **log-likelihood** and **nleaves**
  is robust under trans-dimensional moves (both are well-defined scalars even as
  the parameter dimension changes), unlike per-parameter tau. tau is estimated
  from the walker-averaged autocorrelation function with Sokal's automated
  window (smallest ``M >= c * tau(M)``), as in emcee. A fixed summation window
  (eryn's ``get_integrated_act`` uses 50 lags) caps tau at ~100 and so
  under-reports exactly the slowly mixing chains the check is meant to catch.
* When the sky is sampled, the tau of the cold-chain **sky position** (the
  three Cartesian components of the line-of-sight unit vector, so the
  ``ra=0/2*pi`` wrap doesn't matter) is folded in as well. The likelihood and
  nleaves can decorrelate in a few steps while the sky barely mixes -- e.g. a
  cold chain whose sky only changes via tempering swaps from a small pool of
  states -- which would otherwise pass as "converged" with spiky sky maps.
* **Split-R-hat** (Vehtari et al. 2021, rank-normalised and folded) over the
  split walker chains catches walkers sitting in different modes (e.g. some at
  D=3, others at D=5), which a pooled split-half test can miss when the mix of
  walkers per mode is itself stationary.
* The **nleaves posterior** is the headline output, so a split-half
  total-variation test on p(D) directly certifies the result has converged. The
  TV is compared with its Monte-Carlo noise level at the effective sample size:
  that noise must be below ``pd_tol`` (p(D) is known to that precision) and the
  observed TV must be consistent with it (no significant drift). Railing into
  ``nleaves_max`` is flagged separately, since a posterior piled up at the cap
  can look perfectly stationary.
* Evidence (thermodynamic integration / stepping stone) converges more slowly
  and noisily than the posterior; it is best computed once at the end for model
  comparison, not used as a stopping signal.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

try:
    from eryn.utils.stopping import Stopping as _ErynStopping
except Exception:  # pragma: no cover - eryn optional at import time
    _ErynStopping = object


def _walker_acf(chain_2d):
    """Walker-averaged normalised autocorrelation function of ``(n_steps, n_walkers)``."""
    x = np.asarray(chain_2d, dtype=float)
    n = x.shape[0]
    x = x - x.mean(axis=0, keepdims=True)
    nfft = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft, axis=0)
    acov = np.fft.irfft(f * f.conj(), n=nfft, axis=0)[:n].mean(axis=1)
    if not acov[0] > 0:
        return None                                     # constant chain: tau undefined
    return acov / acov[0]


def _integrated_act(chain_2d, c=5.0):
    """Integrated autocorrelation time of a ``(n_steps, n_walkers)`` chain.

    Sokal's automated windowing on the walker-averaged ACF (emcee's estimator):
    ``tau(M) = 1 + 2 sum_{k=1}^{M} rho_k`` with the smallest ``M >= c tau(M)``.
    Returns NaN for chains too short to estimate or with zero variance.
    """
    if chain_2d.shape[0] < 60:
        return np.nan
    rho = _walker_acf(chain_2d)
    if rho is None:
        return np.nan
    taus = 2.0 * np.cumsum(rho) - 1.0
    m = np.arange(taus.size) < c * taus
    window = int(np.argmin(m)) if not m.all() else taus.size - 1
    return float(taus[window])


def _split_chains(chain_2d):
    """Split each walker in half: ``(n, W)`` -> ``(n // 2, 2W)``."""
    half = chain_2d.shape[0] // 2
    return np.concatenate([chain_2d[:half], chain_2d[half:2 * half]], axis=1)


def _rhat_basic(chains):
    """Classic potential scale reduction of ``(n, m)`` chains (NaN if degenerate)."""
    n = chains.shape[0]
    w = chains.var(axis=0, ddof=1).mean()
    if not w > 0:
        return np.nan
    b = n * chains.mean(axis=0).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def _rank_normalise(chains):
    r = rankdata(chains, method="average").reshape(chains.shape)
    return ndtri((r - 0.375) / (chains.size + 0.25))


def _split_rhat(chain_2d):
    """Rank-normalised, folded split-R-hat (Vehtari et al. 2021) over walkers."""
    x = np.asarray(chain_2d, dtype=float)
    if x.shape[0] < 60 or x.shape[1] < 2:
        return np.nan
    s = _split_chains(x)
    bulk = _rhat_basic(_rank_normalise(s))
    tail = _rhat_basic(_rank_normalise(np.abs(s - np.median(s))))
    vals = [v for v in (bulk, tail) if np.isfinite(v)]
    return max(vals) if vals else np.nan


def _pD(counts, n_max):
    """Normalised histogram of integer wavelet counts over ``0..n_max``."""
    hist = np.bincount(np.asarray(counts, dtype=int).ravel(), minlength=n_max + 1).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _total_variation(p, q):
    n = max(p.size, q.size)
    p = np.pad(p, (0, n - p.size))
    q = np.pad(q, (0, n - q.size))
    return 0.5 * float(np.abs(p - q).sum())


def _tv_noise(p, n_eff_half):
    """Expected split-half TV of two independent ``n_eff_half``-sample draws from ``p``.

    Each bin difference is ~ N(0, 2 p (1 - p) / n), whose mean absolute value is
    ``sqrt(2/pi)`` times its standard deviation.
    """
    if not n_eff_half > 0:
        return np.inf
    sd = np.sqrt(2.0 * p * (1.0 - p) / n_eff_half)
    return 0.5 * float(np.sqrt(2.0 / np.pi) * sd.sum())


class WaveletConvergenceStopping(_ErynStopping):
    """Autocorrelation + split-R-hat + model-order stationarity stopping criterion.

    Parameters
    ----------
    nleaves_branch:
        Branch whose leaf count is the model order (default ``"signal"``).
    nleaves_max:
        Upper bound on the leaf count (for the p(D) histogram and railing check).
    sky_branch:
        Branch holding ``[ra, dec, ...]`` in its first two columns (e.g.
        ``"extrinsic"``). If given, the sky autocorrelation time and R-hat also
        enter the criterion (default ``None``: sky not checked).
    autocorr_mult:
        Require ``chain_length > autocorr_mult * tau`` (emcee rule, default 50).
    target_ess:
        Also require an effective sample size ``n_steps * n_walkers / tau`` above
        this (default 2000).
    tau_rtol:
        Require the tau estimate to be stable between checks to within this
        relative tolerance (default 0.05).
    rhat_tol:
        Require the rank-normalised split-R-hat of every monitored scalar below
        this (default 1.01, Vehtari et al. 2021).
    pd_tol:
        Required precision of p(D): the Monte-Carlo noise level of the split-half
        total variation at the effective sample size must be below this
        (default 0.02).
    pd_nsigma:
        The observed split-half TV must be below ``pd_nsigma`` times that noise
        level, i.e. consistent with no drift (default 3).
    rail_tol:
        Fail if the posterior mass at ``nleaves_max`` exceeds this (default 0.01).
    n_consecutive:
        Number of consecutive passing checks required before stopping (default 2).
    discard_frac:
        Minimum fraction of the stored chain discarded as burn-in for the
        estimates (default 0.3).
    burn_mult:
        Also discard at least ``burn_mult * tau`` steps, using the previous
        check's tau (default 20), capped at half the chain.
    verbose:
        Print a diagnostics line at each check.
    """

    def __init__(self, nleaves_branch="signal", nleaves_max=40, sky_branch=None, autocorr_mult=50,
                 target_ess=2000, tau_rtol=0.05, rhat_tol=1.01, pd_tol=0.02, pd_nsigma=3.0,
                 rail_tol=0.01, n_consecutive=2, discard_frac=0.3, burn_mult=20.0, verbose=True):
        self.nleaves_branch = nleaves_branch
        self.nleaves_max = int(nleaves_max)
        self.sky_branch = sky_branch
        self.autocorr_mult = float(autocorr_mult)
        self.target_ess = float(target_ess)
        self.tau_rtol = float(tau_rtol)
        self.rhat_tol = float(rhat_tol)
        self.pd_tol = float(pd_tol)
        self.pd_nsigma = float(pd_nsigma)
        self.rail_tol = float(rail_tol)
        self.n_consecutive = int(n_consecutive)
        self.discard_frac = float(discard_frac)
        self.burn_mult = float(burn_mult)
        self.verbose = verbose
        self._old_tau = None
        self._consec = 0
        #: filled in at each check, so the driver can report the final state
        self.last = {}

    def _discard(self, n):
        d = int(self.discard_frac * n)
        if self._old_tau is not None and np.isfinite(self._old_tau):
            d = max(d, int(self.burn_mult * self._old_tau))
        return min(d, n // 2)

    def __call__(self, iteration, sample, sampler):
        logl = np.asarray(sampler.get_log_like())            # (n, ntemps, nwalkers)
        n = logl.shape[0]
        if n < 60:
            return False
        discard = self._discard(n)
        cold_logl = logl[discard:, 0, :]                     # (n', nwalkers)
        nleaves = np.asarray(sampler.get_nleaves()[self.nleaves_branch])[discard:, 0, :]
        sky = self._sky_xyz(sampler, discard)

        tau_l = _integrated_act(cold_logl)
        tau_n = _integrated_act(nleaves.astype(float))
        tau_s = _nanmax([_integrated_act(v) for v in sky])
        taus = [t for t in (tau_l, tau_n, tau_s) if np.isfinite(t)]
        if not taus:
            return False
        tau = max(taus)
        n_eff_steps = cold_logl.shape[0]
        n_walkers = cold_logl.shape[1]
        ess = n_eff_steps * n_walkers / tau

        rhat_l = _split_rhat(cold_logl)
        rhat_n = _split_rhat(nleaves.astype(float))
        rhat_s = _nanmax([_split_rhat(v) for v in sky])
        rhat = _nanmax([rhat_l, rhat_n, rhat_s])
        rhat_ok = not np.isfinite(rhat) or rhat < self.rhat_tol

        half = nleaves.shape[0] // 2
        if half > 0:
            p_all = _pD(nleaves, self.nleaves_max)
            tv = _total_variation(_pD(nleaves[:half], self.nleaves_max),
                                  _pD(nleaves[half:], self.nleaves_max))
            tau_pd = tau_n if np.isfinite(tau_n) else tau
            tv_noise = _tv_noise(p_all, half * n_walkers / tau_pd)
            rail = float(p_all[self.nleaves_max]) if p_all.size > self.nleaves_max else 0.0
        else:
            tv, tv_noise, rail = np.inf, np.inf, 0.0
        pd_ok = tv_noise < self.pd_tol and tv <= self.pd_nsigma * tv_noise
        rail_ok = rail <= self.rail_tol

        tau_stable = self._old_tau is not None and abs(tau - self._old_tau) / tau < self.tau_rtol
        self._old_tau = tau

        passed = (n_eff_steps > self.autocorr_mult * tau and ess > self.target_ess
                  and tau_stable and rhat_ok and pd_ok and rail_ok)
        self._consec = self._consec + 1 if passed else 0

        self.last = dict(iteration=int(iteration), discard=int(discard), tau=float(tau),
                         tau_logl=float(tau_l), tau_nleaves=float(tau_n), tau_sky=float(tau_s),
                         ess=float(ess), rhat=float(rhat), rhat_logl=float(rhat_l),
                         rhat_nleaves=float(rhat_n), rhat_sky=float(rhat_s),
                         pD_tv=float(tv), pD_noise=float(tv_noise), pD_rail=float(rail),
                         tau_stable=bool(tau_stable), consecutive=int(self._consec))
        if self.verbose:
            sky_s = f" (sky {tau_s:.1f})" if self.sky_branch is not None else ""
            rail_s = "" if rail_ok else f" RAILING p(Dmax)={rail:.3f}"
            print(f"[converge] it={iteration} tau={tau:.1f}{sky_s} "
                  f"ESS={ess:.0f}/{self.target_ess:.0f} "
                  f"n/tau={n_eff_steps/tau:.1f}/{self.autocorr_mult:.0f} "
                  f"Rhat={rhat:.3f}/{self.rhat_tol} "
                  f"pD_tv={tv:.3f} (noise {tv_noise:.3f}/{self.pd_tol}) "
                  f"stable={tau_stable}{rail_s} "
                  f"pass={self._consec}/{self.n_consecutive}", flush=True)

        return self._consec >= self.n_consecutive

    def _sky_xyz(self, sampler, discard):
        """Cold-chain line-of-sight unit-vector components, each ``(n', nwalkers)``."""
        if self.sky_branch is None:
            return []
        # a list, not a bare string: eryn's get_value mis-handles a str branch name
        chain = sampler.get_chain(discard=discard, temp_index=0, branch_names=[self.sky_branch])
        x = np.asarray(chain[self.sky_branch])[:, :, 0, :2]  # (n', nwalkers, [ra, dec])
        ra, dec = x[..., 0].astype(float), x[..., 1].astype(float)
        return [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def _nanmax(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return max(vals) if vals else np.nan


__all__ = ["WaveletConvergenceStopping"]
