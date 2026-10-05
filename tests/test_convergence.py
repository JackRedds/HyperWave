import numpy as np
import pytest

pytest.importorskip("eryn")

from hyperwave.inference.convergence import (
    WaveletConvergenceStopping,
    _integrated_act,
    _split_rhat,
)


class _StubSampler:
    """logl / nleaves only (no sky branch)."""

    def __init__(self, logl, nleaves):
        self._logl, self._nleaves = logl, nleaves

    def get_log_like(self):
        return self._logl[:, None, :]

    def get_nleaves(self):
        return {"signal": self._nleaves[:, None, :]}


def _ar1(tau, n, nwalkers, rng):
    a = (tau - 1) / (tau + 1)
    x = np.zeros((n, nwalkers))
    e = rng.normal(size=(n, nwalkers))
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def _run(sampler, **kw):
    kw = dict(dict(nleaves_max=10, target_ess=1000, verbose=False), **kw)
    stop = WaveletConvergenceStopping(**kw)
    return [stop(it, None, sampler) for it in range(3)], stop.last


def test_tau_not_capped_by_fixed_window():
    # eryn's fixed 50-lag window reports ~80-90 here; the Sokal window must not.
    x = _ar1(300, 30000, 16, np.random.default_rng(0))
    assert _integrated_act(x) > 200


def test_iid_chain_converges():
    rng = np.random.default_rng(1)
    n, w = 1500, 32
    passed, last = _run(_StubSampler(rng.normal(size=(n, w)), rng.integers(2, 5, size=(n, w))))
    assert passed[-1]
    assert last["rhat"] < 1.01 and last["pD_noise"] < 0.02


def test_walkers_stuck_in_different_modes_blocked_by_rhat():
    # half the walkers sit at D=3, half at D=5, and never move between them:
    # pooled p(D) is perfectly stationary, so only R-hat can catch this.
    rng = np.random.default_rng(2)
    n, w = 1500, 32
    nleaves = np.where(np.arange(w) < w // 2, 3, 5) + rng.integers(0, 2, size=(n, w))
    logl = rng.normal(size=(n, w)) + 5.0 * (np.arange(w) < w // 2)
    passed, last = _run(_StubSampler(logl, nleaves))
    assert not any(passed)
    assert last["rhat"] > 1.1 and last["pD_tv"] < 0.02


def test_drifting_model_order_blocked():
    rng = np.random.default_rng(3)
    n, w = 3000, 32
    nleaves = rng.integers(2, 5, size=(n, w))
    nleaves[n // 2:] += rng.random((n - n // 2, w)) < 0.1     # D creeps up late in the run
    passed, last = _run(_StubSampler(rng.normal(size=(n, w)), nleaves))
    assert not any(passed)
    assert last["pD_tv"] > 3 * last["pD_noise"]


def test_railing_into_nleaves_max_blocked():
    rng = np.random.default_rng(4)
    n, w = 1500, 32
    nleaves = rng.integers(8, 11, size=(n, w))                 # a third of the mass at the cap
    passed, last = _run(_StubSampler(rng.normal(size=(n, w)), nleaves))
    assert not any(passed)
    assert last["pD_rail"] > 0.3


def test_split_rhat_near_one_for_mixed_chain():
    x = _ar1(5, 4000, 32, np.random.default_rng(5))
    assert _split_rhat(x) < 1.01
