import numpy as np
import pytest

pytest.importorskip("eryn")

from hyperwave.inference.convergence import WaveletConvergenceStopping


class _StubSampler:
    """Just enough of eryn's EnsembleSampler for the stopping criterion."""

    def __init__(self, logl, nleaves, sky):
        self._logl, self._nleaves, self._sky = logl, nleaves, sky  # sky: (n, nwalkers, 2)

    def get_log_like(self):
        return self._logl[:, None, :]

    def get_nleaves(self):
        return {"signal": self._nleaves[:, None, :]}

    def get_chain(self, discard=0, temp_index=None, branch_names=None):
        assert temp_index == 0 and branch_names == ["extrinsic"]
        return {"extrinsic": self._sky[discard:, :, None, :]}


def _chain(sky_hold, n=1500, nwalkers=32, seed=0):
    """logl/nleaves decorrelate every step; the sky only changes every ``sky_hold`` steps."""
    rng = np.random.default_rng(seed)
    logl = rng.normal(size=(n, nwalkers))
    nleaves = rng.integers(2, 5, size=(n, nwalkers))
    blocks = -(-n // sky_hold)
    ra = np.repeat(rng.uniform(0, 2 * np.pi, (blocks, nwalkers)), sky_hold, axis=0)[:n]
    dec = np.repeat(np.arcsin(rng.uniform(-1, 1, (blocks, nwalkers))), sky_hold, axis=0)[:n]
    return _StubSampler(logl, nleaves, np.stack([ra, dec], axis=-1))


def _run(sampler, **kw):
    stop = WaveletConvergenceStopping(nleaves_max=10, target_ess=1000, pd_tol=0.05,
                                      verbose=False, **kw)
    return [stop(it, None, sampler) for it in range(3)], stop.last


def test_stuck_sky_blocks_convergence_only_when_checked():
    stuck = _chain(sky_hold=40)
    passed, last = _run(stuck)                        # old behaviour: sky ignored
    assert passed[-1] and np.isnan(last["tau_sky"])
    passed, last = _run(stuck, sky_branch="extrinsic")
    assert not any(passed)
    assert last["tau_sky"] > 20 and last["tau"] == last["tau_sky"]


def test_mixing_sky_still_converges():
    passed, last = _run(_chain(sky_hold=1), sky_branch="extrinsic")
    assert passed[-1] and last["tau_sky"] < 3
