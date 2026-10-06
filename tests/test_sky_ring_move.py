"""The polarisation-carrying sky-ring move is a signal-preserving bijection."""

from __future__ import annotations

import numpy as np

from conftest import requires_lal

T_REF = 1268189526.951953


def _move():
    from hyperwave.inference.wavelet_proposals import WaveletSkyRingMove
    return WaveletSkyRingMove(["H1", "L1"], T_REF)


def _apply(move, x, omega, branch):
    from hyperwave.inference.wavelet_proposals import sky_ring_polarization_map
    return sky_ring_polarization_map(move.detectors, move.reference_time, move.axis,
                                     move.gmst, *x.T, omega, branch)


def _sky(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
                     rng.uniform(0, np.pi, n), rng.uniform(-1, 1, n)], axis=-1)


@requires_lal
def test_ring_move_preserves_detector_signal():
    from hyperwave.detectors.waveforms.wavelets import WaveletTemplate

    duration, fs, start = 4.0, 2048.0, T_REF - 2.0
    freqs = np.fft.rfftfreq(int(duration * fs), 1.0 / fs)
    tmpl = WaveletTemplate(["H1", "L1"], freqs, duration, start, minimum_frequency=20.0,
                           reference_time=T_REF, amplitude_param="amplitude")
    move = _move()
    x = _sky(20, seed=1)
    # high-Q wavelets: the (f + f0) lobe that a phase shift does not carry is negligible
    wav = np.array([[2.0, 150.0, 12.0, 1e-21, 0.4], [2.01, 300.0, 20.0, 5e-22, 2.0]])
    wav = np.broadcast_to(wav, (len(x), 2, 5)).copy()
    omega = np.random.default_rng(2).uniform(0, 2 * np.pi, len(x))
    ra2, dec2, psi2, ell2, c, dt, _, ok = _apply(move, x, omega, np.zeros(len(x)))
    assert ok.all()

    new = wav.copy()
    new[..., 0] -= dt[:, None]
    new[..., 3] *= np.abs(c)[:, None]
    new[..., 4] = np.mod(new[..., 4] + np.angle(c)[:, None], 2 * np.pi)
    h_old = tmpl.project_batch(wav, *x.T)
    h_new = tmpl.project_batch(new, ra2, dec2, psi2, ell2)
    rel = np.linalg.norm(h_new - h_old, axis=(1, 2)) / np.linalg.norm(h_old, axis=(1, 2))
    assert np.max(rel) < 1e-6


@requires_lal
def test_ring_map_is_invertible():
    move = _move()
    x = _sky(50, seed=3)
    omega = np.random.default_rng(4).uniform(0, 2 * np.pi, len(x))
    ra2, dec2, psi2, ell2, c, dt, lj, ok = _apply(move, x, omega, np.zeros(len(x)))
    y = np.stack([ra2, dec2, psi2, ell2], axis=-1)
    # the reverse move rotates by -omega and needs the branch bit that recovers psi
    back_branch = (x[:, 2] >= 0.5 * np.pi).astype(float)
    ra3, dec3, psi3, ell3, c3, dt3, lj3, ok3 = _apply(move, y, -omega, back_branch)
    np.testing.assert_allclose(np.mod(ra3 - x[:, 0] + np.pi, 2 * np.pi) - np.pi, 0, atol=1e-8)
    np.testing.assert_allclose(dec3, x[:, 1], atol=1e-8)
    np.testing.assert_allclose(psi3, x[:, 2], atol=1e-7)
    np.testing.assert_allclose(ell3, x[:, 3], atol=1e-7)
    np.testing.assert_allclose(c3 * c, 1.0, atol=1e-7)
    np.testing.assert_allclose(dt3, -dt, atol=1e-12)
    np.testing.assert_allclose(lj3, -lj, atol=1e-6)


@requires_lal
def test_ring_map_jacobian_matches_finite_differences():
    move = _move()
    x0 = _sky(10, seed=5)
    omega = np.random.default_rng(6).uniform(0, 2 * np.pi, len(x0))
    br = np.zeros(len(x0))
    lj = _apply(move, x0, omega, br)[6]
    h = 1e-6
    for i in range(len(x0)):
        jac = np.empty((4, 4))
        for j in range(4):
            xp, xm = x0[i:i + 1].copy(), x0[i:i + 1].copy()
            xp[0, j] += h
            xm[0, j] -= h
            yp = np.array(_apply(move, xp, omega[i:i + 1], br[i:i + 1])[:4])[:, 0]
            ym = np.array(_apply(move, xm, omega[i:i + 1], br[i:i + 1])[:4])[:, 0]
            d = yp - ym
            d[0] = np.mod(d[0] + np.pi, 2 * np.pi) - np.pi        # ra wraps
            d[2] = np.mod(d[2] + np.pi / 4, np.pi / 2) - np.pi / 4  # psi wraps mod pi/2
            jac[:, j] = d / (2 * h)
        assert np.isclose(np.log(abs(np.linalg.det(jac))), lj[i], atol=1e-4)


@requires_lal
def test_get_proposal_updates_both_branches():
    move = _move()
    ntemps, nwalkers, nleaves = 2, 3, 4
    ext = _sky(ntemps * nwalkers, seed=7).reshape(ntemps, nwalkers, 1, 4)
    rng = np.random.default_rng(8)
    sig = np.stack([rng.uniform(1, 3, (ntemps, nwalkers, nleaves)),
                    rng.uniform(50, 500, (ntemps, nwalkers, nleaves)),
                    rng.uniform(5, 20, (ntemps, nwalkers, nleaves)),
                    rng.uniform(5, 20, (ntemps, nwalkers, nleaves)),
                    rng.uniform(0, 2 * np.pi, (ntemps, nwalkers, nleaves))], axis=-1)
    inds = {"extrinsic": np.ones((ntemps, nwalkers, 1), bool),
            "signal": rng.uniform(size=(ntemps, nwalkers, nleaves)) < 0.6}
    q, factors = move.get_proposal({"extrinsic": ext, "signal": sig},
                                   np.random.RandomState(0), branches_inds=inds)
    assert factors.shape == (ntemps, nwalkers) and np.isfinite(factors).all()
    off = ~inds["signal"]
    np.testing.assert_array_equal(q["signal"][off], sig[off])     # inactive leaves untouched
    assert not np.allclose(q["signal"][inds["signal"]], sig[inds["signal"]])
    assert not np.allclose(q["extrinsic"], ext)
