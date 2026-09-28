import numpy as np
import pytest

from hyperwave.skymap import (
    crossmatch_sky,
    healpix_skymap,
    rasterize_healpix,
    sky_autocorr_time,
    thin_sky_samples,
)

pytest.importorskip("healpy")


def _gaussian_blob(n, ra0=1.0, dec0=0.3, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return ra0 + rng.normal(0, sigma, n) / np.cos(dec0), dec0 + rng.normal(0, sigma, n)


def test_healpix_skymap_tiles_sky_and_normalizes():
    ra, dec = _gaussian_blob(20000)
    order, ipix, probdensity = healpix_skymap(ra, dec)
    area = 4 * np.pi / (12 * 4.0**order)
    assert np.isclose(area.sum(), 4 * np.pi)
    assert np.isclose((probdensity * area).sum(), 1.0)
    # no leaf left over-full unless it is at the finest allowed order
    counts = probdensity * area * len(ra)
    assert np.all((np.round(counts) < 20) | (order == order.max()))


def test_rasterize_conserves_probability():
    ra, dec = _gaussian_blob(5000)
    order, ipix, probdensity = healpix_skymap(ra, dec)
    for raster_order in (3, int(order.max())):
        density = rasterize_healpix(order, ipix, probdensity, raster_order)
        assert np.isclose(density.sum() * 4 * np.pi / len(density), 1.0)


def test_crossmatch_truth_at_center_vs_far_away():
    ra, dec = _gaussian_blob(20000, sigma=0.05)
    near = crossmatch_sky(ra, dec, 1.0, 0.3)
    far = crossmatch_sky(ra, dec, 1.0 + np.pi, -0.3)
    # ~20 samples per pixel leaves ~20% density noise near the peak
    assert near["searched_probability"] < 0.5
    assert far["searched_probability"] > 0.99
    # 2D Gaussian: 90% region has area -2 ln(0.1) * pi * sigma^2 (sr)
    expected = -2 * np.log(0.1) * np.pi * np.degrees(0.05) ** 2
    assert near["contour_areas_deg2"][1] == pytest.approx(expected, rel=0.15)


def test_thinning_uses_autocorrelation():
    rng = np.random.default_rng(1)
    nsteps, nwalkers, repeat = 4000, 10, 25
    # each walker holds each independent draw for `repeat` steps
    ra = np.repeat(rng.uniform(0, 2 * np.pi, (nsteps // repeat, nwalkers)), repeat, axis=0)
    dec = np.repeat(rng.uniform(-1, 1, (nsteps // repeat, nwalkers)), repeat, axis=0)
    tau = sky_autocorr_time(ra.ravel(), dec.ravel(), nwalkers)
    assert 10 < tau < 50
    tra, _ = thin_sky_samples(ra.ravel(), dec.ravel(), nwalkers, tau=tau)
    assert len(tra) == nwalkers * len(range(nsteps - 1, -1, -int(np.ceil(tau))))
