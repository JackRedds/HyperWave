"""Sky-localization metrics from posterior RA/Dec samples.

The wavelet reconstruction scripts (``scripts/*_wavelet_reconstruction.py``)
save posterior sky samples in the ``extrinsic`` array of their output
``*_reconstruction.npz`` file, with columns ``[ra, dec, psi, ellipticity]``
(radians, only populated when run with ``--sample-sky``), and the injected
truth in ``inj_sky`` as ``[ra, dec, psi]``. This module turns those flat
samples into the standard GW sky-localization statistics: the searched area
and searched probability at the injected location, and the credible area at
an arbitrary confidence level.

Two density estimates are available, selected by ``method``:

* ``"histogram"`` (default, no extra dependencies) -- a fixed equal-area sky
  histogram: bins uniform in right ascension and in ``sin(dec)`` have exactly
  equal solid angle, since the sphere's area element is ``d(ra) d(sin dec)``.
* ``"healpix"`` (needs ``healpy``) -- the adaptive multi-order HEALPix
  histogram AMPLFI uses (``ligo.skymap.healpix_tree.adaptive_healpix_histogram``,
  scored as ``ligo.skymap.postprocess.crossmatch`` does), reimplemented here so
  it doesn't drag in ``ligo.skymap``'s ``matplotlib>=3.9`` requirement. A pixel
  is only split while it holds more than ``max_samples_per_pixel`` samples, so
  sparse regions are averaged over big pixels instead of showing raw sampling
  noise, and the numbers are directly comparable to AMPLFI's.

MCMC chains should be thinned by their autocorrelation time before either
estimate (``thin=True``): a chain repeats its state on every rejected
proposal, so it can hold hundreds of copies of each of a few thousand
distinct sky positions, which a histogram renders as isolated spikes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "load_sky_samples",
    "sky_autocorr_time",
    "thin_sky_samples",
    "healpix_skymap",
    "rasterize_healpix",
    "crossmatch_sky",
    "sky_histogram",
    "credible_area",
    "searched_area",
    "searched_probability",
    "sky_localization_summary",
]


def load_sky_samples(path: str, thin: bool = False):
    """Load RA/Dec posterior samples and the injected truth from a reconstruction ``.npz``.

    Returns ``(ra, dec, true_ra, true_dec)`` in radians; the truth values are
    ``None`` if the file has no ``inj_sky`` entry. ``thin=True`` thins the
    chain by its sky autocorrelation time (see :func:`thin_sky_samples`),
    taking the walker count from the saved ``nleaves`` array.
    """
    with np.load(path) as d:
        extrinsic = d["extrinsic"]
        nwalkers = d["nleaves"].shape[1] if "nleaves" in d else None
        true_ra, true_dec = None, None
        if "inj_sky" in d:
            inj_sky = d["inj_sky"]
            true_ra, true_dec = float(inj_sky[0]), float(inj_sky[1])
    if extrinsic.shape[0] == 0:
        raise ValueError(f"{path!r} has no sky posterior samples (run with --sample-sky)")
    ra, dec = extrinsic[:, 0].astype(float), extrinsic[:, 1].astype(float)
    if thin:
        if nwalkers is None:
            raise ValueError(f"{path!r} has no nleaves array to recover the walker count from")
        ra, dec = thin_sky_samples(ra, dec, nwalkers)
    return ra, dec, true_ra, true_dec


def _integrated_time(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time of ``x``, shape ``(nsteps, nwalkers)``.

    The walker-averaged FFT estimator with Sokal's automatic window (the
    same one as ``emcee.autocorr.integrated_time``): the smallest window
    ``M`` with ``M >= c * tau(M)``.
    """
    nsteps = x.shape[0]
    x = x - x.mean(axis=0)
    n = 1 << int(np.ceil(np.log2(2 * nsteps)))
    f = np.fft.rfft(x, n=n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), n=n, axis=0)[:nsteps].real
    var = acf[0]
    ok = var > 0  # a walker stuck the whole time carries no correlation info
    if not ok.any():
        return float(nsteps)
    rho = (acf[:, ok] / var[ok]).mean(axis=1)
    taus = 2.0 * np.cumsum(rho) - 1.0
    m = np.arange(len(taus)) < c * taus
    window = int(np.argmin(m)) if not m.all() else len(taus) - 1
    return float(taus[window])


def sky_autocorr_time(ra: np.ndarray, dec: np.ndarray, nwalkers: int) -> float:
    """Autocorrelation time (in steps) of a flattened ``(nsteps, nwalkers)`` sky chain.

    ``ra``/``dec`` are in the step-major order the reconstruction scripts
    save (``get_chain()[...].reshape(-1, 4)``). Uses the largest time of the
    three Cartesian unit-vector components, which avoids the ``ra=0/2*pi``
    wrap and doesn't depend on where the posterior sits on the sky.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if len(ra) % nwalkers:
        raise ValueError(f"{len(ra)} samples is not a whole number of steps of {nwalkers} walkers")
    nsteps = len(ra) // nwalkers
    xyz = (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    return max(_integrated_time(v.reshape(nsteps, nwalkers)) for v in xyz)


def thin_sky_samples(ra, dec, nwalkers: int, tau: float | None = None):
    """Keep every ``ceil(tau)``-th step of every walker of a flattened sky chain.

    ``tau`` defaults to :func:`sky_autocorr_time`. What's left is close to
    independent draws, which is what a histogram (or AMPLFI's per-pixel
    sample count) assumes it's getting.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if tau is None:
        tau = sky_autocorr_time(ra, dec, nwalkers)
    step = max(1, int(np.ceil(tau)))
    nsteps = len(ra) // nwalkers
    keep = slice(nsteps - 1, None, -step)  # anchor on the last (most converged) step
    ra = ra.reshape(nsteps, nwalkers)[keep][::-1].ravel()
    dec = dec.reshape(nsteps, nwalkers)[keep][::-1].ravel()
    return ra, dec


def _require_healpy():
    try:
        import healpy
    except ImportError as err:
        raise ImportError(
            'method="healpix" needs healpy: pip install "hyperwave[skymap]"'
        ) from err
    return healpy


def healpix_skymap(ra, dec, max_samples_per_pixel: int = 20, max_nside: int = 2048):
    """Adaptive multi-order HEALPix histogram of posterior sky samples.

    Same algorithm and defaults as AMPLFI's ``adaptive_histogram_skymap``
    (``ligo.skymap.healpix_tree.adaptive_healpix_histogram``): starting from
    the 12 base pixels, a pixel is split into its 4 NESTED children while it
    holds at least ``max_samples_per_pixel`` samples. Like ``ligo.skymap``
    (whose tree counts the whole sphere as level 0), the finest leaves stop
    one order short of ``max_nside``, i.e. at ``max_nside / 2``. Returns ``(order, ipix, probdensity)`` for the leaf pixels, with
    ``ipix`` NESTED at each leaf's own ``order`` and ``probdensity`` in
    probability per steradian; together the leaves tile the whole sky.
    """
    hp = _require_healpy()
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    max_order = int(np.log2(max_nside))
    fine = hp.ang2pix(1 << max_order, 0.5 * np.pi - dec, np.mod(ra, 2 * np.pi), nest=True)
    n = len(fine)

    orders, ipixs, counts = [], [], []
    # nodes to examine at the current order: every base pixel to begin with
    nodes = np.arange(12, dtype=np.int64)
    for order in range(max_order + 1):
        parent = fine >> (2 * (max_order - order))
        node_counts = np.bincount(np.searchsorted(nodes, parent[np.isin(parent, nodes)]),
                                  minlength=len(nodes))
        split = (node_counts >= max_samples_per_pixel) & (order < max_order - 1)
        orders.append(np.full((~split).sum(), order))
        ipixs.append(nodes[~split])
        counts.append(node_counts[~split])
        nodes = (4 * nodes[split][:, None] + np.arange(4)).ravel()
        if len(nodes) == 0:
            break
    order = np.concatenate(orders)
    ipix = np.concatenate(ipixs)
    pixarea = 4 * np.pi / (12 * 4.0 ** order)
    probdensity = np.concatenate(counts) / n / pixarea
    return order, ipix, probdensity


def rasterize_healpix(order, ipix, probdensity, raster_order: int):
    """Flatten a :func:`healpix_skymap` to a fixed NESTED HEALPix order.

    Leaves coarser than ``raster_order`` fill all their sub-pixels; finer
    ones are averaged (probability-conserving) into their parent. Returns
    the density per steradian for every pixel at ``raster_order``.
    """
    order = np.asarray(order)
    ipix = np.asarray(ipix, dtype=np.int64)
    npix = 12 * 4**raster_order
    prob = np.zeros(npix)
    leaf_prob = probdensity * 4 * np.pi / (12 * 4.0**order)
    fine = order >= raster_order
    np.add.at(prob, ipix[fine] >> (2 * (order[fine] - raster_order)), leaf_prob[fine])
    for o in np.unique(order[~fine]):
        sel = ~fine & (order == o)
        k = 4 ** (raster_order - int(o))
        idx = (ipix[sel][:, None] * k + np.arange(k)).ravel()
        prob[idx] += np.repeat(leaf_prob[sel] / k, k)
    return prob / (4 * np.pi / npix)


def crossmatch_sky(
    ra, dec, true_ra: float, true_dec: float, contours=(0.5, 0.9),
    max_samples_per_pixel: int = 20, max_nside: int = 2048,
) -> dict:
    """Searched area/probability and credible areas on the :func:`healpix_skymap`.

    Mirrors ``ligo.skymap.postprocess.crossmatch``: pixels are sorted by
    descending density; the searched area/probability are the cumulative
    area/probability up to and including the pixel holding the truth, and
    each credible area is interpolated on the cumulative probability.
    ``contours`` are fractions (``0.9``, not ``90``).
    """
    hp = _require_healpy()
    order, ipix, probdensity = healpix_skymap(
        ra, dec, max_samples_per_pixel=max_samples_per_pixel, max_nside=max_nside
    )
    rank = np.argsort(probdensity, kind="stable")[::-1]
    order, ipix, probdensity = order[rank], ipix[rank], probdensity[rank]
    dA = 4 * np.pi / (12 * 4.0 ** order)
    prob = np.cumsum(probdensity * dA)
    area = np.cumsum(dA) * (180 / np.pi) ** 2

    top = int(order.max())
    true_pix = hp.ang2pix(1 << top, 0.5 * np.pi - true_dec, np.mod(true_ra, 2 * np.pi), nest=True)
    true_idx = int(np.flatnonzero((true_pix >> (2 * (top - order))) == ipix)[0])

    contour_areas = np.interp(contours, np.pad(prob, (1, 0)), np.pad(area, (1, 0)),
                              right=4 * 180**2 / np.pi)
    return {
        "searched_area_deg2": float(area[true_idx]),
        "searched_probability": float(prob[true_idx]),
        "contour_areas_deg2": [float(a) for a in contour_areas],
    }


def _fd_bins(x: np.ndarray, span: float, n_eff: int, floor: int, cap: int = 2000) -> int:
    """Freedman-Diaconis bin count for ``x`` spanning ``span``, clipped to ``[floor, cap]``.

    Ties resolution to how concentrated the samples actually are (via the
    IQR) and to how many *independent* draws support that resolution (via
    ``n_eff**-1/3``), instead of a fixed grid that saturates for a tightly
    localized posterior (see :func:`sky_histogram`). ``floor`` keeps it no
    coarser than the old fixed default; ``cap`` guards against a degenerate
    (near delta-function) IQR demanding an unreasonably fine grid.
    """
    q75, q25 = np.percentile(x, [75, 25])
    spread = (q75 - q25) or 1.349 * np.std(x)
    if spread <= 0:  # all samples identical
        return cap
    width = 2 * spread * n_eff ** (-1 / 3)
    return int(np.clip(np.ceil(span / width), floor, cap))


def sky_histogram(ra: np.ndarray, dec: np.ndarray, nra: int | None = None, ndec: int | None = None):
    """Equal-area sky histogram of posterior samples.

    Bins are uniform in ``ra`` over ``[0, 2*pi)`` and in ``sin(dec)`` over
    ``[-1, 1]``, so every bin has exactly the same solid angle. Returns
    ``(prob, ra_edges, dec_edges, pixel_area_deg2)`` where ``prob`` is the
    fraction of samples in each bin, shape ``(nra, ndec)``, and ``dec_edges``
    is ``arcsin`` of the ``sin(dec)`` bin edges.

    ``nra``/``ndec`` default to a Freedman-Diaconis estimate from the sample
    spread (never coarser than the historical 180x90 default), so a tightly
    localized posterior gets a finer grid automatically instead of saturating
    at one pixel. Pass explicit values to override.

    The bin *count* uses the number of distinct ``(ra, dec)`` pairs, not the
    raw sample count: MCMC chains (e.g. eryn's) hold their previous state on
    every rejected proposal, so a chain can have millions of rows but only a
    few thousand actually-distinct sky positions -- using the raw count would
    demand a grid far finer than the chain can actually resolve, leaving most
    pixels spuriously empty (including, by bad luck, sometimes the one
    containing the truth; see ``truth_window`` on :func:`searched_area`).
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if nra is None or ndec is None:
        n_eff = len(np.unique(np.stack([ra, dec], axis=1), axis=0))
    if nra is None:
        # re-center away from the wrap point so a cluster straddling ra=0/2*pi
        # doesn't look artificially spread out
        mean_dir = np.angle(np.mean(np.exp(1j * ra)))
        ra_centered = np.mod(ra - mean_dir + np.pi, 2 * np.pi)
        nra = _fd_bins(ra_centered, 2 * np.pi, n_eff, floor=180)
    if ndec is None:
        ndec = _fd_bins(np.sin(dec), 2.0, n_eff, floor=90)
    ra_edges = np.linspace(0.0, 2 * np.pi, nra + 1)
    sindec_edges = np.linspace(-1.0, 1.0, ndec + 1)
    counts, _, _ = np.histogram2d(
        np.mod(ra, 2 * np.pi), np.sin(dec), bins=[ra_edges, sindec_edges]
    )
    prob = counts / counts.sum()
    pixel_area_deg2 = (2 * np.pi / nra) * (2.0 / ndec) * (180.0 / np.pi) ** 2
    dec_edges = np.arcsin(sindec_edges)
    return prob, ra_edges, dec_edges, pixel_area_deg2


def _pixel_index(value: float, edges: np.ndarray) -> int:
    idx = np.searchsorted(edges, value, side="right") - 1
    return int(np.clip(idx, 0, len(edges) - 2))


def _truth_density(prob: np.ndarray, i: int, j: int, truth_window: int) -> float:
    """Density "at" the truth pixel, maximized over a ``truth_window``x``truth_window`` neighborhood.

    A chain with a modest effective sample size (see :func:`sky_histogram`)
    leaves genuinely-covered sky area looking like sparse, isolated hits with
    empty gaps between them -- reading the single bin the truth happens to
    fall in is fragile to landing exactly in such a gap right next to real
    support. Taking the best nearby pixel instead (``truth_window=1``
    reproduces the literal single-bin lookup) fixes that without touching
    the density estimate anywhere else on the map, so it doesn't bias
    :func:`credible_area` or a plotted skymap.
    """
    if truth_window <= 1:
        return float(prob[i, j])
    half = truth_window // 2
    nra, ndec = prob.shape
    rows = (np.arange(i - half, i + half + 1)) % nra  # periodic in ra
    cols = np.clip(np.arange(j - half, j + half + 1), 0, ndec - 1)  # clamped at the poles
    return float(prob[np.ix_(rows, cols)].max())


def credible_area(ra, dec, level: float = 0.9, nra: int | None = None, ndec: int | None = None) -> float:
    """Area in deg^2 of the smallest region enclosing ``level`` posterior probability."""
    prob, *_, pixel_area_deg2 = sky_histogram(ra, dec, nra=nra, ndec=ndec)
    sorted_prob = np.sort(prob.ravel())[::-1]
    n_pixels = int(np.searchsorted(np.cumsum(sorted_prob), level)) + 1
    return float(n_pixels * pixel_area_deg2)


def searched_area(
    ra, dec, true_ra: float, true_dec: float,
    nra: int | None = None, ndec: int | None = None, truth_window: int = 3,
) -> float:
    """Area in deg^2 of the smallest credible region that contains the true sky location.

    See :func:`_truth_density` for what ``truth_window`` does and why.
    """
    prob, ra_edges, dec_edges, pixel_area_deg2 = sky_histogram(ra, dec, nra=nra, ndec=ndec)
    i = _pixel_index(np.mod(true_ra, 2 * np.pi), ra_edges)
    j = _pixel_index(true_dec, dec_edges)
    truth_prob = _truth_density(prob, i, j, truth_window)
    n_enclosed = int(np.sum(prob >= truth_prob))
    return float(n_enclosed * pixel_area_deg2)


def searched_probability(
    ra, dec, true_ra: float, true_dec: float,
    nra: int | None = None, ndec: int | None = None, truth_window: int = 3,
) -> float:
    """Posterior probability enclosed by the smallest credible region containing the truth.

    E.g. ``0.42`` means the injected sky location sits inside the 42%
    credible region -- a 10% or 50% area search would miss it, anything at
    or above ~42% would find it. See :func:`_truth_density` for
    ``truth_window``.
    """
    prob, ra_edges, dec_edges, _ = sky_histogram(ra, dec, nra=nra, ndec=ndec)
    i = _pixel_index(np.mod(true_ra, 2 * np.pi), ra_edges)
    j = _pixel_index(true_dec, dec_edges)
    truth_prob = _truth_density(prob, i, j, truth_window)
    return float(np.sum(prob[prob >= truth_prob]))


def sky_localization_summary(
    path: str, nra: int | None = None, ndec: int | None = None, truth_window: int = 3,
    method: str = "histogram", thin: bool = False, max_samples_per_pixel: int = 20,
) -> dict:
    """Searched area/probability and 50%/90% credible areas for one reconstruction file.

    ``method="healpix"`` with ``thin=True`` gives numbers computed the same
    way as AMPLFI's ``CrossMatchStatistics`` (adaptive HEALPix histogram with
    ``max_samples_per_pixel`` samples per pixel); ``nra``/``ndec``/
    ``truth_window`` only apply to ``method="histogram"``.
    """
    ra, dec, true_ra, true_dec = load_sky_samples(path, thin=thin)
    if true_ra is None:
        raise ValueError(f"{path!r} has no injected truth (inj_sky) to search against")
    if method == "healpix":
        cm = crossmatch_sky(ra, dec, true_ra, true_dec, contours=(0.5, 0.9),
                            max_samples_per_pixel=max_samples_per_pixel)
        return {
            "searched_area_deg2": cm["searched_area_deg2"],
            "searched_probability": cm["searched_probability"],
            "credible_area_50_deg2": cm["contour_areas_deg2"][0],
            "credible_area_90_deg2": cm["contour_areas_deg2"][1],
            "n_samples": len(ra),
        }
    if method != "histogram":
        raise ValueError(f'method must be "histogram" or "healpix", not {method!r}')
    return {
        "searched_area_deg2": searched_area(
            ra, dec, true_ra, true_dec, nra=nra, ndec=ndec, truth_window=truth_window
        ),
        "searched_probability": searched_probability(
            ra, dec, true_ra, true_dec, nra=nra, ndec=ndec, truth_window=truth_window
        ),
        "credible_area_50_deg2": credible_area(ra, dec, 0.5, nra=nra, ndec=ndec),
        "credible_area_90_deg2": credible_area(ra, dec, 0.9, nra=nra, ndec=ndec),
        "n_samples": len(ra),
    }
