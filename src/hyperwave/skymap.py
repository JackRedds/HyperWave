"""Sky-localization metrics from posterior RA/Dec samples.

The wavelet reconstruction scripts (``scripts/*_wavelet_reconstruction.py``)
save posterior sky samples in the ``extrinsic`` array of their output
``*_reconstruction.npz`` file, with columns ``[ra, dec, psi, ellipticity]``
(radians, only populated when run with ``--sample-sky``), and the injected
truth in ``inj_sky`` as ``[ra, dec, psi]``. This module turns those flat
samples into the standard GW sky-localization statistics: the searched area
and searched probability at the injected location, and the credible area at
an arbitrary confidence level.

No skymap infrastructure (HEALPix, ``ligo.skymap``) is assumed or required --
posterior density is estimated with a simple equal-area sky histogram: bins
uniform in right ascension and in ``sin(dec)`` have exactly equal solid
angle, since the sphere's area element is ``d(ra) d(sin dec))``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "load_sky_samples",
    "sky_histogram",
    "credible_area",
    "searched_area",
    "searched_probability",
    "sky_localization_summary",
]


def load_sky_samples(path: str):
    """Load RA/Dec posterior samples and the injected truth from a reconstruction ``.npz``.

    Returns ``(ra, dec, true_ra, true_dec)`` in radians; the truth values are
    ``None`` if the file has no ``inj_sky`` entry.
    """
    with np.load(path) as d:
        extrinsic = d["extrinsic"]
        true_ra, true_dec = None, None
        if "inj_sky" in d:
            inj_sky = d["inj_sky"]
            true_ra, true_dec = float(inj_sky[0]), float(inj_sky[1])
    if extrinsic.shape[0] == 0:
        raise ValueError(f"{path!r} has no sky posterior samples (run with --sample-sky)")
    return extrinsic[:, 0], extrinsic[:, 1], true_ra, true_dec


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
    path: str, nra: int | None = None, ndec: int | None = None, truth_window: int = 3
) -> dict:
    """Searched area/probability and 50%/90% credible areas for one reconstruction file."""
    ra, dec, true_ra, true_dec = load_sky_samples(path)
    if true_ra is None:
        raise ValueError(f"{path!r} has no injected truth (inj_sky) to search against")
    return {
        "searched_area_deg2": searched_area(
            ra, dec, true_ra, true_dec, nra=nra, ndec=ndec, truth_window=truth_window
        ),
        "searched_probability": searched_probability(
            ra, dec, true_ra, true_dec, nra=nra, ndec=ndec, truth_window=truth_window
        ),
        "credible_area_50_deg2": credible_area(ra, dec, 0.5, nra=nra, ndec=ndec),
        "credible_area_90_deg2": credible_area(ra, dec, 0.9, nra=nra, ndec=ndec),
    }
