"""Sky-map plotting for posterior RA/Dec samples.

Uses the equal-area histogram (``method="histogram"``) or AMPLFI-style
adaptive HEALPix map (``method="healpix"``, needs ``healpy``) from
:mod:`hyperwave.skymap`, drawn on matplotlib's built-in Mollweide projection.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..skymap import _require_healpy, healpix_skymap, rasterize_healpix, sky_histogram
from .corners import rcparams1, rcparams2
from .style import SIGNAL_COLOR, apply_style


def _wrap_ra(ra):
    """Map RA from [0, 2*pi) to (-pi, pi], the range mollweide axes expect."""
    return np.where(ra > np.pi, ra - 2 * np.pi, ra)


def _healpix_grid(ra, dec, max_samples_per_pixel, credible_levels, nra=720, ndec=360):
    """Rasterize the adaptive HEALPix map onto a regular (ra, dec) plotting grid.

    Returns ``(ra_centers, dec_centers, density, contour_field, levels)``:
    ``density`` in probability per deg^2, and ``contour_field`` the greedy
    credible level of each grid point, so the credible regions are contoured
    at ``levels == credible_levels`` exactly (same greedy sort as
    :func:`hyperwave.skymap.crossmatch_sky`, done on the HEALPix pixels).
    """
    hp = _require_healpy()
    order, ipix, probdensity = healpix_skymap(ra, dec, max_samples_per_pixel=max_samples_per_pixel)
    # nside 512 (~0.1 deg) is already finer than the plotting grid
    raster_order = int(min(order.max(), 9))
    density = rasterize_healpix(order, ipix, probdensity, raster_order)
    rank = np.argsort(density)[::-1]
    cls = np.empty_like(density)
    cls[rank] = np.cumsum(density[rank]) / density.sum()

    ra_centers = np.linspace(-np.pi, np.pi, nra, endpoint=False) + np.pi / nra
    dec_centers = np.linspace(-np.pi / 2, np.pi / 2, ndec, endpoint=False) + np.pi / (2 * ndec)
    rr, dd = np.meshgrid(np.mod(ra_centers, 2 * np.pi), dec_centers, indexing="ij")
    pix = hp.ang2pix(1 << raster_order, 0.5 * np.pi - dd, rr, nest=True)
    per_deg2 = density[pix] * (np.pi / 180) ** 2
    return ra_centers, dec_centers, per_deg2, cls[pix], sorted(credible_levels)


def plot_skymap(
    ra,
    dec,
    true_ra=None,
    true_dec=None,
    nra=None,
    ndec=None,
    credible_levels=(0.5, 0.9),
    cmap="magma_r",
    flip_ra=True,
    title="",
    outpath=None,
    show=True,
    black_background=False,
    panel_scale=0.9,
    preset="prd",
    method="histogram",
    max_samples_per_pixel=20,
):
    """Mollweide sky map of posterior RA/Dec samples (radians).

    Shades the equal-area posterior density and draws contours at each
    confidence level in ``credible_levels``. If ``true_ra``/``true_dec`` are
    given, the injected location is marked with a star. ``flip_ra=True``
    plots East-to-the-left, the usual astronomical convention.

    ``method="healpix"`` shades the adaptive HEALPix map AMPLFI uses (see
    :func:`hyperwave.skymap.healpix_skymap`; ``nra``/``ndec`` are ignored),
    which averages sparse regions over large pixels instead of showing raw
    sampling noise. Pass chain-thinned samples
    (:func:`hyperwave.skymap.thin_sky_samples`) for a fair picture.
    """
    if method == "healpix":
        ra_centers, dec_centers, prob, contour_field, levels = _healpix_grid(
            ra, dec, max_samples_per_pixel, credible_levels
        )
        colorbar_label = r"Posterior probability per deg$^2$"
    elif method == "histogram":
        prob, ra_edges, dec_edges, _ = sky_histogram(ra, dec, nra=nra, ndec=ndec)

        # Reindex to a monotonic RA axis in (-pi, pi] so pcolormesh/contour draw correctly.
        ra_centers = _wrap_ra(0.5 * (ra_edges[:-1] + ra_edges[1:]))
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
        order = np.argsort(ra_centers)
        ra_centers, prob = ra_centers[order], prob[order, :]

        # Density thresholds enclosing each credible level (same greedy sort as
        # hyperwave.skymap.credible_area), contoured for a visual credible region.
        sorted_desc = np.sort(prob.ravel())[::-1]
        cum = np.cumsum(sorted_desc)
        levels = sorted({sorted_desc[np.searchsorted(cum, lvl)] for lvl in credible_levels})
        contour_field = prob
        colorbar_label = "Posterior probability"
    else:
        raise ValueError(f'method must be "histogram" or "healpix", not {method!r}')

    if black_background:
        matplotlib.rcParams.update(rcparams1)
    else:
        matplotlib.rcParams.update(rcparams2)
    apply_style(preset=preset, black_background=black_background,
                transparent=black_background, panel_scale=panel_scale)

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "mollweide"})

    mesh = ax.pcolormesh(ra_centers, dec_centers, prob.T, cmap=cmap, shading="nearest")
    fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, shrink=0.7,
                 label=colorbar_label)

    if levels:
        line_color = "white" if black_background else "black"
        ax.contour(ra_centers, dec_centers, contour_field.T, levels=levels, colors=line_color,
                   linewidths=1.0, alpha=0.8)

    if true_ra is not None and true_dec is not None:
        ax.plot(_wrap_ra(np.mod(true_ra, 2 * np.pi)), true_dec, marker="*",
                markersize=20, color=SIGNAL_COLOR, markeredgecolor="black",
                markeredgewidth=0.6, linestyle="none", label="Injected", zorder=5)
        ax.legend(loc="upper right", fontsize=14)

    if flip_ra:
        ax.invert_xaxis()

    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight", transparent=black_background)
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight", transparent=black_background)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
