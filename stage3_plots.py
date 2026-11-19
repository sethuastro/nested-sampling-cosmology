"""Stage 3 figure: posteriors from nested sampling against the grid.

Run stage3_pantheon.py first. This script only reads its saved results.

Left:  the posterior of Omega_m in flat LambdaCDM. Histogram: weighted
       nested sampling points. Line: direct integration on a grid.
Right: the posterior in the (Omega_m, w) plane for flat wCDM. Points:
       nested sampling. Contours: the 68% and 95% regions from the grid.
       The dashed line is LambdaCDM, w = -1.

Run:  python stage3_plots.py
Writes figures/stage3_posteriors.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#0072B2", "#D55E00"


def credible_levels(density, cell_area, fractions=(0.95, 0.68)):
    """Density values whose contours enclose the given posterior fractions."""
    flat = np.sort(density.ravel())[::-1]
    enclosed = np.cumsum(flat) * cell_area
    return [flat[np.searchsorted(enclosed, f)] for f in fractions]


if __name__ == "__main__":
    s = np.load("results/stage3_samples.npz")
    om, w = s["grid_om"], s["grid_w"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: Omega_m in LambdaCDM (column 0 of the samples).
    ax1.hist(s["LCDM_samples"][:, 0], weights=s["LCDM_weights"], bins=40,
             range=(0.26, 0.41), density=True, color=BLUE, alpha=0.45,
             label="nested sampling")
    ax1.plot(om, s["grid_post_1d"], color="black", lw=1.4, label="grid")
    ax1.set_xlim(0.26, 0.41)
    ax1.set_xlabel(r"$\Omega_m$")
    ax1.set_ylabel("posterior density")
    ax1.set_title(r"Flat $\Lambda$CDM")
    ax1.legend(frameon=False)

    # Right: (Omega_m, w) in wCDM. Draw points in proportion to their
    # posterior weight so that the scatter shows the posterior itself.
    rng = np.random.default_rng(0)
    pick = rng.choice(len(s["wCDM_weights"]), size=3000, p=s["wCDM_weights"])
    ax2.plot(s["wCDM_samples"][pick, 0], s["wCDM_samples"][pick, 1], ".",
             ms=2.5, color=BLUE, alpha=0.4, rasterized=True)
    ax2.plot([], [], "o", ms=4, color=BLUE, label="nested sampling")
    cell = (om[1] - om[0]) * (w[1] - w[0])
    ax2.contour(om, w, s["grid_post_2d"].T,
                levels=credible_levels(s["grid_post_2d"], cell),
                colors="black", linewidths=1.3)
    ax2.plot([], [], "-", color="black", lw=1.3, label="grid, 68% and 95%")
    ax2.axhline(-1.0, color=ORANGE, lw=1.3, ls="--")
    ax2.text(0.02, -0.97, r"$\Lambda$CDM, $w=-1$", color=ORANGE, va="bottom")
    ax2.set_xlim(0.0, 0.55)
    ax2.set_ylim(-1.9, -0.4)
    ax2.set_xlabel(r"$\Omega_m$")
    ax2.set_ylabel(r"$w$")
    ax2.set_title(r"Flat $w$CDM")
    ax2.legend(frameon=False, loc="lower left")

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/stage3_posteriors.png", dpi=200)
    print("saved figures/stage3_posteriors.png")
