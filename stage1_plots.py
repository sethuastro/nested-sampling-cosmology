"""Stage 1 figure: the likelihood as a function of prior mass, L(X).

Nested sampling turns the evidence into a one-dimensional integral,

    Z = integral from 0 to 1 of L(X) dX.

This script plots the curve L(X) that the sampler actually traced out and
compares it with the exact curve for the Gaussian test problem.

The exact curve.  The likelihood contour L = lambda is a sphere of radius
r around the mean. With a uniform prior on the box [-a, a]^d, the prior
mass inside that sphere is its volume divided by the box volume,

    X(r) = V_d r^d / (2a)^d,      V_d = pi^(d/2) / Gamma(d/2 + 1),

as long as the sphere fits inside the box (r <= a). Inverting,

    r(X) = 2a (X / V_d)^(1/d),    L(X) = L_max exp(-r(X)^2 / (2 sigma^2)).

The second panel uses dX = X d(ln X), so that

    Z = integral of L(X) X d(ln X).

L X is therefore the evidence contributed per unit ln X. The peak of this
curve is where the posterior lives. The information H sets its scale: in
these tests the peak lies within a nat or two of ln X = -H (it is not
exactly at -H, as the dotted lines show). Reaching ln X = -H takes N H
iterations, which is why the run length grows in proportion to N and H.

Run:  python stage1_plots.py
Writes figures/stage1_likelihood_vs_prior_mass.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

from gaussian_test import make_problem
from nested_sampler import nested_sampling

A, SIGMA, N_LIVE = 5.0, 1.0, 400
COLOURS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}


def exact_logl_ratio(logx, d, a=A, sigma=SIGMA):
    """ln(L / L_max) as a function of ln X, valid while the sphere fits in the box."""
    log_vd = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1)
    log_r = np.log(2 * a) + (logx - log_vd) / d
    r = np.exp(log_r)
    out = -0.5 * r**2 / sigma**2
    return np.where(r <= a, out, np.nan)


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for d in (1, 2, 3):
        logl, prior, logz_true = make_problem(d=d, a=A, sigma=SIGMA)
        res = nested_sampling(logl, prior, n_live=N_LIVE,
                              rng=np.random.default_rng(42))
        n = res["n_iter"]
        logx = res["logx"]                      # ln X_i = -i/N for dead points
        logl_dead = res["logl"][:n]
        logl_max = -d * np.log(np.sqrt(2 * np.pi) * SIGMA)

        # Panel 1: L / L_max against X.
        ax1.plot(np.exp(logx), np.exp(logl_dead - logl_max), ".", ms=2.5,
                 color=COLOURS[d], alpha=0.6, rasterized=True)
        grid = np.linspace(logx.min(), 0.0, 600)
        ax1.plot(np.exp(grid), np.exp(exact_logl_ratio(grid, d)), "-",
                 color="black", lw=0.9)
        ax1.plot([], [], "o", ms=4, color=COLOURS[d], label=f"{d}D sampler")

        # Panel 2: evidence per unit ln X, normalised to integrate to 1.
        ax2.plot(logx, np.exp(logl_dead + logx - res["logz"]), "-",
                 color=COLOURS[d], lw=1.6,
                 label=f"{d}D:  H = {res['H']:.2f}")
        ax2.axvline(-res["H"], color=COLOURS[d], lw=0.9, ls=":")

    ax1.plot([], [], "-", color="black", lw=0.9, label="exact")
    ax1.set_xscale("log")
    ax1.set_xlabel("prior mass X")
    ax1.set_ylabel(r"$L(X)\,/\,L_{\max}$")
    ax1.set_title("Likelihood against prior mass")
    ax1.legend(frameon=False, loc="lower left")

    ax2.set_xlabel(r"$\ln X$")
    ax2.set_ylabel(r"$L\,X\,/\,Z$   (evidence per unit $\ln X$)")
    ax2.set_title(r"Where the evidence comes from (dotted: $\ln X = -H$)")
    ax2.legend(frameon=False, loc="upper left")

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/stage1_likelihood_vs_prior_mass.png", dpi=200)
    print("saved figures/stage1_likelihood_vs_prior_mass.png")
