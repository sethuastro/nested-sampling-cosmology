"""Stage 2: BlackJAX Nested Slice Sampling on the same Gaussian tests.

Three questions:

A. Does BlackJAX agree with my NumPy sampler and with the analytic ln Z
   in 1, 2 and 3 dimensions?
B. Does it keep working in 5, 10 and 20 dimensions, where rejection
   sampling is far too slow to run?
C. Over repeated runs, is it unbiased, and is its error bar the right size?

Why rejection sampling fails.  A prior draw lands inside the contour with
probability X, so each new point costs about 1/X likelihood calls. My
Stage 1 sampler stops when the live points hold 1% of Z, which for this
problem is at

    X_stop = 0.01 Z / L_max = 0.01 (2 pi sigma^2)^(d/2) / (2a)^d,

and the total cost is about N / X_stop. With a = 5 and sigma = 1 that is
4 x 10^4 x 3.99^d calls for N = 400: it grows by a factor of 4 with every
extra dimension. Slice sampling costs roughly the same per point at any
X, so its total cost grows only gently with d.

Run:  python stage2_blackjax_gaussian.py      (about 5 minutes on a laptop CPU)
Writes results/stage2_gaussian.json and figures/stage2_cost_vs_dimension.png
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blackjax_runner import make_nss
from gaussian_test import make_problem
from nested_sampler import nested_sampling

A, SIGMA, N_LIVE, NUM_DELETE = 5.0, 1.0, 400, 20


def make_jax_problem(d):
    """The same Gaussian-in-a-box problem as gaussian_test.py, written for JAX."""

    def loglikelihood(theta):
        return (-0.5 * jnp.sum(theta**2) / SIGMA**2
                - d * jnp.log(jnp.sqrt(2 * jnp.pi) * SIGMA))

    def logprior(theta):
        # Uniform on [-A, A]^d: -d ln(2A) inside the box, -inf outside.
        return jnp.sum(jax.scipy.stats.uniform.logpdf(theta, -A, 2 * A))

    return logprior, loglikelihood


def rejection_cost(d, n_live=N_LIVE):
    """Predicted likelihood calls for the Stage 1 rejection sampler."""
    x_stop = 0.01 * (2 * np.pi * SIGMA**2) ** (d / 2) / (2 * A) ** d
    return n_live / x_stop


if __name__ == "__main__":
    results = {"settings": {"a": A, "sigma": SIGMA, "n_live": N_LIVE,
                            "num_delete": NUM_DELETE,
                            "num_inner_steps": "max(5, 2 d)"},
               "comparison": {}, "repeated_runs": {}}
    samplers = {}

    print("A and B: one run per dimension")
    for d in (1, 2, 3, 5, 10, 20):
        _, _, logz_true = make_problem(d=d, a=A, sigma=SIGMA)
        logprior, loglikelihood = make_jax_problem(d)
        samplers[d] = make_nss(logprior, loglikelihood,
                               num_inner_steps=max(5, 2 * d),
                               num_delete=NUM_DELETE)
        start = np.random.default_rng(42).uniform(-A, A, size=(N_LIVE, d))
        bj = samplers[d](42, jnp.asarray(start))

        row = {"logz_analytic": float(logz_true),
               "blackjax": {"logz": bj["logz"], "logz_err": bj["logz_err"],
                            "n_calls": bj["n_calls"]},
               "rejection_predicted_calls": rejection_cost(d)}
        line = (f"d={d:2d}  analytic {logz_true:8.3f}   BlackJAX "
                f"{bj['logz']:8.3f} +/- {bj['logz_err']:.3f} "
                f"({(bj['logz'] - logz_true) / bj['logz_err']:+.1f} sigma, "
                f"{bj['n_calls']:,} calls)")

        if d <= 3:      # the NumPy sampler is only practical here
            logl_np, prior_np, _ = make_problem(d=d, a=A, sigma=SIGMA)
            mine = nested_sampling(logl_np, prior_np, n_live=N_LIVE,
                                   rng=np.random.default_rng(42))
            row["numpy"] = {"logz": float(mine["logz"]),
                            "logz_err": float(mine["logz_err"]),
                            "n_calls": int(mine["n_calls"])}
            line += (f"   NumPy {mine['logz']:8.3f} +/- {mine['logz_err']:.3f} "
                     f"({mine['n_calls']:,} calls)")
        else:
            line += f"   NumPy: not run (predicted {rejection_cost(d):.1e} calls)"
        print(line)
        results["comparison"][f"d={d}"] = row

    print("\nC: repeated BlackJAX runs")
    for d, n_runs in ((2, 20), (10, 20)):
        _, _, logz_true = make_problem(d=d, a=A, sigma=SIGMA)
        diff, err = [], []
        for seed in range(100, 100 + n_runs):
            start = np.random.default_rng(seed).uniform(-A, A, size=(N_LIVE, d))
            r = samplers[d](seed, jnp.asarray(start))
            diff.append(r["logz"] - logz_true)
            err.append(r["logz_err"])
        diff, err = np.array(diff), np.array(err)
        s = {"n_runs": n_runs,
             "mean_difference": float(diff.mean()),
             "error_on_mean_difference": float(diff.std(ddof=1) / np.sqrt(n_runs)),
             "scatter_between_runs": float(diff.std(ddof=1)),
             "mean_estimated_error": float(err.mean())}
        results["repeated_runs"][f"d={d}"] = s
        print(f"d={d:2d}: {n_runs} runs   mean difference = "
              f"{s['mean_difference']:+.3f} +/- {s['error_on_mean_difference']:.3f}"
              f"   scatter = {s['scatter_between_runs']:.3f}"
              f"   mean estimated error = {s['mean_estimated_error']:.3f}")

    # Figure: likelihood calls against dimension.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    dims = [int(k[2:]) for k in results["comparison"]]
    grid = np.linspace(1, 20, 100)
    ax.plot(grid, rejection_cost(grid), "-", color="#D55E00", lw=1.3)
    measured = [d for d in dims if "numpy" in results["comparison"][f"d={d}"]]
    ax.plot(measured,
            [results["comparison"][f"d={d}"]["numpy"]["n_calls"] for d in measured],
            "o", color="#D55E00", ms=7)
    ax.plot(dims,
            [results["comparison"][f"d={d}"]["blackjax"]["n_calls"] for d in dims],
            "o-", color="#0072B2", ms=7, lw=1.3)
    ax.text(1.6, 3e12, "rejection sampling\n(points: measured,\nline: predicted)",
            color="#D55E00", va="center")
    ax.text(11, 1.6e5, "BlackJAX nested slice sampling\n(measured)",
            color="#0072B2", va="center")
    ax.set_yscale("log")
    ax.set_ylim(3e4, 1e17)
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.set_xlabel("number of dimensions d")
    ax.set_ylabel("likelihood evaluations")
    ax.set_title("Cost of one run, 400 live points")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("results/stage2_gaussian.json").write_text(json.dumps(results, indent=2))
    fig.savefig("figures/stage2_cost_vs_dimension.png", dpi=200)
    print("\nsaved results/stage2_gaussian.json and "
          "figures/stage2_cost_vs_dimension.png")
