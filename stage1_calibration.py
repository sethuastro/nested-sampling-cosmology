"""Stage 1 check: is the sampler unbiased, and is its error bar honest?

One run cannot answer this. A single result that is 0.3 sigma from the
truth says very little. So we repeat the run with many different random
seeds and look at the "pull" of each run,

    pull = (ln Z_sampler - ln Z_analytic) / sigma_estimated.

If the sampler is unbiased and sqrt(H/N) is the right error, the pulls
follow a standard normal distribution: mean 0 and standard deviation 1.

Run:  python stage1_calibration.py
Writes results/stage1_calibration.json and figures/stage1_calibration.png
"""

import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from gaussian_test import make_problem
from nested_sampler import nested_sampling

N_LIVE = 400
N_RUNS = {1: 100, 2: 50, 3: 20}     # fewer runs in 3D because each is slow
COLOURS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}


def one_run(job):
    d, seed = job
    logl, prior, logz_true = make_problem(d=d)
    res = nested_sampling(logl, prior, n_live=N_LIVE,
                          rng=np.random.default_rng(seed))
    return d, res["logz"] - logz_true, res["logz_err"]


if __name__ == "__main__":
    jobs = [(d, 1000 * d + s) for d, n in N_RUNS.items() for s in range(n)]
    with Pool() as pool:
        out = pool.map(one_run, jobs)

    summary = {}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    grid = np.linspace(-4, 4, 200)
    bins = np.linspace(-4, 4, 17)
    stacked = []
    for d in N_RUNS:
        diff = np.array([o[1] for o in out if o[0] == d])
        err = np.array([o[2] for o in out if o[0] == d])
        pull = diff / err
        stacked.append(pull)
        summary[f"d={d}"] = {
            "n_runs": len(diff),
            "mean_difference": float(diff.mean()),
            "error_on_mean_difference": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "scatter_between_runs": float(diff.std(ddof=1)),
            "mean_estimated_error": float(err.mean()),
            "pull_mean": float(pull.mean()),
            "pull_std": float(pull.std(ddof=1)),
        }
        s = summary[f"d={d}"]
        print(f"d={d}: {s['n_runs']} runs   mean difference = "
              f"{s['mean_difference']:+.4f} +/- {s['error_on_mean_difference']:.4f}"
              f"   scatter = {s['scatter_between_runs']:.4f}"
              f"   mean estimated error = {s['mean_estimated_error']:.4f}"
              f"   pull std = {s['pull_std']:.2f}")

    ax.hist(stacked, bins=bins, stacked=True, density=True,
            color=[COLOURS[d] for d in N_RUNS],
            label=[f"{d}D ({N_RUNS[d]} runs)" for d in N_RUNS],
            edgecolor="white", linewidth=0.6)
    ax.plot(grid, norm.pdf(grid), color="black", lw=1.5,
            label="standard normal")
    ax.set_xlabel(r"pull = ($\ln Z_{\rm sampler} - \ln Z_{\rm analytic}$) / $\sqrt{H/N}$")
    ax.set_ylabel("probability density")
    ax.set_title("Repeated runs: the error estimate is calibrated")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path("results/stage1_calibration.json").write_text(json.dumps(summary, indent=2))
    fig.savefig("figures/stage1_calibration.png", dpi=200)
    print("saved results/stage1_calibration.json and figures/stage1_calibration.png")
