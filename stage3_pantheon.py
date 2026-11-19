"""Stage 3: Bayesian evidence for flat LambdaCDM against flat wCDM, Pantheon+.

Question: do the Pantheon+ supernovae prefer dark energy with a free,
constant equation of state w over a cosmological constant (w = -1)?

Models and priors (all uniform):

    LambdaCDM:  Omega_m in [0.01, 0.99],  M in [-20, -19]
    wCDM:       the same, plus  w in [-3, 1]

Bayes factor, with the sign convention

    ln B = ln Z(wCDM) - ln Z(LambdaCDM),

so a negative ln B favours LambdaCDM.

The script does the calculation twice, by two unrelated methods.

1. Nested sampling with BlackJAX, sampling all parameters including M,
   repeated with several seeds.
2. Direct integration on a grid. M is integrated out analytically (see
   cosmology.make_marginal_loglikelihood), which leaves a 1D integral for
   LambdaCDM and a 2D integral for wCDM. In so few dimensions a grid is
   exact to many digits, so it plays the role that the analytic ln Z
   played in Stages 1 and 2.

Run:  python stage3_pantheon.py        (about 30 minutes on a 2-core CPU)
Writes results/stage3_pantheon.json and results/stage3_samples.npz
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from anesthetic import NestedSamples
from scipy.integrate import simpson
from scipy.special import logsumexp

import cosmology
from blackjax_runner import make_nss

PRIOR = {"omega_m": (0.01, 0.99), "w": (-3.0, 1.0), "M": (-20.0, -19.0)}
MODELS = {"LCDM": ["omega_m", "M"], "wCDM": ["omega_m", "w", "M"]}
N_LIVE, NUM_DELETE, N_RUNS = 1000, 50, 4
GRID_OMEGA_M, GRID_W = 401, 601         # odd, so Simpson's rule applies


def make_model(name, loglikelihood):
    """Log-prior and log-likelihood of one model, as functions of a vector."""
    names = MODELS[name]
    low = jnp.array([PRIOR[p][0] for p in names])
    width = jnp.array([PRIOR[p][1] - PRIOR[p][0] for p in names])

    def logprior(theta):
        return jnp.sum(jax.scipy.stats.uniform.logpdf(theta, low, width))

    if name == "LCDM":
        def loglike(theta):
            return loglikelihood(theta[0], -1.0, theta[1])
    else:
        def loglike(theta):
            return loglikelihood(theta[0], theta[1], theta[2])

    def prior_draw(n, rng):
        return np.asarray(low) + np.asarray(width) * rng.uniform(size=(n, len(names)))

    return logprior, loglike, prior_draw


def weighted_mean_std(x, weights):
    mean = np.sum(weights * x)
    return float(mean), float(np.sqrt(np.sum(weights * (x - mean) ** 2)))


def run_nested_sampling(loglikelihood):
    out, samples = {}, {}
    for name, names in MODELS.items():
        logprior, loglike, prior_draw = make_model(name, loglikelihood)
        d = len(names)
        run = make_nss(logprior, loglike, num_inner_steps=3 * d,
                       num_delete=NUM_DELETE)
        runs = []
        for seed in range(N_RUNS):
            start = prior_draw(N_LIVE, np.random.default_rng(seed))
            r = run(seed, jnp.asarray(start))
            summary = {"seed": seed, "logz": r["logz"], "logz_err": r["logz_err"],
                       "H": r["H"], "n_calls": r["n_calls"],
                       "seconds": r["seconds"]}
            for k, p in enumerate(names):
                summary[p] = weighted_mean_std(r["samples"][:, k], r["weights"])
            runs.append(summary)
            print(f"{name:5s} seed {seed}: ln Z = {r['logz']:.3f} +/- "
                  f"{r['logz_err']:.3f}   H = {r['H']:.2f}   "
                  f"{r['n_calls']:,} calls   {r['seconds']:.0f} s", flush=True)
            if seed == 0:
                samples[name] = r
                # Cross-check ln Z with anesthetic, the Handley group's own
                # post-processing package, from the same dead points.
                ns = NestedSamples(r["samples"], logL=r["logl"],
                                   logL_birth=r["logl_birth"], logzero=np.nan)
                summary["logz_anesthetic"] = float(ns.logZ())
        logz = np.array([r["logz"] for r in runs])
        logz_err = np.array([r["logz_err"] for r in runs])
        out[name] = {
            "parameters": names,
            "runs": runs,
            "logz_mean": float(logz.mean()),
            # Error on the mean of N_RUNS independent runs: the typical
            # single-run error divided by sqrt(N_RUNS). With only a few
            # runs the scatter between them is too noisy to use for this,
            # so it is recorded separately as a consistency check.
            "logz_mean_err": float(logz_err.mean() / np.sqrt(N_RUNS)),
            "logz_scatter": float(logz.std(ddof=1)),
        }
    lnb = out["wCDM"]["logz_mean"] - out["LCDM"]["logz_mean"]
    lnb_err = float(np.hypot(out["wCDM"]["logz_mean_err"],
                             out["LCDM"]["logz_mean_err"]))
    out["lnB_wCDM_minus_LCDM"] = {"value": float(lnb), "err": lnb_err}
    return out, samples


def run_grid(data):
    """Evidence by direct integration, with M integrated out analytically."""
    marginal = cosmology.make_marginal_loglikelihood(
        data, m_low=PRIOR["M"][0], m_high=PRIOR["M"][1])
    om = np.linspace(*PRIOR["omega_m"], GRID_OMEGA_M)
    w = np.linspace(*PRIOR["w"], GRID_W)

    # LambdaCDM: a one-dimensional integral over Omega_m at w = -1.
    f1 = jax.jit(jax.vmap(lambda o: marginal(o, -1.0)))
    logl_1d = np.asarray(f1(jnp.asarray(om)))

    # wCDM: a two-dimensional integral. lax.map evaluates the grid in
    # batches to keep the memory use small.
    oo, ww = np.meshgrid(om, w, indexing="ij")
    f2 = jax.jit(lambda pts: jax.lax.map(lambda p: marginal(p[0], p[1]), pts,
                                         batch_size=512))
    logl_2d = np.asarray(f2(jnp.asarray(np.column_stack([oo.ravel(), ww.ravel()]))))
    logl_2d = logl_2d.reshape(oo.shape)

    def integrate(logl, axes, step=1):
        """ln of the integral of exp(logl) over the grid, times the prior."""
        peak = logl.max()
        value = np.exp(logl - peak)
        for x in reversed(axes):
            value = simpson(value, x=x[::step], axis=-1)
        prior_volume = np.prod([x[-1] - x[0] for x in axes])
        return float(peak + np.log(value / prior_volume))

    logz_l = integrate(logl_1d, [om])
    logz_w = integrate(logl_2d, [om, w])
    # The same integrals on a grid twice as coarse, to show convergence.
    logz_l_coarse = integrate(logl_1d[::2], [om], step=2)
    logz_w_coarse = integrate(logl_2d[::2, ::2], [om, w], step=2)

    # Posterior means and widths from the grid.
    post_1d = np.exp(logl_1d - logl_1d.max())
    post_1d /= simpson(post_1d, x=om)
    post_2d = np.exp(logl_2d - logl_2d.max())
    post_2d /= simpson(simpson(post_2d, x=w, axis=1), x=om)
    p_om = simpson(post_2d, x=w, axis=1)
    p_w = simpson(post_2d, x=om, axis=0)

    def mean_std(x, p):
        mean = simpson(x * p, x=x)
        return float(mean), float(np.sqrt(simpson((x - mean) ** 2 * p, x=x)))

    # Best fit, and how much better wCDM fits than LambdaCDM.
    i, j = np.unravel_index(np.argmax(logl_2d), logl_2d.shape)

    # How ln B changes if the prior on w is made narrower. Each range ends
    # on a grid point, so the integral is just restricted to part of the grid.
    prior_sensitivity = {}
    for lo, hi in ((-3.0, 1.0), (-2.0, 0.0), (-1.5, -0.5)):
        sel = (w >= lo - 1e-9) & (w <= hi + 1e-9)
        lz = integrate(logl_2d[:, sel], [om, w[sel]])
        prior_sensitivity[f"w in [{lo:g}, {hi:g}]"] = lz - logz_l

    out = {
        "grid_points": [GRID_OMEGA_M, GRID_W],
        "logz_LCDM": logz_l,
        "logz_wCDM": logz_w,
        "lnB_wCDM_minus_LCDM": logz_w - logz_l,
        "convergence_change_from_half_grid": {
            "logz_LCDM": logz_l - logz_l_coarse,
            "logz_wCDM": logz_w - logz_w_coarse},
        "LCDM_omega_m": mean_std(om, post_1d),
        "wCDM_omega_m": mean_std(om, p_om),
        "wCDM_w": mean_std(w, p_w),
        "wCDM_best_fit": {"omega_m": float(om[i]), "w": float(w[j])},
        "delta_max_loglike": float(logl_2d.max() - logl_1d.max()),
        "lnB_for_other_w_priors": prior_sensitivity,
    }
    arrays = {"grid_om": om, "grid_w": w, "grid_post_1d": post_1d,
              "grid_post_2d": post_2d}
    return out, arrays


if __name__ == "__main__":
    t0 = time.time()
    data = cosmology.load_pantheon_plus("data")
    print(f"Pantheon+: {len(data['m_b'])} supernovae with z_HD > 0.01")
    loglikelihood = cosmology.make_loglikelihood(data)

    grid, arrays = run_grid(data)
    print(f"grid:  ln Z(LCDM) = {grid['logz_LCDM']:.3f}   "
          f"ln Z(wCDM) = {grid['logz_wCDM']:.3f}   "
          f"ln B = {grid['lnB_wCDM_minus_LCDM']:.3f}", flush=True)

    nested, samples = run_nested_sampling(loglikelihood)
    b = nested["lnB_wCDM_minus_LCDM"]
    print(f"\nnested sampling:  ln Z(LCDM) = {nested['LCDM']['logz_mean']:.3f} "
          f"+/- {nested['LCDM']['logz_mean_err']:.3f}   ln Z(wCDM) = "
          f"{nested['wCDM']['logz_mean']:.3f} +/- "
          f"{nested['wCDM']['logz_mean_err']:.3f}")
    print(f"nested sampling:  ln B = {b['value']:.3f} +/- {b['err']:.3f}"
          f"     grid:  ln B = {grid['lnB_wCDM_minus_LCDM']:.3f}")

    results = {
        "data": {"n_supernovae": int(len(data["m_b"])), "z_min": 0.01,
                 "H0_fixed": cosmology.H0},
        "priors": PRIOR,
        "settings": {"n_live": N_LIVE, "num_delete": NUM_DELETE,
                     "num_inner_steps": "3 d", "n_runs": N_RUNS},
        "nested_sampling": nested,
        "grid": grid,
        "total_seconds": time.time() - t0,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/stage3_pantheon.json").write_text(json.dumps(results, indent=2))

    # Keep the weighted posterior samples of the first run of each model,
    # and the grid posteriors, for stage3_plots.py.
    for name in MODELS:
        arrays[f"{name}_samples"] = samples[name]["samples"]
        arrays[f"{name}_weights"] = samples[name]["weights"]
    np.savez_compressed("results/stage3_samples.npz", **arrays)
    print("saved results/stage3_pantheon.json and results/stage3_samples.npz")
