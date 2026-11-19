"""A thin wrapper around BlackJAX Nested Slice Sampling (NSS).

BlackJAX provides the algorithm (blackjax.nss). This file only adds the
outer loop, the stopping rule and the bookkeeping, so that Stage 2 and
Stage 3 can share them.

The outer loop is the same as in nested_sampler.py:

    remove the lowest-likelihood live points  ->  add L_i w_i to Z
    ->  replace them with new points that have L > L*  ->  repeat.

Two things are different.

1. Replacement. Instead of drawing from the whole prior until a point
   happens to land inside the contour, NSS starts from a live point that
   is already inside and moves it with slice sampling: pick a random
   direction, find the stretch of that line that lies inside the contour,
   and jump to a uniformly chosen point on it. After `num_inner_steps`
   such moves the point has forgotten where it started. The cost per new
   point no longer depends on how small the prior mass X has become.

2. Batching. `num_delete` points are removed and replaced in one step.
   The replacements are independent of each other, so JAX can compute
   them in parallel (jax.vmap), which is what makes the method suited to
   GPUs.

Reference: Yallup, Kroupa & Handley, "Nested Slice Sampling: Vectorized
Nested Sampling for GPU-Accelerated Inference", arXiv:2601.23252.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)   # double precision, as in NumPy

import blackjax
import jax.numpy as jnp
import numpy as np
from blackjax.ns.utils import finalise, log_weights
from jax.scipy.special import logsumexp


def make_nss(logprior_fn, loglikelihood_fn, num_inner_steps, num_delete,
             tol=1e-3):
    """Build a nested slice sampler and return a function that runs it.

    Parameters
    ----------
    logprior_fn, loglikelihood_fn : callables
        Each takes one parameter vector of shape (d,) and returns a scalar.
        The log-prior must be -inf outside the allowed region.
    num_inner_steps : int
        Slice moves per new point. BlackJAX recommends at least
        max(5, 2 d).
    num_delete : int
        Points removed and replaced per step.
    tol : float
        Stop when the live points hold less than this fraction of Z.

    Returns
    -------
    run(seed, initial_positions) -> dict
        The compiled functions are reused between calls, so repeated runs
        with different seeds do not recompile.
    """
    algorithm = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
    )
    init_fn = jax.jit(algorithm.init)
    step_fn = jax.jit(algorithm.step)
    log_tol = float(np.log(tol))

    def run(seed, initial_positions):
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        start = time.time()
        live = init_fn(initial_positions, rng_key=init_key)

        dead = []
        # logZ_live is the evidence still held by the live points,
        # (mean live L) * X. Stop when it is a fraction `tol` of logZ.
        while not live.integrator.logZ_live - live.integrator.logZ < log_tol:
            key, step_key = jax.random.split(key)
            live, info = step_fn(step_key, live)
            dead.append(info)
        elapsed = time.time() - start

        # Likelihood evaluations. Each slice move evaluates both ends of
        # its starting bracket (2), then one point per expansion and one
        # per shrink. Add the evaluations of the initial live points.
        n_slice_moves = sum(int(d.update_info.num_shrink.size) for d in dead)
        n_evals = (len(initial_positions) + 2 * n_slice_moves
                   + sum(int(d.update_info.num_expansions.sum()) for d in dead)
                   + sum(int(d.update_info.num_shrink.sum()) for d in dead))

        # Join the dead points and the final live points into one list.
        final = finalise(live, dead)
        logl = final.particles.loglikelihood

        # Error on ln Z, by simulation: draw the shrinkage factors t_i from
        # their Beta distributions 500 times (same idea as simulate_logz in
        # nested_sampler.py). Each column of logw is one such draw.
        key, weight_key = jax.random.split(key)
        logw = log_weights(weight_key, final, shape=500)
        logz_draws = logsumexp(logw, axis=0)
        logz = float(jnp.mean(logz_draws))

        # Posterior weights p_i = L_i w_i / Z, averaged over the draws.
        logp = logsumexp(logw - logz_draws, axis=1) - np.log(logw.shape[1])
        p = np.exp(np.asarray(logp))
        H = float(np.sum(p * (np.asarray(logl) - logz)))

        return {
            "logz": logz,
            "logz_err": float(jnp.std(logz_draws)),
            "H": H,
            "samples": np.asarray(final.particles.position),
            "logl": np.asarray(logl),
            "logl_birth": np.asarray(final.particles.loglikelihood_birth),
            "weights": p / p.sum(),
            "n_dead": len(dead) * num_delete,
            "logx_final": float(live.integrator.logX),
            "n_calls": int(n_evals),
            "seconds": elapsed,
        }

    return run
