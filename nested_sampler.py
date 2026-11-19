"""A simple nested sampler, written from scratch to learn the algorithm.

Nested sampling (Skilling 2006) computes the Bayesian evidence

    Z = integral of L(theta) * pi(theta) d(theta)

by turning it into a one-dimensional integral over prior mass X:

    Z = integral from 0 to 1 of L(X) dX.

This version uses rejection sampling from the prior to replace live
points. That is the simplest correct choice, but it becomes slow as the
likelihood contours shrink, so it is only practical in low dimensions.
"""

import numpy as np
from scipy.special import logsumexp


def nested_sampling(log_likelihood, prior_sample, n_live=400, dlogz=0.01,
                    max_iter=100_000, rng=None):
    """Run nested sampling and return the log-evidence and its uncertainty.

    Parameters
    ----------
    log_likelihood : callable
        Takes an array of shape (n, d) and returns log L for each row.
    prior_sample : callable
        prior_sample(n, rng) returns n independent draws from the prior,
        as an array of shape (n, d).
    n_live : int
        Number of live points N.
    dlogz : float
        Stop when the remaining live points could add at most this much
        to log Z.
    max_iter : int
        Safety limit on the number of iterations.
    rng : numpy.random.Generator, optional

    Returns
    -------
    dict with keys
        "logz"     : estimated log-evidence
        "logz_err" : estimated uncertainty, sqrt(H / N)
        "H"        : information (KL divergence from prior to posterior), in nats
        "samples"  : dead points followed by the final live points,
                     shape (n_iter + n_live, d)
        "logl"     : log-likelihood of each of those points
        "logwt"    : log of the posterior weight of each of those points
        "logx"     : log prior mass assigned to each dead point, -i/N
        "n_live"   : number of live points N
        "n_iter"   : number of iterations
        "n_calls"  : number of likelihood evaluations
    """
    rng = np.random.default_rng() if rng is None else rng

    # 1. Draw N live points from the prior.
    live = prior_sample(n_live, rng)
    live_logl = log_likelihood(live)
    n_calls = n_live

    dead, dead_logl, dead_logw, dead_logx = [], [], [], []
    logz = -np.inf      # running log Z
    logx_prev = 0.0     # log X_0 = log 1 = 0

    for i in range(1, max_iter + 1):
        # 2. The live point with the lowest likelihood sets the contour L*.
        worst = np.argmin(live_logl)
        logl_star = live_logl[worst]

        # 3. Prior mass shrinks on average by exp(-1/N) per iteration:
        #    X_i = t * X_{i-1}, t ~ Beta(N, 1), E[ln t] = -1/N.
        logx = -i / n_live
        # Weight w_i = X_{i-1} - X_i, computed in log space.
        logw = logx_prev + np.log1p(-np.exp(logx - logx_prev))

        # 4. Add L* w_i to the evidence.
        logz = np.logaddexp(logz, logl_star + logw)

        dead.append(live[worst].copy())
        dead_logl.append(logl_star)
        dead_logw.append(logl_star + logw)
        dead_logx.append(logx)

        # 5. Replace the worst point with a new prior draw that has L > L*
        #    (rejection sampling).
        while True:
            trial = prior_sample(1, rng)
            trial_logl = log_likelihood(trial)[0]
            n_calls += 1
            if trial_logl > logl_star:
                break
        live[worst] = trial[0]
        live_logl[worst] = trial_logl
        logx_prev = logx

        # 6. Stop when the live points can no longer change log Z by more
        #    than dlogz: max(L_live) * X_i is an upper bound on what is left.
        logz_remain = np.max(live_logl) + logx
        if np.logaddexp(logz, logz_remain) - logz < dlogz:
            break

    # 7. Add the remaining live points, each with prior mass X_final / N.
    logw_live = live_logl + logx - np.log(n_live)
    logz = np.logaddexp(logz, logsumexp(logw_live))

    samples = np.vstack([np.array(dead), live])
    logl_all = np.concatenate([np.array(dead_logl), live_logl])
    # Normalised posterior weights p_i = L_i w_i / Z.
    logwt = np.concatenate([np.array(dead_logw), logw_live]) - logz

    # 8. Information H = sum_i p_i ln(L_i / Z), and the error on log Z.
    H = np.sum(np.exp(logwt) * (logl_all - logz))

    return {
        "logz": logz,
        "logz_err": np.sqrt(max(H, 0.0) / n_live),
        "H": H,
        "samples": samples,
        "logl": logl_all,
        "logwt": logwt,
        "logx": np.array(dead_logx),
        "n_live": n_live,
        "n_iter": i,
        "n_calls": n_calls,
    }


def simulate_logz(logl_dead, logl_live, n_live, n_samples=1000, rng=None):
    """Estimate the uncertainty on log Z by simulating the shrinkage.

    The sampler above replaces each shrinkage factor t_i by its mean,
    ln X_i = -i/N. Here we instead draw every t_i from its true
    distribution, Beta(n_i, 1), many times. Each draw gives one possible
    set of prior masses X_i and therefore one possible value of log Z.
    The spread of those values is the uncertainty, with no approximation
    beyond the algorithm itself.

    The final live points are treated as if they were removed one at a
    time, so the number of live points falls N, N-1, ..., 1.

    Returns an array of n_samples values of log Z.
    """
    rng = np.random.default_rng() if rng is None else rng
    logl = np.concatenate([logl_dead, np.sort(logl_live)])
    n = np.concatenate([np.full(len(logl_dead), n_live),
                        np.arange(len(logl_live), 0, -1)])
    # If u ~ Uniform(0, 1) then u**(1/n) ~ Beta(n, 1), so ln t = ln(u) / n.
    logt = np.log(rng.uniform(size=(n_samples, len(logl)))) / n
    logx = np.cumsum(logt, axis=1)
    logx_prev = np.concatenate([np.zeros((n_samples, 1)), logx[:, :-1]], axis=1)
    logw = logx_prev + np.log1p(-np.exp(logt))      # ln(X_{i-1} - X_i)
    return logsumexp(logl + logw, axis=1)
