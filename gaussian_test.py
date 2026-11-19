"""Check the sampler against a Gaussian likelihood with known evidence.

Likelihood: normalised d-dimensional Gaussian, mean mu, width sigma.
Prior: uniform on the box [-a, a]^d.

    Z = (2a)^(-d) * prod_k [Phi((a - mu_k)/sigma) - Phi((-a - mu_k)/sigma)]

If the Gaussian sits well inside the box, Z is very close to (2a)^(-d),
so ln Z is close to -d ln(2a).
"""

import numpy as np
from scipy.stats import norm

from nested_sampler import nested_sampling


def make_problem(d=2, a=5.0, sigma=1.0, mu=None):
    mu = np.zeros(d) if mu is None else np.asarray(mu, dtype=float)

    def log_likelihood(theta):
        r2 = np.sum((theta - mu) ** 2, axis=1)
        return -0.5 * r2 / sigma**2 - d * np.log(np.sqrt(2 * np.pi) * sigma)

    def prior_sample(n, rng):
        return rng.uniform(-a, a, size=(n, d))

    mass = norm.cdf((a - mu) / sigma) - norm.cdf((-a - mu) / sigma)
    logz_true = -d * np.log(2 * a) + np.sum(np.log(mass))
    return log_likelihood, prior_sample, logz_true


if __name__ == "__main__":
    for d in (1, 2, 3):
        logl, prior, logz_true = make_problem(d=d)
        res = nested_sampling(logl, prior, n_live=400,
                              rng=np.random.default_rng(42))
        pull = (res["logz"] - logz_true) / res["logz_err"]
        print(f"d={d}:  ln Z = {res['logz']:.3f} +/- {res['logz_err']:.3f}"
              f"   analytic = {logz_true:.3f}   difference = {pull:+.2f} sigma"
              f"   ({res['n_iter']} iterations, {res['n_calls']} likelihood calls)")
