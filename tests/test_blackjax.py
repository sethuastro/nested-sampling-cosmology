import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp

from blackjax_runner import make_nss
from gaussian_test import make_problem
from stage2_blackjax_gaussian import A, SIGMA, make_jax_problem


@pytest.mark.parametrize("d", [2, 5])
def test_blackjax_recovers_analytic_evidence(d):
    _, _, logz_true = make_problem(d=d, a=A, sigma=SIGMA)
    logprior, loglikelihood = make_jax_problem(d)
    run = make_nss(logprior, loglikelihood, num_inner_steps=max(5, 2 * d),
                   num_delete=20)
    start = np.random.default_rng(d).uniform(-A, A, size=(400, d))
    res = run(d, jnp.asarray(start))
    assert abs(res["logz"] - logz_true) < 4 * res["logz_err"]
    assert np.isclose(res["weights"].sum(), 1.0)


def test_blackjax_posterior_width():
    d = 2
    logprior, loglikelihood = make_jax_problem(d)
    run = make_nss(logprior, loglikelihood, num_inner_steps=5, num_delete=20)
    start = np.random.default_rng(0).uniform(-A, A, size=(400, d))
    res = run(0, jnp.asarray(start))
    mean = np.sum(res["weights"][:, None] * res["samples"], axis=0)
    var = np.sum(res["weights"][:, None] * (res["samples"] - mean) ** 2, axis=0)
    assert np.allclose(mean, 0.0, atol=0.15)
    assert np.allclose(np.sqrt(var), SIGMA, atol=0.15)
