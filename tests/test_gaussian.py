import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_test import make_problem
from nested_sampler import nested_sampling


@pytest.mark.parametrize("d", [1, 2, 3])
def test_recovers_analytic_evidence(d):
    logl, prior, logz_true = make_problem(d=d)
    res = nested_sampling(logl, prior, n_live=400,
                          rng=np.random.default_rng(d))
    assert abs(res["logz"] - logz_true) < 4 * res["logz_err"]


def test_offset_gaussian():
    logl, prior, logz_true = make_problem(d=2, mu=[1.5, -2.0])
    res = nested_sampling(logl, prior, n_live=400,
                          rng=np.random.default_rng(7))
    assert abs(res["logz"] - logz_true) < 4 * res["logz_err"]


def test_posterior_mean_and_width():
    logl, prior, _ = make_problem(d=2, mu=[1.0, -1.0])
    res = nested_sampling(logl, prior, n_live=400,
                          rng=np.random.default_rng(3))
    w = np.exp(res["logwt"])
    mean = np.sum(w[:, None] * res["samples"], axis=0)
    var = np.sum(w[:, None] * (res["samples"] - mean) ** 2, axis=0)
    assert np.allclose(mean, [1.0, -1.0], atol=0.15)
    assert np.allclose(np.sqrt(var), [1.0, 1.0], atol=0.15)


def test_simulated_error_agrees_with_sqrt_h_over_n():
    from nested_sampler import simulate_logz

    logl, prior, logz_true = make_problem(d=2)
    res = nested_sampling(logl, prior, n_live=400,
                          rng=np.random.default_rng(5))
    n = res["n_iter"]
    draws = simulate_logz(res["logl"][:n], res["logl"][n:], 400,
                          n_samples=2000, rng=np.random.default_rng(0))
    assert abs(draws.mean() - res["logz"]) < 0.02
    assert abs(draws.std() - res["logz_err"]) < 0.2 * res["logz_err"]
