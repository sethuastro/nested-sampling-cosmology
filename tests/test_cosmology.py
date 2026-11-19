import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad, simpson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

import cosmology

HAVE_DATA = (ROOT / "data" / "Pantheon+SH0ES_STAT+SYS.cov").exists()
needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason="run python data/download_pantheon.py first")


def mu_reference(z_hd, z_hel, omega_m, w):
    """Distance modulus by adaptive quadrature, independent of cosmology.py."""
    def integrand(z):
        return 1 / np.sqrt(omega_m * (1 + z) ** 3
                           + (1 - omega_m) * (1 + z) ** (3 * (1 + w)))
    integral = quad(integrand, 0, z_hd, epsabs=1e-13, epsrel=1e-13)[0]
    return 5 * np.log10((1 + z_hel) * cosmology.C_KM_S / cosmology.H0 * integral) + 25


@pytest.mark.parametrize("omega_m, w", [(0.3, -1.0), (0.01, -3.0), (0.99, 1.0),
                                        (0.5, -0.5), (0.01, 1.0)])
def test_distance_modulus_matches_quadrature(omega_m, w):
    z = np.array([0.01, 0.1, 0.5, 1.0, 2.26])
    z_hel = z + 0.001
    ours = np.asarray(cosmology.make_distance_modulus(z, z_hel)(omega_m, w))
    ref = np.array([mu_reference(a, b, omega_m, w) for a, b in zip(z, z_hel)])
    assert np.allclose(ours, ref, atol=1e-9)


def test_einstein_de_sitter_distance():
    # For Omega_m = 1 the integral is exact: 2 [1 - (1+z)^(-1/2)].
    z = jnp.array([0.5, 1.0, 2.0])
    mu = np.asarray(cosmology.make_distance_modulus(z, z)(1.0, -1.0))
    d_lum = (1 + z) * cosmology.C_KM_S / cosmology.H0 * 2 * (1 - (1 + z) ** -0.5)
    assert np.allclose(mu, 5 * np.log10(np.asarray(d_lum)) + 25, atol=1e-9)


@needs_data
def test_selection_keeps_1590_supernovae():
    data = cosmology.load_pantheon_plus(ROOT / "data")
    assert len(data["m_b"]) == 1590
    assert data["cov"].shape == (1590, 1590)
    assert data["z_hd"].min() > 0.01


@needs_data
def test_loglikelihood_matches_numpy():
    data = cosmology.load_pantheon_plus(ROOT / "data")
    loglike = cosmology.make_loglikelihood(data)
    omega_m, w, abs_mag = 0.31, -0.95, -19.33
    mu = np.array([mu_reference(a, b, omega_m, w)
                   for a, b in zip(data["z_hd"], data["z_hel"])])
    r = data["m_b"] - mu - abs_mag
    expected = (-0.5 * r @ np.linalg.solve(data["cov"], r)
                - 0.5 * (len(r) * np.log(2 * np.pi)
                         + np.linalg.slogdet(data["cov"])[1]))
    assert np.isclose(float(loglike(omega_m, w, abs_mag)), expected, atol=1e-6)


@needs_data
def test_marginalising_over_m_matches_brute_force():
    data = cosmology.load_pantheon_plus(ROOT / "data")
    loglike = cosmology.make_loglikelihood(data)
    marginal = cosmology.make_marginal_loglikelihood(data, -20.0, -19.0)
    m_grid = np.linspace(-20.0, -19.0, 20001)
    for omega_m, w in [(0.33, -1.0), (0.29, -0.9), (0.9, 0.5)]:
        logl = np.asarray(jax.vmap(lambda m: loglike(omega_m, w, m))(jnp.asarray(m_grid)))
        brute = logl.max() + np.log(simpson(np.exp(logl - logl.max()), x=m_grid))
        assert np.isclose(float(marginal(omega_m, w)), brute, atol=1e-6)
