"""Supernova cosmology for Stage 3: distances and the Pantheon+ likelihood.

The data.  Each Type Ia supernova gives a standardised apparent magnitude
m_i at redshift z_i. If all supernovae have the same absolute magnitude M,

    m_i = mu(z_i) + M,        mu = 5 log10(d_L / 10 pc),

where mu is the distance modulus and d_L the luminosity distance.

The model.  For a flat universe containing matter (density Omega_m) and
dark energy with a constant equation of state w,

    E(z)^2 = [H(z)/H0]^2 = Omega_m (1+z)^3 + (1 - Omega_m) (1+z)^(3(1+w)).

LambdaCDM is the special case w = -1, where the dark-energy term is
constant. Radiation is ignored: it matters only at much higher redshift.
The luminosity distance is

    d_L = (1 + z_hel) (c / H0) * integral from 0 to z_HD of dz' / E(z').

z_HD is the redshift corrected for our motion and for peculiar velocities
("Hubble diagram" redshift); it sets the cosmological distance. z_hel is
the redshift actually observed from the Sun, which is what stretches and
dims the light, so it goes in the (1 + z) factor.

H0 and M cannot be separated.  Changing H0 shifts every mu by the same
constant, exactly as changing M does. Supernovae alone measure only the
combination M - 5 log10(H0). So H0 is fixed at 70 km/s/Mpc here and M is
a free parameter. A different H0 would shift M and change nothing else.

The likelihood.  The measurement errors are correlated (shared
calibration, shared systematics), so the likelihood is a multivariate
Gaussian with the published covariance matrix C (statistical + systematic):

    ln L = -1/2 r^T C^-1 r - 1/2 ln det(2 pi C),     r_i = m_i - mu(z_i) - M.

Data: Pantheon+ (Scolnic et al. 2022; Brout et al. 2022), from
github.com/PantheonPlusSH0ES/DataRelease.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

C_KM_S = 299792.458     # speed of light in km/s
H0 = 70.0               # km/s/Mpc, fixed (see the note above)

# Gauss-Legendre quadrature: nodes u_k and weights g_k on [0, 1], so that
# the integral of f(u) from 0 to 1 is approximately sum_k g_k f(u_k).
# 20 nodes give mu to better than 1e-9 mag over the whole prior range.
_nodes, _weights = np.polynomial.legendre.leggauss(20)
GL_NODES = jnp.asarray(0.5 * (_nodes + 1.0))
GL_WEIGHTS = jnp.asarray(0.5 * _weights)


def make_distance_modulus(z_hd, z_hel):
    """Return a function mu(omega_m, w) for a fixed set of supernovae.

    The integral over z' from 0 to z is done by substituting z' = z u, so
    that every supernova uses the same nodes u_k on [0, 1]:

        integral_0^z dz'/E(z') = z * sum_k g_k / E(z u_k).

    The redshifts never change, so ln(1 + z u_k) is computed once here.
    Inside the returned function the powers of (1 + z) are then written as
    (1+z)^p = exp(p ln(1+z)), which is several times faster than
    recomputing a general power for every likelihood call.
    """
    z_hd = jnp.asarray(z_hd)
    z_hel = jnp.asarray(z_hel)
    log1pz = jnp.log1p(z_hd[:, None] * GL_NODES[None, :])   # shape (n_sn, 20)
    matter = jnp.exp(3.0 * log1pz)                          # (1+z)^3

    def distance_modulus(omega_m, w=-1.0):
        # E(z)^2 = Omega_m (1+z)^3 + (1 - Omega_m) (1+z)^(3(1+w))
        e2 = omega_m * matter + (1 - omega_m) * jnp.exp(3.0 * (1 + w) * log1pz)
        integral = z_hd * jnp.sum(GL_WEIGHTS / jnp.sqrt(e2), axis=1)
        d_lum_mpc = (1 + z_hel) * (C_KM_S / H0) * integral
        # 10 pc = 1e-5 Mpc, so mu = 5 log10(d_L / Mpc) + 25.
        return 5.0 * jnp.log10(d_lum_mpc) + 25.0

    return distance_modulus


def load_pantheon_plus(data_dir="data", z_min=0.01):
    """Load Pantheon+ magnitudes, redshifts and the covariance matrix.

    Supernovae with z_HD <= 0.01 are removed, as in the Pantheon+ cosmology
    analysis: at such low redshift the peculiar velocity of the host galaxy
    is a large fraction of its recession velocity. This leaves 1590 of the
    1701 light curves.
    """
    data_dir = Path(data_dir)
    table = np.genfromtxt(data_dir / "Pantheon+SH0ES.dat", names=True,
                          dtype=None, encoding="utf-8")
    raw = np.loadtxt(data_dir / "Pantheon+SH0ES_STAT+SYS.cov")
    n_all = int(raw[0])                      # first line is the matrix size
    cov = raw[1:].reshape(n_all, n_all)
    cov = 0.5 * (cov + cov.T)                # remove rounding asymmetry

    keep = table["zHD"] > z_min
    cov = cov[np.ix_(keep, keep)]
    return {
        "z_hd": np.asarray(table["zHD"][keep], dtype=float),
        "z_hel": np.asarray(table["zHEL"][keep], dtype=float),
        "m_b": np.asarray(table["m_b_corr"][keep], dtype=float),
        "cov": cov,
    }


def make_loglikelihood(data):
    """Return ln L(omega_m, w, M) for the Pantheon+ data.

    C^-1 and ln det(2 pi C) do not depend on the parameters, so they are
    computed once here.
    """
    distance_modulus = make_distance_modulus(data["z_hd"], data["z_hel"])
    m_b = jnp.asarray(data["m_b"])
    cov_inv = jnp.asarray(np.linalg.inv(data["cov"]))
    n = len(data["m_b"])
    log_norm = -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(data["cov"])[1])

    def loglikelihood(omega_m, w, abs_mag):
        r = m_b - distance_modulus(omega_m, w) - abs_mag
        return -0.5 * r @ (cov_inv @ r) + log_norm

    return loglikelihood


def make_marginal_loglikelihood(data, m_low, m_high):
    """Return ln L(omega_m, w) with the absolute magnitude M integrated out.

    Used only as an independent check on the nested sampling result.

    Write r = y - M 1, with y_i = m_i - mu(z_i) and 1 a vector of ones:

        r^T C^-1 r = a - 2 b M + c M^2,
        a = y^T C^-1 y,   b = 1^T C^-1 y,   c = 1^T C^-1 1.

    Completing the square,

        r^T C^-1 r = (a - b^2/c) + c (M - b/c)^2,

    so as a function of M the likelihood is a Gaussian centred on
    M_hat = b/c with standard deviation s = 1/sqrt(c). Integrating it over
    the uniform prior on M, which has density 1/(m_high - m_low), gives

        L_marg = norm * exp(-(a - b^2/c)/2) * sqrt(2 pi / c) / (m_high - m_low)
                 * [Phi((m_high - M_hat)/s) - Phi((m_low - M_hat)/s)].

    The last factor is the fraction of that Gaussian that lies inside the
    prior range. It is 1 wherever the fit is good, and keeping it makes
    this exactly the same integral that nested sampling estimates.
    """
    distance_modulus = make_distance_modulus(data["z_hd"], data["z_hel"])
    m_b = jnp.asarray(data["m_b"])
    cov_inv = jnp.asarray(np.linalg.inv(data["cov"]))
    ones = jnp.ones(len(data["m_b"]))
    cinv_ones = cov_inv @ ones
    c = ones @ cinv_ones
    s = 1.0 / jnp.sqrt(c)
    n = len(data["m_b"])
    log_norm = -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(data["cov"])[1])

    def marginal_loglikelihood(omega_m, w):
        y = m_b - distance_modulus(omega_m, w)
        a = y @ (cov_inv @ y)
        b = cinv_ones @ y
        m_hat = b / c
        # ln[Phi(hi) - Phi(lo)], computed stably even when it is tiny.
        log_fraction = jax.scipy.special.logsumexp(
            jnp.array([jax.scipy.stats.norm.logcdf((m_high - m_hat) / s),
                       jax.scipy.stats.norm.logcdf((m_low - m_hat) / s)]),
            b=jnp.array([1.0, -1.0]))
        return (-0.5 * (a - b**2 / c) + 0.5 * jnp.log(2 * jnp.pi / c)
                - jnp.log(m_high - m_low) + log_fraction + log_norm)

    return marginal_loglikelihood
