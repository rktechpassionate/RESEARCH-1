"""Independent numerical checks of the project's Fourier convention.

The reference integrals here start from the definition of the Fourier-side
profile. They do not reuse the closed-form polynomial in the package.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

from weil_gaussian import (
    autocorrelation,
    autocorrelation_at_zero,
    fourier_profile,
    inverse_profile,
    total_mass,
)


def _reference_fourier(frequency: mp.mpf, center: mp.mpf, width: mp.mpf) -> mp.mpf:
    return (frequency**2 + mp.mpf("0.25")) * (
        mp.exp(-((frequency - center) / width) ** 2 / 2)
        + mp.exp(-((frequency + center) / width) ** 2 / 2)
    )


def _integrate_frequency(integrand, center: mp.mpf) -> mp.mpf:
    """Split at Gaussian peaks so quadrature does not miss a narrow peak."""
    points = [-mp.inf, -center, 0, center, mp.inf] if center else [-mp.inf, 0, mp.inf]
    return mp.quad(integrand, points)


@pytest.mark.parametrize(
    "center,width,position",
    [(0.0, 1.0, 0.0), (0.0, 1.0, 1.3), (2.0, 0.6, 0.9), (3.2, 1.1, 2.0)],
)
def test_inverse_profile_against_fourier_integral(center, width, position):
    with mp.workdps(45):
        a, s, x = (mp.mpf(str(value)) for value in (center, width, position))
        expected = _integrate_frequency(
            lambda xi: _reference_fourier(xi, a, s) * mp.cos(xi * x), a
        ) / (2 * mp.pi)
    assert inverse_profile(position, center, width) == pytest.approx(
        float(expected), rel=2e-12, abs=2e-12
    )


@pytest.mark.parametrize(
    "center,width,position",
    [(0.0, 1.0, 1.25), (0.8, 1.2, 0.5), (2.4, 0.7, 1.1), (4.5, 1.8, 2.0)],
)
def test_autocorrelation_against_fourier_integral(center, width, position):
    with mp.workdps(45):
        a, s, x = (mp.mpf(str(value)) for value in (center, width, position))
        expected = _integrate_frequency(
            lambda xi: _reference_fourier(xi, a, s) ** 2 * mp.cos(xi * x), a
        ) / (2 * mp.pi)
    assert autocorrelation(position, center, width) == pytest.approx(
        float(expected), rel=2e-12, abs=2e-12
    )


@pytest.mark.parametrize("center,width", [(0.0, 1.0), (1.4, 0.8)])
def test_zero_lag_matches_parseval(center, width):
    with mp.workdps(45):
        a, s = mp.mpf(str(center)), mp.mpf(str(width))
        spectral_energy = _integrate_frequency(
            lambda xi: _reference_fourier(xi, a, s) ** 2, a
        ) / (2 * mp.pi)
        time_energy = mp.quad(
            lambda x: inverse_profile(float(x), center, width) ** 2,
            [-mp.inf, 0, mp.inf],
        )
    observed = autocorrelation_at_zero(center, width)
    assert observed == pytest.approx(float(spectral_energy), rel=2e-12, abs=2e-12)
    assert observed == pytest.approx(float(time_energy), rel=2e-12, abs=2e-12)
    assert autocorrelation(0.0, center, width) == pytest.approx(observed)


@pytest.mark.parametrize("center,width", [(0.0, 1.0), (0.8, 1.2), (2.4, 0.7)])
def test_total_mass_is_zero_frequency_square(center, width):
    expected = fourier_profile(0.0, center, width) ** 2
    assert total_mass(center, width) == pytest.approx(expected, rel=1e-14)
    if center == 0:
        assert total_mass(center, width) == pytest.approx(0.25)


@pytest.mark.parametrize("center,width", [(0.0, 1.0), (1.4, 0.8), (3.2, 1.1)])
def test_admissibility_symmetry_and_correlation_bound(center, width):
    assert fourier_profile(0.5j, center, width) == pytest.approx(0j, abs=1e-14)
    assert fourier_profile(-0.5j, center, width) == pytest.approx(0j, abs=1e-14)
    zero_lag = autocorrelation_at_zero(center, width)
    for x in (0.0, 0.2, 0.8, 2.0):
        assert fourier_profile(x, center, width) == pytest.approx(
            fourier_profile(-x, center, width), rel=1e-14
        )
        assert inverse_profile(x, center, width) == pytest.approx(
            inverse_profile(-x, center, width), rel=1e-14, abs=1e-14
        )
        phi = autocorrelation(x, center, width)
        assert phi == pytest.approx(autocorrelation(-x, center, width), abs=1e-13)
        assert abs(phi) <= zero_lag + 1e-12


def test_vector_inputs_match_scalar_evaluations():
    positions = np.array([-2.0, -0.3, 0.0, 0.7, 2.1])
    for function in (fourier_profile, inverse_profile, autocorrelation):
        vector = function(positions, 1.2, 0.9)
        assert isinstance(vector, np.ndarray)
        assert vector.shape == positions.shape
        np.testing.assert_allclose(
            vector, [function(float(x), 1.2, 0.9) for x in positions], rtol=1e-13
        )


@pytest.mark.parametrize("center,width", [(-0.1, 1.0), (0.0, 0.0), (0.0, -1.0), (math.nan, 1.0), (0.0, math.inf)])
def test_invalid_parameters_are_rejected(center, width):
    for function in (fourier_profile, inverse_profile, autocorrelation):
        with pytest.raises(ValueError):
            function(0.0, center, width)
    for function in (total_mass, autocorrelation_at_zero):
        with pytest.raises(ValueError):
            function(center, width)
