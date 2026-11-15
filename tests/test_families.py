"""Independent checks of general polynomial-Gaussian families."""

from dataclasses import FrozenInstanceError
import math

import mpmath as mp
import numpy as np
import pytest

from weil_gaussian.core import autocorrelation, fourier_profile, inverse_profile
from weil_gaussian.families import PolynomialGaussianFamily


def _reference_spectrum(t, center, width, zeros):
    """Construct the defining zero polynomial with mpmath, not NumPy."""
    y = t * t
    polynomial = mp.mpf(1)
    for value in zeros:
        z = mp.mpc(str(value.real), str(value.imag))
        root = z * z
        if value.real == 0.0 or value.imag == 0.0:
            polynomial *= y - mp.re(root)
        else:
            polynomial *= (y - root) * (y - mp.conj(root))
    return polynomial * (
        mp.exp(-((t - center) / width) ** 2 / 2)
        + mp.exp(-((t + center) / width) ** 2 / 2)
    )


def _reference_transform(position, center, width, zeros, squared):
    with mp.workdps(40):
        x = mp.mpf(str(position))
        a = mp.mpf(str(center))
        s = mp.mpf(str(width))
        intervals = [0, mp.inf] if a == 0 else [0, a, mp.inf]

        def integrand(t):
            value = _reference_spectrum(t, a, s, zeros)
            return (value * value if squared else value) * mp.cos(t * x)

        # Even real spectrum: integral over R / (2*pi) = integral_0^inf / pi.
        return float(mp.re(mp.quad(integrand, intervals) / mp.pi))


@pytest.mark.parametrize(
    "zeros,expected",
    [
        ((), (1.0,)),
        ((0.5j,), (0.25, 1.0)),
        ((0.5j, 1.5j), (0.5625, 2.5, 1.0)),
        ((1 + 2j,), (25.0, 6.0, 1.0)),
        ((0.5j, 0.5j), (0.0625, 0.5, 1.0)),
    ],
)
def test_zero_polynomial_coefficients(zeros, expected):
    family = PolynomialGaussianFamily(1.2, 0.9, zeros)
    assert family.coefficients == pytest.approx(expected)


@pytest.mark.parametrize(
    "zero",
    [0.5j, -0.5j, 1.5j, -2.0j, 1.0 + 2.0j, -1.0 + 2.0j, 2.0, -3.0],
)
def test_prescribed_zero_and_symmetry_closure(zero):
    family = PolynomialGaussianFamily(0.8, 1.1, (zero,))
    for partner in (zero, -zero, zero.conjugate(), -zero.conjugate()):
        assert abs(family.fourier(partner)) < 1e-10


def test_exact_complex_root_short_circuits_gaussian_overflow():
    family = PolynomialGaussianFamily(0.0, 1.0, (100j,))
    with np.errstate(over="raise", invalid="raise"):
        assert family.fourier(100j) == 0j
        assert family.fourier(-100j) == 0j


def test_mixed_array_short_circuits_only_prescribed_roots():
    family = PolynomialGaussianFamily(0.0, 1.0, (100j,))
    frequencies = np.array([100j, 0j, -100j, 1j])
    with np.errstate(over="raise", invalid="raise"):
        observed = family.fourier(frequencies)
    assert observed.shape == frequencies.shape
    assert observed[0] == 0j
    assert observed[2] == 0j
    assert observed[1] == pytest.approx(20_000.0)
    assert observed[3] != 0j
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(FloatingPointError):
            family.fourier(101j)


@pytest.mark.parametrize(
    "zeros,admissible",
    [
        ((), False),
        ((0.5j,), True),
        ((-0.5j, 1 + 2j), True),
        ((0.5000000000000001j,), False),
        ((1 + 2j,), False),
    ],
)
def test_weil_admissibility_uses_root_construction(zeros, admissible):
    family = PolynomialGaussianFamily(0.8, 1.1, zeros)
    assert family.is_weil_admissible is admissible


@pytest.mark.parametrize("center,width", [(0.0, 1.0), (1.5, 0.7), (3.0, 2.0)])
def test_default_family_matches_original_functions(center, width):
    family = PolynomialGaussianFamily(center, width)
    grid = np.array([-2.0, -0.4, 0.0, 0.9, 2.3])
    np.testing.assert_allclose(
        family.fourier(grid), fourier_profile(grid, center, width)
    )
    np.testing.assert_allclose(
        family.inverse(grid), inverse_profile(grid, center, width)
    )
    np.testing.assert_allclose(
        family.autocorrelation(grid), autocorrelation(grid, center, width),
        rtol=2e-13, atol=2e-13,
    )


@pytest.mark.parametrize(
    "center,width,position",
    [(0.0, 1.0, 0.8), (2.0, 0.7, 0.3), (1.2, 1.5, 1.8)],
)
def test_zero_free_family_has_elementary_gaussian_forms(center, width, position):
    family = PolynomialGaussianFamily(center, width, ())
    x, a, s = position, center, width
    expected_inverse = (
        math.sqrt(2 / math.pi) * s * math.exp(-s * s * x * x / 2) * math.cos(a * x)
    )
    expected_correlation = (
        s / math.sqrt(math.pi) * math.exp(-s * s * x * x / 4)
        * (math.cos(a * x) + math.exp(-a * a / (s * s)))
    )
    assert family.inverse(x) == pytest.approx(expected_inverse, rel=1e-13)
    assert family.autocorrelation(x) == pytest.approx(expected_correlation, rel=1e-13)


@pytest.mark.parametrize(
    "center,width,position,zeros",
    [
        (0.0, 1.0, 0.7, (0.5j, 1.5j)),
        (1.2, 0.9, 0.6, (0.5j, 1.0 + 2.0j)),
        (2.3, 1.1, 1.4, (1.0 + 0.5j,)),
    ],
)
def test_inverse_matches_independent_high_precision_quadrature(
    center, width, position, zeros
):
    family = PolynomialGaussianFamily(center, width, zeros)
    expected = _reference_transform(position, center, width, zeros, squared=False)
    assert family.inverse(position) == pytest.approx(expected, rel=3e-11, abs=1e-9)


@pytest.mark.parametrize(
    "center,width,position,zeros",
    [
        (0.0, 1.0, 0.7, (0.5j, 1.5j)),
        (1.2, 0.9, 0.6, (0.5j, 1.0 + 2.0j)),
        (2.3, 1.1, 1.4, (1.0 + 0.5j,)),
    ],
)
def test_autocorrelation_matches_independent_high_precision_quadrature(
    center, width, position, zeros
):
    family = PolynomialGaussianFamily(center, width, zeros)
    expected = _reference_transform(position, center, width, zeros, squared=True)
    assert family.autocorrelation(position) == pytest.approx(
        expected, rel=3e-11, abs=1e-8
    )


@pytest.mark.parametrize(
    "center,width,zeros", [(0.0, 1.0, ()), (1.2, 0.9, (0.5j, 1 + 2j))]
)
def test_real_even_profiles_and_correlation_bound(center, width, zeros):
    family = PolynomialGaussianFamily(center, width, zeros)
    assert family.autocorrelation(0.0) > 0
    for x in (0.0, 0.3, 0.9, 2.0):
        assert family.fourier(x) == pytest.approx(family.fourier(-x), rel=1e-12)
        assert family.inverse(x) == pytest.approx(family.inverse(-x), rel=1e-12)
        assert family.autocorrelation(x) == pytest.approx(
            family.autocorrelation(-x), rel=1e-11, abs=1e-10
        )
        assert abs(family.autocorrelation(x)) <= family.autocorrelation(0.0) + 1e-8


def test_vector_inputs_match_scalar_values_and_preserve_shape():
    family = PolynomialGaussianFamily(1.2, 0.9, (0.5j, 1 + 2j))
    grid = np.array([[-1.0, 0.0, 0.5], [1.2, 2.0, 3.1]])
    for method in (family.fourier, family.inverse, family.autocorrelation):
        observed = method(grid)
        expected = np.array([[method(float(x)) for x in row] for row in grid])
        assert observed.shape == grid.shape
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_complex_fourier_evenness_and_conjugation():
    family = PolynomialGaussianFamily(1.0, 0.8, (0.5j, 1 + 2j))
    for frequency in (0.3 + 0.7j, -1.2 + 0.4j, 2.0 - 0.2j):
        value = family.fourier(frequency)
        assert family.fourier(-frequency) == pytest.approx(value, rel=2e-12)
        assert family.fourier(frequency.conjugate()) == pytest.approx(
            value.conjugate(), rel=2e-12
        )


def test_autocorrelation_mass_matches_zero_frequency_square():
    family = PolynomialGaussianFamily(1.0, 1.0, (0.5j, 1.5j))
    with mp.workdps(25):
        observed = mp.quad(
            lambda x: family.autocorrelation(float(x)), [-20, 0, 20]
        )
    assert float(observed) == pytest.approx(family.fourier(0.0) ** 2, rel=2e-10)


@pytest.mark.parametrize(
    "center,width,zeros",
    [
        (-1.0, 1.0, (0.5j,)),
        (1.0, 0.0, (0.5j,)),
        (math.nan, 1.0, (0.5j,)),
        (1.0, 1.0, (complex(math.inf, 0),)),
        (1.0, 1.0, None),
        (1.0, 1.0, 0.5j),
    ],
)
def test_invalid_parameters_and_zero_sequences_rejected(center, width, zeros):
    with pytest.raises((ValueError, TypeError)):
        PolynomialGaussianFamily(center, width, zeros)


def test_invalid_evaluation_inputs_rejected():
    family = PolynomialGaussianFamily(1.0, 1.0, (0.5j, 1 + 2j))
    for method in (family.fourier, family.inverse, family.autocorrelation):
        with pytest.raises(ValueError):
            method(float("nan"))
    for method in (family.inverse, family.autocorrelation):
        with pytest.raises(ValueError):
            method(1 + 1j)


def test_frozen_family_rejects_mutation():
    family = PolynomialGaussianFamily(1.0, 1.0, (0.5j,))
    with pytest.raises(FrozenInstanceError):
        family.center = 3.0
