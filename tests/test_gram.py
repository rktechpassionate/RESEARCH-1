"""Independent checks for L2 Gram and finite Weil diagnostics."""

from __future__ import annotations

import math
import warnings

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import IntegrationWarning

from weil_gaussian.families import PolynomialGaussianFamily
from weil_gaussian.balance import balance_diagnostic
from weil_gaussian.gram import (
    cross_autocorrelation,
    finite_weil_diagnostics,
    finite_weil_matrix,
    grid_gram_diagnostics,
    l2_gram_diagnostics,
    l2_gram_matrix,
)


def _mp_fourier(family: PolynomialGaussianFamily, frequency: mp.mpf) -> mp.mpf:
    """Reference product from the requested zeros, not package coefficients."""
    polynomial = mp.mpf("1")
    for root in family.zeros:
        z = mp.mpc(str(root.real), str(root.imag))
        square = z * z
        if mp.im(square) == 0:
            polynomial *= frequency * frequency - mp.re(square)
        else:
            polynomial *= (frequency * frequency - square) * (
                frequency * frequency - mp.conj(square)
            )
    a, s = mp.mpf(str(family.center)), mp.mpf(str(family.width))
    gaussian = mp.exp(-((frequency - a) / s) ** 2 / 2) + mp.exp(
        -((frequency + a) / s) ** 2 / 2
    )
    return mp.re(polynomial * gaussian)


def _mp_cross(left, right, position):
    x = mp.mpf(str(position))
    points = sorted(
        {mp.mpf("0"), mp.mpf(str(left.center)), mp.mpf(str(right.center))}
    ) + [mp.inf]
    return mp.quad(
        lambda t: _mp_fourier(left, t)
        * _mp_fourier(right, t)
        * mp.cos(t * x),
        points,
    ) / mp.pi


@pytest.mark.parametrize(
    "center,width,position",
    [(0.0, 1.0, 0.0), (0.8, 0.7, 0.4), (2.0, 1.2, 1.1), (3.0, 0.6, 2.0)],
)
def test_cross_diagonal_matches_family_autocorrelation(center, width, position):
    family = PolynomialGaussianFamily(center, width)
    assert cross_autocorrelation(family, family, position) == pytest.approx(
        family.autocorrelation(position), rel=2e-12, abs=2e-12
    )


@pytest.mark.parametrize("position", [0.0, 0.2, 1.0, -1.4])
def test_cross_pairing_is_symmetric_and_even(position):
    left = PolynomialGaussianFamily(0.8, 0.7, (0.5j, 0.3 + 0.4j))
    right = PolynomialGaussianFamily(1.4, 1.1, (0.5j, 0.9j))
    value = cross_autocorrelation(left, right, position)
    assert value == pytest.approx(
        cross_autocorrelation(right, left, position), rel=2e-12, abs=2e-12
    )
    assert value == pytest.approx(
        cross_autocorrelation(left, right, -position), rel=2e-12, abs=2e-12
    )


@pytest.mark.parametrize(
    "left,right,position",
    [
        (PolynomialGaussianFamily(0.0, 1.0), PolynomialGaussianFamily(0.7, 0.9), 0.0),
        (
            PolynomialGaussianFamily(1.1, 0.8, (0.5j, 0.6j)),
            PolynomialGaussianFamily(1.4, 1.2, (0.5j, 0.3 + 0.2j)),
            0.5,
        ),
        (
            PolynomialGaussianFamily(2.0, 0.7, (0.3 + 0.4j,)),
            PolynomialGaussianFamily(1.5, 1.0, (0.5j,)),
            1.2,
        ),
    ],
)
def test_cross_pairing_against_independent_fourier_integral(left, right, position):
    with mp.workdps(35):
        reference = _mp_cross(left, right, position)
    assert cross_autocorrelation(left, right, position) == pytest.approx(
        float(reference), rel=5e-11, abs=5e-11
    )


def test_gram_matrix_is_symmetric_and_positive_semidefinite():
    families = (
        PolynomialGaussianFamily(0.0, 1.0),
        PolynomialGaussianFamily(0.9, 0.8),
        PolynomialGaussianFamily(1.8, 1.2, (0.5j, 0.6j)),
        PolynomialGaussianFamily(2.4, 0.7, (0.3 + 0.4j,)),
    )
    matrix = l2_gram_matrix(families)
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    assert matrix.shape == (4, 4)
    assert not matrix.flags.writeable


def test_gram_diagonal_matches_independent_parseval_integral():
    families = (
        PolynomialGaussianFamily(0.0, 1.0),
        PolynomialGaussianFamily(1.4, 0.8, (0.5j, 0.7j)),
    )
    matrix = l2_gram_matrix(families)
    with mp.workdps(35):
        for index, family in enumerate(families):
            reference = _mp_cross(family, family, 0.0)
            assert matrix[index, index] == pytest.approx(
                float(reference), rel=5e-11, abs=5e-11
            )


def test_grid_order_and_diagnostics():
    result = grid_gram_diagnostics([0.0, 1.0], [0.7, 1.2], zeros=(0.5j,))
    assert [(f.center, f.width) for f in result.families] == [
        (0.0, 0.7), (0.0, 1.2), (1.0, 0.7), (1.0, 1.2)
    ]
    assert result.matrix.shape == (4, 4)
    assert result.eigenvalues.shape == (4,)
    assert result.is_positive_semidefinite
    assert result.effective_rank == 4
    assert math.isfinite(result.condition_number)
    assert not result.eigenvalues.flags.writeable


def test_duplicate_families_have_rank_one_and_infinite_condition_number():
    family = PolynomialGaussianFamily(1.0, 0.8)
    result = l2_gram_diagnostics((family, family))
    assert result.effective_rank == 1
    assert math.isinf(result.condition_number)
    assert result.eigenvalues[0] == pytest.approx(0.0, abs=1e-12)


def test_near_dependent_grid_reports_poor_conditioning():
    left = PolynomialGaussianFamily(1.0, 0.8)
    right = PolynomialGaussianFamily(1.0 + 1e-7, 0.8)
    result = l2_gram_diagnostics((left, right))
    assert result.effective_rank == 1
    assert math.isinf(result.condition_number)


def test_tiny_full_rank_gram_uses_relative_spectral_scale():
    family = PolynomialGaussianFamily(0.0, 1e-12, ())
    result = l2_gram_diagnostics((family,))
    assert 0 < result.matrix[0, 0] < 1e-10
    assert result.effective_rank == 1
    assert result.condition_number == pytest.approx(1.0)
    assert result.is_positive_semidefinite


@pytest.mark.parametrize(
    "families",
    [(), [1, 2], [PolynomialGaussianFamily(0.0, 1.0), "bad"]],
)
def test_gram_rejects_empty_or_wrong_family_entries(families):
    with pytest.raises((TypeError, ValueError)):
        l2_gram_matrix(families)


@pytest.mark.parametrize("position", [math.nan, math.inf, "bad"])
def test_cross_pairing_rejects_invalid_position(position):
    family = PolynomialGaussianFamily(0.0, 1.0)
    with pytest.raises(ValueError):
        cross_autocorrelation(family, family, position)


@pytest.mark.parametrize("tolerance", [0, -1e-4, 1.0, math.nan, math.inf])
def test_diagnostics_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError):
        l2_gram_diagnostics((PolynomialGaussianFamily(0.0, 1.0),), tolerance)


@pytest.mark.parametrize(
    "centers,widths",
    [([], [1.0]), ([0.0], []), ([-1.0], [1.0]), ([0.0], [0.0])],
)
def test_grid_rejects_empty_or_invalid_parameters(centers, widths):
    with pytest.raises(ValueError):
        grid_gram_diagnostics(centers, widths)


def test_finite_weil_rejects_nonadmissible_family_even_when_other_is_admissible():
    admissible = PolynomialGaussianFamily(0.4, 0.8, (0.5j,))
    not_admissible = PolynomialGaussianFamily(0.6, 0.9, (0.7j,))
    with pytest.raises(ValueError, match="i/2"):
        finite_weil_matrix((admissible, not_admissible), 8)


@pytest.mark.parametrize("bad_cutoff", [1, 2.5, True])
def test_finite_weil_requires_integer_cutoff(bad_cutoff):
    with pytest.raises((TypeError, ValueError)):
        finite_weil_matrix((PolynomialGaussianFamily(0.0, 1.0),), bad_cutoff)


def test_finite_weil_matrices_are_symmetric_and_decompose_exactly():
    families = (PolynomialGaussianFamily(0.4, 0.8), PolynomialGaussianFamily(1.2, 0.9))
    result = finite_weil_diagnostics(families, 8)
    for matrix in (
        result.archimedean_matrix,
        result.finite_prime_matrix,
        result.balance_matrix,
        result.integration_error_estimate,
    ):
        np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
        assert not matrix.flags.writeable
    np.testing.assert_allclose(
        result.balance_matrix,
        result.archimedean_matrix - result.finite_prime_matrix,
        rtol=0,
        atol=0,
    )
    assert result.cutoff == 8
    assert result.eigenvalues.shape == (2,)


def test_finite_weil_prime_cross_term_against_independent_factorization_and_fourier():
    left = PolynomialGaussianFamily(0.2, 0.8)
    right = PolynomialGaussianFamily(0.7, 0.9)
    result = finite_weil_diagnostics((left, right), 5)
    with mp.workdps(35):
        reference = mp.fsum(
            2 * mp.log(prime) / mp.sqrt(value) * _mp_cross(left, right, mp.log(value))
            for value, prime in ((2, 2), (3, 3), (4, 2), (5, 5))
        )
    assert result.finite_prime_matrix[0, 1] == pytest.approx(
        float(reference), rel=5e-11, abs=5e-11
    )


def test_finite_weil_archimedean_diagonal_against_independent_quadrature():
    family = PolynomialGaussianFamily(0.6, 0.8)
    result = finite_weil_diagnostics((family,), 3)
    with mp.workdps(28):
        integral = mp.quad(
            lambda t: mp.re(mp.digamma(mp.mpf("0.25") + mp.mpc(0, 0.5) * t))
            * _mp_fourier(family, t) ** 2,
            [0, mp.mpf("0.6"), mp.mpf("12")],
        ) / mp.pi
        reference = integral - _mp_cross(family, family, 0) * mp.log(mp.pi)
    assert result.archimedean_matrix[0, 0] == pytest.approx(
        float(reference), rel=5e-9, abs=5e-9
    )


def test_finite_weil_rejects_nonconverged_archimedean_quadrature(monkeypatch):
    def warning_quad(*_args, **_kwargs):
        warnings.warn("forced nonconvergence", IntegrationWarning)
        return 0.0, 0.0

    monkeypatch.setattr("weil_gaussian.gram.quad", warning_quad)
    with pytest.raises(FloatingPointError, match="did not converge"):
        finite_weil_matrix((PolynomialGaussianFamily(0.0, 1.0),), 2)


@pytest.mark.parametrize("error", [1.0, math.nan])
def test_finite_weil_rejects_invalid_archimedean_error_estimate(monkeypatch, error):
    monkeypatch.setattr("weil_gaussian.gram.quad", lambda *_args, **_kwargs: (0.0, error))
    with pytest.raises(FloatingPointError):
        finite_weil_matrix((PolynomialGaussianFamily(0.0, 1.0),), 2)


@pytest.mark.parametrize("center,width,cutoff", [(0.0, 1.0, 20), (1.0, 0.8, 20), (3.0, 1.2, 50)])
def test_finite_weil_diagonal_matches_scalar_balance(center, width, cutoff):
    family = PolynomialGaussianFamily(center, width)
    matrix = finite_weil_diagnostics((family,), cutoff)
    scalar = balance_diagnostic(family, cutoff)
    assert matrix.archimedean_matrix[0, 0] == pytest.approx(
        scalar.archimedean.value, rel=1e-11, abs=1e-11
    )
    assert matrix.finite_prime_matrix[0, 0] == pytest.approx(
        scalar.finite_prime_term, rel=1e-11, abs=1e-11
    )
    assert matrix.balance_matrix[0, 0] == pytest.approx(
        scalar.finite_balance, rel=1e-11, abs=1e-11
    )
    assert matrix.prime_tail_bound is None
