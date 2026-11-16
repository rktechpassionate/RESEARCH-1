"""Variational checks for represented finite-cutoff Rayleigh--Ritz problems."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weil_gaussian.extremal import ExtremalResult, solve_extremal_rayleigh_ritz
from weil_gaussian.families import PolynomialGaussianFamily
from weil_gaussian.gram import FiniteWeilDiagnostics, finite_weil_diagnostics


@pytest.fixture(scope="module")
def basis() -> tuple[PolynomialGaussianFamily, ...]:
    return tuple(
        PolynomialGaussianFamily(0.4, 0.8, zeros)
        for zeros in (
            (0.5j,),
            (0.5j, 1.0),
            (0.5j, 1.0, 2.0),
            (0.5j, 1.0, 2.0, 3.0),
        )
    )


@pytest.fixture(scope="module")
def diagnostic(
    basis: tuple[PolynomialGaussianFamily, ...],
) -> FiniteWeilDiagnostics:
    return finite_weil_diagnostics(basis, 20)


@pytest.fixture(scope="module")
def free(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> ExtremalResult:
    return solve_extremal_rayleigh_ritz(basis, 20, finite_diagnostic=diagnostic)


@pytest.fixture(scope="module")
def constrained(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> ExtremalResult:
    return solve_extremal_rayleigh_ritz(
        basis, 20, zeros=(0.7 + 0.3j,), finite_diagnostic=diagnostic
    )


@pytest.fixture(scope="module")
def complex_constrained(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> ExtremalResult:
    return solve_extremal_rayleigh_ritz(
        basis,
        20,
        zeros=(0.7 + 0.3j,),
        coefficient_field="complex",
        finite_diagnostic=diagnostic,
    )


def _quotient(result: ExtremalResult, coefficients: np.ndarray) -> float:
    numerator = np.vdot(coefficients, result.weil_matrix @ coefficients)
    denominator = np.vdot(coefficients, result.gram_matrix @ coefficients)
    return float(np.real(numerator / denominator))


def test_pairing_matrices_are_real_symmetric_and_finite(free: ExtremalResult) -> None:
    for matrix in (free.gram_matrix, free.weil_matrix):
        assert matrix.shape == (4, 4)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(matrix, matrix.T)
        assert not matrix.flags.writeable
    assert np.linalg.eigvalsh(free.gram_matrix)[0] > 0


def test_unconstrained_subspace_is_whole_basis(free: ExtremalResult) -> None:
    assert free.constraint_rank == 0
    assert free.subspace_dimension == 4
    assert free.constraint_matrix.shape == (0, 4)
    np.testing.assert_array_equal(free.nullspace_basis, np.eye(4))


@pytest.mark.parametrize("endpoint", ("minimum", "maximum"))
def test_unconstrained_eigenpairs_obey_full_generalized_equation(
    free: ExtremalResult, endpoint: str
) -> None:
    pair = getattr(free, endpoint)
    residual = free.weil_matrix @ pair.coefficients - pair.eigenvalue * (
        free.gram_matrix @ pair.coefficients
    )
    assert np.linalg.norm(residual) < 1e-12
    assert pair.full_residual_norm == pytest.approx(np.linalg.norm(residual))
    assert pair.projected_residual_norm == pytest.approx(pair.full_residual_norm)


@pytest.mark.parametrize("endpoint", ("minimum", "maximum"))
def test_extremal_vector_is_g_normalized_and_matches_quotient(
    free: ExtremalResult, endpoint: str
) -> None:
    pair = getattr(free, endpoint)
    assert pair.metric_norm == pytest.approx(1.0, abs=2e-13)
    assert pair.rayleigh_quotient == pytest.approx(pair.eigenvalue, abs=1e-12)
    assert pair.rayleigh_quotient == pytest.approx(
        _quotient(free, pair.coefficients), abs=1e-12
    )
    assert not pair.coefficients.flags.writeable


def test_extreme_vectors_are_g_orthogonal(free: ExtremalResult) -> None:
    overlap = free.minimum.coefficients @ free.gram_matrix @ free.maximum.coefficients
    assert overlap == pytest.approx(0.0, abs=1e-12)


def test_unconstrained_bounds_cover_independent_trial_vectors(
    free: ExtremalResult,
) -> None:
    trials = np.random.default_rng(1847).normal(size=(100, 4))
    for vector in trials:
        value = _quotient(free, vector)
        assert free.minimum.eigenvalue - 1e-12 <= value
        assert value <= free.maximum.eigenvalue + 1e-12


@settings(max_examples=35, deadline=None)
@given(
    vector=st.lists(
        st.floats(min_value=-2, max_value=2, allow_nan=False),
        min_size=4,
        max_size=4,
    )
)
def test_rayleigh_variational_property_for_real_trials(
    free: ExtremalResult, vector: list[float]
) -> None:
    coefficients = np.asarray(vector)
    if np.linalg.norm(coefficients) < 1e-10:
        return
    value = _quotient(free, coefficients)
    assert free.minimum.eigenvalue - 2e-12 <= value
    assert value <= free.maximum.eigenvalue + 2e-12


@pytest.mark.parametrize("endpoint", ("minimum", "maximum"))
@pytest.mark.parametrize("frequency", (-0.8, 0.0, 0.9))
def test_common_envelope_polynomial_reconstructs_optimal_profile(
    basis: tuple[PolynomialGaussianFamily, ...],
    free: ExtremalResult,
    endpoint: str,
    frequency: float,
) -> None:
    pair = getattr(free, endpoint)
    assert pair.polynomial_coefficients is not None
    polynomial = np.polynomial.polynomial.polyval(
        frequency * frequency, pair.polynomial_coefficients
    )
    gaussian = math.exp(-0.5 * ((frequency - 0.4) / 0.8) ** 2) + math.exp(
        -0.5 * ((frequency + 0.4) / 0.8) ** 2
    )
    combination = sum(
        coefficient * family.fourier(frequency)
        for coefficient, family in zip(pair.coefficients, basis, strict=True)
    )
    assert polynomial * gaussian == pytest.approx(combination, rel=1e-12, abs=1e-12)


def test_constrained_real_subspace_uses_two_equations(
    constrained: ExtremalResult,
) -> None:
    assert constrained.coefficient_field == "real"
    assert constrained.constraint_matrix.shape == (2, 4)
    assert constrained.constraint_rank == 2
    assert constrained.subspace_dimension == 2
    np.testing.assert_allclose(
        constrained.nullspace_basis.T @ constrained.nullspace_basis,
        np.eye(2),
        atol=1e-13,
    )


@pytest.mark.parametrize("endpoint", ("minimum", "maximum"))
def test_constrained_zero_and_projected_residual_reach_machine_scale(
    constrained: ExtremalResult, endpoint: str
) -> None:
    pair = getattr(constrained, endpoint)
    assert np.linalg.norm(constrained.constraint_matrix @ pair.coefficients) < 1e-13
    assert pair.max_zero_residual < 1e-13
    assert pair.projected_residual_norm < 1e-13
    assert pair.relative_projected_residual < 1e-13
    assert pair.metric_norm == pytest.approx(1.0, abs=2e-13)


def test_constrained_full_residual_need_not_vanish(
    constrained: ExtremalResult,
) -> None:
    # Constraint multipliers live normal to the feasible subspace.
    assert constrained.minimum.full_residual_norm > 1.0
    assert constrained.maximum.full_residual_norm > 1.0
    assert constrained.minimum.projected_residual_norm < 1e-13


def test_constrained_extrema_interlace_unconstrained_extrema(
    constrained: ExtremalResult, free: ExtremalResult
) -> None:
    assert free.minimum.eigenvalue < constrained.minimum.eigenvalue
    assert constrained.maximum.eigenvalue < free.maximum.eigenvalue


def test_constrained_bounds_cover_trials_in_nullspace(
    constrained: ExtremalResult,
) -> None:
    trials = np.random.default_rng(877).normal(size=(60, 2))
    for coordinates in trials:
        vector = constrained.nullspace_basis @ coordinates
        value = _quotient(constrained, vector)
        assert constrained.minimum.eigenvalue - 1e-12 <= value
        assert value <= constrained.maximum.eigenvalue + 1e-12


def test_complex_mode_is_a_distinct_hermitian_extension(
    complex_constrained: ExtremalResult, constrained: ExtremalResult
) -> None:
    result = complex_constrained
    assert result.coefficient_field == "complex"
    assert result.constraint_matrix.shape == (1, 4)
    assert result.constraint_rank == 1
    assert result.subspace_dimension == 3
    assert result.nullspace_basis.dtype.kind == "c"
    assert result.minimum.eigenvalue < constrained.minimum.eigenvalue
    assert result.maximum.eigenvalue > constrained.maximum.eigenvalue


@pytest.mark.parametrize("endpoint", ("minimum", "maximum"))
def test_complex_mode_zero_and_projected_residuals_are_small(
    complex_constrained: ExtremalResult, endpoint: str
) -> None:
    result = complex_constrained
    pair = getattr(result, endpoint)
    assert np.linalg.norm(result.constraint_matrix @ pair.coefficients) < 1e-13
    assert pair.max_zero_residual < 1e-13
    assert pair.projected_residual_norm < 1e-13
    assert pair.metric_norm == pytest.approx(1.0, abs=2e-13)
    assert np.max(np.abs(pair.coefficients.imag)) > 1e-3


def test_complex_extremum_bounds_complex_trial_vectors(
    complex_constrained: ExtremalResult,
) -> None:
    rng = np.random.default_rng(818)
    for _ in range(50):
        coordinates = rng.normal(size=3) + 1j * rng.normal(size=3)
        vector = complex_constrained.nullspace_basis @ coordinates
        value = _quotient(complex_constrained, vector)
        assert complex_constrained.minimum.eigenvalue - 1e-12 <= value
        assert value <= complex_constrained.maximum.eigenvalue + 1e-12


def test_redundant_common_zero_does_not_shrink_subspace(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
    free: ExtremalResult,
) -> None:
    result = solve_extremal_rayleigh_ritz(
        basis, 20, zeros=(0.5j,), finite_diagnostic=diagnostic
    )
    assert result.constraint_rank == 0
    np.testing.assert_allclose(result.eigenvalues, free.eigenvalues, atol=1e-12)
    assert result.minimum.max_zero_residual == 0


def test_repeated_constraint_does_not_change_rank(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
    constrained: ExtremalResult,
) -> None:
    result = solve_extremal_rayleigh_ritz(
        basis,
        20,
        zeros=(0.7 + 0.3j, 0.7 + 0.3j),
        finite_diagnostic=diagnostic,
    )
    assert result.constraint_rank == 2
    np.testing.assert_allclose(result.eigenvalues, constrained.eigenvalues, atol=1e-12)


def test_cutoff_changes_only_finite_problem(
    basis: tuple[PolynomialGaussianFamily, ...], free: ExtremalResult
) -> None:
    other = solve_extremal_rayleigh_ritz(basis, 50)
    assert other.cutoff == 50
    np.testing.assert_allclose(other.gram_matrix, free.gram_matrix, rtol=0, atol=0)
    assert abs(other.minimum.eigenvalue - free.minimum.eigenvalue) > 0.1
    assert other.prime_tail_bound is None


def test_one_family_has_identical_minimum_and_maximum() -> None:
    one = (PolynomialGaussianFamily(0.4, 0.8),)
    result = solve_extremal_rayleigh_ritz(one, 20)
    assert result.subspace_dimension == 1
    assert result.minimum.eigenvalue == pytest.approx(result.maximum.eigenvalue)
    assert result.minimum.projected_residual_norm < 1e-12


def test_polynomial_coefficients_unavailable_for_mixed_envelopes() -> None:
    mixed = (PolynomialGaussianFamily(0.4, 0.8), PolynomialGaussianFamily(1.0, 0.9))
    result = solve_extremal_rayleigh_ritz(mixed, 20)
    assert result.minimum.polynomial_coefficients is None
    assert result.maximum.polynomial_coefficients is None


@pytest.mark.parametrize("bad_basis", ((), ("wrong",)))
def test_bad_basis_is_rejected(bad_basis: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        solve_extremal_rayleigh_ritz(bad_basis, 20)  # type: ignore[arg-type]


def test_basis_must_satisfy_common_i_over_two_zero() -> None:
    family = PolynomialGaussianFamily(0.4, 0.8, (1.0,))
    with pytest.raises(ValueError, match="i/2"):
        solve_extremal_rayleigh_ritz((family,), 20)


@pytest.mark.parametrize(
    ("cutoff", "error", "message"),
    (
        (1, ValueError, "at least 2"),
        (True, TypeError, "integer"),
        (1.5, TypeError, "integer"),
    ),
)
def test_invalid_cutoff_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...],
    cutoff: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        solve_extremal_rayleigh_ritz(basis, cutoff)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_zero", (complex(math.nan, 0), complex(0, math.inf)))
def test_nonfinite_complex_constraint_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...], bad_zero: complex
) -> None:
    with pytest.raises(ValueError, match="zeros"):
        solve_extremal_rayleigh_ritz(basis, 20, zeros=(bad_zero,))


def test_overdetermined_constraints_are_rejected(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> None:
    with pytest.raises(ValueError, match="no nonzero"):
        solve_extremal_rayleigh_ritz(
            basis,
            20,
            zeros=(0.7 + 0.3j, 1.1 + 0.6j),
            finite_diagnostic=diagnostic,
        )


def test_rank_truncation_cannot_silently_break_zero_constraint(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> None:
    with pytest.raises(FloatingPointError, match="zero-constraint residual"):
        solve_extremal_rayleigh_ritz(
            basis,
            20,
            zeros=(0.7 + 0.3j,),
            constraint_rtol=0.5,
            finite_diagnostic=diagnostic,
        )


@pytest.mark.parametrize("name", ("constraint_rtol", "metric_rtol"))
@pytest.mark.parametrize("value", (0.0, 1.0, math.nan))
def test_invalid_rank_tolerance_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...], name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=name):
        solve_extremal_rayleigh_ritz(basis, 20, **{name: value})


def test_unknown_coefficient_field_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...],
) -> None:
    with pytest.raises(ValueError, match="coefficient_field"):
        solve_extremal_rayleigh_ritz(basis, 20, coefficient_field="quaternion")  # type: ignore[arg-type]


def test_dependent_basis_is_rejected() -> None:
    family = PolynomialGaussianFamily(0.4, 0.8)
    with pytest.raises(ValueError, match="singular or ill-conditioned"):
        solve_extremal_rayleigh_ritz((family, family), 20)


def test_precomputed_diagnostic_must_match_basis_and_cutoff(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
) -> None:
    with pytest.raises(ValueError, match="families or cutoff"):
        solve_extremal_rayleigh_ritz(basis, 21, finite_diagnostic=diagnostic)
    with pytest.raises(ValueError, match="families or cutoff"):
        solve_extremal_rayleigh_ritz(basis[::-1], 20, finite_diagnostic=diagnostic)


@pytest.mark.parametrize(
    ("matrix", "message"),
    (
        (np.zeros((3, 3)), "wrong shape"),
        (np.full((4, 4), math.nan), "nonfinite"),
        (np.triu(np.ones((4, 4))), "symmetric"),
    ),
)
def test_malformed_precomputed_matrix_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
    matrix: np.ndarray,
    message: str,
) -> None:
    altered = replace(diagnostic, balance_matrix=matrix)
    with pytest.raises((ValueError, FloatingPointError), match=message):
        solve_extremal_rayleigh_ritz(basis, 20, finite_diagnostic=altered)


@pytest.mark.parametrize("bad_error", (math.nan, -1.0))
def test_malformed_precomputed_integration_error_is_rejected(
    basis: tuple[PolynomialGaussianFamily, ...],
    diagnostic: FiniteWeilDiagnostics,
    bad_error: float,
) -> None:
    errors = np.asarray(diagnostic.integration_error_estimate).copy()
    errors[0, 0] = bad_error
    altered = replace(diagnostic, integration_error_estimate=errors)
    with pytest.raises(ValueError, match="integration error estimates"):
        solve_extremal_rayleigh_ritz(basis, 20, finite_diagnostic=altered)
