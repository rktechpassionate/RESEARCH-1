"""Rayleigh--Ritz extrema for represented finite-cutoff pairing matrices.

The mathematical L2 Gram is positive semidefinite. Its entries here are
evaluated in float64; the finite Weil matrix also contains numerical
archimedean integration and a truncated prime sum. The omitted prime tail is
unbounded. These computed eigenpairs do not certify a complete Weil form or
say anything about zeta zeros.

Additional complex-frequency zeros constrain real basis coefficients by
their real and imaginary equations. An optional complex coefficient mode
gives an algebraic Hermitian extension, without asserting that its resulting
functions are real-even. Both modes use an orthonormal numerical nullspace.
For a constrained optimum, only the residual projected into that nullspace
must vanish: the full generalized residual can be large because it contains
the Lagrange-multiplier contribution.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from .arithmetic import _cutoff
from .families import PolynomialGaussianFamily
from .gram import FiniteWeilDiagnostics, finite_weil_diagnostics, l2_gram_matrix

CoefficientField: TypeAlias = Literal["real", "complex"]
CoefficientArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True)
class ExtremalEigenpair:
    """One G-normalized eigenpair of the represented, possibly constrained form."""

    eigenvalue: float
    rayleigh_quotient: float
    coefficients: CoefficientArray
    polynomial_coefficients: tuple[float | complex, ...] | None
    metric_norm: float
    full_residual_norm: float
    projected_residual_norm: float
    relative_projected_residual: float
    zero_residuals: tuple[float, ...]
    max_zero_residual: float


@dataclass(frozen=True)
class ExtremalResult:
    """Extrema for a computed finite-dimensional, finite-cutoff problem.

    ``nullspace_basis`` has orthonormal columns spanning the represented
    coefficient constraints. ``eigenvalues`` are ascending generalized
    eigenvalues on that subspace. A small ``projected_residual_norm`` is the
    correct stationarity check; ``full_residual_norm`` need not be small when
    ``prescribed_zeros`` is nonempty. The metric condition number is for the
    projected computed Gram. No bound for omitted prime powers is available.
    """

    families: tuple[PolynomialGaussianFamily, ...]
    cutoff: int
    prescribed_zeros: tuple[complex, ...]
    coefficient_field: CoefficientField
    gram_matrix: NDArray[np.float64]
    weil_matrix: NDArray[np.float64]
    integration_error_estimate: NDArray[np.float64]
    constraint_matrix: CoefficientArray
    nullspace_basis: CoefficientArray
    constraint_rank: int
    subspace_dimension: int
    metric_condition_number: float
    metric_relative_tolerance: float
    eigenvalues: NDArray[np.float64]
    minimum: ExtremalEigenpair
    maximum: ExtremalEigenpair
    prime_tail_bound: None = None


def _families(
    values: Iterable[PolynomialGaussianFamily],
) -> tuple[PolynomialGaussianFamily, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError("families must be an iterable") from exc
    if not items:
        raise ValueError("at least one basis family is required")
    if not all(isinstance(item, PolynomialGaussianFamily) for item in items):
        raise TypeError("every basis family must be a PolynomialGaussianFamily")
    if not all(item.is_weil_admissible for item in items):
        raise ValueError("every basis family must contain the i/2 zero")
    return items


def _zeros(values: Iterable[complex]) -> tuple[complex, ...]:
    try:
        requested = tuple(complex(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("zeros must be finite complex frequencies") from exc
    if not all(
        math.isfinite(value.real) and math.isfinite(value.imag) for value in requested
    ):
        raise ValueError("zeros must be finite complex frequencies")
    return requested


def _relative_tolerance(value: float, name: str) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in (0, 1)") from exc
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError(f"{name} must be finite and in (0, 1)")
    return tolerance


def _constraint_subspace(
    families: tuple[PolynomialGaussianFamily, ...],
    zeros: tuple[complex, ...],
    tolerance: float,
    coefficient_field: CoefficientField,
) -> tuple[CoefficientArray, CoefficientArray, int]:
    size = len(families)
    if not zeros:
        if coefficient_field == "real":
            return (
                np.zeros((0, size), dtype=np.float64),
                np.eye(size, dtype=np.float64),
                0,
            )
        return (
            np.zeros((0, size), dtype=np.complex128),
            np.eye(size, dtype=np.complex128),
            0,
        )
    values_matrix = np.empty((len(zeros), size), dtype=np.complex128)
    for row, zero in enumerate(zeros):
        values = np.asarray(
            [complex(family.fourier(zero)) for family in families],
            dtype=np.complex128,
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("constraint evaluation is nonfinite")
        values_matrix[row] = values
    matrix: CoefficientArray
    if coefficient_field == "real":
        real_matrix = np.empty((2 * len(zeros), size), dtype=np.float64)
        real_matrix[0::2] = values_matrix.real
        real_matrix[1::2] = values_matrix.imag
        matrix = real_matrix
    else:
        matrix = values_matrix

    # Scaling equations does not change their nullspace and prevents one
    # large Fourier value from masking an independent small equation.
    scaled_rows: list[CoefficientArray] = []
    for equation in matrix:
        scale = float(np.linalg.norm(equation))
        if not math.isfinite(scale):
            raise FloatingPointError("constraint row norm is nonfinite")
        if scale > 0:
            scaled_rows.append(equation / scale)
    if not scaled_rows:
        return matrix, cast(CoefficientArray, np.eye(size, dtype=matrix.dtype)), 0
    if coefficient_field == "real":
        scaled_real = np.asarray(np.stack(scaled_rows), dtype=np.float64)
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_real, full_matrices=True
        )
    else:
        scaled_complex = np.asarray(np.stack(scaled_rows), dtype=np.complex128)
        _, singular_values, right_vectors = np.linalg.svd(
            scaled_complex, full_matrices=True
        )
    threshold = max(
        tolerance, np.finfo(float).eps * max(len(scaled_rows), size)
    ) * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank == size:
        raise ValueError("zero constraints leave no nonzero coefficient vector")
    nullspace = cast(CoefficientArray, right_vectors[rank:].conj().T)
    return matrix, nullspace, rank


def _combined_polynomial(
    families: tuple[PolynomialGaussianFamily, ...],
    coefficients: CoefficientArray,
    coefficient_field: CoefficientField,
) -> tuple[float | complex, ...] | None:
    first = families[0]
    if any(
        family.center != first.center or family.width != first.width
        for family in families[1:]
    ):
        return None
    degree_count = max(len(family.coefficients) for family in families)
    result: list[float | complex] = []
    for degree in range(degree_count):
        terms = [
            complex(coefficient) * family.coefficients[degree]
            for coefficient, family in zip(coefficients, families, strict=True)
            if degree < len(family.coefficients)
        ]
        real = math.fsum(term.real for term in terms)
        if coefficient_field == "real":
            result.append(real)
        else:
            result.append(complex(real, math.fsum(term.imag for term in terms)))
    return tuple(result)


def _eigenpair(
    eigenvalue: float,
    reduced_coefficients: CoefficientArray,
    nullspace: CoefficientArray,
    gram: NDArray[np.float64],
    weil: NDArray[np.float64],
    families: tuple[PolynomialGaussianFamily, ...],
    zeros: tuple[complex, ...],
    denominator_scale: float,
    coefficient_field: CoefficientField,
) -> ExtremalEigenpair:
    coefficients = cast(CoefficientArray, np.asarray(nullspace @ reduced_coefficients))
    pivot = int(np.argmax(np.abs(coefficients)))
    pivot_value = complex(coefficients[pivot])
    if coefficient_field == "real":
        if pivot_value.real < 0:
            coefficients = -coefficients
    elif pivot_value != 0:
        coefficients = cast(
            CoefficientArray,
            coefficients * (pivot_value.conjugate() / abs(pivot_value)),
        )
    metric_norm = float(np.real(np.vdot(coefficients, gram @ coefficients)))
    if not math.isfinite(metric_norm) or metric_norm <= 0:
        raise FloatingPointError("generalized eigenvector lost positive G norm")
    quotient = float(np.real(np.vdot(coefficients, weil @ coefficients))) / metric_norm
    residual = weil @ coefficients - eigenvalue * (gram @ coefficients)
    projected = nullspace.conj().T @ residual
    full_norm = float(np.linalg.norm(residual))
    projected_norm = float(np.linalg.norm(projected))
    relative_projected = projected_norm / max(
        1.0, denominator_scale * float(np.linalg.norm(coefficients))
    )
    zero_residuals = tuple(
        abs(
            sum(
                complex(coefficient) * complex(family.fourier(zero))
                for coefficient, family in zip(coefficients, families, strict=True)
            )
        )
        for zero in zeros
    )
    if not all(math.isfinite(value) for value in zero_residuals):
        raise FloatingPointError("zero residual is nonfinite")
    coefficients.setflags(write=False)
    return ExtremalEigenpair(
        eigenvalue=float(eigenvalue),
        rayleigh_quotient=quotient,
        coefficients=coefficients,
        polynomial_coefficients=_combined_polynomial(
            families, coefficients, coefficient_field
        ),
        metric_norm=metric_norm,
        full_residual_norm=full_norm,
        projected_residual_norm=projected_norm,
        relative_projected_residual=relative_projected,
        zero_residuals=zero_residuals,
        max_zero_residual=max(zero_residuals, default=0.0),
    )


def _check_constraint_residuals(
    pair: ExtremalEigenpair,
    families: tuple[PolynomialGaussianFamily, ...],
    zeros: tuple[complex, ...],
    tolerance: float,
) -> None:
    """Reject a truncated SVD nullspace that fails represented equations."""
    # The rank threshold may be user-adjusted, but an accepted solution must
    # still satisfy the original equations close to float64 precision.
    relative_limit = max(64 * np.finfo(float).eps, min(8 * tolerance, 1e-12))
    for zero, residual in zip(zeros, pair.zero_residuals, strict=True):
        scale = math.fsum(
            abs(complex(coefficient)) * abs(complex(family.fourier(zero)))
            for coefficient, family in zip(pair.coefficients, families, strict=True)
        )
        if residual > relative_limit * max(1.0, scale):
            raise FloatingPointError(
                "zero-constraint residual exceeds represented-subspace tolerance"
            )


def solve_extremal_rayleigh_ritz(
    families: Sequence[PolynomialGaussianFamily],
    cutoff: int,
    *,
    zeros: Sequence[complex] = (),
    coefficient_field: CoefficientField = "real",
    constraint_rtol: float = 1e-12,
    metric_rtol: float = 1e-12,
    finite_diagnostic: FiniteWeilDiagnostics | None = None,
) -> ExtremalResult:
    """Optimize c* W c / c* G c over a finite coefficient subspace.

    The unconstrained case uses ``zeros=()``. For real coefficients, each
    complex frequency adds real and imaginary linear equations and retains
    real-even test functions. ``coefficient_field='complex'`` adds one complex
    equation per frequency and is only an algebraic Hermitian extension; the
    resulting functions need not be real-even. The nullspace is computed by
    SVD and the projected generalized eigenproblem by ``scipy.linalg.eigh``
    (Cholesky reduction). Ill-conditioned or nonpositive reduced Grams are rejected;
    no ridge is silently added to the objective. Supplied precomputed finite
    diagnostics must describe the same ordered families and cutoff.

    The returned eigenvalues optimize the *computed* matrices under the
    represented constraints. They are not certified extrema of exact
    integrals or of the full, infinite Weil form.
    """
    items = _families(families)
    limit = _cutoff(cutoff)
    requested = _zeros(zeros)
    if coefficient_field not in ("real", "complex"):
        raise ValueError("coefficient_field must be 'real' or 'complex'")
    constraint_tolerance = _relative_tolerance(constraint_rtol, "constraint_rtol")
    metric_tolerance = _relative_tolerance(metric_rtol, "metric_rtol")
    constraints, nullspace, rank = _constraint_subspace(
        items, requested, constraint_tolerance, coefficient_field
    )
    gram = l2_gram_matrix(items)
    if finite_diagnostic is None:
        diagnostic = finite_weil_diagnostics(items, limit)
    else:
        if not isinstance(finite_diagnostic, FiniteWeilDiagnostics):
            raise TypeError("finite_diagnostic must be FiniteWeilDiagnostics")
        if finite_diagnostic.families != items or finite_diagnostic.cutoff != limit:
            raise ValueError("finite_diagnostic families or cutoff do not match")
        diagnostic = finite_diagnostic
    weil = diagnostic.balance_matrix
    integration_errors = diagnostic.integration_error_estimate
    shape = (len(items), len(items))
    if gram.shape != shape or weil.shape != shape or integration_errors.shape != shape:
        raise ValueError("pairing matrices have the wrong shape")
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(weil)):
        raise FloatingPointError("pairing matrices are nonfinite")
    if not np.all(np.isfinite(integration_errors)) or np.any(integration_errors < 0):
        raise ValueError("integration error estimates must be finite and nonnegative")
    if not np.allclose(gram, gram.T, atol=1e-12, rtol=1e-12) or not np.allclose(
        weil, weil.T, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("pairing matrices must be symmetric")
    reduced_gram = nullspace.conj().T @ gram @ nullspace
    reduced_weil = nullspace.conj().T @ weil @ nullspace
    reduced_gram = 0.5 * (reduced_gram + reduced_gram.conj().T)
    reduced_weil = 0.5 * (reduced_weil + reduced_weil.conj().T)
    metric_eigenvalues = np.linalg.eigvalsh(reduced_gram)
    smallest = float(metric_eigenvalues[0])
    largest = float(metric_eigenvalues[-1])
    if not math.isfinite(smallest) or not math.isfinite(largest):
        raise FloatingPointError("projected Gram spectrum is nonfinite")
    if smallest <= metric_tolerance * largest:
        raise ValueError("projected Gram is singular or ill-conditioned")
    metric_condition = largest / smallest
    try:
        eigenvalues, reduced_vectors = eigh(reduced_weil, reduced_gram)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError("generalized eigensolver rejected the projected Gram") from exc
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(reduced_vectors)):
        raise FloatingPointError("generalized eigensolver returned nonfinite data")
    denominator_scale = float(np.linalg.norm(weil, 2)) + float(
        np.max(np.abs(eigenvalues))
    ) * float(np.linalg.norm(gram, 2))
    if not math.isfinite(denominator_scale):
        raise FloatingPointError("residual normalization scale is nonfinite")
    minimum = _eigenpair(
        float(eigenvalues[0]),
        reduced_vectors[:, 0],
        nullspace,
        gram,
        weil,
        items,
        requested,
        denominator_scale,
        coefficient_field,
    )
    maximum = _eigenpair(
        float(eigenvalues[-1]),
        reduced_vectors[:, -1],
        nullspace,
        gram,
        weil,
        items,
        requested,
        denominator_scale,
        coefficient_field,
    )
    for pair in (minimum, maximum):
        _check_constraint_residuals(pair, items, requested, constraint_tolerance)
    for array in (constraints, nullspace, eigenvalues):
        array.setflags(write=False)
    return ExtremalResult(
        families=items,
        cutoff=limit,
        prescribed_zeros=requested,
        coefficient_field=coefficient_field,
        gram_matrix=gram,
        weil_matrix=weil,
        integration_error_estimate=diagnostic.integration_error_estimate,
        constraint_matrix=constraints,
        nullspace_basis=nullspace,
        constraint_rank=rank,
        subspace_dimension=nullspace.shape[1],
        metric_condition_number=metric_condition,
        metric_relative_tolerance=metric_tolerance,
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        minimum=minimum,
        maximum=maximum,
    )
