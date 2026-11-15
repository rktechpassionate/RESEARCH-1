"""L2 Gram matrices and finite-cutoff Weil-form diagnostics.

The L2 Gram matrix is positive semidefinite for every parameter family by
construction. The finite Weil matrix is a numerical K - P_N calculation:
without a prime-tail bound it does not certify positivity or RH.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Iterable, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.polynomial import polyval
from scipy.integrate import IntegrationWarning, quad
from scipy.special import digamma

from .arithmetic import _cutoff, _primes_through
from .families import PolynomialGaussianFamily


@dataclass(frozen=True)
class GramDiagnostics:
    """Spectrum of an unconditional L2 Gram matrix."""

    families: tuple[PolynomialGaussianFamily, ...]
    matrix: np.ndarray
    eigenvalues: np.ndarray
    effective_rank: int
    condition_number: float
    is_positive_semidefinite: bool
    relative_tolerance: float


@dataclass(frozen=True)
class FiniteWeilDiagnostics:
    """Finite K - P_N matrix; its spectrum is exploratory, not a certificate."""

    families: tuple[PolynomialGaussianFamily, ...]
    cutoff: int
    archimedean_matrix: np.ndarray
    finite_prime_matrix: np.ndarray
    balance_matrix: np.ndarray
    eigenvalues: np.ndarray
    integration_error_estimate: np.ndarray
    prime_tail_bound: None = None


def _families(values: Iterable[PolynomialGaussianFamily]) -> tuple[PolynomialGaussianFamily, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError("families must be an iterable of PolynomialGaussianFamily") from exc
    if not result:
        raise ValueError("at least one family is required")
    if not all(isinstance(item, PolynomialGaussianFamily) for item in result):
        raise TypeError("every family must be a PolynomialGaussianFamily")
    return result


def _position(value: float) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("position must be a finite real number") from exc
    if not math.isfinite(x):
        raise ValueError("position must be a finite real number")
    return x


def _components(left: PolynomialGaussianFamily, right: PolynomialGaussianFamily):
    first = left.width * left.width
    second = right.width * right.width
    total = first + second
    tau2 = first * second / total
    if not math.isfinite(total) or not math.isfinite(tau2) or tau2 <= 0:
        raise FloatingPointError("widths are outside reliable double-precision range")
    tau = math.sqrt(tau2)
    for a in (-left.center, left.center):
        for b in (-right.center, right.center):
            mu = (a * second + b * first) / total
            attenuation = math.exp(-0.5 * ((a - b) ** 2) / total)
            yield mu, tau, attenuation


def _quadrature_order(left: PolynomialGaussianFamily, right: PolynomialGaussianFamily) -> int:
    # P_i(xi) and P_j(xi) have degrees 2m_i and 2m_j. Hermite quadrature
    # integrates their product exactly when 2n-1 >= 2(m_i+m_j).
    order = max(12, len(left.coefficients) + len(right.coefficients) - 1)
    if order > 96:
        raise ValueError("polynomial degree exceeds the stable quadrature limit")
    return order


def cross_autocorrelation(
    left: PolynomialGaussianFamily,
    right: PolynomialGaussianFamily,
    position: float,
) -> float:
    """Evaluate the cross-correlation at a real lag.

    This is (2 pi)^-1 times the Fourier integral of F_left F_right exp(i xi x).
    Gaussian-product completion and Hermite quadrature give an exact polynomial
    integral up to floating-point rounding, without sampling the real line.
    """
    _families((left, right))
    x = _position(position)
    nodes, weights = hermgauss(_quadrature_order(left, right))
    left_coeff = np.asarray(left.coefficients, dtype=float)
    right_coeff = np.asarray(right.coefficients, dtype=float)
    values: list[complex] = []
    for mu, tau, attenuation in _components(left, right):
        tau2 = tau * tau
        shifted = mu + 1j * tau2 * x + math.sqrt(2.0) * tau * nodes
        polynomial = polyval(shifted * shifted, left_coeff) * polyval(
            shifted * shifted, right_coeff
        )
        moment = np.dot(weights, polynomial)
        values.append(
            math.sqrt(2.0)
            * tau
            * attenuation
            * np.exp(1j * mu * x - 0.5 * tau2 * x * x)
            * moment
            / (2.0 * math.pi)
        )
    result = sum(values)
    if not np.isfinite(result):
        raise FloatingPointError("cross-correlation overflowed")
    if abs(result.imag) > 1e-9 * max(1.0, abs(result.real)):
        raise FloatingPointError("cross-correlation lost conjugate symmetry")
    return float(result.real)


def l2_gram_matrix(families: Sequence[PolynomialGaussianFamily]) -> np.ndarray:
    """Return G_ij = integral f_i(x) f_j(x) dx = Phi_ij(0).

    The returned symmetric matrix is positive semidefinite in exact arithmetic
    for every finite family list. A tiny negative eigenvalue can appear through
    ordinary floating-point rounding.
    """
    items = _families(families)
    matrix = np.empty((len(items), len(items)), dtype=float)
    for i, left in enumerate(items):
        for j in range(i, len(items)):
            value = cross_autocorrelation(left, items[j], 0.0)
            matrix[i, j] = matrix[j, i] = value
    matrix.setflags(write=False)
    return matrix


def l2_gram_diagnostics(
    families: Sequence[PolynomialGaussianFamily], relative_tolerance: float = 1e-10
) -> GramDiagnostics:
    """Return eigenvalues, effective rank, and condition number for an L2 Gram."""
    items = _families(families)
    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("relative_tolerance must be finite and in (0, 1)")
    matrix = l2_gram_matrix(items)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("Gram eigenvalues are nonfinite")
    scale = float(np.max(np.abs(eigenvalues)))
    # Use the actual spectral scale. An absolute floor of one would mark a
    # small but full-rank Gram matrix as singular.
    threshold = tolerance * scale if scale > 0 else 0.0
    rank = int(np.count_nonzero(eigenvalues > threshold))
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    condition = largest / smallest if scale > 0 and smallest > threshold else math.inf
    eigenvalues.setflags(write=False)
    return GramDiagnostics(
        items,
        matrix,
        eigenvalues,
        rank,
        condition,
        smallest >= -threshold,
        tolerance,
    )


def grid_gram_diagnostics(
    centers: Sequence[float],
    widths: Sequence[float],
    zeros: Sequence[complex] = (0.5j,),
    relative_tolerance: float = 1e-10,
) -> GramDiagnostics:
    """Build the Cartesian parameter grid in center-major order and diagnose it."""
    centers = tuple(centers)
    widths = tuple(widths)
    if not centers or not widths:
        raise ValueError("centers and widths must both be nonempty")
    roots = tuple(zeros)
    items = tuple(
        PolynomialGaussianFamily(center, width, roots)
        for center in centers
        for width in widths
    )
    return l2_gram_diagnostics(items, relative_tolerance)


def _admissible_for_weil(family: PolynomialGaussianFamily) -> bool:
    # An exact requested zero is essential. Evaluating F(i/2) alone is unsafe:
    # a very small Gaussian value can look like a zero through underflow.
    return family.is_weil_admissible


def _archimedean_cross(
    left: PolynomialGaussianFamily, right: PolynomialGaussianFamily
) -> tuple[float, float]:
    """Numerically integrate the cross archimedean term and error estimate."""
    def integrand(frequency: float) -> float:
        f_left = float(left.fourier(frequency))
        f_right = float(right.fourier(frequency))
        return float(np.real(digamma(0.25 + 0.5j * frequency))) * f_left * f_right

    breaks = {0.0}
    for family in (left, right):
        breaks.add(family.center)
        breaks.add(max(0.0, family.center - 6.0 * family.width))
        breaks.add(family.center + 6.0 * family.width)
    ordered = sorted(point for point in breaks if math.isfinite(point))
    if not ordered or ordered[0] != 0:
        ordered.insert(0, 0.0)
    intervals = list(zip(ordered[:-1], ordered[1:])) + [(ordered[-1], math.inf)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        pieces = [
            quad(integrand, lower, upper, epsabs=1e-10, epsrel=1e-10, limit=250)
            for lower, upper in intervals
            if upper > lower
        ]
    if any(issubclass(warning.category, IntegrationWarning) for warning in caught):
        raise FloatingPointError("archimedean quadrature did not converge")
    integral = math.fsum(value for value, _ in pieces) / math.pi
    estimated_error = math.fsum(error for _, error in pieces) / math.pi
    if estimated_error > max(1e-9, 1e-9 * abs(integral)):
        raise FloatingPointError("archimedean quadrature exceeded tolerance")
    result = integral - cross_autocorrelation(left, right, 0.0) * math.log(math.pi)
    return result, estimated_error


def _prime_powers(limit: int):
    for prime in _primes_through(limit):
        value = prime
        while value <= limit:
            yield value, math.log(prime)
            if value > limit // prime:
                break
            value *= prime


def finite_weil_diagnostics(
    families: Sequence[PolynomialGaussianFamily], cutoff: int
) -> FiniteWeilDiagnostics:
    """Compute the exploratory cross-family matrix K_ij - P_ij,N.

    Here K_ij is the cross archimedean integral minus Phi_ij(0) log(pi), and
    P_ij,N = 2 sum_{n<=N} Lambda(n) Phi_ij(log n) / sqrt(n). Every family must
    enforce F(+/-i/2)=0. The omitted prime tail is NOT bounded; eigenvalues
    of this finite matrix cannot establish Weil positivity or RH.
    """
    items = _families(families)
    limit = _cutoff(cutoff)
    if not all(_admissible_for_weil(family) for family in items):
        raise ValueError("finite Weil diagnostics require a zero at +i/2 and -i/2")
    powers = tuple(_prime_powers(limit))
    size = len(items)
    arch = np.empty((size, size), dtype=float)
    prime = np.empty((size, size), dtype=float)
    errors = np.empty((size, size), dtype=float)
    for i, left in enumerate(items):
        for j in range(i, size):
            right = items[j]
            arch_value, error = _archimedean_cross(left, right)
            prime_value = math.fsum(
                2.0
                * log_prime
                / math.sqrt(value)
                * cross_autocorrelation(left, right, math.log(value))
                for value, log_prime in powers
            )
            arch[i, j] = arch[j, i] = arch_value
            prime[i, j] = prime[j, i] = prime_value
            errors[i, j] = errors[j, i] = error
    balance = arch - prime
    eigenvalues = np.linalg.eigvalsh(balance)
    for array in (arch, prime, errors, balance, eigenvalues):
        if not np.all(np.isfinite(array)):
            raise FloatingPointError("finite Weil calculation is nonfinite")
        array.setflags(write=False)
    return FiniteWeilDiagnostics(
        items, limit, arch, prime, balance, eigenvalues, errors
    )


def finite_weil_matrix(
    families: Sequence[PolynomialGaussianFamily], cutoff: int
) -> np.ndarray:
    """Return only the exploratory K - P_N matrix; see finite_weil_diagnostics."""
    return finite_weil_diagnostics(families, cutoff).balance_matrix
