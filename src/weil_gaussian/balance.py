"""Numerical archimedean and finite prime-power balance diagnostics.

The finite arithmetic sum has no certified bound for omitted prime powers.
Consequently the finite balance is not the Weil quadratic form, a positivity
certificate, or a test of the Riemann Hypothesis.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from scipy.integrate import IntegrationWarning, quad
from scipy.special import digamma

from .arithmetic import _cutoff, _primes_through
from .families import PolynomialGaussianFamily


@dataclass(frozen=True)
class ArchimedeanResult:
    """Numerical archimedean term and non-rigorous quadrature error estimate."""

    value: float
    quadrature_error_estimate: float
    converged: bool


@dataclass(frozen=True)
class BalanceDiagnostic:
    """Comparison with a finite prime-power sum through an integer cutoff.

    The finite_balance equals archimedean.value minus finite_prime_term.
    The omitted prime tail has no bound here, represented explicitly by None.
    """

    archimedean: ArchimedeanResult
    finite_prime_term: float
    finite_balance: float
    cutoff: int
    prime_tail_bound: None = None


def _family(family: PolynomialGaussianFamily) -> PolynomialGaussianFamily:
    if not isinstance(family, PolynomialGaussianFamily):
        raise TypeError("family must be a PolynomialGaussianFamily")
    return family


def _admissible(family: PolynomialGaussianFamily) -> None:
    if not family.is_weil_admissible:
        raise ValueError(
            "Weil balance requires a family with Fourier zeros at both +/-i/2"
        )


def _tolerance(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _real_scalar(value: float | complex, description: str) -> float:
    converted = complex(value)
    if not math.isfinite(converted.real) or not math.isfinite(converted.imag):
        raise FloatingPointError(f"{description} is not finite")
    if abs(converted.imag) > 1e-11 * max(1.0, abs(converted.real)):
        raise FloatingPointError(f"{description} is not real within rounding error")
    return converted.real


def _integration_intervals(
    center: float, width: float
) -> tuple[tuple[float, float], ...]:
    """Split near the positive Gaussian peak so narrow peaks are not missed."""
    upper_peak = center + 12.0 * width
    if not math.isfinite(upper_peak) or upper_peak <= center:
        raise ValueError("parameter scale exceeds supported float integration range")
    candidates = (0.0, max(0.0, center - 12.0 * width), center, upper_peak)
    points: list[float] = []
    for point in candidates:
        if math.isfinite(point) and (not points or point > points[-1]):
            points.append(point)
    if len(points) == 1:
        raise ValueError("parameter scale exceeds supported float integration range")
    return tuple(zip(points, (*points[1:], math.inf), strict=True))


def archimedean_component(
    family: PolynomialGaussianFamily,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> ArchimedeanResult:
    r"""Numerically evaluate the archimedean term K.

    K = (1/pi) integral_0^inf Re psi(1/4+i*t/2) F(t)^2 dt
        - Phi(0) log(pi).

    This normalization requires F(+/-i/2)=0. The returned quadrature error
    is an adaptive estimate, not a mathematically certified enclosure.
    For large polynomial degrees, roundoff in the Fourier profile can exceed
    this quadrature estimate; the two errors must not be conflated.
    """
    family = _family(family)
    _admissible(family)
    abs_tol = _tolerance(abs_tol, "abs_tol")
    rel_tol = _tolerance(rel_tol, "rel_tol")
    intervals = _integration_intervals(family.center, family.width)

    def integrand(frequency: float) -> float:
        profile = _real_scalar(family.fourier(frequency), "Fourier profile")
        weight = float(digamma(0.25 + 0.5j * frequency).real)
        result = weight * profile * profile
        if not math.isfinite(result):
            raise FloatingPointError("archimedean integrand is not finite")
        return result

    values: list[float] = []
    errors: list[float] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        for lower, upper in intervals:
            value, error = quad(
                integrand,
                lower,
                upper,
                epsabs=abs_tol * math.pi / len(intervals),
                epsrel=rel_tol,
                limit=250,
            )
            values.append(value)
            errors.append(error)

    integral = math.fsum(values) / math.pi
    error_estimate = math.fsum(errors) / math.pi
    zero_lag = _real_scalar(family.autocorrelation(0.0), "zero-lag autocorrelation")
    result = integral - zero_lag * math.log(math.pi)
    if not math.isfinite(result) or not math.isfinite(error_estimate):
        raise FloatingPointError("archimedean term is not finite")
    warned = any(issubclass(warning.category, IntegrationWarning) for warning in caught)
    target = max(abs_tol, rel_tol * abs(integral))
    return ArchimedeanResult(
        value=result,
        quadrature_error_estimate=error_estimate,
        converged=not warned and error_estimate <= target,
    )


def finite_prime_component(family: PolynomialGaussianFamily, cutoff: int) -> float:
    r"""Return the finite prime-power term through the integer cutoff.

    The value is 2 sum_{n<=cutoff} Lambda(n) Phi(log n) / sqrt(n).
    There is no estimate or bound for the omitted terms n>cutoff.
    """
    family = _family(family)
    cutoff = _cutoff(cutoff)
    terms: list[float] = []
    for prime in _primes_through(cutoff):
        weight = 2.0 * math.log(prime)
        power = prime
        while power <= cutoff:
            overlap = _real_scalar(
                family.autocorrelation(math.log(power)), "autocorrelation"
            )
            contribution = weight * overlap / math.sqrt(power)
            if not math.isfinite(contribution):
                raise FloatingPointError("prime-power contribution is not finite")
            terms.append(contribution)
            if power > cutoff // prime:
                break
            power *= prime
    return math.fsum(terms)


def balance_diagnostic(
    family: PolynomialGaussianFamily,
    cutoff: int,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> BalanceDiagnostic:
    """Compare the archimedean term with a finite prime-power sum.

    The difference is cutoff-dependent, not the full Weil quadratic form.
    A None prime_tail_bound makes the missing control machine-readable.
    """
    family = _family(family)
    _admissible(family)
    cutoff = _cutoff(cutoff)
    archimedean = archimedean_component(family, abs_tol=abs_tol, rel_tol=rel_tol)
    finite_prime = finite_prime_component(family, cutoff)
    return BalanceDiagnostic(
        archimedean=archimedean,
        finite_prime_term=finite_prime,
        finite_balance=archimedean.value - finite_prime,
        cutoff=cutoff,
    )
