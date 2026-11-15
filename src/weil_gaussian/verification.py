"""Independent double-precision checks of polynomial-Gaussian identities.

The adaptive and Gauss-Hermite integrations use different evaluations of the
defining Fourier integral. Their reported errors are numerical indicators,
not rigorous enclosures. Neither check addresses a Weil positivity claim.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.polynomial import polyval
from scipy.integrate import IntegrationWarning, quad

from .families import PolynomialGaussianFamily

Quantity = Literal["inverse", "autocorrelation"]


@dataclass(frozen=True)
class VerificationCheck:
    """One identity check and its independently computed error indicators.

    ``adaptive_error_estimate`` is SciPy QUADPACK's local error estimate,
    summed over intervals. ``hermite_refinement_delta`` compares orders N and
    2N; it is a convergence indicator, not an upper bound on the true error.
    Both schemes integrate an infinite-domain formula, so no finite-cutoff
    Gaussian tail bound applies.
    """

    quantity: Quantity
    position: float
    analytic: float
    adaptive: float
    gauss_hermite: float
    adaptive_error_estimate: float
    hermite_refinement_delta: float
    analytic_adaptive_abs_error: float
    analytic_hermite_abs_error: float
    cross_scheme_discrepancy: float
    acceptance_threshold: float
    passed: bool
    finite_cutoff_tail_bound: None = None


@dataclass(frozen=True)
class VerificationReport:
    """Comparisons for all requested positions, inverse then autocorrelation."""

    family: PolynomialGaussianFamily
    positions: tuple[float, ...]
    checks: tuple[VerificationCheck, ...]
    absolute_tolerance: float
    relative_tolerance: float
    hermite_base_order: int
    hermite_reported_order: int
    passed: bool


def _finite_position(value: float) -> float:
    try:
        position = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("position must be a finite real number") from exc
    if not math.isfinite(position):
        raise ValueError("position must be a finite real number")
    return position


def _tolerance(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _adaptive_integral(
    family: PolynomialGaussianFamily, position: float, quantity: Quantity
) -> tuple[float, float]:
    """Integrate [0, infinity) with adaptive Gauss-Kronrod subintervals.

    Evenness gives the factor 1/pi. Splits expose the positive Gaussian peak
    to QUADPACK; its error estimate is useful evidence but not a certificate.
    """

    center = family.center
    width = family.width
    endpoint = center + 8.0 * width
    if not math.isfinite(endpoint) or endpoint <= center:
        raise FloatingPointError("Gaussian scales exceed reliable quadrature range")
    breaks = sorted(
        {
            0.0,
            max(0.0, center - 8.0 * width),
            max(0.0, center - 4.0 * width),
            max(0.0, center - 2.0 * width),
            center,
            center + 2.0 * width,
            center + 4.0 * width,
            endpoint,
        }
    )
    intervals = [
        (lower, upper) for lower, upper in zip(breaks, breaks[1:], strict=False)
    ]
    intervals.append((endpoint, math.inf))

    def integrand(frequency: float) -> float:
        spectrum = float(family.fourier(frequency))
        value = spectrum * math.cos(frequency * position)
        if quantity == "autocorrelation":
            value *= spectrum
        if not math.isfinite(value):
            raise FloatingPointError("quadrature integrand became nonfinite")
        return value

    pieces: list[tuple[float, float]] = []
    for lower, upper in intervals:
        if upper <= lower:
            continue
        # A request below the roundoff floor can trigger QUADPACK's warning
        # despite a stable integral. Retry that interval at a less ambitious
        # tolerance, retaining the resulting (often larger) error estimate.
        for requested_error in (1e-13, 1e-12, 1e-11):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                piece = quad(
                    integrand,
                    lower,
                    upper,
                    epsabs=requested_error,
                    epsrel=requested_error,
                    limit=250,
                )
            if not any(
                issubclass(item.category, IntegrationWarning) for item in caught
            ):
                pieces.append(piece)
                break
        else:
            raise FloatingPointError("adaptive quadrature did not converge")
    estimate = math.fsum(value for value, _ in pieces) / math.pi
    error_estimate = math.fsum(error for _, error in pieces) / math.pi
    if not math.isfinite(estimate) or not math.isfinite(error_estimate):
        raise FloatingPointError("adaptive quadrature returned nonfinite output")
    return estimate, error_estimate


def _hermite_integral(
    family: PolynomialGaussianFamily,
    position: float,
    quantity: Quantity,
    order: int,
) -> float:
    """Integrate the real Gaussian decomposition using unshifted Hermite nodes.

    For the inverse, t=b+sqrt(2)*s*u for b=+/-a converts each Gaussian to
    exp(-u**2). For autocorrelation, the two diagonal terms use t=b+s*u;
    the cross term uses b=0, scale=s and weight 2*exp(-a**2/s**2).
    Gauss-Hermite convergence is fast for these entire cosine integrands, but
    order doubling remains an empirical check rather than a rigorous bound.
    """

    nodes, weights = hermgauss(order)
    coefficients = np.asarray(family.coefficients, dtype=np.float64)
    center = family.center
    width = family.width

    def component(shift: float, scale: float, amplitude: float) -> float:
        frequency = shift + scale * nodes
        polynomial = polyval(frequency * frequency, coefficients)
        if quantity == "autocorrelation":
            polynomial *= polynomial
        samples = polynomial * np.cos(frequency * position)
        value = amplitude * scale * float(np.dot(weights, samples))
        if not math.isfinite(value):
            raise FloatingPointError("Gauss-Hermite quadrature became nonfinite")
        return value

    if quantity == "inverse":
        scale = math.sqrt(2.0) * width
        integral = math.fsum(
            (component(-center, scale, 1.0), component(center, scale, 1.0))
        )
    else:
        cross_weight = 2.0 * math.exp(-((center / width) ** 2))
        integral = math.fsum(
            (
                component(-center, width, 1.0),
                component(center, width, 1.0),
                component(0.0, width, cross_weight),
            )
        )
    result = integral / (2.0 * math.pi)
    if not math.isfinite(result):
        raise FloatingPointError("Gauss-Hermite quadrature returned nonfinite output")
    return result


def verify_family(
    family: PolynomialGaussianFamily,
    positions: Iterable[float] = (0.0, 0.8, 1.4),
    *,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-10,
    hermite_order: int = 48,
) -> VerificationReport:
    """Compare two closed-form identities against two numerical schemes.

    The adaptive method uses SciPy's Gauss-Kronrod QUADPACK integrator. The
    separate Gauss-Hermite method samples the full real-line Gaussian
    decomposition at orders ``hermite_order`` and ``2*hermite_order``. Its
    reported value is the higher-order result. A check passes when each
    analytic discrepancy, cross-scheme discrepancy, adaptive error estimate,
    and Hermite refinement delta is at most ``atol + rtol * scale``. Passing
    means numerical agreement at the stated tolerance, not a proved bound.
    """

    if not isinstance(family, PolynomialGaussianFamily):
        raise TypeError("family must be a PolynomialGaussianFamily")
    try:
        requested = tuple(_finite_position(value) for value in positions)
    except TypeError as exc:
        raise TypeError("positions must be an iterable of finite reals") from exc
    if not requested:
        raise ValueError("positions must be nonempty")
    abs_tol = _tolerance(absolute_tolerance, "absolute_tolerance")
    rel_tol = _tolerance(relative_tolerance, "relative_tolerance")
    if isinstance(hermite_order, bool) or not isinstance(hermite_order, int):
        raise ValueError("hermite_order must be an integer from 12 through 48")
    if not 12 <= hermite_order <= 48:
        raise ValueError("hermite_order must be an integer from 12 through 48")

    checks: list[VerificationCheck] = []
    for position in requested:
        for quantity in ("inverse", "autocorrelation"):
            analytic = float(
                family.inverse(position)
                if quantity == "inverse"
                else family.autocorrelation(position)
            )
            if not math.isfinite(analytic):
                raise FloatingPointError("analytic identity returned nonfinite output")
            adaptive, adaptive_error = _adaptive_integral(family, position, quantity)
            coarse = _hermite_integral(family, position, quantity, hermite_order)
            hermite = _hermite_integral(family, position, quantity, 2 * hermite_order)
            analytic_adaptive_error = abs(analytic - adaptive)
            analytic_hermite_error = abs(analytic - hermite)
            scheme_discrepancy = abs(adaptive - hermite)
            refinement_delta = abs(hermite - coarse)
            threshold = abs_tol + rel_tol * max(
                abs(analytic), abs(adaptive), abs(hermite)
            )
            passed = (
                max(
                    analytic_adaptive_error,
                    analytic_hermite_error,
                    scheme_discrepancy,
                    adaptive_error,
                    refinement_delta,
                )
                <= threshold
            )
            checks.append(
                VerificationCheck(
                    quantity=quantity,
                    position=position,
                    analytic=analytic,
                    adaptive=adaptive,
                    gauss_hermite=hermite,
                    adaptive_error_estimate=adaptive_error,
                    hermite_refinement_delta=refinement_delta,
                    analytic_adaptive_abs_error=analytic_adaptive_error,
                    analytic_hermite_abs_error=analytic_hermite_error,
                    cross_scheme_discrepancy=scheme_discrepancy,
                    acceptance_threshold=threshold,
                    passed=passed,
                )
            )
    return VerificationReport(
        family=family,
        positions=requested,
        checks=tuple(checks),
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
        hermite_base_order=hermite_order,
        hermite_reported_order=2 * hermite_order,
        passed=all(check.passed for check in checks),
    )
