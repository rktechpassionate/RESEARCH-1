"""Outward-rounded analytic bounds for spatial Gaussian tails.

The certificate bounds an exact mathematical family whose Python float inputs
are interpreted as binary rationals. It does not bound float64 roundoff, a
frequency-domain quadrature cutoff, or the infinite prime-power tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, DecimalException, localcontext
from fractions import Fraction
from typing import Literal, TypeAlias

from .families import PolynomialGaussianFamily

SpatialQuantity: TypeAlias = Literal["inverse", "autocorrelation"]
_RComplex: TypeAlias = tuple[Fraction, Fraction]
_ZERO: _RComplex = (Fraction(0), Fraction(0))
_ONE: _RComplex = (Fraction(1), Fraction(0))


@dataclass(frozen=True)
class TailCertificate:
    """Upper bound on the integral of an exact spatial profile outside ±R.

    The independently auditable inequality is

        integral_(|x|>R) |profile(x)| dx
            <= rational_prefactor * exp(-gaussian_rate * R**2).

    The decimal bound encloses its right-hand side from above. It does not
    enclose a float64 evaluation of the profile.
    """

    quantity: SpatialQuantity
    radius: float
    bound_decimal: str
    working_digits: int
    envelope_degree: int
    radius_ratio: str
    gaussian_rate_ratio: str
    rational_prefactor_ratio: str
    method: str = "exact-rational derivative envelope, Gaussian Mills recurrence, outward-rounded Decimal"
    certified_analytic_bound: bool = True
    prime_tail_bound: None = None


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _add(left: _RComplex, right: _RComplex) -> _RComplex:
    return left[0] + right[0], left[1] + right[1]


def _scale(value: _RComplex, factor: Fraction) -> _RComplex:
    return value[0] * factor, value[1] * factor


def _real_polynomial_product(
    left: tuple[Fraction, ...], right: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, first in enumerate(left):
        for j, second in enumerate(right):
            result[i + j] += first * second
    return tuple(result)


def _exact_polynomial(family: PolynomialGaussianFamily) -> tuple[Fraction, ...]:
    """Construct P(y) directly from exact float inputs, without float products."""
    coefficients: tuple[Fraction, ...] = (Fraction(1),)
    for zero in family.zeros:
        real = Fraction.from_float(zero.real)
        imaginary = Fraction.from_float(zero.imag)
        squared_real = real * real - imaginary * imaginary
        squared_imaginary = 2 * real * imaginary
        factor: tuple[Fraction, ...]
        if squared_imaginary == 0:
            factor = (-squared_real, Fraction(1))
        else:
            factor = (
                squared_real * squared_real + squared_imaginary * squared_imaginary,
                -2 * squared_real,
                Fraction(1),
            )
        coefficients = _real_polynomial_product(coefficients, factor)
    return coefficients


def _next_derivative(
    coefficients: list[_RComplex], beta: Fraction, center: Fraction
) -> list[_RComplex]:
    """q_(n+1)=q_n'+(-2 beta*x+i center)q_n, exactly over Q+iQ."""
    result = [_ZERO for _ in range(len(coefficients) + 1)]
    for index, value in enumerate(coefficients):
        if index:
            result[index - 1] = _add(result[index - 1], _scale(value, Fraction(index)))
        result[index] = _add(result[index], (-center * value[1], center * value[0]))
        result[index + 1] = _add(result[index + 1], _scale(value, -2 * beta))
    return result


def _operator_polynomial(
    coefficients: tuple[Fraction, ...], beta: Fraction, center: Fraction
) -> tuple[_RComplex, ...]:
    """Polynomial multiplying exp(-beta*x²+i*center*x) after P(-D²)."""
    degree = 2 * (len(coefficients) - 1)
    result = [_ZERO for _ in range(degree + 1)]
    derivative = [_ONE]
    for order in range(degree + 1):
        if order % 2 == 0:
            index = order // 2
            weight = coefficients[index] * (-1 if index % 2 else 1)
            for power, value in enumerate(derivative):
                result[power] = _add(result[power], _scale(value, weight))
        if order < degree:
            derivative = _next_derivative(derivative, beta, center)
    return tuple(result)


def _envelope(
    family: PolynomialGaussianFamily, quantity: SpatialQuantity
) -> tuple[Fraction, Fraction, tuple[Fraction, ...]]:
    width = Fraction.from_float(family.width)
    center = Fraction.from_float(family.center)
    polynomial = _exact_polynomial(family)
    if quantity == "inverse":
        beta = width * width / 2
        # sqrt(2/pi) < 4/5 follows from the classical pi>25/8.
        constant = Fraction(4, 5) * width
        transformed = _operator_polynomial(polynomial, beta, center)
        envelope = tuple(abs(real) + abs(imag) for real, imag in transformed)
    else:
        beta = width * width / 4
        # 1/sqrt(pi) < 3/5 follows from pi>3>25/9.
        constant = Fraction(3, 5) * width
        squared = _real_polynomial_product(polynomial, polynomial)
        shifted = _operator_polynomial(squared, beta, center)
        unshifted = _operator_polynomial(squared, beta, Fraction(0))
        # The cross-Gaussian attenuation exp(-a²/s²) is at most one.
        envelope = tuple(
            abs(first[0]) + abs(first[1]) + abs(second[0]) + abs(second[1])
            for first, second in zip(shifted, unshifted, strict=True)
        )
    return beta, constant, envelope


def _tail_prefactor(
    coefficients: tuple[Fraction, ...],
    beta: Fraction,
    constant: Fraction,
    radius: Fraction,
) -> Fraction:
    """Return exact A with two-sided L1 tail <= A*exp(-beta*R²).

    Mills' inequality gives J0 <= exp(-beta R²)/(2 beta R), while J1 is
    exp(-beta R²)/(2 beta). Integration by parts supplies the recurrence
    Jk = R^(k-1) exp(-beta R²)/(2 beta) + (k-1)J_(k-2)/(2 beta).
    """
    moments = [Fraction(1, 1) / (2 * beta * radius)]
    if len(coefficients) >= 2:
        moments.append(Fraction(1, 1) / (2 * beta))
    for order in range(2, len(coefficients)):
        moments.append(
            radius ** (order - 1) / (2 * beta)
            + Fraction(order - 1, 1) * moments[order - 2] / (2 * beta)
        )
    return (
        2
        * constant
        * sum(
            (
                coefficient * moment
                for coefficient, moment in zip(coefficients, moments, strict=True)
            ),
            Fraction(0),
        )
    )


def _outward_decimal(prefactor: Fraction, exponent: Fraction, digits: int) -> str:
    """Enclose prefactor*exp(-exponent) using Decimal's correct exp rounding."""
    try:
        with localcontext() as context:
            context.prec = digits
            context.rounding = ROUND_CEILING
            context.Emin = -999999
            context.Emax = 999999
            # Rounding a negative exponent toward +infinity is essential.
            exponent_argument = Decimal(-exponent.numerator) / Decimal(
                exponent.denominator
            )
            # Decimal.exp rounds correctly (ROUND_HALF_EVEN); next_plus
            # gives an upper endpoint.
            exponential_upper = exponent_argument.exp().next_plus(context)
            prefactor_upper = Decimal(prefactor.numerator) / Decimal(
                prefactor.denominator
            )
            upper = (prefactor_upper * exponential_upper).next_plus(context)
            if not upper.is_finite() or upper <= 0:
                raise ValueError("tail bound is outside the decimal range")
            return str(upper)
    except DecimalException as exc:
        raise ValueError("tail bound is outside the decimal range") from exc


def certify_spatial_tail(
    family: PolynomialGaussianFamily,
    radius: float,
    *,
    quantity: SpatialQuantity = "inverse",
    digits: int = 80,
) -> TailCertificate:
    """Bound the L1 tail of f(x) or Phi(x) over |x|>R.

    The exact family uses the supplied binary floats interpreted as rationals.
    This bounds x-domain truncation only; it does not certify roundoff in
    float64 evaluation, a frequency-domain quadrature cutoff, or a truncated
    prime-power sum. The rational fields identify the exact inequality, and
    bound_decimal is its outward-rounded upper endpoint.
    """
    if not isinstance(family, PolynomialGaussianFamily):
        raise TypeError("family must be a PolynomialGaussianFamily")
    if quantity not in ("inverse", "autocorrelation"):
        raise ValueError("quantity must be 'inverse' or 'autocorrelation'")
    if (
        isinstance(digits, bool)
        or not isinstance(digits, int)
        or not 32 <= digits <= 200
    ):
        raise ValueError("digits must be an integer from 32 to 200")
    try:
        radius_float = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("radius must be a finite positive real number") from exc
    if isinstance(radius, bool) or not math.isfinite(radius_float) or radius_float <= 0:
        raise ValueError("radius must be a finite positive real number")
    radius_exact = Fraction.from_float(radius_float)
    beta, constant, coefficients = _envelope(family, quantity)
    prefactor = _tail_prefactor(coefficients, beta, constant, radius_exact)
    if prefactor <= 0:
        raise ValueError("tail envelope must have a positive prefactor")
    bound = _outward_decimal(prefactor, beta * radius_exact * radius_exact, digits)
    return TailCertificate(
        quantity=quantity,
        radius=radius_float,
        bound_decimal=bound,
        working_digits=digits,
        envelope_degree=len(coefficients) - 1,
        radius_ratio=_ratio(radius_exact),
        gaussian_rate_ratio=_ratio(beta),
        rational_prefactor_ratio=_ratio(prefactor),
    )
