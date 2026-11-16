"""Independent checks of outward-rounded spatial Gaussian tail bounds."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict
from decimal import Decimal
from fractions import Fraction
from typing import Any

import mpmath as mp
import pytest

from weil_gaussian.certification import (
    _envelope,
    certify_spatial_tail,
)
from weil_gaussian.families import PolynomialGaussianFamily


def _as_mp(value: Fraction) -> Any:
    return mp.mpf(value.numerator) / value.denominator


def _ratio_mp(value: str) -> Any:
    numerator, denominator = value.split("/")
    return mp.mpf(numerator) / mp.mpf(denominator)


def _exact_profile(family: PolynomialGaussianFamily, quantity: str, x: Any) -> Any:
    """Use the paper's elementary formulas, independently of the envelope."""
    a = mp.mpf(family.center)
    s = mp.mpf(family.width)
    if not family.zeros:
        if quantity == "inverse":
            return s * mp.sqrt(2 / mp.pi) * mp.exp(-s * s * x * x / 2) * mp.cos(a * x)
        return (
            s
            / mp.sqrt(mp.pi)
            * mp.exp(-s * s * x * x / 4)
            * (mp.cos(a * x) + mp.exp(-a * a / (s * s)))
        )
    assert family.zeros == (0.5j,)
    if quantity == "inverse":
        return (
            s
            * mp.sqrt(2 / mp.pi)
            * mp.exp(-s * s * x * x / 2)
            * (
                (a * a + s * s + mp.mpf("0.25") - s**4 * x * x) * mp.cos(a * x)
                - 2 * a * s * s * x * mp.sin(a * x)
            )
        )

    def polynomial(z: Any) -> Any:
        return (
            z**4
            - (3 * s * s + mp.mpf("0.5")) * z * z
            + 3 * s**4 / 4
            + s * s / 4
            + mp.mpf("0.0625")
        )

    shifted = 1j * a - s * s * x / 2
    unshifted = -s * s * x / 2
    return (
        s
        / mp.sqrt(mp.pi)
        * mp.exp(-s * s * x * x / 4)
        * (
            mp.re(mp.exp(1j * a * x) * polynomial(shifted))
            + mp.exp(-a * a / (s * s)) * polynomial(unshifted)
        )
    )


@pytest.mark.parametrize("zeros", [(), (0.5j,)])
@pytest.mark.parametrize("quantity", ["inverse", "autocorrelation"])
@pytest.mark.parametrize("radius", [2.0, 4.0])
def test_certificate_exceeds_independent_high_precision_spatial_tail(
    zeros: tuple[complex, ...], quantity: str, radius: float
) -> None:
    family = PolynomialGaussianFamily(0.0, 1.0, zeros)
    certificate = certify_spatial_tail(family, radius, quantity=quantity)
    with mp.workdps(70):
        direct = 2 * mp.quad(
            lambda x: abs(_exact_profile(family, quantity, x)),
            [radius, radius + 3, mp.inf],
        )
        assert mp.mpf(certificate.bound_decimal) > direct


@pytest.mark.parametrize(
    "family",
    [
        PolynomialGaussianFamily(0.0, 1.0, ()),
        PolynomialGaussianFamily(0.75, 1.25, (0.5j,)),
        PolynomialGaussianFamily(0.75, 1.25, (0.5j, 0.5 + 0.25j)),
    ],
)
@pytest.mark.parametrize("quantity", ["inverse", "autocorrelation"])
@pytest.mark.parametrize("radius", [0.5, 3.0, 8.0])
def test_decimal_endpoint_encloses_exact_rational_gaussian_expression(
    family: PolynomialGaussianFamily, quantity: str, radius: float
) -> None:
    certificate = certify_spatial_tail(family, radius, quantity=quantity)
    with mp.workdps(140):
        prefactor = _ratio_mp(certificate.rational_prefactor_ratio)
        rate = _ratio_mp(certificate.gaussian_rate_ratio)
        exact_radius = _ratio_mp(certificate.radius_ratio)
        target = prefactor * mp.exp(-rate * exact_radius * exact_radius)
        assert mp.mpf(certificate.bound_decimal) >= target


@pytest.mark.parametrize("quantity", ["inverse", "autocorrelation"])
def test_multizero_envelope_exceeds_independent_fourier_integral(
    quantity: str,
) -> None:
    family = PolynomialGaussianFamily(0.75, 1.0, (0.5j, 0.5 + 0.25j))
    rate, constant, coefficients = _envelope(family, quantity)
    with mp.workdps(65):
        a = mp.mpf(family.center)
        s = mp.mpf(family.width)

        def spectrum(t: Any) -> Any:
            square = t * t
            polynomial = mp.mpc(1)
            for zero in family.zeros:
                root = mp.mpc(zero.real, zero.imag) ** 2
                polynomial *= square - root
                if mp.im(root) != 0:
                    polynomial *= square - mp.conj(root)
            gaussians = mp.exp(-(((t - a) / s) ** 2) / 2) + mp.exp(
                -(((t + a) / s) ** 2) / 2
            )
            return mp.re(polynomial) * gaussians

        for x in (mp.mpf("0.5"), mp.mpf("2.0")):

            def integrand(t: Any, position: Any = x) -> Any:
                profile = spectrum(t)
                if quantity == "autocorrelation":
                    profile *= profile
                return profile * mp.cos(t * position)

            direct = mp.quad(integrand, [0, a, a + 8 * s, mp.inf]) / mp.pi
            envelope = (
                _as_mp(constant)
                * mp.exp(-_as_mp(rate) * x * x)
                * sum(
                    _as_mp(value) * x**degree
                    for degree, value in enumerate(coefficients)
                )
            )
            assert abs(direct) < envelope


@pytest.mark.parametrize("quantity", ["inverse", "autocorrelation"])
def test_selected_large_radius_bounds_decrease(quantity: str) -> None:
    family = PolynomialGaussianFamily(0.75, 1.25, (0.5j, 0.5 + 0.25j))
    bounds = [
        Decimal(certify_spatial_tail(family, radius, quantity=quantity).bound_decimal)
        for radius in (2.0, 5.0, 10.0, 20.0)
    ]
    assert all(
        left > right > 0 for left, right in zip(bounds, bounds[1:], strict=False)
    )


def test_rational_certificate_identity_and_metadata() -> None:
    family = PolynomialGaussianFamily(0.0, 1.0, ())
    inverse = certify_spatial_tail(family, 2.0)
    correlation = certify_spatial_tail(family, 2.0, quantity="autocorrelation")
    assert inverse.gaussian_rate_ratio == "1/2"
    assert inverse.rational_prefactor_ratio == "4/5"
    assert correlation.gaussian_rate_ratio == "1/4"
    assert correlation.rational_prefactor_ratio == "12/5"
    assert inverse.radius_ratio == "2/1"
    assert inverse.envelope_degree == 0
    assert correlation.envelope_degree == 0
    assert inverse.certified_analytic_bound is True
    assert inverse.prime_tail_bound is None
    assert (
        json.loads(json.dumps(asdict(inverse)))["bound_decimal"]
        == inverse.bound_decimal
    )
    with pytest.raises(FrozenInstanceError):
        inverse.radius = 3.0  # type: ignore[misc]


@pytest.mark.parametrize("digits", [32, 80, 200])
def test_requested_decimal_precision_still_encloses_exact_expression(
    digits: int,
) -> None:
    family = PolynomialGaussianFamily(1.25, 0.75, (0.5j,))
    certificate = certify_spatial_tail(family, 5.0, digits=digits)
    with mp.workdps(260):
        target = _ratio_mp(certificate.rational_prefactor_ratio) * mp.exp(
            -_ratio_mp(certificate.gaussian_rate_ratio)
            * _ratio_mp(certificate.radius_ratio) ** 2
        )
        assert mp.mpf(certificate.bound_decimal) > target


def test_extreme_gaussian_decay_keeps_a_positive_upper_endpoint() -> None:
    family = PolynomialGaussianFamily(0.0, 1e200, ())
    certificate = certify_spatial_tail(family, 1.0)
    upper = Decimal(certificate.bound_decimal)
    assert upper > 0
    assert upper.adjusted() < -1000


def test_decimal_underflow_endpoint_still_encloses_the_true_expression() -> None:
    certificate = certify_spatial_tail(PolynomialGaussianFamily(0.0, 1.0, ()), 5000.0)
    with mp.workdps(100):
        exact = _ratio_mp(certificate.rational_prefactor_ratio) * mp.exp(
            -_ratio_mp(certificate.gaussian_rate_ratio)
            * _ratio_mp(certificate.radius_ratio) ** 2
        )
        assert mp.mpf(certificate.bound_decimal) > exact > 0


def test_subnormal_width_produces_finite_certificate() -> None:
    family = PolynomialGaussianFamily(0.0, math.nextafter(0.0, 1.0), ())
    certificate = certify_spatial_tail(family, 1.0)
    assert Decimal(certificate.bound_decimal).is_finite()
    assert Decimal(certificate.bound_decimal) > 0


@pytest.mark.parametrize(
    "radius", [0.0, -1.0, math.nan, math.inf, -math.inf, True, None]
)
def test_invalid_radius_is_rejected(radius: Any) -> None:
    with pytest.raises(ValueError, match="radius"):
        certify_spatial_tail(PolynomialGaussianFamily(0.0, 1.0), radius)


@pytest.mark.parametrize("digits", [31, 201, True, 80.0, "80", None])
def test_invalid_precision_is_rejected(digits: Any) -> None:
    with pytest.raises(ValueError, match="digits"):
        certify_spatial_tail(PolynomialGaussianFamily(0.0, 1.0), 2.0, digits=digits)


def test_invalid_quantity_and_family_are_rejected() -> None:
    family = PolynomialGaussianFamily(0.0, 1.0)
    with pytest.raises(ValueError, match="quantity"):
        certify_spatial_tail(family, 2.0, quantity="mass")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="family"):
        certify_spatial_tail(object(), 2.0)  # type: ignore[arg-type]
