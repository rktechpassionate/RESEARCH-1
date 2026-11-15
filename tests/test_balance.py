"""Independent checks for numerical archimedean and finite-prime diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import math
import warnings

import mpmath as mp
import pytest
from scipy.integrate import IntegrationWarning

from weil_gaussian.balance import (
    archimedean_component,
    balance_diagnostic,
    finite_prime_component,
)
from weil_gaussian.families import PolynomialGaussianFamily


def _mp_spectrum(t: mp.mpf, center: float, width: float, zeros: tuple[complex, ...]):
    a, s = mp.mpf(str(center)), mp.mpf(str(width))
    polynomial = mp.mpf(1)
    for z in zeros:
        root = mp.mpc(str(z.real), str(z.imag))
        square = root * root
        if square.imag == 0:
            polynomial *= t * t - square.real
        else:
            polynomial *= (t * t - square) * (t * t - mp.conj(square))
    return polynomial * (
        mp.exp(-((t - a) / s) ** 2 / 2)
        + mp.exp(-((t + a) / s) ** 2 / 2)
    )


def _mp_integral(integrand, center: float, width: float):
    a, s = mp.mpf(str(center)), mp.mpf(str(width))
    points = [mp.mpf(0), a, a + 12 * s, mp.inf] if a else [mp.mpf(0), 12 * s, mp.inf]
    return mp.quad(integrand, points)


@pytest.mark.parametrize(
    "center,width,zeros",
    [
        (0.0, 1.0, (0.5j,)),
        (0.8, 1.2, (0.5j, 1.5j)),
        (1.1, 0.9, (0.5j, 0.7 + 0.9j)),
    ],
)
def test_archimedean_against_independent_mpmath(center, width, zeros):
    family = PolynomialGaussianFamily(center, width, zeros)
    with mp.workdps(45):
        spectrum = lambda t: _mp_spectrum(t, center, width, zeros)
        weighted = _mp_integral(
            lambda t: mp.re(mp.digamma(mp.mpf("0.25") + 0.5j * t))
            * spectrum(t) ** 2,
            center,
            width,
        )
        spectral_energy = _mp_integral(
            lambda t: spectrum(t) ** 2, center, width
        )
        expected = weighted / mp.pi - spectral_energy / mp.pi * mp.log(mp.pi)
    actual = archimedean_component(family)
    assert actual.converged
    assert actual.quadrature_error_estimate >= 0
    assert actual.value == pytest.approx(float(mp.re(expected)), rel=2e-10, abs=2e-10)


def _naive_von_mangoldt(n: int) -> float:
    factors: set[int] = set()
    candidate = 2
    remaining = n
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            factors.add(candidate)
            while remaining % candidate == 0:
                remaining //= candidate
        candidate += 1
    if remaining > 1:
        factors.add(remaining)
    return math.log(next(iter(factors))) if len(factors) == 1 else 0.0


@pytest.mark.parametrize("cutoff", [2, 3, 10])
def test_finite_prime_component_against_direct_fourier_integral(cutoff):
    center, width, zeros = 0.6, 1.1, (0.5j, 1.2j)
    family = PolynomialGaussianFamily(center, width, zeros)
    with mp.workdps(40):
        terms = []
        for n in range(2, cutoff + 1):
            von_mangoldt = _naive_von_mangoldt(n)
            if von_mangoldt == 0:
                continue
            log_n = mp.log(n)
            phi = _mp_integral(
                lambda t: _mp_spectrum(t, center, width, zeros) ** 2
                * mp.cos(t * log_n),
                center,
                width,
            ) / mp.pi
            terms.append(2 * mp.mpf(str(von_mangoldt)) * phi / mp.sqrt(n))
        expected = mp.fsum(terms)
    actual = finite_prime_component(family, cutoff)
    assert actual == pytest.approx(float(mp.re(expected)), rel=2e-10, abs=2e-10)


def test_diagnostic_keeps_finite_tail_uncertainty_explicit():
    family = PolynomialGaussianFamily(0.0, 1.0)
    result = balance_diagnostic(family, 30)
    assert result.cutoff == 30
    assert result.prime_tail_bound is None
    assert result.archimedean.converged
    assert result.finite_balance == pytest.approx(
        result.archimedean.value - result.finite_prime_term
    )
    assert asdict(result)["prime_tail_bound"] is None


@pytest.mark.parametrize("zeros", [(), (1.2j,), (0.8 + 1.1j,)])
def test_weil_balance_rejects_missing_plus_minus_i_over_two(zeros):
    family = PolynomialGaussianFamily(0.5, 1.0, zeros)
    with pytest.raises(ValueError, match="Fourier zeros"):
        archimedean_component(family)
    with pytest.raises(ValueError, match="Fourier zeros"):
        balance_diagnostic(family, 10)
    assert math.isfinite(finite_prime_component(family, 10))


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, "bad"])
def test_bad_quadrature_tolerance_is_rejected(bad):
    family = PolynomialGaussianFamily(0.0, 1.0)
    with pytest.raises(ValueError, match="abs_tol"):
        archimedean_component(family, abs_tol=bad)
    with pytest.raises(ValueError, match="rel_tol"):
        archimedean_component(family, rel_tol=bad)


@pytest.mark.parametrize("cutoff", [True, 0, 1, 2.5, "10"])
def test_bad_cutoff_is_rejected(cutoff):
    family = PolynomialGaussianFamily(0.0, 1.0)
    with pytest.raises((TypeError, ValueError)):
        finite_prime_component(family, cutoff)
    with pytest.raises((TypeError, ValueError)):
        balance_diagnostic(family, cutoff)


@pytest.mark.parametrize("value", [None, object(), (0.5j,)])
def test_bad_family_type_is_rejected(value):
    with pytest.raises(TypeError, match="PolynomialGaussianFamily"):
        archimedean_component(value)
    with pytest.raises(TypeError, match="PolynomialGaussianFamily"):
        finite_prime_component(value, 10)
    with pytest.raises(TypeError, match="PolynomialGaussianFamily"):
        balance_diagnostic(value, 10)


def test_integration_warning_is_reported_not_silently_discarded(monkeypatch):
    from weil_gaussian import balance

    def warning_quad(*args, **kwargs):
        warnings.warn("mock convergence warning", IntegrationWarning)
        return 1.0, 0.01

    monkeypatch.setattr(balance, "quad", warning_quad)
    result = balance.archimedean_component(PolynomialGaussianFamily(0, 1))
    assert not result.converged
    assert result.quadrature_error_estimate > 0
