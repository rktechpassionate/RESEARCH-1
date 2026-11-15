"""Independent checks for the finite prime-power contribution."""

from __future__ import annotations

import math

import mpmath as mp
import pytest

from weil_gaussian.arithmetic import prime_power_terms, truncated_prime_term


def _trial_division_von_mangoldt(n: int) -> float:
    """Small-N reference: factor each integer without using the package sieve."""
    factor = 2
    remaining = n
    distinct = []
    while factor * factor <= remaining:
        if remaining % factor == 0:
            distinct.append(factor)
            while remaining % factor == 0:
                remaining //= factor
        factor += 1
    if remaining > 1:
        distinct.append(remaining)
    return math.log(distinct[0]) if len(distinct) == 1 else 0.0


def test_prime_power_enumeration_matches_trial_division():
    terms = prime_power_terms(30, center=1.2, width=0.9)
    expected_values = [
        n for n in range(2, 31) if _trial_division_von_mangoldt(n) != 0
    ]
    assert [term.value for term in terms] == expected_values
    for term in terms:
        assert term.value == term.prime**term.exponent
        assert term.von_mangoldt == pytest.approx(
            _trial_division_von_mangoldt(term.value)
        )


def test_truncated_sum_against_trial_division_and_direct_fourier_integral():
    # This oracle uses naive factorization and the defining Fourier integral
    # for Phi. It does not call the package's prime sieve or closed form Phi.
    center = mp.mpf("1.3")
    width = mp.mpf("0.9")

    def spectrum(t):
        return (t * t + mp.mpf("0.25")) * (
            mp.exp(-((t - center) / width) ** 2 / 2)
            + mp.exp(-((t + center) / width) ** 2 / 2)
        )

    with mp.workdps(35):
        direct_terms = []
        for n in range(2, 11):
            prime_weight = _trial_division_von_mangoldt(n)
            if prime_weight == 0:
                continue
            x = mp.log(n)
            phi = mp.quad(
                lambda t: spectrum(t) ** 2 * mp.cos(t * x),
                [0, center, center + 14 * width],
            ) / mp.pi
            direct_terms.append(2 * prime_weight * phi / mp.sqrt(n))
        expected = float(mp.fsum(direct_terms))

    assert truncated_prime_term(10, 1.3, 0.9) == pytest.approx(
        expected, rel=2e-12, abs=2e-12
    )
    assert truncated_prime_term(9, 1.3, 0.9) == pytest.approx(expected)


@pytest.mark.parametrize("bad_limit", [True, 0, 1, 3.5, "10"])
def test_limit_must_be_integer_at_least_two(bad_limit):
    with pytest.raises((TypeError, ValueError)):
        prime_power_terms(bad_limit, 1.0, 1.0)
    with pytest.raises((TypeError, ValueError)):
        truncated_prime_term(bad_limit, 1.0, 1.0)


@pytest.mark.parametrize("center,width", [(-1.0, 1.0), (1.0, 0.0), (1.0, math.inf)])
def test_parameter_domain_is_enforced(center, width):
    with pytest.raises(ValueError):
        prime_power_terms(10, center, width)
    with pytest.raises(ValueError):
        truncated_prime_term(10, center, width)
