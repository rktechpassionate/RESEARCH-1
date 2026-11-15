"""Finite prime-power sums associated with the Gaussian autocorrelation.

Only the *truncated* arithmetic side is computed here. In the Fourier
convention used by this package, the standard explicit-formula prime term
has coefficient 2. No tail bound, zero sum, or RH test is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Iterator

from .core import _parameters, autocorrelation


@dataclass(frozen=True)
class PrimePowerTerm:
    """One term in a finite prime-power sum.

    ``value = prime**exponent`` and ``von_mangoldt = log(prime)``.
    ``contribution`` includes the explicit-formula factor 2.
    """

    value: int
    prime: int
    exponent: int
    von_mangoldt: float
    contribution: float


def _cutoff(limit: int) -> int:
    if isinstance(limit, bool):
        raise TypeError("limit must be an integer")
    try:
        cutoff = operator.index(limit)
    except TypeError as exc:
        raise TypeError("limit must be an integer") from exc
    if cutoff < 2:
        raise ValueError("limit must be at least 2")
    return cutoff


def _primes_through(limit: int) -> Iterator[int]:
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for prime in range(2, math.isqrt(limit) + 1):
        if sieve[prime]:
            start = prime * prime
            sieve[start::prime] = b"\x00" * ((limit - start) // prime + 1)
    return (value for value in range(2, limit + 1) if sieve[value])


def _terms(limit: int, center: float, width: float) -> Iterator[PrimePowerTerm]:
    for prime in _primes_through(limit):
        prime_log = math.log(prime)
        value = prime
        exponent = 1
        while value <= limit:
            contribution = (
                2.0
                * prime_log
                / math.sqrt(value)
                * autocorrelation(math.log(value), center, width)
            )
            yield PrimePowerTerm(value, prime, exponent, prime_log, contribution)
            if value > limit // prime:
                break
            value *= prime
            exponent += 1


def prime_power_terms(
    limit: int, center: float, width: float
) -> tuple[PrimePowerTerm, ...]:
    r"""Return all terms with ``prime**exponent <= limit``, sorted by value.

    The returned values belong to the finite sum

    ``2 * sum_{2 <= n <= limit} Lambda(n) / sqrt(n) * Phi(log(n))``.

    ``Lambda(n)`` is ``log(p)`` when ``n=p**k`` for a prime ``p`` and integer
    ``k>=1``, and zero otherwise. This function does not estimate the omitted
    infinite tail.
    """
    limit = _cutoff(limit)
    center, width = _parameters(center, width)
    return tuple(sorted(_terms(limit, center, width), key=lambda term: term.value))


def truncated_prime_term(limit: int, center: float, width: float) -> float:
    r"""Compute the finite prime term through the integer ``limit``.

    This returns ``2 * sum_{n<=limit} Lambda(n) Phi(log(n)) / sqrt(n)``.
    Because no bound is provided for ``n>limit``, this result alone cannot be
    substituted for the full explicit formula or used to test RH.
    """
    limit = _cutoff(limit)
    center, width = _parameters(center, width)
    return math.fsum(term.contribution for term in _terms(limit, center, width))
