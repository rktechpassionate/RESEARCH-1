"""Compare closed forms with independent Fourier integrals.

Run after ``pip install -e '.[test]'``. This checks Gaussian identities only;
it does not test a zeta zero criterion or the Riemann Hypothesis.
"""

from __future__ import annotations

import math

import mpmath as mp

from weil_gaussian import autocorrelation, inverse_profile

CASES = [(0.0, 1.0, 0.8), (2.5, 0.7, 0.3), (5.0, 2.0, 1.2)]


def direct_fourier_integrals(center: float, width: float, position: float):
    """Integrate the defining spectrum without using library closed forms."""
    a, s, x = map(lambda value: mp.mpf(str(value)), (center, width, position))

    def spectrum(frequency):
        return (frequency**2 + mp.mpf("0.25")) * (
            mp.exp(-(((frequency - a) / s) ** 2) / 2)
            + mp.exp(-(((frequency + a) / s) ** 2) / 2)
        )

    # Gaussian tails beyond this cutoff are below the precision used below.
    cutoff = a + 14 * s
    intervals = [mp.mpf("0"), cutoff] if a == 0 else [mp.mpf("0"), a, cutoff]
    inverse = mp.quad(lambda t: spectrum(t) * mp.cos(t * x), intervals) / mp.pi
    correlation = mp.quad(lambda t: spectrum(t) ** 2 * mp.cos(t * x), intervals) / mp.pi
    return inverse, correlation


def main() -> None:
    mp.mp.dps = 40
    print("center  width  position  inverse error  autocorrelation error")
    for center, width, position in CASES:
        direct_inverse, direct_correlation = direct_fourier_integrals(
            center, width, position
        )
        inverse_error = abs(inverse_profile(position, center, width) - direct_inverse)
        correlation_error = abs(
            autocorrelation(position, center, width) - direct_correlation
        )
        print(
            f"{center:6g}  {width:5g}  {position:8g}  "
            f"{float(inverse_error):13.3e}  {float(correlation_error):21.3e}"
        )
        if not all(
            math.isfinite(float(error)) and error < mp.mpf("1e-8")
            for error in (inverse_error, correlation_error)
        ):
            raise SystemExit("Closed form differs from the defining integral")


if __name__ == "__main__":
    main()
