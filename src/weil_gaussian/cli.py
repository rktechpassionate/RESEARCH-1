"""Command-line diagnostics for the Gaussian family.

The verifier compares closed forms against independent numerical Fourier
integrals. The prime command deliberately computes only a finite sum.
Neither command is a Riemann Hypothesis test.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from typing import Sequence

import mpmath as mp
import numpy as np

from .arithmetic import prime_power_terms
from .balance import balance_diagnostic
from .families import PolynomialGaussianFamily


def _finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _nonnegative_float(value: str) -> float:
    result = _finite_float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _bounded_integer(value: str, *, low: int, high: int) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not low <= result <= high:
        raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
    return result


def _dps(value: str) -> int:
    return _bounded_integer(value, low=20, high=200)


def _limit(value: str) -> int:
    return _bounded_integer(value, low=2, high=1_000_000)


def _shown(value: str) -> int:
    return _bounded_integer(value, low=0, high=100)


def _points(value: str) -> int:
    return _bounded_integer(value, low=10, high=1_000_000)


def _repeats(value: str) -> int:
    return _bounded_integer(value, low=1, high=50)


def _complex_zero(value: str) -> complex:
    try:
        result = complex(value.replace("i", "j"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a complex frequency such as 0.5j or 1+0.5j"
        ) from exc
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise argparse.ArgumentTypeError("zero must be finite")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Create the same parser used by the console script and module entry."""
    parser = argparse.ArgumentParser(
        prog="weil-gaussian",
        description="Reproducible Gaussian identity and finite prime-power diagnostics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser(
        "verify", help="compare analytic formulas with high-precision quadrature"
    )
    verify.add_argument("--center", type=_nonnegative_float, default=1.2)
    verify.add_argument("--width", type=_positive_float, default=0.9)
    verify.add_argument(
        "--zero",
        type=_complex_zero,
        action="append",
        dest="zeros",
        metavar="Z",
        help="complex zero generator; repeat to build a multi-zero family",
    )
    verify.add_argument(
        "--position",
        type=_finite_float,
        action="append",
        dest="positions",
        metavar="X",
        help="position to check; repeat for multiple positions",
    )
    verify.add_argument("--dps", type=_dps, default=40, help="quadrature digits")
    verify.add_argument(
        "--tolerance",
        type=_positive_float,
        default=1e-9,
        help="maximum acceptable absolute error",
    )
    verify.add_argument("--json", action="store_true", help="emit structured JSON")

    prime = commands.add_parser(
        "prime-sum", help="show a finite prime-power sum, without a tail bound"
    )
    prime.add_argument("--center", type=_nonnegative_float, default=1.2)
    prime.add_argument("--width", type=_positive_float, default=0.9)
    prime.add_argument(
        "--limit", type=_limit, required=True, help="integer cutoff from 2 to 1000000"
    )
    prime.add_argument(
        "--show-terms", type=_shown, default=5, help="number of initial terms to show"
    )
    prime.add_argument("--json", action="store_true", help="emit structured JSON")

    benchmark = commands.add_parser(
        "benchmark", help="time deterministic vectorized profile evaluations"
    )
    benchmark.add_argument("--center", type=_nonnegative_float, default=1.2)
    benchmark.add_argument("--width", type=_positive_float, default=0.9)
    benchmark.add_argument(
        "--zero",
        type=_complex_zero,
        action="append",
        dest="zeros",
        metavar="Z",
        help="complex zero generator; repeat to benchmark a multi-zero family",
    )
    benchmark.add_argument("--points", type=_points, default=10_001)
    benchmark.add_argument("--repeats", type=_repeats, default=5)
    benchmark.add_argument("--json", action="store_true", help="emit structured JSON")
    return parser


def _direct_integrals(
    center: float, width: float, position: float, zeros: tuple[complex, ...]
) -> tuple[mp.mpf, mp.mpf]:
    """Integrate the defining spectrum, without using either closed form."""
    a = mp.mpf(str(center))
    s = mp.mpf(str(width))
    x = mp.mpf(str(position))

    def spectrum(t: mp.mpf) -> mp.mpf:
        polynomial = mp.mpc(1)
        for zero in zeros:
            root = mp.mpc(str(zero.real), str(zero.imag)) ** 2
            polynomial *= t * t - root
            if mp.im(root) != 0:
                polynomial *= t * t - mp.conj(root)
        return mp.re(polynomial) * (
            mp.exp(-((t - a) / s) ** 2 / 2)
            + mp.exp(-((t + a) / s) ** 2 / 2)
        )

    # The finite endpoint places a Gaussian tail below working precision.
    # This is an independent numerical check, not a certified error bound.
    tail_widths = mp.sqrt(2 * (mp.mp.dps + 12) * mp.log(10))
    end = a + tail_widths * s
    if a > 8 * s:
        points = [mp.mpf("0"), a - 8 * s, a, end]
    elif a > 0:
        points = [mp.mpf("0"), a, end]
    else:
        points = [mp.mpf("0"), end]
    direct_inverse = mp.quad(lambda t: spectrum(t) * mp.cos(t * x), points) / mp.pi
    direct_correlation = (
        mp.quad(lambda t: spectrum(t) ** 2 * mp.cos(t * x), points) / mp.pi
    )
    return direct_inverse, direct_correlation


def _verify(args: argparse.Namespace) -> int:
    positions = args.positions if args.positions is not None else [0.0, 0.8, 1.4]
    family = PolynomialGaussianFamily(
        args.center, args.width, tuple(args.zeros or (0.5j,))
    )
    rows: list[dict[str, object]] = []
    with mp.workdps(args.dps):
        for x in positions:
            direct_inverse, direct_correlation = _direct_integrals(
                args.center, args.width, x, family.zeros
            )
            for name, analytic, direct in (
                ("inverse", family.inverse(x), direct_inverse),
                (
                    "autocorrelation",
                    family.autocorrelation(x),
                    direct_correlation,
                ),
            ):
                analytic_value = float(analytic)
                direct_value = float(direct)
                error = float(abs(mp.mpf(str(analytic_value)) - direct))
                analytic_value = analytic_value if math.isfinite(analytic_value) else None
                direct_value = direct_value if math.isfinite(direct_value) else None
                error = error if math.isfinite(error) else None
                passed = error is not None and error <= args.tolerance
                rows.append(
                    {
                        "quantity": name,
                        "position": x,
                        "analytic": analytic_value,
                        "quadrature": direct_value,
                        "absolute_error": error,
                        "passed": passed,
                    }
                )
    all_passed = all(bool(row["passed"]) for row in rows)
    if args.json:
        print(
            json.dumps(
                {
                    "command": "verify",
                    "center": args.center,
                    "width": args.width,
                    "zero_generators": [str(zero) for zero in family.zeros],
                    "quadrature_digits": args.dps,
                    "absolute_tolerance": args.tolerance,
                    "rows": rows,
                    "passed": all_passed,
                    "claim": "numerical identity comparison only; not an RH test",
                },
                allow_nan=False,
                indent=2,
            )
        )
    else:
        print(
            "quantity          position       analytic     quadrature       abs error  status"
        )
        for row in rows:
            analytic = row["analytic"]
            error = row["absolute_error"]
            analytic_text = f"{analytic:>13.6g}" if analytic is not None else f"{'nonfinite':>13}"
            direct = row["quadrature"]
            direct_text = f"{direct:>13.6g}" if direct is not None else f"{'nonfinite':>13}"
            error_text = f"{error:>13.3e}" if error is not None else f"{'nonfinite':>13}"
            print(
                f"{row['quantity']:<16} {row['position']:>8.3g}  "
                f"{analytic_text}  {direct_text}  "
                f"{error_text}  {'PASS' if row['passed'] else 'FAIL'}"
            )
        print(
            f"{'PASS' if all_passed else 'FAIL'}: "
            f"{len(rows)} comparisons at {args.dps} digits; "
            "numerical check only, not an RH test."
        )
    return 0 if all_passed else 1


def _prime_sum(args: argparse.Namespace) -> int:
    terms = prime_power_terms(args.limit, args.center, args.width)
    diagnostic = balance_diagnostic(
        PolynomialGaussianFamily(args.center, args.width), args.limit
    )
    finite_sum = diagnostic.finite_prime_term
    shown = terms[: args.show_terms]
    if args.json:
        print(
            json.dumps(
                {
                    "command": "prime-sum",
                    "center": args.center,
                    "width": args.width,
                    "cutoff": args.limit,
                    "prime_power_count": len(terms),
                    "finite_prime_term": finite_sum,
                    "archimedean_term": diagnostic.archimedean.value,
                    "quadrature_error_estimate": diagnostic.archimedean.quadrature_error_estimate,
                    "quadrature_converged": diagnostic.archimedean.converged,
                    "finite_balance": diagnostic.finite_balance,
                    "prime_tail_bound": None,
                    "shown_terms": [
                        {
                            "value": term.value,
                            "prime": term.prime,
                            "exponent": term.exponent,
                            "von_mangoldt": term.von_mangoldt,
                            "contribution": term.contribution,
                        }
                        for term in shown
                    ],
                    "omitted_display_terms": len(terms) - len(shown),
                    "no_tail_bound": True,
                    "claim": "cutoff-dependent balance only; no prime-tail bound or RH test",
                },
                allow_nan=False,
                indent=2,
            )
        )
    else:
        print(
            f"Finite prime-power diagnostic: center={args.center:g}, "
            f"width={args.width:g}, cutoff={args.limit}"
        )
        print(f"Prime-power terms through cutoff: {len(terms)}")
        print(
            "2 * sum Lambda(n) Phi(log n) / sqrt(n) "
            f"= {finite_sum:.12g}"
        )
        print(f"Archimedean K = {diagnostic.archimedean.value:.12g}")
        print(f"Finite P_N = {finite_sum:.12g}")
        print(f"Cutoff-dependent K - P_N = {diagnostic.finite_balance:.12g}")
        print(
            "Quadrature error estimate = "
            f"{diagnostic.archimedean.quadrature_error_estimate:.3g}; "
            f"converged={diagnostic.archimedean.converged}"
        )
        if shown:
            print("       n       p   k    contribution")
            for term in shown:
                print(
                    f"{term.value:>8} {term.prime:>7} {term.exponent:>3} "
                    f"{term.contribution:>15.7g}"
                )
        print("No prime-tail bound; not the full Weil form or an RH test.")
    return 0 if diagnostic.archimedean.converged else 1


def _benchmark(args: argparse.Namespace) -> int:
    """Measure wall-clock times on a fixed grid, without making speed claims."""
    grid = np.linspace(-8.0, 8.0, args.points)
    family = PolynomialGaussianFamily(
        args.center, args.width, tuple(args.zeros or (0.5j,))
    )
    operations = (
        ("fourier_profile", family.fourier),
        ("inverse_profile", family.inverse),
        ("autocorrelation", family.autocorrelation),
    )
    measurements: list[dict[str, object]] = []
    for name, function in operations:
        function(grid)  # warm up imports and NumPy dispatch
        samples: list[float] = []
        last_values = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            last_values = function(grid)
            samples.append(time.perf_counter() - started)
        assert last_values is not None
        measurements.append(
            {
                "function": name,
                "median_seconds": statistics.median(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
                "checksum": float(np.sum(last_values, dtype=np.float64)),
            }
        )
    report = {
        "command": "benchmark",
        "center": args.center,
        "width": args.width,
        "zero_generators": [str(zero) for zero in family.zeros],
        "points": args.points,
        "repeats": args.repeats,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "grid": "linspace(-8, 8, points)",
        "measurements": measurements,
        "note": "wall-clock measurements on this machine; not comparable across machines",
    }
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2))
    else:
        print(
            f"Deterministic grid: {args.points} points in [-8, 8]; "
            f"center={args.center:g}, width={args.width:g}, repeats={args.repeats}"
        )
        print(f"Python {report['python_version']}; NumPy {report['numpy_version']}")
        print("function                 median (ms)       min (ms)       max (ms)")
        for row in measurements:
            print(
                f"{row['function']:<24} "
                f"{1000 * row['median_seconds']:>11.3f} "
                f"{1000 * row['min_seconds']:>14.3f} "
                f"{1000 * row['max_seconds']:>14.3f}"
            )
        print("Wall-clock measurements on this machine; not comparable across machines.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return an exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            return _verify(args)
        if args.command == "prime-sum":
            return _prime_sum(args)
        if args.command == "benchmark":
            return _benchmark(args)
    except (ArithmeticError, OverflowError, ValueError) as exc:
        print(f"numerical evaluation failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")
