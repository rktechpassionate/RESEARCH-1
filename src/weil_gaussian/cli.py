"""Command-line diagnostics for the Gaussian family.

The verifier compares closed forms against independent numerical Fourier
integrals. The prime command deliberately computes only a finite sum.
Neither command is a Riemann Hypothesis test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict

import mpmath as mp
import numpy as np

from .arithmetic import prime_power_terms
from .balance import balance_diagnostic
from .certification import certify_spatial_tail
from .families import PolynomialGaussianFamily
from .verification import verify_family


class VerifyRow(TypedDict):
    """One analytic versus direct quadrature comparison."""

    quantity: str
    position: float
    analytic: float | None
    quadrature: float | None
    absolute_error: float | None
    adaptive: float
    gauss_hermite: float
    adaptive_error_estimate: float
    hermite_refinement_delta: float
    analytic_adaptive_abs_error: float
    analytic_hermite_abs_error: float
    cross_scheme_discrepancy: float
    acceptance_threshold: float
    scheme_passed: bool
    finite_cutoff_tail_bound: None
    passed: bool


class BenchmarkMeasurement(TypedDict):
    """Wall-clock measurements for one vectorized operation."""

    function: str
    median_seconds: float
    min_seconds: float
    max_seconds: float
    checksum: float


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
        "--hermite-order",
        type=lambda value: _bounded_integer(value, low=12, high=48),
        default=48,
        help="base Gauss-Hermite order (reported result uses twice this order)",
    )
    verify.add_argument(
        "--tolerance",
        type=_positive_float,
        default=1e-9,
        help="maximum acceptable absolute error",
    )
    verify.add_argument("--json", action="store_true", help="emit structured JSON")
    verify.add_argument(
        "--tail-radius",
        type=_positive_float,
        metavar="R",
        help="certify exact-model spatial L1 tails outside |x| > R",
    )

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

    extremal = commands.add_parser(
        "extremal",
        help="optimize a finite-cutoff Weil Rayleigh quotient on a Gaussian basis",
    )
    extremal.add_argument("--center", type=_nonnegative_float, default=0.4)
    extremal.add_argument("--width", type=_positive_float, default=0.8)
    extremal.add_argument(
        "--cutoff",
        type=lambda value: _bounded_integer(value, low=2, high=1000),
        action="append",
        dest="cutoffs",
        metavar="N",
        help="finite prime-power cutoff; repeat (default: 20 and 50)",
    )
    extremal.add_argument(
        "--zero",
        type=_complex_zero,
        action="append",
        dest="prescribed_zeros",
        metavar="Z",
        help="complex frequency where the optimal combination must vanish; repeat",
    )
    extremal.add_argument("--json", action="store_true", help="emit structured JSON")

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

    reproduce = commands.add_parser(
        "reproduce", help="regenerate deterministic paper tables and diagnostics"
    )
    reproduce.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="write reproducibility.json and reproducibility.md to DIR",
    )
    reproduce.add_argument(
        "--check",
        action="store_true",
        help="check regenerated golden vectors against the committed fixture",
    )
    reproduce.add_argument(
        "--golden-file",
        type=Path,
        metavar="PATH",
        help="fixture to compare with --check (default: packaged golden vectors)",
    )
    reproduce.add_argument(
        "--json", action="store_true", help="emit structured JSON instead of Markdown"
    )
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
            mp.exp(-(((t - a) / s) ** 2) / 2) + mp.exp(-(((t + a) / s) ** 2) / 2)
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
    two_scheme = verify_family(
        family,
        positions,
        absolute_tolerance=args.tolerance,
        relative_tolerance=0.0,
        hermite_order=args.hermite_order,
    )
    rows: list[VerifyRow] = []
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
                analytic_float = float(analytic)
                direct_float = float(direct)
                error_float = float(abs(mp.mpf(str(analytic_float)) - direct))
                analytic_value = (
                    analytic_float if math.isfinite(analytic_float) else None
                )
                direct_value = direct_float if math.isfinite(direct_float) else None
                error = error_float if math.isfinite(error_float) else None
                scheme_check = two_scheme.checks[len(rows)]
                if scheme_check.quantity != name or scheme_check.position != x:
                    raise AssertionError("verification scheme row order changed")
                passed = (
                    error is not None
                    and error <= args.tolerance
                    and scheme_check.passed
                )
                rows.append(
                    {
                        "quantity": name,
                        "position": x,
                        "analytic": analytic_value,
                        "quadrature": direct_value,
                        "absolute_error": error,
                        "adaptive": scheme_check.adaptive,
                        "gauss_hermite": scheme_check.gauss_hermite,
                        "adaptive_error_estimate": scheme_check.adaptive_error_estimate,
                        "hermite_refinement_delta": scheme_check.hermite_refinement_delta,
                        "analytic_adaptive_abs_error": scheme_check.analytic_adaptive_abs_error,
                        "analytic_hermite_abs_error": scheme_check.analytic_hermite_abs_error,
                        "cross_scheme_discrepancy": scheme_check.cross_scheme_discrepancy,
                        "acceptance_threshold": scheme_check.acceptance_threshold,
                        "scheme_passed": scheme_check.passed,
                        "finite_cutoff_tail_bound": None,
                        "passed": passed,
                    }
                )
    all_passed = all(bool(row["passed"]) for row in rows)
    spatial_tails = (
        [
            asdict(certify_spatial_tail(family, args.tail_radius, quantity=quantity))
            for quantity in ("inverse", "autocorrelation")
        ]
        if args.tail_radius is not None
        else []
    )
    if args.json:
        print(
            json.dumps(
                {
                    "command": "verify",
                    "center": args.center,
                    "width": args.width,
                    "zero_generators": [str(zero) for zero in family.zeros],
                    "quadrature_digits": args.dps,
                    "gauss_hermite_base_order": two_scheme.hermite_base_order,
                    "gauss_hermite_reported_order": two_scheme.hermite_reported_order,
                    "absolute_tolerance": args.tolerance,
                    "relative_tolerance": 0.0,
                    "schemes": [
                        "mpmath finite-endpoint quadrature",
                        "adaptive Gauss-Kronrod on [0, infinity)",
                        "Gauss-Hermite on the Gaussian decomposition",
                    ],
                    "rows": rows,
                    "spatial_tail_certificates": spatial_tails,
                    "passed": all_passed,
                    "error_certification": (
                        "observed discrepancies and integration estimates; "
                        "not rigorous enclosures"
                    ),
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
            analytic_text = (
                f"{analytic:>13.6g}" if analytic is not None else f"{'nonfinite':>13}"
            )
            direct = row["quadrature"]
            direct_text = (
                f"{direct:>13.6g}" if direct is not None else f"{'nonfinite':>13}"
            )
            error_text = (
                f"{error:>13.3e}" if error is not None else f"{'nonfinite':>13}"
            )
            print(
                f"{row['quantity']:<16} {row['position']:>8.3g}  "
                f"{analytic_text}  {direct_text}  "
                f"{error_text}  {'PASS' if row['passed'] else 'FAIL'}"
            )
        print(
            "\nIndependent schemes: adaptive Gauss-Kronrod and Gauss-Hermite "
            f"(orders {two_scheme.hermite_base_order}/{two_scheme.hermite_reported_order})"
        )
        print(
            "quantity          position     adaptive      Hermite   "
            "adaptive err   Hermite err  status"
        )
        for row in rows:
            print(
                f"{row['quantity']:<16} {row['position']:>8.3g} "
                f"{row['adaptive']:>12.6g} {row['gauss_hermite']:>12.6g} "
                f"{row['analytic_adaptive_abs_error']:>14.3e} "
                f"{row['analytic_hermite_abs_error']:>13.3e} "
                f"{'PASS' if row['scheme_passed'] else 'FAIL'}"
            )
        print(
            "Errors and refinement estimates are numerical indicators, not rigorous bounds."
        )
        for certificate in spatial_tails:
            print(
                f"Certified exact-model spatial L1 tail |x| > {certificate['radius']:g} "
                f"for {certificate['quantity']}: "
                f"<= {certificate['bound_decimal']}"
            )
        if spatial_tails:
            print(
                "These spatial bounds do not enclose numerical Fourier quadrature "
                "roundoff or the omitted prime-power tail."
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
        print(f"2 * sum Lambda(n) Phi(log n) / sqrt(n) = {finite_sum:.12g}")
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


def _extremal(args: argparse.Namespace) -> int:
    """Report unconstrained and constrained finite-cutoff Rayleigh extrema."""
    from .repro import extremal_demo_report

    report = extremal_demo_report(
        center=args.center,
        width=args.width,
        cutoffs=tuple(args.cutoffs or (20, 50)),
        prescribed_zeros=tuple(args.prescribed_zeros or (0.7 + 0.3j,)),
    )
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(
            "Finite-cutoff Rayleigh-Ritz on polynomial-Gaussian basis: "
            f"center={args.center:g}, width={args.width:g}"
        )
        print("Prescribed zero frequencies: " + ", ".join(report["prescribed_zeros"]))
        print(
            "cutoff  mode            minimum       maximum     "
            "max projected residual  max zero residual"
        )
        for row in report["cutoffs"]:
            for mode in ("unconstrained", "constrained"):
                result = row[mode]
                largest_projected = max(
                    result["minimum"]["projected_residual_norm"],
                    result["maximum"]["projected_residual_norm"],
                )
                largest_zero = max(
                    result["minimum"]["max_zero_residual"],
                    result["maximum"]["max_zero_residual"],
                )
                zero_text = f"{largest_zero:.3e}" if mode == "constrained" else "n/a"
                print(
                    f"{row['cutoff']:>6}  {mode:<13} "
                    f"{result['minimum']['rayleigh_quotient']:>13.6g} "
                    f"{result['maximum']['rayleigh_quotient']:>13.6g} "
                    f"{largest_projected:>22.3e} {zero_text:>18}"
                )
                for endpoint in ("minimum", "maximum"):
                    vector = result[endpoint]
                    print(
                        f"  {mode} {endpoint} basis weights: "
                        f"{vector['basis_coefficients']}"
                    )
                    if vector["polynomial_coefficients_in_xi_squared"] is not None:
                        print(
                            f"  {mode} {endpoint} polynomial coefficients "
                            "(ascending powers of xi^2): "
                            f"{vector['polynomial_coefficients_in_xi_squared']}"
                        )
        print(
            "Constrained stationarity uses the projected residual; the full "
            "generalized residual can be nonzero."
        )
        print(
            "Finite prime-power cutoff only: no omitted-prime-tail bound or RH claim."
        )
    return 0


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
    measurements: list[BenchmarkMeasurement] = []
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
        print(
            "Wall-clock measurements on this machine; not comparable across machines."
        )
    return 0


def _write_text_atomic(destination: Path, contents: str) -> None:
    """Replace one requested artifact only after its complete contents are on disk."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _reproduce(args: argparse.Namespace) -> int:
    """Regenerate fixed paper artifacts, with optional golden-vector verification."""
    from .repro import (
        generate_golden_vectors,
        render_markdown,
        reproduce_report,
        validate_golden_vectors,
    )

    golden = generate_golden_vectors()
    report: dict[str, Any] = reproduce_report(golden=golden)
    check_result: dict[str, Any] | None = None
    if args.check or args.golden_file is not None:
        fixture_path = args.golden_file or (
            resources.files("weil_gaussian")
            .joinpath("data")
            .joinpath("golden_vectors.json")
        )
        if not fixture_path.is_file():
            problem = (
                f"fixture missing: {fixture_path}; reinstall or pass --golden-file"
            )
            check_result = {
                "passed": False,
                "problems": [problem],
                "max_absolute_error": None,
            }
            print(
                f"golden-vector fixture missing: {fixture_path}; "
                "reinstall or pass --golden-file",
                file=sys.stderr,
            )
        else:
            try:
                expected = json.loads(fixture_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeError) as exc:
                check_result = {
                    "passed": False,
                    "problems": [f"fixture is not valid JSON: {exc}"],
                    "max_absolute_error": None,
                }
            if check_result is None and not isinstance(expected, dict):
                check_result = {
                    "passed": False,
                    "problems": ["fixture is not a JSON object"],
                    "max_absolute_error": None,
                }
            if check_result is None:
                result = validate_golden_vectors(expected, golden)
                check_result = {
                    "passed": result["all_passed"] is True,
                    "problems": result["problems"],
                    "max_absolute_error": result["max_absolute_error"],
                }
            if check_result["passed"]:
                print("golden-vector check passed", file=sys.stderr)
            else:
                problems = "; ".join(check_result["problems"])
                print(f"golden-vector check failed: {problems}", file=sys.stderr)

    if check_result is not None:
        report["golden_vector_check"] = check_result
        report["all_passed"] = report["all_passed"] is True and check_result["passed"]
    passed = report["all_passed"] is True

    json_text = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(report).rstrip("\n") + "\n"
    if check_result is not None:
        status = "passed" if check_result["passed"] else "failed"
        markdown_text += f"\nGolden-vector fixture check: {status}.\n"
        if check_result["problems"]:
            markdown_text += (
                "\n"
                + "\n".join(f"- {problem}" for problem in check_result["problems"])
                + "\n"
            )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(output_dir / "reproducibility.json", json_text)
        _write_text_atomic(output_dir / "reproducibility.md", markdown_text)

    print(json_text if args.json else markdown_text, end="")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and return an exit code."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            return _verify(args)
        if args.command == "prime-sum":
            return _prime_sum(args)
        if args.command == "extremal":
            return _extremal(args)
        if args.command == "benchmark":
            return _benchmark(args)
        if args.command == "reproduce":
            return _reproduce(args)
    except (ArithmeticError, OSError, OverflowError, ValueError) as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")
