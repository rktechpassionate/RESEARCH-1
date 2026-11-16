"""Deterministic, high-precision references for the manuscript's numerical table.

The reference values are obtained from the *defining* Fourier integrals with
mpmath, rather than from the closed forms used by the package.  They are
numerical checks, not computer-assisted proofs or certified tail bounds.
There is no randomness in this pipeline; the seed field is therefore null.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import mpmath as mp
import numpy as np

from .families import PolynomialGaussianFamily
from .gram import finite_weil_diagnostics, l2_gram_matrix


@dataclass(frozen=True)
class _Case:
    label: str
    center: str
    width: str
    zeros: tuple[tuple[str, str], ...]
    position: str
    frequency: str


_CASES = (
    _Case("base_origin", "1.2", "0.9", (("0", "0.5"),), "0", "0.4"),
    _Case("base_offset", "1.2", "0.9", (("0", "0.5"),), "0.8", "1.1"),
    _Case(
        "two_imaginary_zeros",
        "0.7",
        "1.1",
        (("0", "0.5"), ("0", "1.0")),
        "0.8",
        "0.4",
    ),
    _Case(
        "complex_zero_orbit",
        "1.0",
        "0.8",
        (("0", "0.5"), ("0.7", "0.3")),
        "0.4",
        "1.5",
    ),
)
_GRAM_INDICES = (0, 2, 3)
_FLOAT64_ABS_TOL = 5e-12
_FLOAT64_REL_TOL = 5e-12
_REFERENCE_ABS_TOL = "1e-48"
_REFERENCE_REL_TOL = "1e-48"
_EXTREMAL_ABS_TOL = 1e-8
_EXTREMAL_REL_TOL = 1e-8
_REFERENCE_METHOD = (
    "mpmath tanh-sinh quadrature of inverse, autocorrelation and Gram Fourier "
    "integrals on [0, infinity); direct spectrum and F(0)^2 mass"
)
_FOURIER_CONVENTION = (
    "F(xi)=integral f(x) exp(-i xi x) dx; f(x)=integral F(xi) exp(i xi x) dxi/(2 pi)"
)
_EXTREMAL_CENTER = 0.4
_EXTREMAL_WIDTH = 0.8
_EXTREMAL_CUTOFFS = (20, 50)
_EXTREMAL_ZEROS = (0.7 + 0.3j,)
_EXTREMAL_BASIS_ZEROS = (
    (0.5j,),
    (0.5j, 1.0),
    (0.5j, 1.0, 2.0),
    (0.5j, 1.0, 2.0, 3.0),
)
_SPATIAL_TAIL_RADIUS = 8.0


def _extremal_basis(
    center: float, width: float
) -> tuple[PolynomialGaussianFamily, ...]:
    """Fixed polynomial basis sharing one Gaussian envelope."""
    return tuple(
        PolynomialGaussianFamily(center, width, tuple(complex(z) for z in zeros))
        for zeros in _EXTREMAL_BASIS_ZEROS
    )


def _eigenpair_report(pair: Any) -> dict[str, Any]:
    return {
        "eigenvalue": float(pair.eigenvalue),
        "rayleigh_quotient": float(pair.rayleigh_quotient),
        "basis_coefficients": [float(value) for value in pair.coefficients],
        "polynomial_coefficients_in_xi_squared": (
            [float(value) for value in pair.polynomial_coefficients]
            if pair.polynomial_coefficients is not None
            else None
        ),
        "metric_norm": float(pair.metric_norm),
        "full_residual_norm": float(pair.full_residual_norm),
        "projected_residual_norm": float(pair.projected_residual_norm),
        "relative_projected_residual": float(pair.relative_projected_residual),
        "zero_residuals": [float(value) for value in pair.zero_residuals],
        "max_zero_residual": float(pair.max_zero_residual),
    }


def _extremal_solution_report(solution: Any) -> dict[str, Any]:
    return {
        "eigenvalues": [float(value) for value in solution.eigenvalues],
        "constraint_rank": solution.constraint_rank,
        "subspace_dimension": solution.subspace_dimension,
        "metric_condition_number": float(solution.metric_condition_number),
        "metric_relative_tolerance": float(solution.metric_relative_tolerance),
        "gram_matrix": solution.gram_matrix.tolist(),
        "finite_weil_matrix": solution.weil_matrix.tolist(),
        "constraint_matrix": solution.constraint_matrix.tolist(),
        "integration_error_estimate": solution.integration_error_estimate.tolist(),
        "prime_tail_bound": None,
        "minimum": _eigenpair_report(solution.minimum),
        "maximum": _eigenpair_report(solution.maximum),
    }


def extremal_demo_report(
    *,
    center: float = _EXTREMAL_CENTER,
    width: float = _EXTREMAL_WIDTH,
    cutoffs: tuple[int, ...] = _EXTREMAL_CUTOFFS,
    prescribed_zeros: tuple[complex, ...] = _EXTREMAL_ZEROS,
) -> dict[str, Any]:
    """Reproduce finite-cutoff Rayleigh extrema on a common-envelope basis.

    This is a float64 numerical diagnostic. In particular it does not bound
    the omitted prime powers or certify the complete Weil form.
    """
    from .extremal import solve_extremal_rayleigh_ritz

    if not cutoffs:
        raise ValueError("at least one finite prime cutoff is required")
    basis = _extremal_basis(center, width)
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        diagnostic = finite_weil_diagnostics(basis, cutoff)
        unconstrained = solve_extremal_rayleigh_ritz(
            basis, cutoff, finite_diagnostic=diagnostic
        )
        constrained = solve_extremal_rayleigh_ritz(
            basis,
            cutoff,
            zeros=prescribed_zeros,
            finite_diagnostic=diagnostic,
        )
        rows.append(
            {
                "cutoff": cutoff,
                "unconstrained": _extremal_solution_report(unconstrained),
                "constrained": _extremal_solution_report(constrained),
            }
        )
    return {
        "basis_center": basis[0].center,
        "basis_width": basis[0].width,
        "basis": [
            {
                "zero_generators": [str(zero) for zero in family.zeros],
                "polynomial_coefficients_in_xi_squared": list(family.coefficients),
            }
            for family in basis
        ],
        "prescribed_zeros": [str(zero) for zero in prescribed_zeros],
        "cutoffs": rows,
        "claim": (
            "float64 Rayleigh extrema for computed finite-cutoff pairing only; "
            "no omitted-prime-tail bound or RH claim"
        ),
        "stationarity": (
            "constrained stationary residual is projected into the nullspace; "
            "the full residual need not vanish"
        ),
    }


def _extremal_regression(report: Mapping[str, Any]) -> dict[str, Any]:
    """Store float64 finite-matrix outputs separately from mpmath references."""
    return {
        "kind": "float64 finite-cutoff regression; not an independent high-precision W reference",
        "basis_center": report["basis_center"],
        "basis_width": report["basis_width"],
        "basis": report["basis"],
        "prescribed_zeros": report["prescribed_zeros"],
        "cutoffs": [
            {
                "cutoff": row["cutoff"],
                **{
                    mode: {
                        endpoint: {
                            "rayleigh_quotient": row[mode][endpoint][
                                "rayleigh_quotient"
                            ],
                            "basis_coefficients": row[mode][endpoint][
                                "basis_coefficients"
                            ],
                        }
                        for endpoint in ("minimum", "maximum")
                    }
                    for mode in ("unconstrained", "constrained")
                },
            }
            for row in report["cutoffs"]
        ],
    }


def _spatial_tail_report() -> dict[str, Any]:
    """Certificate for the exact model, not numerical or omitted-prime error."""
    from .certification import certify_spatial_tail

    family = _family(_CASES[0])
    return {
        "family_case_id": _CASES[0].label,
        "radius": _SPATIAL_TAIL_RADIUS,
        "certificates": [
            asdict(
                certify_spatial_tail(family, _SPATIAL_TAIL_RADIUS, quantity=quantity)
            )
            for quantity in ("inverse", "autocorrelation")
        ],
        "claim": (
            "rigorous upper bounds for exact-model spatial L1 tails |x|>R; "
            "not Fourier quadrature rounding or omitted prime-power tails"
        ),
    }


def _digits(dps: int) -> int:
    if isinstance(dps, bool) or not isinstance(dps, int) or not 60 <= dps <= 80:
        raise ValueError("dps must be an integer from 60 to 80")
    return dps


def _family(case: _Case) -> PolynomialGaussianFamily:
    return PolynomialGaussianFamily(
        float(case.center),
        float(case.width),
        tuple(complex(float(real), float(imag)) for real, imag in case.zeros),
    )


def _case_metadata(case: _Case) -> dict[str, Any]:
    return {
        "id": case.label,
        "center": case.center,
        "width": case.width,
        "zeros": [{"real": real, "imag": imag} for real, imag in case.zeros],
        "position": case.position,
        "frequency": case.frequency,
    }


def _provenance(dps: int) -> dict[str, Any]:
    return {
        "project_version": "1.1.0",
        "working_digits": dps,
        "output_digits": dps - 10,
        "mpmath_version": mp.__version__,
        "reference_method": _REFERENCE_METHOD,
        "fourier_convention": _FOURIER_CONVENTION,
        "random_seed": None,
    }


def _tolerances() -> dict[str, Any]:
    return {
        "float64_absolute": _FLOAT64_ABS_TOL,
        "float64_relative": _FLOAT64_REL_TOL,
        "reference_absolute": _REFERENCE_ABS_TOL,
        "reference_relative": _REFERENCE_REL_TOL,
        "finite_extremal_absolute": _EXTREMAL_ABS_TOL,
        "finite_extremal_relative": _EXTREMAL_REL_TOL,
    }


def _spectrum(frequency: Any, case: _Case) -> Any:
    """Evaluate the defining polynomial and Gaussian pair in mp arithmetic."""
    polynomial = mp.mpc(1)
    square = frequency * frequency
    for real, imag in case.zeros:
        root_squared = mp.mpc(real, imag) ** 2
        polynomial *= square - root_squared
        if mp.im(root_squared) != 0:
            polynomial *= square - mp.conj(root_squared)
    center = mp.mpf(case.center)
    width = mp.mpf(case.width)
    gaussian_pair = mp.exp(-(((frequency - center) / width) ** 2) / 2) + mp.exp(
        -(((frequency + center) / width) ** 2) / 2
    )
    return mp.re(polynomial) * gaussian_pair


def _quadrature_points(*cases: _Case) -> list[Any]:
    centers = sorted({mp.mpf(case.center) for case in cases if case.center != "0"})
    return [mp.mpf("0"), *centers, mp.inf]


def _product_spectrum(t: Any, left: _Case, right: _Case) -> Any:
    return _spectrum(t, left) * _spectrum(t, right)


def _integral(function: Callable[[Any], Any], *cases: _Case) -> Any:
    # Tanh-sinh is independent of the package's Hermite/closed-form route.
    return mp.quad(function, _quadrature_points(*cases), method="tanh-sinh") / mp.pi


def _decimal(value: Any, digits: int) -> str:
    return str(mp.nstr(value, digits, strip_zeros=False))


def _reference_case(case: _Case, digits: int) -> dict[str, Any]:
    x = mp.mpf(case.position)
    frequency = mp.mpf(case.frequency)
    fourier = _spectrum(frequency, case)
    inverse = _integral(lambda t: _spectrum(t, case) * mp.cos(t * x), case)
    correlation = _integral(lambda t: _spectrum(t, case) ** 2 * mp.cos(t * x), case)
    mass = _spectrum(mp.mpf("0"), case) ** 2
    return {
        **_case_metadata(case),
        "reference": {
            "fourier": _decimal(fourier, digits),
            "inverse": _decimal(inverse, digits),
            "autocorrelation": _decimal(correlation, digits),
            "mass": _decimal(mass, digits),
        },
    }


def _reference_gram(cases: tuple[_Case, ...], digits: int) -> dict[str, Any]:
    size = len(cases)
    matrix = mp.matrix(size)
    for i, left in enumerate(cases):
        for j in range(i, size):
            right = cases[j]
            value = _integral(
                partial(_product_spectrum, left=left, right=right), left, right
            )
            matrix[i, j] = value
            matrix[j, i] = value
    eigenvalues, _ = mp.eigsy(matrix)
    return {
        "case_ids": [case.label for case in cases],
        "matrix": [
            [_decimal(matrix[i, j], digits) for j in range(size)] for i in range(size)
        ],
        "eigenvalues": [_decimal(eigenvalues[i], digits) for i in range(size)],
        "condition_number": _decimal(eigenvalues[size - 1] / eigenvalues[0], digits),
    }


def generate_golden_vectors(dps: int = 70) -> dict[str, Any]:
    """Regenerate all fixed paper vectors from defining Fourier integrals.

    The output uses decimal strings for reference numbers to preserve more
    information than JSON binary floats.  The two tolerances describe the
    expected comparison with this package's float64 formulas; they are not
    interval-arithmetic certificates or promises for arbitrary hardware.
    """
    _digits(dps)
    with mp.workdps(dps):
        extremal = extremal_demo_report()
        return {
            "schema_version": 2,
            "provenance": _provenance(dps),
            "tolerances": _tolerances(),
            "cases": [_reference_case(case, dps - 10) for case in _CASES],
            "gram": _reference_gram(
                tuple(_CASES[index] for index in _GRAM_INDICES), dps - 10
            ),
            "finite_extremal_regression": _extremal_regression(extremal),
            "spatial_tail_certificate": _spatial_tail_report(),
        }


def _within_float64(analytic: float, reference: str) -> tuple[float, bool]:
    with mp.workdps(80):
        error = float(abs(mp.mpf(analytic) - mp.mpf(reference)))
    threshold = _FLOAT64_ABS_TOL + _FLOAT64_REL_TOL * abs(float(reference))
    return error, math.isfinite(error) and error <= threshold


def _compare_reference(
    old_value: Any, new_value: Any, absolute: Any, relative: Any
) -> tuple[float | None, bool]:
    """Compare decimal vectors at higher precision than either vector stores."""
    with mp.workdps(80):
        old_mp = mp.mpf(old_value)
        new_mp = mp.mpf(new_value)
        if not mp.isfinite(old_mp) or not mp.isfinite(new_mp):
            return None, False
        error = abs(old_mp - new_mp)
        return float(error), bool(error <= absolute + relative * abs(new_mp))


def _finite_decimal(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"golden {label} must be a decimal string")
    try:
        finite = bool(mp.isfinite(mp.mpf(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"golden {label} is not a decimal number") from exc
    if not finite:
        raise ValueError(f"golden {label} must be finite")


def _compare_extremal_regression(
    committed: Mapping[str, Any], regenerated: Mapping[str, Any]
) -> tuple[float, list[str]]:
    """Compare finite float64 diagnostics separately from mpmath references."""
    problems: list[str] = []
    maximum_error = 0.0
    try:
        for field in (
            "kind",
            "basis_center",
            "basis_width",
            "basis",
            "prescribed_zeros",
        ):
            if committed[field] != regenerated[field]:
                problems.append(f"finite extremal {field} differs")
        old_rows = committed["cutoffs"]
        new_rows = regenerated["cutoffs"]
        if len(old_rows) != len(new_rows):
            problems.append("finite extremal cutoff count differs")
        for old, new in zip(old_rows, new_rows, strict=False):
            if old["cutoff"] != new["cutoff"]:
                problems.append("finite extremal cutoff differs")
            for mode in ("unconstrained", "constrained"):
                for endpoint in ("minimum", "maximum"):
                    old_pair = old[mode][endpoint]
                    new_pair = new[mode][endpoint]
                    for field in ("rayleigh_quotient", "basis_coefficients"):
                        old_values = old_pair[field]
                        new_values = new_pair[field]
                        if field == "rayleigh_quotient":
                            old_values = [old_values]
                            new_values = [new_values]
                        if len(old_values) != len(new_values):
                            problems.append(
                                f"finite extremal {mode}.{endpoint}.{field} length differs"
                            )
                        for index, (old_value, new_value) in enumerate(
                            zip(old_values, new_values, strict=False)
                        ):
                            first = float(old_value)
                            second = float(new_value)
                            label = (
                                f"finite extremal {mode}.{endpoint}.{field}[{index}]"
                            )
                            if not math.isfinite(first) or not math.isfinite(second):
                                problems.append(f"{label} is nonfinite")
                                continue
                            error = abs(first - second)
                            maximum_error = max(maximum_error, error)
                            threshold = _EXTREMAL_ABS_TOL + _EXTREMAL_REL_TOL * abs(
                                second
                            )
                            if error > threshold:
                                problems.append(f"{label} differs")
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        problems.append(f"malformed finite extremal regression: {exc}")
    return maximum_error, problems


def _compare_spatial_tail_certificate(
    committed: Mapping[str, Any], regenerated: Mapping[str, Any]
) -> list[str]:
    """The exact-rational/Decimal certificate should be bytewise reproducible."""
    return [] if committed == regenerated else ["spatial tail certificate differs"]


def _validate_precomputed_layout(golden: Mapping[str, Any], dps: int) -> None:
    """Reject supplied references whose labels do not identify our fixed cases."""
    if not isinstance(golden, Mapping):
        raise ValueError("golden vectors must be an object")
    if golden.get("schema_version") != 2:
        raise ValueError("golden schema_version must be 2")
    provenance = golden.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("golden provenance is missing")
    expected_provenance = _provenance(dps)
    if set(provenance) != set(expected_provenance):
        raise ValueError("golden provenance fields do not match fixed run")
    if (
        not isinstance(provenance["mpmath_version"], str)
        or not provenance["mpmath_version"]
    ):
        raise ValueError("golden mpmath_version must be a nonempty string")
    for field, expected in expected_provenance.items():
        if field != "mpmath_version" and provenance.get(field) != expected:
            raise ValueError(f"golden provenance.{field} does not match fixed run")
    if golden.get("tolerances") != _tolerances():
        raise ValueError("golden tolerances do not match fixed run")
    vectors = golden.get("cases")
    if not isinstance(vectors, list) or len(vectors) != len(_CASES):
        raise ValueError("golden cases must match the fixed case count")
    for case, vector in zip(_CASES, vectors, strict=True):
        if not isinstance(vector, Mapping):
            raise ValueError(f"golden {case.label} must be an object")
        for field, expected in _case_metadata(case).items():
            if vector.get(field) != expected:
                raise ValueError(
                    f"golden {case.label}.{field} does not match fixed case"
                )
        references = vector.get("reference")
        if not isinstance(references, Mapping):
            raise ValueError(f"golden {case.label}.reference is missing")
        for quantity in ("fourier", "inverse", "autocorrelation", "mass"):
            _finite_decimal(references.get(quantity), f"{case.label}.{quantity}")
    gram = golden.get("gram")
    if not isinstance(gram, Mapping):
        raise ValueError("golden Gram is missing")
    expected_ids = [_CASES[index].label for index in _GRAM_INDICES]
    if gram.get("case_ids") != expected_ids:
        raise ValueError("golden Gram case IDs do not match fixed families")
    matrix = gram.get("matrix")
    size = len(expected_ids)
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("golden Gram matrix has the wrong shape")
    for index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("golden Gram matrix has the wrong shape")
        for column, value in enumerate(row):
            _finite_decimal(value, f"Gram matrix[{index},{column}]")
    eigenvalues = gram.get("eigenvalues")
    if not isinstance(eigenvalues, list) or len(eigenvalues) != size:
        raise ValueError("golden Gram eigenvalues have the wrong shape")
    for index, value in enumerate(eigenvalues):
        _finite_decimal(value, f"Gram eigenvalues[{index}]")
    _finite_decimal(gram.get("condition_number"), "Gram condition number")
    if not isinstance(golden.get("finite_extremal_regression"), Mapping):
        raise ValueError("golden finite_extremal_regression is missing")
    if not isinstance(golden.get("spatial_tail_certificate"), Mapping):
        raise ValueError("golden spatial_tail_certificate is missing")


def reproduce_report(
    dps: int = 70, golden: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute references and compare every paper row with package output."""
    _digits(dps)
    if golden is None:
        golden = generate_golden_vectors(dps)
    _validate_precomputed_layout(golden, dps)
    rows: list[dict[str, Any]] = []
    for case, vector in zip(_CASES, golden["cases"], strict=True):
        family = _family(case)
        analytic = {
            "fourier": float(family.fourier(float(case.frequency))),
            "inverse": float(family.inverse(float(case.position))),
            "autocorrelation": float(family.autocorrelation(float(case.position))),
            "mass": float(family.fourier(0.0)) ** 2,
        }
        errors: dict[str, float] = {}
        passes: dict[str, bool] = {}
        for quantity, value in analytic.items():
            errors[quantity], passes[quantity] = _within_float64(
                value, vector["reference"][quantity]
            )
        rows.append(
            {
                **_case_metadata(case),
                "reference": vector["reference"],
                "analytic": analytic,
                "absolute_error": errors,
                "within_tolerance": passes,
                "max_absolute_error": max(errors.values()),
            }
        )

    gram_families = tuple(_family(_CASES[index]) for index in _GRAM_INDICES)
    matrix = l2_gram_matrix(gram_families)
    reference_matrix = golden["gram"]["matrix"]
    matrix_errors = [
        _within_float64(float(matrix[i, j]), reference_matrix[i][j])
        for i in range(len(gram_families))
        for j in range(len(gram_families))
    ]
    eigenvalues = np.linalg.eigvalsh(matrix)
    reference_eigenvalues = golden["gram"]["eigenvalues"]
    eigen_errors = [
        _within_float64(float(value), reference)
        for value, reference in zip(eigenvalues, reference_eigenvalues, strict=True)
    ]
    condition = float(eigenvalues[-1] / eigenvalues[0])
    condition_error, condition_passed = _within_float64(
        condition, golden["gram"]["condition_number"]
    )
    gram_passed = all(passed for _, passed in matrix_errors + eigen_errors)
    gram_passed = gram_passed and condition_passed
    gram = {
        "case_ids": golden["gram"]["case_ids"],
        "reference_matrix": reference_matrix,
        "matrix": matrix.tolist(),
        "reference_eigenvalues": reference_eigenvalues,
        "eigenvalues": eigenvalues.tolist(),
        "reference_condition_number": golden["gram"]["condition_number"],
        "condition_number": condition,
        "condition_number_absolute_error": condition_error,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_matrix_absolute_error": max(error for error, _ in matrix_errors),
        "max_eigenvalue_absolute_error": max(error for error, _ in eigen_errors),
        "within_tolerance": gram_passed,
    }
    extremal = extremal_demo_report()
    finite_regression = _extremal_regression(extremal)
    finite_error, finite_problems = _compare_extremal_regression(
        golden["finite_extremal_regression"], finite_regression
    )
    spatial_tail = _spatial_tail_report()
    spatial_problems = _compare_spatial_tail_certificate(
        golden["spatial_tail_certificate"], spatial_tail
    )
    return {
        "schema_version": golden["schema_version"],
        "provenance": {**golden["provenance"], "numpy_version": np.__version__},
        "tolerances": golden["tolerances"],
        "rows": rows,
        "gram": gram,
        "extremal": extremal,
        "finite_extremal_regression_check": {
            "passed": not finite_problems,
            "max_absolute_error": finite_error,
            "problems": finite_problems,
        },
        "spatial_tail": spatial_tail,
        "spatial_tail_certificate_check": {
            "passed": not spatial_problems,
            "problems": spatial_problems,
        },
        "all_passed": all(all(row["within_tolerance"].values()) for row in rows)
        and gram_passed
        and not finite_problems
        and not spatial_problems,
    }


def validate_golden_vectors(
    committed: Mapping[str, Any], regenerated: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Check committed vectors numerically without requiring byte identity.

    Version metadata may differ across machines. Case labels and parameters
    must match exactly; decimal references must meet the declared high-
    precision comparison tolerances. A missing or malformed field fails the
    check with a descriptive problem entry.
    """
    if regenerated is None:
        try:
            dps = int(committed["provenance"]["working_digits"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("committed vectors lack working_digits") from exc
        regenerated = generate_golden_vectors(dps)
    problems: list[str] = []
    max_error = 0.0
    finite_error = 0.0
    if committed.get("schema_version") != regenerated.get("schema_version"):
        problems.append("schema_version differs")
    try:
        if not isinstance(committed.get("provenance"), Mapping):
            raise ValueError("provenance must be an object")
        tolerances = committed["tolerances"]
        if tolerances != regenerated["tolerances"]:
            problems.append("reference tolerances differ")
        absolute = mp.mpf(tolerances["reference_absolute"])
        relative = mp.mpf(tolerances["reference_relative"])
        if not mp.isfinite(absolute) or not mp.isfinite(relative):
            raise ValueError("reference tolerances must be finite")
        if absolute <= 0 or relative <= 0:
            raise ValueError("reference tolerances must be positive")
        if set(committed["provenance"]) != set(regenerated["provenance"]):
            problems.append("provenance fields differ")
        if not isinstance(
            committed["provenance"].get("mpmath_version"), str
        ) or not committed["provenance"].get("mpmath_version"):
            problems.append("provenance.mpmath_version is missing")
        for field in (
            "project_version",
            "working_digits",
            "output_digits",
            "reference_method",
            "fourier_convention",
            "random_seed",
        ):
            if committed["provenance"][field] != regenerated["provenance"][field]:
                problems.append(f"provenance.{field} differs")
        old_cases = committed["cases"]
        new_cases = regenerated["cases"]
        if len(old_cases) != len(new_cases):
            problems.append("case count differs")
        for old, new in zip(old_cases, new_cases, strict=False):
            for field in ("id", "center", "width", "zeros", "position", "frequency"):
                if old[field] != new[field]:
                    problems.append(f"{new['id']}.{field} differs")
            for quantity in ("fourier", "inverse", "autocorrelation", "mass"):
                error, within = _compare_reference(
                    old["reference"][quantity],
                    new["reference"][quantity],
                    absolute,
                    relative,
                )
                if error is None:
                    problems.append(f"{new['id']}.{quantity} reference is nonfinite")
                    continue
                max_error = max(max_error, error)
                if not within:
                    problems.append(f"{new['id']}.{quantity} reference differs")
        old_gram = committed["gram"]
        new_gram = regenerated["gram"]
        if old_gram["case_ids"] != new_gram["case_ids"]:
            problems.append("Gram case IDs differ")
        for field in ("matrix", "eigenvalues", "condition_number"):
            old_values = old_gram[field]
            new_values = new_gram[field]
            if field == "matrix":
                old_values = [value for row in old_values for value in row]
                new_values = [value for row in new_values for value in row]
            if field == "condition_number":
                old_values = [old_values]
                new_values = [new_values]
            if len(old_values) != len(new_values):
                problems.append(f"Gram {field} length differs")
            for index, (old_value, new_value) in enumerate(
                zip(old_values, new_values, strict=False)
            ):
                error, within = _compare_reference(
                    old_value, new_value, absolute, relative
                )
                if error is None:
                    problems.append(f"Gram {field}[{index}] is nonfinite")
                    continue
                max_error = max(max_error, error)
                if not within:
                    problems.append(f"Gram {field}[{index}] differs")
        old_extremal = committed["finite_extremal_regression"]
        new_extremal = regenerated["finite_extremal_regression"]
        finite_error, finite_problems = _compare_extremal_regression(
            old_extremal, new_extremal
        )
        problems.extend(finite_problems)
        old_tail = committed["spatial_tail_certificate"]
        new_tail = regenerated["spatial_tail_certificate"]
        problems.extend(_compare_spatial_tail_certificate(old_tail, new_tail))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        problems.append(f"malformed vectors: {exc}")
    return {
        "all_passed": not problems,
        "max_absolute_error": max_error,
        "max_finite_extremal_absolute_error": finite_error,
        "problems": problems,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the regenerated numerical rows in a stable Markdown table."""
    lines = [
        "# Weil Gaussian Lab reproducibility report",
        "",
        "The references use high-precision quadrature of the defining Fourier integrals.",
        "Observed errors are numerical checks, not formal error bounds or a zeta-zero claim.",
        "",
        "| Case | a | sigma | x | f(x) | Phi(x) | Max absolute error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['center']} | {row['width']} | {row['position']} "
            f"| {row['analytic']['inverse']:.10g} "
            f"| {row['analytic']['autocorrelation']:.10g} "
            f"| {row['max_absolute_error']:.3e} |"
        )
    gram = report["gram"]
    lines.extend(
        [
            "",
            f"Gram minimum eigenvalue: {gram['min_eigenvalue']:.10g}.",
            f"Gram condition number: {gram['condition_number']:.10g}.",
            f"Maximum Gram matrix absolute error: {gram['max_matrix_absolute_error']:.3e}.",
            "",
            "## Finite-cutoff Rayleigh–Ritz diagnostics",
            "",
            "These optimize represented finite matrices, not the complete Weil form.",
            "Constrained stationarity uses a projected residual; the full residual need not vanish.",
            "",
            "| Cutoff | Mode | Minimum quotient | Maximum quotient | Max projected residual | Max zero residual |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["extremal"]["cutoffs"]:
        for mode, label in (("unconstrained", "free"), ("constrained", "constrained")):
            result = row[mode]
            projected = max(
                result[endpoint]["projected_residual_norm"]
                for endpoint in ("minimum", "maximum")
            )
            zero = max(
                result[endpoint]["max_zero_residual"]
                for endpoint in ("minimum", "maximum")
            )
            zero_text = f"{zero:.3e}" if mode == "constrained" else "n/a"
            lines.append(
                f"| {row['cutoff']} | {label} | "
                f"{result['minimum']['rayleigh_quotient']:.6f} | "
                f"{result['maximum']['rayleigh_quotient']:.6f} | "
                f"{projected:.3e} | {zero_text} |"
            )
    lines.extend(
        [
            "",
            "## Analytic spatial L1 tail certificates",
            "",
            "These bound exact-model inverse/autocorrelation integrals outside |x| > R;",
            "they do not bound numerical roundoff, frequency quadrature, or omitted prime powers.",
        ]
    )
    for certificate in report["spatial_tail"]["certificates"]:
        lines.append(
            f"- {certificate['quantity']} outside |x| > {certificate['radius']:g}: "
            f"at most {certificate['bound_decimal']}"
        )
    lines.extend(
        [
            "",
            f"All checks passed: {str(report['all_passed']).lower()}.",
            "",
        ]
    )
    return "\n".join(lines)
