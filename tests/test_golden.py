"""Independent high-precision references for the manuscript's fixed cases."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
import pytest

from weil_gaussian.families import PolynomialGaussianFamily
from weil_gaussian.gram import l2_gram_matrix
from weil_gaussian.repro import (
    generate_golden_vectors,
    render_markdown,
    reproduce_report,
    validate_golden_vectors,
)


@pytest.fixture(scope="module")
def committed() -> dict[str, Any]:
    path = Path(__file__).parent / "data" / "golden_vectors.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def regenerated() -> dict[str, Any]:
    return generate_golden_vectors()


@pytest.fixture(scope="module")
def report(regenerated: dict[str, Any]) -> dict[str, Any]:
    return reproduce_report(golden=regenerated)


def _family(case: dict[str, Any]) -> PolynomialGaussianFamily:
    return PolynomialGaussianFamily(
        float(case["center"]),
        float(case["width"]),
        tuple(
            complex(float(zero["real"]), float(zero["imag"])) for zero in case["zeros"]
        ),
    )


def _value(case: dict[str, Any], quantity: str) -> float:
    family = _family(case)
    if quantity == "fourier":
        return float(family.fourier(float(case["frequency"])))
    if quantity == "inverse":
        return float(family.inverse(float(case["position"])))
    if quantity == "autocorrelation":
        return float(family.autocorrelation(float(case["position"])))
    if quantity == "mass":
        return float(family.fourier(0.0)) ** 2
    raise AssertionError(f"unexpected quantity: {quantity}")


@pytest.mark.parametrize("case_index", range(4))
@pytest.mark.parametrize("quantity", ("fourier", "inverse", "autocorrelation", "mass"))
def test_closed_form_agrees_with_committed_reference(
    committed: dict[str, Any], case_index: int, quantity: str
) -> None:
    case = committed["cases"][case_index]
    actual = _value(case, quantity)
    reference = float(case["reference"][quantity])
    tolerances = committed["tolerances"]
    assert math.isclose(
        actual,
        reference,
        rel_tol=tolerances["float64_relative"],
        abs_tol=tolerances["float64_absolute"],
    )


@pytest.mark.parametrize("case_index", range(4))
def test_prescribed_zeros_are_exact(committed: dict[str, Any], case_index: int) -> None:
    case = committed["cases"][case_index]
    family = _family(case)
    assert family.is_weil_admissible
    for zero in family.zeros:
        for root in (zero, -zero, zero.conjugate(), -zero.conjugate()):
            assert family.fourier(root) == 0


@pytest.mark.parametrize("case_index", range(4))
def test_references_have_high_precision_decimal_strings(
    committed: dict[str, Any], case_index: int
) -> None:
    values = committed["cases"][case_index]["reference"]
    for value in values.values():
        assert isinstance(value, str)
        assert mp.isfinite(mp.mpf(value))
        assert len(value.split("e")[0].replace(".", "").replace("-", "")) >= 58


@pytest.mark.parametrize("row", range(3))
@pytest.mark.parametrize("column", range(3))
def test_gram_entries_agree_with_independent_reference(
    committed: dict[str, Any], row: int, column: int
) -> None:
    lookup = {case["id"]: case for case in committed["cases"]}
    gram = committed["gram"]
    families = [_family(lookup[label]) for label in gram["case_ids"]]
    matrix = l2_gram_matrix(families)
    expected = float(gram["matrix"][row][column])
    tolerances = committed["tolerances"]
    assert math.isclose(
        float(matrix[row, column]),
        expected,
        rel_tol=tolerances["float64_relative"],
        abs_tol=tolerances["float64_absolute"],
    )


@pytest.mark.parametrize("eigen_index", range(3))
def test_gram_eigenvalues_agree_with_reference(
    committed: dict[str, Any], eigen_index: int
) -> None:
    lookup = {case["id"]: case for case in committed["cases"]}
    gram = committed["gram"]
    matrix = l2_gram_matrix([_family(lookup[label]) for label in gram["case_ids"]])
    eigenvalue = float(np.linalg.eigvalsh(matrix)[eigen_index])
    expected = float(gram["eigenvalues"][eigen_index])
    tolerances = committed["tolerances"]
    assert math.isclose(
        eigenvalue,
        expected,
        rel_tol=tolerances["float64_relative"],
        abs_tol=tolerances["float64_absolute"],
    )


def test_golden_regeneration_matches_committed_numerics(
    committed: dict[str, Any], regenerated: dict[str, Any]
) -> None:
    check = validate_golden_vectors(committed, regenerated)
    assert check["all_passed"], check["problems"]
    assert check["max_absolute_error"] < 1e-48


def test_reference_values_stabilize_when_working_precision_increases(
    regenerated: dict[str, Any],
) -> None:
    higher_precision = generate_golden_vectors(80)
    with mp.workdps(80):
        for first, second in zip(
            regenerated["cases"], higher_precision["cases"], strict=True
        ):
            for quantity in ("fourier", "inverse", "autocorrelation", "mass"):
                low = mp.mpf(first["reference"][quantity])
                high = mp.mpf(second["reference"][quantity])
                assert abs(low - high) < mp.mpf("1e-55") * max(1, abs(high))
        for low_row, high_row in zip(
            regenerated["gram"]["matrix"],
            higher_precision["gram"]["matrix"],
            strict=True,
        ):
            for low_value, high_value in zip(low_row, high_row, strict=True):
                low = mp.mpf(low_value)
                high = mp.mpf(high_value)
                assert abs(low - high) < mp.mpf("1e-55") * max(1, abs(high))


def test_reference_gram_is_positive_and_well_conditioned(
    committed: dict[str, Any],
) -> None:
    gram = committed["gram"]
    eigenvalues = [mp.mpf(value) for value in gram["eigenvalues"]]
    assert all(value > 0 for value in eigenvalues)
    assert mp.almosteq(
        eigenvalues[-1] / eigenvalues[0], mp.mpf(gram["condition_number"])
    )
    for row in range(3):
        for column in range(3):
            assert gram["matrix"][row][column] == gram["matrix"][column][row]


def test_report_checks_all_values(report: dict[str, Any]) -> None:
    assert report["all_passed"]
    assert len(report["rows"]) == 4
    assert report["gram"]["within_tolerance"]
    assert report["gram"]["min_eigenvalue"] > 0
    assert all(all(row["within_tolerance"].values()) for row in report["rows"])
    assert report["provenance"]["random_seed"] is None


def test_report_serializes_and_renders_without_nondeterministic_timestamp(
    report: dict[str, Any],
) -> None:
    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    assert json.loads(serialized)["all_passed"] is True
    rendered = render_markdown(report)
    assert rendered.count("| base_") == 2
    assert "| two_imaginary_zeros |" in rendered
    assert "| complex_zero_orbit |" in rendered
    assert "All checks passed: true." in rendered
    assert "2026-" not in rendered


@pytest.mark.parametrize("bad_dps", (59, 81, True, 70.0))
def test_invalid_working_precision_is_rejected(bad_dps: Any) -> None:
    with pytest.raises(ValueError, match="dps"):
        generate_golden_vectors(bad_dps)


def test_precomputed_golden_requires_matching_precision(
    regenerated: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="working_digits"):
        reproduce_report(dps=60, golden=regenerated)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("id", "other_case"),
        ("center", "8.2"),
        ("width", "1.7"),
        ("zeros", [{"real": "0", "imag": "1.5"}]),
        ("position", "0.8"),
        ("frequency", "2.4"),
    ),
)
def test_precomputed_case_metadata_cannot_mislabel_report(
    regenerated: dict[str, Any], field: str, replacement: Any
) -> None:
    altered = deepcopy(regenerated)
    altered["cases"][0][field] = replacement
    with pytest.raises(ValueError, match=f"base_origin.{field}"):
        reproduce_report(golden=altered)


def test_precomputed_gram_ids_cannot_mislabel_report(
    regenerated: dict[str, Any],
) -> None:
    altered = deepcopy(regenerated)
    altered["gram"]["case_ids"].reverse()
    with pytest.raises(ValueError, match="Gram case IDs"):
        reproduce_report(golden=altered)


def test_precomputed_gram_shape_is_checked(regenerated: dict[str, Any]) -> None:
    altered = deepcopy(regenerated)
    altered["gram"]["matrix"].pop()
    with pytest.raises(ValueError, match="Gram matrix"):
        reproduce_report(golden=altered)


def test_precomputed_nonfinite_reference_is_rejected(
    regenerated: dict[str, Any],
) -> None:
    altered = deepcopy(regenerated)
    altered["cases"][0]["reference"]["inverse"] = "nan"
    with pytest.raises(ValueError, match="base_origin.inverse must be finite"):
        reproduce_report(golden=altered)


def test_precomputed_provenance_cannot_claim_unverified_status(
    regenerated: dict[str, Any],
) -> None:
    altered = deepcopy(regenerated)
    altered["provenance"]["certified"] = True
    with pytest.raises(ValueError, match="provenance fields"):
        reproduce_report(golden=altered)


def test_precomputed_wrong_condition_number_fails_report(
    regenerated: dict[str, Any],
) -> None:
    altered = deepcopy(regenerated)
    altered["gram"]["condition_number"] = "1.0"
    report = reproduce_report(golden=altered)
    assert not report["all_passed"]
    assert not report["gram"]["within_tolerance"]
    assert report["gram"]["condition_number_absolute_error"] > 1_000


@pytest.mark.parametrize(
    ("section", "key"),
    (
        ("cases", "reference"),
        ("gram", "matrix"),
        ("tolerances", "reference_absolute"),
        ("provenance", "reference_method"),
    ),
)
def test_modified_fixture_is_rejected(
    committed: dict[str, Any],
    regenerated: dict[str, Any],
    section: str,
    key: str,
) -> None:
    altered = deepcopy(committed)
    if section == "cases":
        altered["cases"][0]["reference"]["inverse"] = "999"
    elif section == "gram":
        altered["gram"]["matrix"][0][0] = "999"
    elif section == "tolerances":
        altered["tolerances"][key] = "1e-3"
    else:
        altered["provenance"][key] = "wrong method"
    assert not validate_golden_vectors(altered, regenerated)["all_passed"]


def test_nonfinite_reference_is_rejected(
    committed: dict[str, Any], regenerated: dict[str, Any]
) -> None:
    altered = deepcopy(committed)
    altered["cases"][0]["reference"]["inverse"] = "nan"
    result = validate_golden_vectors(altered, regenerated)
    assert not result["all_passed"]
    assert "nonfinite" in " ".join(result["problems"])


def test_malformed_fixture_provenance_returns_failed_check(
    committed: dict[str, Any], regenerated: dict[str, Any]
) -> None:
    altered = deepcopy(committed)
    altered["provenance"] = []
    result = validate_golden_vectors(altered, regenerated)
    assert not result["all_passed"]
    assert "provenance must be an object" in " ".join(result["problems"])


def test_different_mpmath_version_metadata_is_allowed(
    committed: dict[str, Any], regenerated: dict[str, Any]
) -> None:
    altered = deepcopy(committed)
    altered["provenance"]["mpmath_version"] = "some-other-version"
    assert validate_golden_vectors(altered, regenerated)["all_passed"]
