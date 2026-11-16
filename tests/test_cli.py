"""CLI behavior and process exit codes for reproducible diagnostics."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest

from weil_gaussian import autocorrelation, cli, truncated_prime_term
from weil_gaussian.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _module(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "weil_gaussian", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_module_help_lists_all_commands():
    result = _module("--help")
    assert result.returncode == 0
    assert "verify" in result.stdout
    assert "prime-sum" in result.stdout
    assert "benchmark" in result.stdout
    assert "reproduce" in result.stdout
    assert "extremal" in result.stdout


def test_extremal_json_reports_both_subspaces_and_polynomial(capsys):
    assert main(["extremal", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [row["cutoff"] for row in report["cutoffs"]] == [20, 50]
    assert report["prescribed_zeros"] == ["(0.7+0.3j)"]
    assert "no omitted-prime-tail bound" in report["claim"]
    for row in report["cutoffs"]:
        for mode in ("unconstrained", "constrained"):
            solution = row[mode]
            assert solution["prime_tail_bound"] is None
            assert (
                solution["minimum"]["rayleigh_quotient"]
                <= solution["maximum"]["rayleigh_quotient"]
            )
            for endpoint in ("minimum", "maximum"):
                pair = solution[endpoint]
                assert len(pair["basis_coefficients"]) == 4
                assert len(pair["polynomial_coefficients_in_xi_squared"]) == 5
                assert pair["metric_norm"] == pytest.approx(1, abs=1e-12)
                assert pair["projected_residual_norm"] < 1e-12
                if mode == "constrained":
                    assert pair["max_zero_residual"] < 1e-12


def test_extremal_text_distinguishes_projected_and_full_residual(capsys):
    assert main(["extremal", "--cutoff", "20"]) == 0
    output = capsys.readouterr().out
    assert "-8.6986" in output
    assert "Constrained stationarity uses the projected residual" in output
    assert "full generalized residual can be nonzero" in output
    assert "polynomial coefficients" in output


def test_extremal_custom_cutoff_and_zero(capsys):
    assert main(["extremal", "--cutoff", "10", "--zero", "0.4j", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [row["cutoff"] for row in report["cutoffs"]] == [10]
    assert report["prescribed_zeros"] == ["0.4j"]
    assert report["cutoffs"][0]["constrained"]["minimum"]["max_zero_residual"] < 1e-12


@pytest.mark.parametrize("cutoff", ("1", "1001", "nan", "2.5"))
def test_extremal_rejects_invalid_cutoff(cutoff):
    result = _module("extremal", "--cutoff", cutoff)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_verify_json_includes_distinct_spatial_tail_certificates(capsys):
    assert main(["verify", "--position", "0.8", "--tail-radius", "8", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    tails = report["spatial_tail_certificates"]
    assert [entry["quantity"] for entry in tails] == ["inverse", "autocorrelation"]
    assert all(entry["certified_analytic_bound"] for entry in tails)
    assert all(entry["prime_tail_bound"] is None for entry in tails)
    assert all(float(entry["bound_decimal"]) > 0 for entry in tails)
    assert all(row["finite_cutoff_tail_bound"] is None for row in report["rows"])


def test_verify_text_labels_spatial_tail_scope(capsys):
    assert main(["verify", "--position", "0.8", "--tail-radius", "8"]) == 0
    output = capsys.readouterr().out
    assert "Certified exact-model spatial L1 tail" in output
    assert "do not enclose numerical Fourier quadrature" in output


@pytest.mark.parametrize("radius", ("0", "-2", "nan"))
def test_verify_rejects_invalid_tail_radius(radius):
    result = _module("verify", "--tail-radius", radius)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_verify_text_output_has_independent_check_and_status(capsys):
    assert main(["verify", "--position", "0.8"]) == 0
    output = capsys.readouterr().out
    assert "inverse" in output
    assert "autocorrelation" in output
    assert "abs error" in output
    assert "PASS: 2 comparisons" in output
    assert "not an RH test" in output


def test_verify_repeated_positions_generate_all_rows(capsys):
    assert main(["verify", "--position", "0", "--position", "1.1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [row["position"] for row in report["rows"]] == [0, 0, 1.1, 1.1]
    assert all(row["passed"] for row in report["rows"])


def test_verify_json_reports_high_precision_comparison(capsys):
    assert (
        main(
            [
                "verify",
                "--center",
                "2.5",
                "--width",
                "0.7",
                "--position",
                "0.3",
                "--dps",
                "45",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["quadrature_digits"] == 45
    assert report["passed"] is True
    assert {row["quantity"] for row in report["rows"]} == {
        "inverse",
        "autocorrelation",
    }
    assert max(row["absolute_error"] for row in report["rows"]) < 1e-12
    assert report["gauss_hermite_reported_order"] == 96
    assert len(report["schemes"]) == 3
    assert "not rigorous enclosures" in report["error_certification"]
    for row in report["rows"]:
        assert row["adaptive_error_estimate"] >= 0
        assert row["hermite_refinement_delta"] >= 0
        assert row["analytic_adaptive_abs_error"] < 1e-9
        assert row["analytic_hermite_abs_error"] < 1e-9
        assert row["cross_scheme_discrepancy"] < 1e-9
        assert row["finite_cutoff_tail_bound"] is None


def test_verify_text_shows_independent_schemes_and_limit(capsys):
    assert main(["verify", "--position", "0.8", "--hermite-order", "24"]) == 0
    output = capsys.readouterr().out
    assert "adaptive Gauss-Kronrod and Gauss-Hermite (orders 24/48)" in output
    assert "not rigorous bounds" in output


@pytest.mark.parametrize("order", ["11", "49", "12.5", "nan"])
def test_verify_rejects_invalid_hermite_order(order):
    result = _module("verify", "--hermite-order", order)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_verify_multi_zero_family_against_independent_quadrature(capsys):
    assert (
        main(
            [
                "verify",
                "--zero",
                "0.5j",
                "--zero",
                "1+0.5j",
                "--position",
                "0.8",
                "--dps",
                "45",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["zero_generators"] == ["0.5j", "(1+0.5j)"]
    assert report["passed"] is True
    assert max(row["absolute_error"] for row in report["rows"]) < 1e-9


def test_verify_arbitrary_complex_zero_family_without_weil_claim(capsys):
    assert main(["verify", "--zero", "1+0.5i", "--position", "0.3", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["zero_generators"] == ["(1+0.5j)"]
    assert report["passed"] is True
    assert "not an RH test" in report["claim"]


def test_verify_strict_tolerance_exits_nonzero(capsys):
    assert main(["verify", "--position", "0.8", "--tolerance", "1e-30", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert any(not row["passed"] for row in report["rows"])


def test_verify_zero_center_case(capsys):
    assert (
        main(["verify", "--center", "0", "--width", "1", "--position", "0", "--json"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["center"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["verify", "--width", "-0.5"],
        ["verify", "--center", "-1"],
        ["verify", "--position", "nan"],
        ["verify", "--dps", "19"],
        ["verify", "--dps", "201"],
        ["verify", "--tolerance", "0"],
        ["verify", "--zero", "x+yj"],
        ["verify", "--zero", "nanj"],
    ],
)
def test_verify_rejects_bad_arguments(args):
    result = _module(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_prime_json_matches_finite_sum_and_has_scope_fields(capsys):
    assert (
        main(
            [
                "prime-sum",
                "--center",
                "1.3",
                "--width",
                "0.9",
                "--limit",
                "30",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["finite_prime_term"] == pytest.approx(
        truncated_prime_term(30, 1.3, 0.9)
    )
    assert report["prime_power_count"] == 16
    assert report["no_tail_bound"] is True
    assert "no prime-tail bound" in report["claim"]
    assert math.isfinite(report["archimedean_term"])
    assert report["finite_balance"] == pytest.approx(
        report["archimedean_term"] - report["finite_prime_term"]
    )
    assert report["prime_tail_bound"] is None
    assert report["quadrature_converged"] is True
    assert report["shown_terms"][0]["value"] == 2
    assert report["shown_terms"][0]["contribution"] == pytest.approx(
        2 * math.log(2) / math.sqrt(2) * autocorrelation(math.log(2), 1.3, 0.9)
    )


def test_prime_text_output_has_cutoff_and_warning(capsys):
    assert main(["prime-sum", "--limit", "10"]) == 0
    output = capsys.readouterr().out
    assert "cutoff=10" in output
    assert "Archimedean K" in output
    assert "Cutoff-dependent K - P_N" in output
    assert "No prime-tail bound" in output
    assert "not the full Weil form" in output


def test_prime_zero_display_terms(capsys):
    assert main(["prime-sum", "--limit", "20", "--show-terms", "0", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["shown_terms"] == []
    assert report["omitted_display_terms"] == report["prime_power_count"]


@pytest.mark.parametrize(
    "args",
    [
        ["prime-sum"],
        ["prime-sum", "--limit", "1"],
        ["prime-sum", "--limit", "2.5"],
        ["prime-sum", "--limit", "1000001"],
        ["prime-sum", "--limit", "10", "--show-terms", "-1"],
        ["prime-sum", "--limit", "10", "--width", "inf"],
    ],
)
def test_prime_rejects_bad_arguments(args):
    result = _module(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_module_verify_process_returns_success():
    result = _module("verify", "--center", "1.2", "--width", "0.9", "--position", "0.8")
    assert result.returncode == 0
    assert "PASS:" in result.stdout


def test_module_verify_process_returns_failure_when_tolerance_is_too_strict():
    result = _module(
        "verify",
        "--center",
        "1.2",
        "--width",
        "0.9",
        "--position",
        "0.8",
        "--tolerance",
        "1e-30",
    )
    assert result.returncode == 1
    assert "FAIL:" in result.stdout


def test_module_prime_json_is_parseable():
    result = _module("prime-sum", "--limit", "12", "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["cutoff"] == 12
    assert report["no_tail_bound"] is True


def test_benchmark_json_has_reproducibility_metadata(capsys):
    assert main(["benchmark", "--points", "101", "--repeats", "2", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["points"] == 101
    assert report["repeats"] == 2
    assert report["grid"] == "linspace(-8, 8, points)"
    assert report["python_version"]
    assert report["numpy_version"]
    assert len(report["measurements"]) == 3
    for row in report["measurements"]:
        assert row["min_seconds"] <= row["median_seconds"] <= row["max_seconds"]
        assert math.isfinite(row["checksum"])


def test_benchmark_multi_zero_json(capsys):
    assert (
        main(
            [
                "benchmark",
                "--zero",
                "0.5j",
                "--zero",
                "1+0.5j",
                "--points",
                "10",
                "--repeats",
                "1",
                "--json",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["zero_generators"] == ["0.5j", "(1+0.5j)"]
    assert all(row["median_seconds"] >= 0 for row in report["measurements"])


def test_benchmark_text_labels_local_wall_clock(capsys):
    assert main(["benchmark", "--points", "10", "--repeats", "1"]) == 0
    output = capsys.readouterr().out
    assert "median (ms)" in output
    assert "not comparable across machines" in output


@pytest.mark.parametrize(
    "args",
    [
        ["benchmark", "--points", "9"],
        ["benchmark", "--points", "1000001"],
        ["benchmark", "--repeats", "0"],
        ["benchmark", "--repeats", "51"],
        ["benchmark", "--zero", "nan"],
        ["benchmark", "--width", "0"],
    ],
)
def test_benchmark_rejects_bad_arguments(args):
    result = _module(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_module_benchmark_process_runs():
    result = _module("benchmark", "--points", "100", "--repeats", "1", "--json")
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["points"] == 100
    assert len(report["measurements"]) == 3


def test_missing_command_is_usage_error():
    result = _module()
    assert result.returncode == 2
    assert "required" in result.stderr


def test_reproduce_subprocess_writes_matching_json_and_markdown_artifacts(tmp_path):
    result = _module("reproduce", "--check", "--json", "--output-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "golden-vector check passed" in result.stderr
    json_path = tmp_path / "reproducibility.json"
    markdown_path = tmp_path / "reproducibility.md"
    assert json_path.read_text(encoding="utf-8") == result.stdout
    report = json.loads(result.stdout)
    assert report["all_passed"] is True
    assert report["golden_vector_check"] == {
        "passed": True,
        "problems": [],
        "max_absolute_error": 0.0,
    }
    assert report["provenance"]["random_seed"] is None
    assert len(report["rows"]) == 4
    assert len(report["gram"]["eigenvalues"]) == 3
    assert [row["cutoff"] for row in report["extremal"]["cutoffs"]] == [20, 50]
    assert report["finite_extremal_regression_check"]["passed"]
    assert report["spatial_tail_certificate_check"]["passed"]
    assert len(report["spatial_tail"]["certificates"]) == 2
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Weil Gaussian Lab reproducibility report" in markdown
    for row in report["rows"]:
        assert row["id"] in markdown
    assert "All checks passed: true." in markdown
    assert "Finite-cutoff Rayleigh–Ritz diagnostics" in markdown
    assert "Analytic spatial L1 tail certificates" in markdown
    assert "Golden-vector fixture check: passed." in markdown


def test_packaged_golden_fixture_is_identical_to_test_reference():
    packaged = resources.files("weil_gaussian").joinpath("data", "golden_vectors.json")
    committed = ROOT / "tests/data/golden_vectors.json"
    assert packaged.read_bytes() == committed.read_bytes()


def test_reproduce_without_output_dir_only_prints_markdown(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert main(["reproduce"]) == 0
    output = capsys.readouterr().out
    assert output.startswith("# Weil Gaussian Lab reproducibility report")
    assert list(tmp_path.iterdir()) == []


def test_reproduce_check_detects_changed_high_precision_reference(tmp_path, capsys):
    fixture = json.loads(
        (ROOT / "tests/data/golden_vectors.json").read_text(encoding="utf-8")
    )
    fixture["cases"][0]["reference"]["inverse"] = "0"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(fixture), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    assert (
        main(
            [
                "reproduce",
                "--check",
                "--golden-file",
                str(changed),
                "--output-dir",
                str(artifacts),
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["all_passed"] is False
    assert report["golden_vector_check"]["passed"] is False
    assert (
        "base_origin.inverse reference differs"
        in report["golden_vector_check"]["problems"]
    )
    assert (
        json.loads((artifacts / "reproducibility.json").read_text(encoding="utf-8"))
        == report
    )
    markdown = (artifacts / "reproducibility.md").read_text(encoding="utf-8")
    assert "All checks passed: false." in markdown
    assert "Golden-vector fixture check: failed." in markdown
    assert "base_origin.inverse reference differs" in captured.err


@pytest.mark.parametrize(
    "section", ("finite_extremal_regression", "spatial_tail_certificate")
)
def test_reproduce_check_fails_for_tampered_v11_fixture_section(
    tmp_path, capsys, section: str
):
    fixture = json.loads(
        (ROOT / "tests/data/golden_vectors.json").read_text(encoding="utf-8")
    )
    if section == "finite_extremal_regression":
        fixture[section]["cutoffs"][0]["constrained"]["minimum"][
            "rayleigh_quotient"
        ] += 1
    else:
        fixture[section]["certificates"][0]["bound_decimal"] = "0"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(fixture), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    assert (
        main(
            [
                "reproduce",
                "--check",
                "--golden-file",
                str(changed),
                "--output-dir",
                str(artifacts),
                "--json",
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert not report["all_passed"]
    assert not report["golden_vector_check"]["passed"]
    assert any(
        "finite extremal" in problem or "spatial tail certificate" in problem
        for problem in report["golden_vector_check"]["problems"]
    )
    assert not json.loads(
        (artifacts / "reproducibility.json").read_text(encoding="utf-8")
    )["all_passed"]
    assert "All checks passed: false." in (artifacts / "reproducibility.md").read_text(
        encoding="utf-8"
    )


def test_reproduce_check_rejects_missing_fixture(tmp_path, capsys):
    fixture = tmp_path / "missing.json"
    assert main(["reproduce", "--golden-file", str(fixture), "--json"]) == 1
    captured = capsys.readouterr()
    assert "golden-vector fixture missing" in captured.err
    report = json.loads(captured.out)
    assert report["all_passed"] is False
    assert report["golden_vector_check"]["passed"] is False
    assert "fixture missing" in report["golden_vector_check"]["problems"][0]


@pytest.mark.parametrize(
    ("contents", "problem"),
    [("{", "not valid JSON"), ("[]", "not a JSON object")],
)
def test_reproduce_check_rejects_malformed_fixture(tmp_path, capsys, contents, problem):
    fixture = tmp_path / "invalid.json"
    fixture.write_text(contents, encoding="utf-8")
    assert main(["reproduce", "--golden-file", str(fixture), "--json"]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["all_passed"] is False
    assert report["golden_vector_check"]["passed"] is False
    assert problem in report["golden_vector_check"]["problems"][0]
    assert "golden-vector check failed" in captured.err


def test_reproduce_rejects_output_dir_that_is_a_file(tmp_path, capsys):
    destination = tmp_path / "occupied"
    destination.write_text("keep", encoding="utf-8")
    assert main(["reproduce", "--output-dir", str(destination)]) == 1
    captured = capsys.readouterr()
    assert "command failed" in captured.err
    assert destination.read_text(encoding="utf-8") == "keep"


def test_atomic_artifact_write_preserves_old_file_on_replace_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "reproducibility.json"
    destination.write_text("original", encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(cli.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        cli._write_text_atomic(destination, "updated")
    assert destination.read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["reproducibility.json"]
