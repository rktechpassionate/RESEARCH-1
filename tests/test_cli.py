"""CLI behavior and process exit codes for reproducible diagnostics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from weil_gaussian import autocorrelation, truncated_prime_term
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
    assert main(
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
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["quadrature_digits"] == 45
    assert report["passed"] is True
    assert {row["quantity"] for row in report["rows"]} == {
        "inverse",
        "autocorrelation",
    }
    assert max(row["absolute_error"] for row in report["rows"]) < 1e-12


def test_verify_multi_zero_family_against_independent_quadrature(capsys):
    assert main(
        [
            "verify", "--zero", "0.5j", "--zero", "1+0.5j",
            "--position", "0.8", "--dps", "45", "--json",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["zero_generators"] == ["0.5j", "(1+0.5j)"]
    assert report["passed"] is True
    assert max(row["absolute_error"] for row in report["rows"]) < 1e-9


def test_verify_arbitrary_complex_zero_family_without_weil_claim(capsys):
    assert main(
        ["verify", "--zero", "1+0.5i", "--position", "0.3", "--json"]
    ) == 0
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
    assert main(["verify", "--center", "0", "--width", "1", "--position", "0", "--json"]) == 0
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
    assert main(
        ["prime-sum", "--center", "1.3", "--width", "0.9", "--limit", "30", "--json"]
    ) == 0
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
    result = _module(
        "verify", "--center", "1.2", "--width", "0.9", "--position", "0.8"
    )
    assert result.returncode == 0
    assert "PASS:" in result.stdout


def test_module_verify_process_returns_failure_when_tolerance_is_too_strict():
    result = _module(
        "verify", "--center", "1.2", "--width", "0.9", "--position", "0.8",
        "--tolerance", "1e-30"
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
    assert main(
        [
            "benchmark", "--zero", "0.5j", "--zero", "1+0.5j",
            "--points", "10", "--repeats", "1", "--json",
        ]
    ) == 0
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
