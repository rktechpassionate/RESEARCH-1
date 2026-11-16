"""Keep the manuscript's rounded numerical table tied to the report API."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

from weil_gaussian.repro import reproduce_report

_SCIENTIFIC_TEX = re.compile(r"\$([0-9.]+)\\times10\^\{(-?\d+)\}\$")
_CASE_LABEL = re.compile(r"\\texttt\{([a-z\\_]+)\}")


def _table_with_label(source: str, label: str) -> str:
    for table in re.findall(r"\\begin\{table\}.*?\\end\{table\}", source, re.DOTALL):
        if rf"\label{{{label}}}" in table:
            return table
    raise AssertionError(f"paper table {label} is missing")


def _scientific_value(text: str) -> float:
    match = _SCIENTIFIC_TEX.search(text)
    assert match is not None, f"missing TeX scientific number: {text}"
    return float(match.group(1)) * 10 ** int(match.group(2))


def test_paper_numerical_table_matches_reproducibility_report() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "paper" / "gaussian_identities.tex"
    ).read_text(encoding="utf-8")
    report = reproduce_report()
    table = _table_with_label(source, "tab:golden")
    paper_rows = [
        line
        for line in table.splitlines()
        if line.startswith(r"\texttt{") and "&" in line
    ]
    assert len(paper_rows) == len(report["rows"])
    for line, row in zip(paper_rows, report["rows"], strict=True):
        cells = [cell.strip() for cell in line.split("&")]
        assert len(cells) == 7
        label = _CASE_LABEL.fullmatch(cells[0])
        assert label is not None
        assert label.group(1).replace(r"\_", "_") == row["id"]
        assert cells[1:4] == [row["center"], row["width"], row["position"]]
        assert cells[4] == f"{row['analytic']['inverse']:.10g}"
        assert cells[5] == f"{row['analytic']['autocorrelation']:.10g}"
        paper_error = _scientific_value(cells[6])
        observed_error = row["max_absolute_error"]
        tolerance = report["tolerances"]["float64_absolute"] + report["tolerances"][
            "float64_relative"
        ] * max(abs(float(value)) for value in row["reference"].values())
        assert 0 < paper_error < tolerance
        assert observed_error < tolerance
        if observed_error > 0:
            # The manuscript labels this as one representative float64 run.
            assert 0.1 < paper_error / observed_error < 10

    eigen_and_condition = re.search(
        r"smallest eigenvalue of \$([0-9.]+)\$ and\s+condition number \$([0-9.]+)\$",
        source,
    )
    assert eigen_and_condition is not None
    assert eigen_and_condition.group(1) == f"{report['gram']['min_eigenvalue']:.10g}"
    assert eigen_and_condition.group(2) == f"{report['gram']['condition_number']:.10g}"
    matrix_error = re.search(
        r"maximum observed matrix-entry absolute error is\s+(\$[0-9.]+\\times10\^\{-?\d+\}\$)",
        source,
    )
    assert matrix_error is not None
    paper_matrix_error = _scientific_value(matrix_error.group(1))
    observed_matrix_error = report["gram"]["max_matrix_absolute_error"]
    assert 0.1 < paper_matrix_error / observed_matrix_error < 10


def test_paper_extremal_table_matches_reproducible_finite_diagnostics() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "paper" / "gaussian_identities.tex"
    ).read_text(encoding="utf-8")
    table = _table_with_label(source, "tab:extremal")
    report = reproduce_report()
    paper_rows = [line for line in table.splitlines() if re.match(r"^(20|50) &", line)]
    expected = [
        (row, mode, label)
        for row in report["extremal"]["cutoffs"]
        for mode, label in (
            ("unconstrained", "Unconstrained"),
            ("constrained", r"$F_c(0.7+0.3i)=0$"),
        )
    ]
    assert len(paper_rows) == len(expected)
    for line, (row, mode, label) in zip(paper_rows, expected, strict=True):
        cells = [cell.strip().rstrip("\\") for cell in line.split("&")]
        assert cells == [
            str(row["cutoff"]),
            label,
            f"{row[mode]['minimum']['rayleigh_quotient']:.6f}",
            f"{row[mode]['maximum']['rayleigh_quotient']:.6f}",
        ]


def test_paper_spatial_tail_figures_are_rounded_up_from_certificates() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "paper" / "gaussian_identities.tex"
    ).read_text(encoding="utf-8")
    report = reproduce_report()
    figures = re.search(
        r"certificate gives .*?\\le (\d+\.\d+)\\times10\^\{(-?\d+)\}.*?"
        r"\\le (\d+\.\d+)\\times10\^\{(-?\d+)\}",
        source,
        re.DOTALL,
    )
    assert figures is not None
    for index, certificate in enumerate(report["spatial_tail"]["certificates"]):
        mantissa = figures.group(1 + 2 * index)
        exponent = figures.group(2 + 2 * index)
        displayed = Decimal(mantissa) * (Decimal(10) ** int(exponent))
        certified = Decimal(certificate["bound_decimal"])
        assert displayed >= certified > 0
        assert displayed < certified * Decimal("1.001")
