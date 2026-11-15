"""Keep the manuscript's rounded numerical table tied to the report API."""

from __future__ import annotations

import re
from pathlib import Path

from weil_gaussian.repro import reproduce_report

_SCIENTIFIC_TEX = re.compile(r"\$([0-9.]+)\\times10\^\{(-?\d+)\}\$")
_CASE_LABEL = re.compile(r"\\texttt\{([a-z\\_]+)\}")


def _scientific_value(text: str) -> float:
    match = _SCIENTIFIC_TEX.search(text)
    assert match is not None, f"missing TeX scientific number: {text}"
    return float(match.group(1)) * 10 ** int(match.group(2))


def test_paper_numerical_table_matches_reproducibility_report() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "paper" / "gaussian_identities.tex"
    ).read_text(encoding="utf-8")
    report = reproduce_report()
    table = re.search(r"\\begin\{table\}.*?\\end\{table\}", source, re.DOTALL)
    assert table is not None
    assert r"\label{tab:golden}" in table.group()
    paper_rows = [
        line
        for line in table.group().splitlines()
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
