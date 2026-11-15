# Weil Gaussian Lab

[![CI](https://github.com/rktechpassionate/RESEARCH-1/actions/workflows/ci.yml/badge.svg)](https://github.com/rktechpassionate/RESEARCH-1/actions/workflows/ci.yml)
![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)
[![MIT license](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Weil Gaussian Lab 1.0.0 is an MIT-licensed, typed Python library for
reproducible experiments with polynomial-Gaussian test functions. It combines
analytic Fourier and autocorrelation formulas, multiple prescribed complex
zeros, $L^2$ Gram diagnostics, finite prime-power calculations, two numerical
quadrature cross-checks, and committed high-precision reference vectors. The
[technical note (PDF)](paper/gaussian_identities.pdf) gives definitions,
derivations, and a numerical table matching the CLI report; its
[LaTeX source](paper/gaussian_identities.tex) is included.

**Scope:** the \(L^2\) Gram matrix is positive semidefinite by construction.
Finite Weil-form and prime-balance calculations are numerical diagnostics,
without a certified infinite prime tail. Neither they nor the test suite
establish the Riemann Hypothesis. See [research scope](docs/RESEARCH_SCOPE.md)
for the claim boundaries and the correction to an older draft.

## Mathematical model

The Fourier convention is

$$
\widehat f(\xi)=\int_{\mathbb R}f(x)e^{-i\xi x}\,dx,\qquad
f(x)=\frac{1}{2\pi}\int_{\mathbb R}\widehat f(\xi)e^{i\xi x}\,d\xi.
$$

For center \(a\geq0\), width \(\sigma>0\), and specified complex zeros \(Z\),
the spectrum is a real-even polynomial in \(\xi\) times a pair of shifted
Gaussians:

$$
F_{a,\sigma,Z}(\xi)=P_Z(\xi^2)
\left(e^{-(\xi-a)^2/(2\sigma^2)}
     +e^{-(\xi+a)^2/(2\sigma^2)}\right).
$$

Each requested zero enforces its negative; non-axis complex zeros also enforce
their conjugates. This keeps the spectrum real and even on the real line.
With \(Z=\{i/2\}\), \(P_Z(\xi^2)=\xi^2+1/4\), matching the original family.
Repeated requested zeros increase multiplicity. The package evaluates the
inverse transform and autocorrelation analytically for these families.
The family reports whether it has the \(i/2\) zero required by the
Weil-named diagnostics; a custom zero set without it remains useful for
Gaussian calculations but is rejected by those diagnostics.

![Normalized Fourier, inverse-transform, and autocorrelation profiles for three centers](docs/figures/profiles.png)

The figure normalizes each curve separately to compare shape, not amplitude.
Recreate it with the optional plotting extra:

~~~bash
python -m pip install -e '.[plot]'
python examples/plot_profiles.py
~~~

## Install and quickstart

Python 3.10 or newer is required.

~~~bash
python -m pip install -e '.[dev]'
weil-gaussian verify --center 1.2 --width 0.9 --position 0.8 --dps 40
python -m weil_gaussian prime-sum --center 1.2 --width 0.9 --limit 100
weil-gaussian reproduce --check --output-dir artifacts
weil-gaussian benchmark --points 100 --repeats 2
~~~

A `reproduce` run prints the paper's fixed numerical table and writes
`artifacts/reproducibility.json` and `artifacts/reproducibility.md`.
An excerpt from one run of the generated Markdown report is:

~~~text
| Case | a | sigma | x | f(x) | Phi(x) | Max absolute error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base_origin | 1.2 | 0.9 | 0 | 1.795240262 | 3.64469161 | 3.064e-16 |
| base_offset | 1.2 | 0.9 | 0.8 | -0.04490139209 | -0.06085600649 | 3.827e-16 |
| two_imaginary_zeros | 0.7 | 1.1 | 0.8 | -2.906565367 | -29.2549345 | 3.282e-15 |
| complex_zero_orbit | 1.0 | 0.8 | 0.4 | 9.318736459 | 265.1303927 | 1.787e-13 |

Gram minimum eigenvalue: 0.3224654265.
Gram condition number: 1883.346262.
Maximum Gram matrix absolute error: 1.291e-12.
All checks passed: true.
~~~

`reproduce --json` prints the structured report; `--check` compares
regenerated high-precision values with
[the committed golden vectors](tests/data/golden_vectors.json), whose
byte-identical copy ships in the wheel. The values
shown above are double-precision analytic results compared with 70-digit
references from direct spectrum evaluation and Fourier integrals. Each
maximum error includes the Fourier profile,
inverse transform, autocorrelation, and total mass for that case. These
case-specific errors have stated tolerances; they do not guarantee
machine-epsilon accuracy for all parameter choices or platforms. The cases
use no randomness, so no seed is needed. The observed error digits can vary
slightly across numerical platforms. The float64 comparison criterion is
`5e-12 + 5e-12 × |reference|`; committed decimal vectors are checked at
`1e-48` absolute plus relative tolerance.

`verify` compares analytic inverse and autocorrelation values with
high-precision `mpmath` integration and two float64 quadrature routes:
adaptive Gauss–Kronrod and Gauss–Hermite. Its errors, adaptive estimates,
order-doubling differences, and pass thresholds are numerical checks, not
rigorous error certificates. Repeat `--zero` for a custom multi-zero
family, for example `--zero 0.5j --zero 1+0.5j`.

The finite prime-power demonstration reports an archimedean component and
a cutoff-dependent balance. Abbreviated output:

~~~text
Finite prime-power diagnostic: center=1.2, width=0.9, cutoff=100
Prime-power terms through cutoff: 35
2 * sum Lambda(n) Phi(log n) / sqrt(n) = -3.09114500231
Archimedean K = -4.51535568786
Finite P_N = -3.09114500231
Cutoff-dependent K - P_N = -1.42421068555
Quadrature error estimate = 1.68e-12; converged=True
... individual prime-power terms ...
No prime-tail bound; not the full Weil form or an RH test.
~~~

The benchmark reports median and range of wall-clock timings plus Python and
NumPy versions; timings vary by machine and are not cross-machine performance
claims. All four commands support help and JSON output for reproducible
scripts.

The Python API also accepts NumPy-compatible arrays:

~~~python
import numpy as np
from weil_gaussian import PolynomialGaussianFamily

family = PolynomialGaussianFamily(
    center=2.0, width=0.8, zeros=(0.5j, 1.0 + 0.5j)
)
frequency = np.linspace(-8, 8, 201)
position = np.linspace(-5, 5, 201)
spectrum = family.fourier(frequency)
signal = family.inverse(position)
overlap = family.autocorrelation(position)
~~~

## Architecture

| Component | Purpose | Interpretation |
| --- | --- | --- |
| [Core formulas](src/weil_gaussian/core.py) | Original two-parameter spectrum, inverse, autocorrelation, and exact mass | Analytic identities |
| [Multi-zero families](src/weil_gaussian/families.py) | Real-even polynomial construction and analytic transforms | Zeros of the chosen test function |
| [Gram diagnostics](src/weil_gaussian/gram.py) | Pairings, eigenvalues, and conditioning across parameter grids | \(L^2\) positivity is unconditional; finite Weil results are exploratory |
| [Arithmetic](src/weil_gaussian/arithmetic.py) | Enumerated prime-power terms through a cutoff | Finite sum, no tail estimate |
| [Balance diagnostics](src/weil_gaussian/balance.py) | Archimedean integral versus finite prime-power term | Numerical balance, not the full Weil value |
| [Verification](src/weil_gaussian/verification.py) | Two independent quadrature schemes and explicit residuals | Numerical evidence, not an integral error certificate |
| [Reproducibility](src/weil_gaussian/repro.py) | Fixed paper cases, high-precision spectrum evaluation and integrals, and golden vectors | Deterministic case-specific regression |
| [CLI](src/weil_gaussian/cli.py) | Verification, finite prime balance, benchmark, and paper artifacts | Human-readable and structured output |
| [Tests](tests) and [CI](.github/workflows/ci.yml) | Fourier invariants, golden vectors, Ruff, strict mypy, pytest, and LaTeX | Regression evidence, not proof of all claims |

The \(L^2\) Gram entries use
\(G_{ij}=(2\pi)^{-1}\int_{\mathbb R}F_i(\xi)F_j(\xi)\,d\xi\).
Its eigenvalues and condition number describe linear dependence of the chosen
test functions. Relative eigenvalues and cumulative spectral energy show
spectral decay. A sensitivity diagnostic compares two computed Grams and
uses a Weyl bound based on their matrix difference; this bound does not
include a certified enclosure for errors in the entries. The finite Weil
matrix and balance output depend on a specified prime cutoff; their
eigenvalues cannot certify spectral positivity for the infinite problem.

See [architecture and numerical contract](docs/ARCHITECTURE.md) for module
boundaries, normalization invariants, error interpretation, PEP 561 type
information, and PR review.

## Verify and contribute

~~~bash
weil-gaussian reproduce --check --output-dir artifacts
python -m pytest -q
ruff check .
ruff format --check .
mypy
python examples/compare_with_quadrature.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error gaussian_identities.tex
~~~

CI runs Ruff lint and format, strict mypy and pytest on Python 3.10, 3.11,
and 3.12, plus the manuscript build. Tests compare closed forms with
independent defining integrals and exercise symmetry, normalization,
parameter limits, invalid inputs, CLI behavior, and matrix diagnostics.
The library's regular arrays use double-precision NumPy values; extreme
parameters can lose relative precision or overflow. Report sensitive cases
with a high-precision reference and a stated error estimate.

See [CONTRIBUTING.md](CONTRIBUTING.md) for mathematical and code review
requirements, [SECURITY.md](SECURITY.md) for private vulnerability reporting,
[LICENSE](LICENSE) for reuse, [CHANGELOG.md](CHANGELOG.md) for release history,
and [CITATION.cff](CITATION.cff) for citation.
