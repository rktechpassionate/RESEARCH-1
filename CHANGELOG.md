# Changelog

Changes are grouped by the software versions represented in this repository.
The original manuscript drafts predate the packaged library. Their broader
claims are historical material, not claims of the current release; see
[research scope](docs/RESEARCH_SCOPE.md).

## 1.1.0

- Add constrained Rayleigh–Ritz optimization on finite polynomial-Gaussian
  bases, using real coefficients for real-even test functions and an optional
  complex-coefficient algebraic extension. The solver reports generalized
  eigenpairs, the projected stationarity residual, the full residual, and
  prescribed-zero constraint residuals separately.
- Add analytic two-sided spatial-tail bounds for the inverse transform and
  autocorrelation. Exact rational Gaussian envelopes and outward-rounded
  decimal evaluation bound the truncation beyond a positive radius.
- Extend the CLI, reproducibility report, golden-vector regressions, tests,
  and manuscript with the new finite-dimensional and tail calculations.
  Neither addition bounds the omitted infinite prime-power tail.

## 1.0.0

- Add a deterministic `reproduce` command and committed high-precision golden
  vectors. The command emits machine-readable JSON and a Markdown report for
  the numerical examples in the [technical note](paper/gaussian_identities.tex).
- Cross-check selected inverse Fourier and autocorrelation values with adaptive
  Gauss–Kronrod and Gauss–Hermite quadrature. Report numerical error indicators
  and cross-scheme discrepancies without presenting them as rigorous bounds.
- Add relative spectral decay and perturbation sensitivity diagnostics for
  computed $L^2$ Gram matrices, with Weyl bounds on those matrices.
- Publish PEP 561 type information and require Ruff, strict mypy, pytest on
  Python 3.10–3.12, and a LaTeX build in CI.
- Document the module boundaries, Fourier convention, error interpretation,
  and paper-to-code workflow.

## 0.2.0

- Extend the original Gaussian family to real-even polynomial-Gaussian spectra
  with multiple requested complex zeros and analytic inverse/autocorrelation
  evaluation.
- Add $L^2$ Gram and finite-cutoff Weil matrix diagnostics, finite
  prime-power/archimedean balance, a CLI verifier and benchmark, tests, and CI.
- Correct the prime-side factor under the stated Fourier convention and
  explicitly separate finite diagnostics from claims about the full Weil form.
- Add the MIT license, citation metadata, contribution and security guidance,
  and a LaTeX technical note.

## 0.1.0 local prototype and initial drafts

- Begin with local LaTeX research drafts and the two-parameter Gaussian
  test-function calculations. Version 0.1.0 was a local prototype, not a
  published package release. The earlier draft is retained in Git history
  for provenance; the current technical note and
  [research scope](docs/RESEARCH_SCOPE.md) state the supported results.
- Establish the Fourier convention and closed-form inverse transform,
  autocorrelation, total mass, and zero-lag normalization later carried into
  the Python library.
