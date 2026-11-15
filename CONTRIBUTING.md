# Contributing

We welcome corrections, independent checks, and small, reviewable extensions.
The priority is mathematical traceability: a reader should be able to connect
each numerical result to a definition, parameter domain, and error estimate.

## Start with an issue

For a suspected formula error, use the
[mathematical correction template](.github/ISSUE_TEMPLATE/mathematical-correction.yml).
Identify the equation or function, give the parameters and Fourier convention,
and provide an independent derivation or reproducible counterexample. For
software failures, use the bug template. Security issues should follow
[SECURITY.md](SECURITY.md).

## Prepare a change

1. Install the project and test dependencies with
   **python -m pip install -e '.[test]'**.
2. Keep mathematical assumptions explicit. The convention is
   \( \widehat f(\xi)=\int f(x)e^{-i\xi x}\,dx \), with inverse factor
   \(1/(2\pi)\). Track every factor of \(2\), \(\pi\), and sign through
   Fourier inversion, autocorrelation, and prime-power sums.
3. Add a test that uses an independent oracle where practical: quadrature
   from the defining spectrum, symbolic differentiation, or an independently
   written prime-power enumerator. A test that repeats the implementation's
   algebra is weaker evidence.
4. Exercise scalar and array inputs, \(a=0\), narrow and broad widths,
   repeated or conjugate-related zeros, and invalid inputs when relevant.
   For Gram matrices, check symmetry or Hermiticity and distinguish the
   unconditional \(L^2\) Gram from exploratory finite Weil diagnostics.
5. For numerical claims, state the precision, quadrature subdivisions,
   stopping criterion, and the observed error. A finite prime cutoff does
   not bound the omitted infinite tail.
6. Update the technical note and README when the API or mathematical
   definitions change. Keep the note's author attribution and citation
   metadata consistent.

Run the checks before opening a pull request:

~~~bash
python -m pytest -q
python examples/compare_with_quadrature.py
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error gaussian_identities.tex
~~~

Use the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) to explain
the change and show the checks. CI runs independent numerical tests and builds
the note. Codex-assisted review may flag missing tests, unsafe inputs, or
normalization changes; a maintainer must still verify mathematical reasoning.
Numerical agreement is evidence for an implementation, not a proof of a
general theorem or the Riemann Hypothesis.
