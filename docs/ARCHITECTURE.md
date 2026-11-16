# Architecture and numerical contract

Weil Gaussian Lab separates exact identities, numerical cross-checks, and
finite arithmetic diagnostics. The [technical note](../paper/gaussian_identities.tex)
derives the formulas. The [research scope](RESEARCH_SCOPE.md) records which
claims are proved and which outputs depend on numerical integration or a
prime cutoff.

## Module map

| Module | Responsibility | Main contract |
| --- | --- | --- |
| [`core.py`](../src/weil_gaussian/core.py) | Original two-parameter Gaussian formulas | Analytic Fourier, inverse, autocorrelation, and normalization identities |
| [`families.py`](../src/weil_gaussian/families.py) | Real-even polynomial factors for requested complex zeros | Analytic transforms via Hermite polynomials; arrays in double precision |
| [`gram.py`](../src/weil_gaussian/gram.py) | Cross-correlations, $L^2$ Grams, spectra, parameter-grid and perturbation diagnostics | An exact $L^2$ Gram is positive semidefinite; reported eigenvalues belong to computed matrices |
| [`extremal.py`](../src/weil_gaussian/extremal.py) | Constrained Rayleigh–Ritz reduction of the finite $W^{(N)}$ and $L^2$ Gram pencil | Numerical extrema on a selected finite basis and cutoff; no infinite-prime-tail conclusion |
| [`certification.py`](../src/weil_gaussian/certification.py) | Rational Gaussian envelopes and outward-rounded spatial-tail bounds | Two-sided $L^1$ tails of $f$ or $\Phi$ beyond a positive radius, not the omitted prime-power tail |
| [`verification.py`](../src/weil_gaussian/verification.py) | Quadrature cross-checks | Adaptive Gauss–Kronrod and Gauss–Hermite estimates, residuals, and pass/fail thresholds |
| [`repro.py`](../src/weil_gaussian/repro.py) | Deterministic paper data and golden-vector comparison | Fixed cases and precision, structured reports and regression vectors |
| [`arithmetic.py`](../src/weil_gaussian/arithmetic.py), [`balance.py`](../src/weil_gaussian/balance.py) | Prime-power enumeration and archimedean integration | Finite-cutoff outputs; no omitted-prime-tail bound |
| [`cli.py`](../src/weil_gaussian/cli.py) | `verify`, `prime-sum`, `extremal`, `benchmark`, `reproduce` | Human-readable and JSON output; nonzero exit on a failed check |

The package ships a `py.typed` marker for PEP 561. Public interfaces are
statically checked with strict mypy. The marker conveys type information; it
does not assert that a floating-point calculation is mathematically exact.

## Fourier normalization

Every formula and test uses

$$
\widehat f(\xi)=\int_{\mathbb R}f(x)e^{-i\xi x}\,dx,\qquad
f(x)=\frac{1}{2\pi}\int_{\mathbb R}\widehat f(\xi)e^{i\xi x}\,d\xi.
$$

For a real-even profile $F$, autocorrelation has transform $F^2$. Thus

$$
\Phi(x)=\frac{1}{2\pi}\int_{\mathbb R}F(\xi)^2e^{i\xi x}\,d\xi,
\quad
\Phi(0)=\|f\|_2^2,
\quad
\int_{\mathbb R}\Phi(x)\,dx=F(0)^2.
$$

For two families, the cross-correlation at zero is the $L^2$ Gram entry
$G_{ij}=\langle f_i,f_j\rangle$. Real-even symmetry, Parseval's $1/(2\pi)$
factor, and the factor $2$ in the finite prime-power term are explicit
regression invariants. A change to the Fourier convention requires updating
the derivation, implementation, golden vectors, and tests together.

## Numerical evidence and limits

`verify` compares analytic inverse and autocorrelation values against two
numerical integrations of their Fourier definitions. Adaptive SciPy `quad`
uses Gauss–Kronrod with intervals split near Gaussian peaks. Gauss–Hermite
uses Gaussian-product decomposition and an order-doubling comparison. Each
check records absolute residuals, the difference between the two schemes,
the adaptive integrator's error estimate, the Hermite order-doubling delta,
and the pass threshold. These indicators expose disagreement; neither is a
rigorous enclosure for the exact real integral. High-precision `mpmath`
values in [`tests/data/golden_vectors.json`](../tests/data/golden_vectors.json)
provide a separate regression reference for specific parameter/position
cases. Decimal strings retain reference precision, while explicitly stored
float64 tolerances govern cross-platform comparisons. The criterion is
$5\times10^{-12}+5\times10^{-12}|\text{reference}|$ for float64 values;
regenerated decimal references are checked at $10^{-48}$ absolute plus
relative tolerance. Observed float64 error digits can vary by platform.
These tolerances are not universal machine-epsilon guarantees for
arbitrary inputs.

For a computed symmetric Gram $G$, `gram.py` diagonalizes $G$ with
`numpy.linalg.eigvalsh`. The condition number is
$\lambda_{\max}/\lambda_{\min}$ only when the smallest eigenvalue exceeds
the configured relative threshold times the spectral scale; otherwise it is
infinite. The relative spectrum divides the positive-part eigenvalues,
ordered from largest to smallest, by the largest. Cumulative spectral energy
is the cumulative positive-part sum divided by its total.

For two computed matrices $G$ and $G+E$, the perturbation diagnostic
reports $\|E\|_2$ and uses Weyl's inequality:

$$
|\lambda_i(G+E)-\lambda_i(G)|\leq\|E\|_2.
$$

When $\lambda_{\min}(G)>\|E\|_2$, it also reports the upper condition
estimate $(\lambda_{\max}+\|E\|_2)/(\lambda_{\min}-\|E\|_2)$; otherwise
the upper estimate is infinite. These are matrix-level bounds for the
computed entries. They do not include a rigorous enclosure for all floating
point and quadrature error relative to an exact Gram.

The finite Weil diagnostic forms $W^{(N)}=K-B^{(N)}$, where $B^{(N)}$
sums prime powers through $N$. It cannot stand in for the complete Weil
form because there is no certified bound for the omitted prime tail. Its
eigenvalues are cutoff dependent and make no claim about zeta zeros.

## Constrained finite-basis optimization

For real basis coefficients $c$, `extremal.py` optimizes
$c^{\mathsf T}W^{(N)}c/(c^{\mathsf T}Gc)$ over a selected finite family.
Here $G_{ij}=\langle f_i,f_j\rangle$ is an exact $L^2$ Gram expression;
`gram.py` evaluates its entries in floating point. $W^{(N)}$ includes a
numerically integrated archimedean term and prime powers only through $N$.
Neither reported matrix is an interval enclosure of its exact entries.

For each requested complex Fourier zero $z$, the real and imaginary parts
of $(F_j(z))_j$ form linear rows of a real constraint matrix $A$. An
orthonormal numerical basis $Z$ for $\ker A$ reduces the pencil to
$(Z^{\mathsf T}W^{(N)}Z,Z^{\mathsf T}GZ)$. This imposes the selected
linear constraints without a penalty term. The reduced metric must be
positive definite and adequately conditioned; ill-conditioned cases are
rejected instead of silently changing the quotient with a ridge term.
The solver reports coefficient vectors normalized in the computed $G$
metric, Rayleigh quotients, constraint residuals, and both full and
projected generalized residuals. At a constrained optimum,
$Z^{\mathsf T}(W^{(N)}c-\lambda Gc)$ is small. The full residual can be
nonzero because the constraints have Lagrange multipliers. A small
projected residual validates the numerical finite problem only.
The API also offers `coefficient_field="complex"` as a Hermitian algebraic
extension. One prescribed frequency then gives one complex linear equation;
the resulting combination need not be a real-even Weil test function. The
CLI and paper example use the real-coefficient setting.

## Analytic spatial-tail certificate

`certification.py` bounds the two-sided absolute spatial integral
$\int_{|x|>R}|h(x)|\,dx$ for $R>0$ and either $h=f_{P,a,\sigma}$ or its
autocorrelation $h=\Phi_{P,a,\sigma}$. The proof expands Gaussian
derivatives from the exact binary-rational input values, takes absolute
values coefficientwise, and uses a Gaussian moment recurrence. The base
moment is expressed with `erfc` and bounded by the Gaussian Mills
inequality. Decimal operations round upward for the reported bound.
The implementation first obtains an exact rational inequality
$T\leq A\exp(-\alpha R^2)$. It rounds the negative exponent upward,
uses `Decimal.exp`'s correctly rounded result followed by `next_plus`,
rounds the positive rational prefactor upward, and multiplies with upward
rounding. The returned decimal is therefore an upper endpoint for this
analytic expression at the requested precision.
Coefficientwise absolute values and the Mills inequality may make the
bound conservative, especially at high polynomial degree.
The certificate concerns the exact mathematical function defined by the
supplied finite floats. It does not enclose the separate double-precision
implementation's roundoff, a frequency-domain quadrature error, or the
prime-power tail omitted from $W^{(N)}$.
The optional `verify --tail-radius R` field prints both spatial certificates
beside the quadrature diagnostics with separate labels and assumptions.

## Paper-to-code workflow

~~~bash
python -m pip install -e '.[dev]'
weil-gaussian reproduce --check
weil-gaussian reproduce --output-dir artifacts
weil-gaussian extremal --cutoff 20 --cutoff 50 --zero 0.7+0.3j --json
python -m pytest -q
ruff check .
ruff format --check .
mypy
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error gaussian_identities.tex
~~~

`reproduce` regenerates the fixed numerical cases and writes JSON and
Markdown artifacts. The manuscript's numerical table cites those same cases.
`--check` rejects drift from the committed reference vectors, including with
an installed wheel that carries the same fixture. A discrepancy
should be investigated for changed math, precision, dependencies, or platform
behavior before refreshing the reference file. Machine and dependency versions
are relevant when comparing double-precision results; wall-clock benchmark
timings are local measurements, not cross-machine performance results.

CI requires Ruff lint and format, strict mypy, pytest on Python 3.10–3.12,
and a clean LaTeX PDF build. Codex or another automated reviewer can inspect
PRs for Fourier normalization, sign changes, precision loss, input validation,
and reproducibility. A human mathematical review remains responsible for
derivations and claim language; a passing test suite supports regression
confidence rather than proving general statements.
