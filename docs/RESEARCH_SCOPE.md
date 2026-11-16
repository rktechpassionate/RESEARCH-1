# Research scope and claim status

## Exact identities and proven properties

Version 1.1.0 implements a real-even polynomial-Gaussian Fourier family whose
root factors can enforce multiple prescribed complex frequencies and their
necessary negative and conjugate partners. The inverse transform and
autocorrelation follow from Gaussian differentiation. The original
two-parameter family is the special case with zeros at $\pm i/2$. The
[technical note](../paper/gaussian_identities.tex) derives these formulas.

At zero lag, the cross-autocorrelation of any finite collection of these
families is its $L^2$ Gram matrix. This matrix is symmetric and positive
semidefinite by the ordinary inner-product identity, independently of RH.
Its eigenvalues, condition number, relative spectral decay, and computed
perturbation sensitivity describe the numerical independence of the chosen
test functions. Weyl bounds apply to the computed Gram matrices; they do not
enclose every floating-point or quadrature error relative to an exact Gram.
Numerical tests compare selected formulas against independently integrated
Fourier definitions; the tests are regression evidence, not a proof of all
mathematical statements.

For a fixed finite family and cutoff, the constrained Rayleigh–Ritz
principle identifies extremal quotients of the exact matrix pencil when
the chosen $L^2$ Gram is positive definite. The software computes the
entries and null-space basis numerically. Its projected generalized
residual and imposed-zero residual check that computed finite problem;
the full generalized residual need not vanish under constraints.

The package also derives an analytic upper bound for the two-sided
*spatial* $L^1$ tails of the inverse Fourier transform $f(x)$ and its
autocorrelation $\Phi(x)$ beyond $|x|>R>0$. Exact rational coefficient
envelopes and outward-rounded decimal evaluation provide the bound for
the mathematical family defined by the supplied finite inputs. This
does not enclose roundoff in the separate floating implementation or
provide a frequency-domain quadrature-error bound.

The Fourier transform convention is

$$
\widehat f(\xi)=\int_{\mathbb R}f(x)e^{-i\xi x}\,dx,
\qquad
f(x)=\frac1{2\pi}\int_{\mathbb R}\widehat f(\xi)e^{i\xi x}\,d\xi.
$$

For the original family, the exact total-mass identity is
$\int\Phi(x)\,dx=\widehat f(0)^2
=\tfrac14e^{-a^2/\sigma^2}$ for $a\geq0,\sigma>0$.

## Numerical diagnostics and their limits

For a family with the required $\pm i/2$ Fourier zeros, the package
numerically evaluates the archimedean integral $K$ and a **finite**
prime-power sum
$P_N=2\sum_{2\leq n\leq N}\Lambda(n)\Phi(\log n)/\sqrt n$.
It reports $K-P_N$, adaptive quadrature error estimates, and an explicit
missing prime-tail bound value of null. The corresponding cross-family
finite Weil matrix and its eigenvalues are also cutoff-dependent.
The extremal solver uses this same finite matrix, so its quotient signs
remain cutoff-dependent.
Families without the $\pm i/2$ zeros are rejected by these Weil-named
diagnostics. The coefficient $2$ follows the Fourier convention above.

The `verify` command compares analytic inverse-transform and autocorrelation
values with adaptive Gauss–Kronrod integration and a separate Gauss–Hermite
calculation. It reports residuals, an adaptive integration error estimate,
an order-doubling difference, and cross-scheme discrepancy. These are
empirical checks, not certified error enclosures. The committed
[golden vectors](../tests/data/golden_vectors.json) record high-precision
spectrum evaluations and integral values for specific cases, with double-precision
comparison tolerances. The `reproduce` command regenerates those cases and
paper artifacts; they do not guarantee machine-epsilon accuracy across all
parameters or systems.

No certified bound is supplied for prime powers beyond $N$; the new
spatial-tail bound is unrelated to that arithmetic omission. Adaptive
quadrature error estimates are not rigorous enclosures. Hence these finite
outputs do **not** give the complete Weil form, a positivity certificate,
a prime-sum inequality, an RH test, or a claim about zeta zeros.

## Historical correction

The older manuscript preserved in Git history contains broader RH-conditional
claims that have **not** been incorporated into this release. During an
independent normalization check, its prime-side coefficient did not match the
standard explicit formula under the manuscript's stated Fourier convention:
the historical draft wrote $(1/\pi)\sum_{n\geq2}\Lambda(n)n^{-1/2}
\Phi(\log n)$ where the corresponding standard prime term carries a factor
of $2$ multiplying that sum. For this convention, see the explicit formula
in [Conrey, *The Riemann Hypothesis*, Notices AMS (2003)](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/conreyRH.pdf).
Any result depending on this step requires a fresh derivation and review
before it can be presented as established.

Future work may derive a certified bound for the omitted prime tail and
rigorous error control for frequency-domain numerical integration. Any
conditional theorem would require separate mathematical review.
