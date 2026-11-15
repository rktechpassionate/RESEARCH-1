# Research scope and claim status

## Exact identities and proven properties

Version 0.2.0 implements a real-even polynomial-Gaussian Fourier family whose
root factors can enforce multiple prescribed complex frequencies and their
necessary negative and conjugate partners. The inverse transform and
autocorrelation follow from Gaussian differentiation. The original
two-parameter family is the special case with zeros at $\pm i/2$. The
[technical note](../paper/gaussian_identities.tex) derives these formulas.

At zero lag, the cross-autocorrelation of any finite collection of these
families is its $L^2$ Gram matrix. This matrix is symmetric and positive
semidefinite by the ordinary inner-product identity, independently of RH.
Its eigenvalues and condition number describe the numerical independence
of the chosen test functions. Numerical tests compare selected formulas
against independently integrated Fourier definitions; the tests are
regression evidence, not a proof of all mathematical statements.

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
Families without the $\pm i/2$ zeros are rejected by these Weil-named
diagnostics. The coefficient $2$ follows the Fourier convention above.

No certified bound is supplied for prime powers beyond $N$; adaptive
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
rigorous error control for numerical integration. Any conditional theorem
would require separate mathematical review.
