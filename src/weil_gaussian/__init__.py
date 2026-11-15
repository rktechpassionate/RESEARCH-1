"""Computations for the Gaussian family in the Weil framework.

The transform convention throughout is

    f_hat(xi) = integral_R f(x) exp(-i xi x) dx,
    f(x) = (1 / 2 pi) integral_R f_hat(xi) exp(i xi x) dxi.

The functions implement explicit identities and finite arithmetic sums for
this family. They do not establish the Riemann Hypothesis.
"""

from .arithmetic import PrimePowerTerm, prime_power_terms, truncated_prime_term
from .balance import (
    ArchimedeanResult,
    BalanceDiagnostic,
    archimedean_component,
    balance_diagnostic,
    finite_prime_component,
)
from .core import (
    autocorrelation,
    autocorrelation_at_zero,
    fourier_profile,
    inverse_profile,
    total_mass,
)
from .families import PolynomialGaussianFamily
from .gram import (
    FiniteWeilDiagnostics,
    GramDiagnostics,
    GramSensitivityDiagnostics,
    cross_autocorrelation,
    finite_weil_diagnostics,
    finite_weil_matrix,
    grid_gram_diagnostics,
    l2_gram_diagnostics,
    l2_gram_matrix,
    l2_gram_sensitivity,
)
from .repro import (
    generate_golden_vectors,
    render_markdown,
    reproduce_report,
    validate_golden_vectors,
)
from .verification import VerificationCheck, VerificationReport, verify_family

__all__ = [
    "ArchimedeanResult",
    "BalanceDiagnostic",
    "FiniteWeilDiagnostics",
    "GramDiagnostics",
    "GramSensitivityDiagnostics",
    "PrimePowerTerm",
    "PolynomialGaussianFamily",
    "VerificationCheck",
    "VerificationReport",
    "autocorrelation",
    "autocorrelation_at_zero",
    "archimedean_component",
    "balance_diagnostic",
    "cross_autocorrelation",
    "finite_prime_component",
    "finite_weil_diagnostics",
    "finite_weil_matrix",
    "fourier_profile",
    "generate_golden_vectors",
    "grid_gram_diagnostics",
    "inverse_profile",
    "l2_gram_diagnostics",
    "l2_gram_matrix",
    "l2_gram_sensitivity",
    "prime_power_terms",
    "render_markdown",
    "reproduce_report",
    "total_mass",
    "truncated_prime_term",
    "validate_golden_vectors",
    "verify_family",
]
