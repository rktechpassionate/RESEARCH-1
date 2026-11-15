"""Randomized invariants for polynomial-Gaussian families.

These complement the fixed high-precision quadrature cases with a broad,
reproducible search over moderate, numerically well-conditioned parameters.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weil_gaussian import (
    PolynomialGaussianFamily,
    autocorrelation,
    fourier_profile,
    inverse_profile,
    l2_gram_diagnostics,
)

centers = st.floats(min_value=0.0, max_value=4.0, allow_nan=False, allow_infinity=False)
widths = st.floats(min_value=0.4, max_value=2.0, allow_nan=False, allow_infinity=False)
positions = st.floats(
    min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False
)


@settings(max_examples=40, deadline=None)
@given(center=centers, width=widths, position=positions)
def test_default_family_matches_legacy_closed_forms(center, width, position):
    family = PolynomialGaussianFamily(center, width)
    assert family.fourier(position) == pytest.approx(
        fourier_profile(position, center, width), rel=2e-10, abs=2e-10
    )
    assert family.inverse(position) == pytest.approx(
        inverse_profile(position, center, width), rel=2e-10, abs=2e-10
    )
    assert family.autocorrelation(position) == pytest.approx(
        autocorrelation(position, center, width), rel=2e-10, abs=2e-10
    )


@settings(max_examples=40, deadline=None)
@given(center=centers, width=widths, position=positions)
def test_multizero_evenness_and_autocorrelation_bound(center, width, position):
    family = PolynomialGaussianFamily(center, width, (0.5j, 1.5j))
    assert family.fourier(position) == pytest.approx(
        family.fourier(-position), rel=1e-10, abs=1e-10
    )
    assert family.inverse(position) == pytest.approx(
        family.inverse(-position), rel=1e-10, abs=1e-10
    )
    assert family.autocorrelation(position) == pytest.approx(
        family.autocorrelation(-position), rel=1e-10, abs=1e-10
    )
    zero_lag = family.autocorrelation(0.0)
    assert math.isfinite(zero_lag) and zero_lag > 0
    assert abs(family.autocorrelation(position)) <= zero_lag * (1 + 1e-10)


@settings(max_examples=30, deadline=None)
@given(center=centers, width=widths)
def test_random_complex_roots_are_enforced_with_conjugate_closure(center, width):
    z = 0.8 + 0.6j
    family = PolynomialGaussianFamily(center, width, (0.5j, z))
    for root in (z, -z, z.conjugate(), -z.conjugate(), 0.5j, -0.5j):
        assert abs(family.fourier(root)) <= 1e-8


@settings(max_examples=30, deadline=None)
@given(
    first_center=centers,
    first_width=widths,
    second_center=centers,
    second_width=widths,
)
def test_random_l2_gram_is_symmetric_and_positive_semidefinite(
    first_center, first_width, second_center, second_width
):
    families = (
        PolynomialGaussianFamily(first_center, first_width, (0.5j,)),
        PolynomialGaussianFamily(second_center, second_width, (0.5j, 1.5j)),
    )
    diagnostic = l2_gram_diagnostics(families)
    matrix = diagnostic.matrix
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=0)
    assert diagnostic.is_positive_semidefinite
    assert diagnostic.eigenvalues[0] >= -1e-10 * max(
        1.0, abs(diagnostic.eigenvalues[-1])
    )
