"""Regression checks for two independent Fourier quadrature schemes."""

from __future__ import annotations

import math
import warnings
from dataclasses import FrozenInstanceError

import pytest
from scipy.integrate import IntegrationWarning

from weil_gaussian.families import PolynomialGaussianFamily
from weil_gaussian.verification import verify_family


@pytest.mark.parametrize(
    "center,width,zeros,position",
    [
        (0.0, 1.0, (), 0.0),
        (0.0, 1.0, (), 1.0),
        (1.2, 0.9, (), 0.8),
        (2.0, 0.7, (), 2.0),
        (0.0, 1.0, (0.5j,), 0.8),
        (1.2, 0.9, (0.5j,), 1.4),
        (2.0, 0.7, (0.5j,), 0.3),
        (1.5, 1.2, (0.5j,), 2.0),
        (0.0, 1.0, (0.5j, 1.5j), 0.7),
        (1.2, 0.9, (0.5j, 1.5j), 0.3),
        (2.0, 0.7, (0.5j, 1.5j), 1.2),
        (1.2, 0.9, (0.5j, 1 + 0.5j), 0.8),
        (0.0, 0.8, (0.5j, 1 + 0.5j), 1.4),
        (2.0, 1.3, (0.5j, 1 + 0.5j), 0.4),
        (1.2, 0.9, (1 + 2j,), 0.6),
        (0.5, 1.1, (1 + 2j,), 1.2),
        (0.0, 1.0, (0.5j, 0.5j), 0.9),
        (1.5, 0.7, (0.5j, 0.5j), 1.3),
        (3.0, 1.2, (0.5j, 1 + 2j), 0.5),
        (0.7, 0.6, (0.5j, 1.5j, 2.5j), 1.1),
        (1.0, 0.8, (0.5j, 2.0), 0.4),
        (0.8, 1.4, (-0.5j, 1 - 0.5j), 1.3),
        (8.0, 0.5, (0.5j,), 0.4),
        (10.0, 0.3, (0.5j, 1.5j), 0.7),
        (4.0, 0.2, (0.5j, 1 + 0.5j), 0.8),
        (0.1, 3.0, (0.5j, 1 + 0.5j), 0.2),
    ],
)
def test_two_numerical_schemes_agree_with_both_closed_forms(
    center: float, width: float, zeros: tuple[complex, ...], position: float
) -> None:
    family = PolynomialGaussianFamily(center, width, zeros)
    report = verify_family(family, (position,))
    assert report.passed
    assert report.hermite_base_order == 48
    assert report.hermite_reported_order == 96
    assert [check.quantity for check in report.checks] == [
        "inverse",
        "autocorrelation",
    ]
    for check in report.checks:
        assert check.position == position
        assert check.passed
        assert check.adaptive_error_estimate >= 0
        assert check.hermite_refinement_delta >= 0
        assert check.finite_cutoff_tail_bound is None
        assert (
            max(
                check.analytic_adaptive_abs_error,
                check.analytic_hermite_abs_error,
                check.cross_scheme_discrepancy,
                check.adaptive_error_estimate,
                check.hermite_refinement_delta,
            )
            <= check.acceptance_threshold
        )


@pytest.mark.parametrize(
    "center,width,position",
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.2),
        (1.2, 0.9, 0.8),
        (2.0, 0.7, 1.3),
        (3.0, 1.5, 0.2),
    ],
)
def test_zero_free_quadrature_against_elementary_gaussian_form(
    center: float, width: float, position: float
) -> None:
    family = PolynomialGaussianFamily(center, width, ())
    inverse = (
        math.sqrt(2.0 / math.pi)
        * width
        * math.exp(-0.5 * (width * position) ** 2)
        * math.cos(center * position)
    )
    correlation = (
        width
        / math.sqrt(math.pi)
        * math.exp(-0.25 * (width * position) ** 2)
        * (math.cos(center * position) + math.exp(-((center / width) ** 2)))
    )
    report = verify_family(family, (position,))
    for check, expected in zip(report.checks, (inverse, correlation), strict=True):
        assert check.analytic == pytest.approx(expected, rel=5e-14, abs=5e-14)
        assert check.adaptive == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert check.gauss_hermite == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_report_preserves_position_order_and_evenness() -> None:
    family = PolynomialGaussianFamily(1.2, 0.9, (0.5j, 1 + 0.5j))
    report = verify_family(family, (0.0, -0.8, 0.8))
    assert report.positions == (0.0, -0.8, 0.8)
    assert [check.position for check in report.checks] == [
        0.0,
        0.0,
        -0.8,
        -0.8,
        0.8,
        0.8,
    ]
    for negative, positive in zip(report.checks[2:4], report.checks[4:6], strict=True):
        assert negative.adaptive == pytest.approx(positive.adaptive, rel=2e-13)
        assert negative.gauss_hermite == pytest.approx(
            positive.gauss_hermite, rel=2e-13
        )


def test_low_hermite_order_exposes_nonconvergence() -> None:
    family = PolynomialGaussianFamily(1.2, 0.9, (0.5j, 1 + 0.5j))
    coarse = verify_family(family, (2.0,), hermite_order=12)
    refined = verify_family(family, (2.0,), hermite_order=48)
    assert not coarse.passed
    assert refined.passed
    assert max(check.hermite_refinement_delta for check in coarse.checks) > 1e-4
    assert max(check.hermite_refinement_delta for check in refined.checks) < 1e-8


def test_strict_threshold_is_reported_as_failure() -> None:
    family = PolynomialGaussianFamily(1.2, 0.9)
    report = verify_family(
        family, (0.8,), absolute_tolerance=0.0, relative_tolerance=0.0
    )
    assert not report.passed
    assert all(check.acceptance_threshold == 0 for check in report.checks)
    assert any(not check.passed for check in report.checks)


def test_wrong_analytic_formula_is_detected_without_changing_quadratures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = PolynomialGaussianFamily(1.2, 0.9, (0.5j,))
    original = PolynomialGaussianFamily.inverse
    monkeypatch.setattr(
        PolynomialGaussianFamily,
        "inverse",
        lambda self, position: original(self, position) + 0.125,
    )
    report = verify_family(family, (0.8,))
    inverse = report.checks[0]
    assert not report.passed
    assert not inverse.passed
    assert inverse.analytic_adaptive_abs_error > 0.12
    assert inverse.analytic_hermite_abs_error > 0.12
    assert inverse.cross_scheme_discrepancy < 1e-10


def test_adaptive_convergence_warning_is_not_silently_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weil_gaussian.verification as module

    def warned_quad(*_args: object, **_kwargs: object) -> tuple[float, float]:
        warnings.warn("test warning", IntegrationWarning, stacklevel=1)
        return 0.0, 1.0

    monkeypatch.setattr(module, "quad", warned_quad)
    with pytest.raises(FloatingPointError, match="did not converge"):
        verify_family(PolynomialGaussianFamily(1.0, 1.0), (0.0,))


@pytest.mark.parametrize("positions", [(), [], [math.nan], [math.inf], [1 + 1j]])
def test_invalid_or_empty_position_sets_rejected(positions: object) -> None:
    with pytest.raises(ValueError):
        verify_family(PolynomialGaussianFamily(1.0, 1.0), positions)


def test_noniterable_position_set_rejected() -> None:
    with pytest.raises(TypeError, match="iterable"):
        verify_family(PolynomialGaussianFamily(1.0, 1.0), 0.8)


@pytest.mark.parametrize(
    "absolute,relative",
    [
        (-1.0, 1e-10),
        (math.nan, 1e-10),
        (math.inf, 1e-10),
        (1e-9, -1.0),
        (1e-9, math.nan),
        (1e-9, math.inf),
    ],
)
def test_invalid_tolerances_rejected(absolute: float, relative: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        verify_family(
            PolynomialGaussianFamily(1.0, 1.0),
            (0.0,),
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )


@pytest.mark.parametrize("order", [True, 11, 49, 12.5, "48"])
def test_invalid_hermite_order_rejected(order: object) -> None:
    with pytest.raises(ValueError, match="hermite_order"):
        verify_family(PolynomialGaussianFamily(1.0, 1.0), (0.0,), hermite_order=order)


def test_family_type_rejected() -> None:
    with pytest.raises(TypeError, match="PolynomialGaussianFamily"):
        verify_family(None, (0.0,))


def test_report_and_check_are_immutable() -> None:
    report = verify_family(PolynomialGaussianFamily(0.0, 1.0, ()), (0.0,))
    with pytest.raises(FrozenInstanceError):
        report.passed = False
    with pytest.raises(FrozenInstanceError):
        report.checks[0].passed = False
