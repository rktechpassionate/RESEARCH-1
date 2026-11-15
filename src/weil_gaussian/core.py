"""Closed-form Gaussian profiles under the manuscript's Fourier convention."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _parameters(center: float, width: float) -> tuple[float, float]:
    """Check the manuscript's domain: center >= 0 and width > 0."""
    center = float(center)
    width = float(width)
    if not math.isfinite(center) or center < 0:
        raise ValueError("center must be a finite nonnegative number")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be a finite positive number")
    return center, width


def _real_input(values: Any, name: str) -> np.ndarray:
    """Convert a real scalar or array while rejecting complex/nonfinite data."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain only finite values")
    return data


def _scalar_or_array(result: np.ndarray, input_data: np.ndarray) -> Any:
    return result.item() if input_data.ndim == 0 else result


def fourier_profile(frequency: Any, center: float, width: float) -> Any:
    r"""Return the entire Fourier-side Gaussian profile.

    With ``a=center`` and ``s=width``, this is

    ``(xi**2 + 1/4) * (exp(-(xi-a)**2/(2*s**2)) +
    exp(-(xi+a)**2/(2*s**2)))``.

    ``frequency`` may be a real or complex scalar or array. In particular,
    the profile vanishes at ``xi=+/-0.5j``. Large imaginary inputs can
    overflow in ordinary double precision.
    """
    center, width = _parameters(center, width)
    frequency = np.asarray(frequency)
    data = np.asarray(
        frequency, dtype=np.complex128 if np.iscomplexobj(frequency) else np.float64
    )
    if not np.all(np.isfinite(data)):
        raise ValueError("frequency must contain only finite values")
    shifted_minus = (data - center) / width
    shifted_plus = (data + center) / width
    result = (data * data + 0.25) * (
        np.exp(-0.5 * shifted_minus * shifted_minus)
        + np.exp(-0.5 * shifted_plus * shifted_plus)
    )
    return _scalar_or_array(result, data)


def inverse_profile(position: Any, center: float, width: float) -> Any:
    r"""Return the real inverse Fourier transform of :func:`fourier_profile`.

    This is the closed-form action of ``-d²/dx² + 1/4`` on the inverse
    transform of the paired Gaussian peaks. ``position`` is real. A scalar
    input gives a float; an array input gives a NumPy array.
    """
    center, width = _parameters(center, width)
    x = _real_input(position, "position")
    w2 = width * width
    polynomial = center * center + w2 + 0.25 - (w2 * x) ** 2
    oscillation = polynomial * np.cos(center * x) - 2 * center * w2 * x * np.sin(
        center * x
    )
    result = (
        (2 * width / math.sqrt(2 * math.pi))
        * np.exp(-0.5 * (width * x) ** 2)
        * oscillation
    )
    return _scalar_or_array(result, x)


def autocorrelation(position: Any, center: float, width: float) -> Any:
    r"""Return ``Phi(x) = (f * f_tilde)(x)`` in closed form.

    ``f`` is :func:`inverse_profile` and ``f_tilde(x)=f(-x)``. For this even,
    real family, ``Phi`` is the inverse transform of ``fourier_profile**2``.
    The implementation applies ``(-d²/dx² + 1/4)**2`` to the inverse
    transform of the squared Gaussian pair; it does not integrate or
    interpolate a sampled signal.
    """
    center, width = _parameters(center, width)
    x = _real_input(position, "position")
    w2 = width * width
    base = 0.75 * w2 * w2 + 0.25 * w2 + 0.0625
    linear = -0.5 * w2 * x
    z = linear + 1j * center
    polynomial = z**4 - (3 * w2 + 0.5) * z**2 + base
    cross_polynomial = linear**4 - (3 * w2 + 0.5) * linear**2 + base
    result = (
        width
        / math.sqrt(math.pi)
        * np.exp(-0.25 * (width * x) ** 2)
        * (
            np.real(np.exp(1j * center * x) * polynomial)
            + math.exp(-((center / width) ** 2)) * cross_polynomial
        )
    )
    return _scalar_or_array(result, x)


def total_mass(center: float, width: float) -> float:
    r"""Return ``integral_R Phi(x) dx = f_hat(0)**2`` exactly.

    The result is ``exp(-center**2 / width**2) / 4``. It follows directly
    from the Fourier transform at frequency zero, independent of numerical
    quadrature.
    """
    center, width = _parameters(center, width)
    return 0.25 * math.exp(-((center / width) ** 2))


def autocorrelation_at_zero(center: float, width: float) -> float:
    r"""Return ``Phi(0) = integral_R f(x)**2 dx`` in closed form."""
    center, width = _parameters(center, width)
    w2 = width * width
    base = 0.75 * w2 * w2 + 0.25 * w2 + 0.0625
    diagonal = center**4 + (3 * w2 + 0.5) * center**2 + base
    return width / math.sqrt(math.pi) * (
        diagonal + math.exp(-((center / width) ** 2)) * base
    )
