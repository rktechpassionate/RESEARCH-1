"""Real-even polynomial-Gaussian families with prescribed complex zeros.

The inverse and autocorrelation formulas use Gaussian derivative identities,
so evaluation does not require a sampled FFT or numerical inversion.
All numeric evaluations use NumPy double precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
from numpy.polynomial.hermite_e import hermeval

from .core import _parameters, _real_input, _scalar_or_array


def _zero_generators(values: tuple[complex, ...]) -> tuple[complex, ...]:
    try:
        generators = tuple(complex(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "zeros must be a sequence of finite complex frequencies"
        ) from exc
    if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in generators):
        raise ValueError("zeros must contain only finite complex frequencies")
    return generators


def _polynomial_coefficients(zeros: tuple[complex, ...]) -> tuple[float, ...]:
    """Return real coefficients c_m of P(xi) = sum c_m * xi**(2*m)."""
    coefficients = np.array([1.0])
    for zero in zeros:
        square = zero * zero
        if square.imag == 0.0:
            # Real or purely imaginary generator: xi**2 - zero**2.
            factor = np.array([-square.real, 1.0])
        else:
            # Pair the squared root with its conjugate for real coefficients.
            factor = np.array([abs(square) ** 2, -2.0 * square.real, 1.0])
        coefficients = np.convolve(coefficients, factor)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("zero polynomial exceeds double-precision range")
    return tuple(float(coefficient) for coefficient in coefficients)


def _even_hermite_coefficients(
    coefficients: tuple[float, ...], scale: float
) -> np.ndarray:
    """Encode sum c_m (-1)^m scale^(2m) He_(2m) for hermeval."""
    weights = np.zeros(2 * len(coefficients) - 1, dtype=np.float64)
    for degree, coefficient in enumerate(coefficients):
        weights[2 * degree] = coefficient * (-1.0) ** degree * scale ** (2 * degree)
    return weights


@dataclass(frozen=True)
class PolynomialGaussianFamily:
    r"""Gaussian pair multiplied by an even real polynomial with chosen zeros.

    For center a >= 0 and width s > 0, the Fourier profile is
    P(xi) * [exp(-(xi-a)^2/(2s^2)) + exp(-(xi+a)^2/(2s^2))].

    Each generator z in zeros contributes xi^2-z^2. If z^2 is nonreal,
    its conjugate factor is added automatically. Thus the profile vanishes
    at +/-z and, for general complex z, +/-conjugate(z), while remaining
    real and even on the real line. Repeating a generator increases its
    multiplicity. An empty sequence gives the Gaussian pair.

    The default zeros=(0.5j,) reproduces the original Gaussian profile.
    Complex-frequency evaluation is available through fourier; inverse and
    autocorrelation accept real positions only.
    """

    center: float
    width: float
    zeros: tuple[complex, ...] = (0.5j,)
    coefficients: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        center, width = _parameters(self.center, self.width)
        zeros = _zero_generators(self.zeros)
        coefficients = _polynomial_coefficients(zeros)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "zeros", zeros)
        object.__setattr__(self, "coefficients", coefficients)

    @property
    def is_weil_admissible(self) -> bool:
        """Whether specified generators enforce F(+/-i/2)=0 exactly.

        This checks the root construction, not a floating-point value of
        the polynomial that might happen to be close to zero.
        """
        return any(z.real == 0.0 and abs(z.imag) == 0.5 for z in self.zeros)

    def fourier(self, frequency: Any) -> Any:
        """Evaluate the entire Fourier profile at real or complex frequencies."""
        raw = np.asarray(frequency)
        data = np.asarray(
            raw, dtype=np.complex128 if np.iscomplexobj(raw) else np.float64
        )
        if not np.all(np.isfinite(data)):
            raise ValueError("frequency must contain only finite values")
        # At prescribed complex roots, the Gaussian can exceed float range.
        # Evaluate those entries at a harmless frequency and return exact
        # zeros, rather than forming the undefined floating product 0 * inf.
        root_mask = np.zeros(data.shape, dtype=bool)
        for zero in self.zeros:
            root_mask |= (
                (data == zero)
                | (data == -zero)
                | (data == zero.conjugate())
                | (data == -zero.conjugate())
            )
        safe_data = np.where(root_mask, 0, data)
        square = safe_data * safe_data
        polynomial = np.ones_like(safe_data)
        for zero in self.zeros:
            root = zero * zero
            if root.imag == 0.0:
                polynomial = polynomial * (square - root.real)
            elif np.iscomplexobj(safe_data):
                polynomial = polynomial * (square - root) * (square - root.conjugate())
            else:
                # On the real line the conjugate pair is a real sum of squares.
                polynomial = polynomial * (
                    (square - root.real) ** 2 + root.imag**2
                )
        centered = (safe_data - self.center) / self.width
        reflected = (safe_data + self.center) / self.width
        result = polynomial * (
            np.exp(-0.5 * centered * centered)
            + np.exp(-0.5 * reflected * reflected)
        )
        result = np.where(root_mask, 0, result)
        return _scalar_or_array(result, data)

    def inverse(self, position: Any) -> Any:
        r"""Evaluate the real inverse Fourier transform of fourier.

        The convention is f(x) = (2*pi)^(-1) integral F(xi) exp(i*xi*x) dxi.
        Even Hermite polynomials evaluate derivatives for powers of xi.
        """
        x = _real_input(position, "position")
        scale = self.width
        argument = scale * x - 1j * self.center / scale
        weights = _even_hermite_coefficients(self.coefficients, scale)
        polynomial = hermeval(argument, weights)
        result = (
            math.sqrt(2.0 / math.pi)
            * scale
            * np.exp(-0.5 * (scale * x) ** 2)
            * np.real(np.exp(1j * self.center * x) * polynomial)
        )
        return _scalar_or_array(result, x)

    def autocorrelation(self, position: Any) -> Any:
        r"""Evaluate Phi(x)=integral f(t) f(t-x) dt in closed form.

        On the real frequency line F is real and even, so the Fourier
        transform of Phi is F**2. The three Gaussian terms in that square
        are inverted using even Hermite derivatives.
        """
        x = _real_input(position, "position")
        scale = self.width / math.sqrt(2.0)
        squared_coefficients = tuple(np.convolve(self.coefficients, self.coefficients))
        weights = _even_hermite_coefficients(squared_coefficients, scale)
        argument = scale * x - 1j * self.center / scale
        centered_term = np.real(
            np.exp(1j * self.center * x) * hermeval(argument, weights)
        )
        cross_term = hermeval(scale * x, weights)
        result = (
            self.width
            / math.sqrt(math.pi)
            * np.exp(-0.25 * (self.width * x) ** 2)
            * (
                centered_term
                + math.exp(-((self.center / self.width) ** 2)) * cross_term
            )
        )
        return _scalar_or_array(result, x)
