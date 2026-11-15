"""Recreate the shape-comparison figure in docs/figures/profiles.png.

Install the optional plotting dependency with ``pip install -e '.[plot]'``.
Each curve is normalized independently to show how its shape changes with
the center parameter; the figure does not compare absolute amplitudes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from weil_gaussian import (
    autocorrelation,
    autocorrelation_at_zero,
    fourier_profile,
    inverse_profile,
)


def main() -> None:
    width = 1.0
    centers = (0.0, 1.5, 3.0)
    frequency = np.linspace(-8.0, 8.0, 1201)
    position = np.linspace(-6.0, 6.0, 1201)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), layout="constrained")
    colors = ("#205072", "#2d936c", "#c65d35")

    for center, color in zip(centers, colors, strict=True):
        spectrum = fourier_profile(frequency, center, width)
        signal = inverse_profile(position, center, width)
        overlap = autocorrelation(position, center, width)
        label = rf"$a={center:g}$"
        axes[0].plot(frequency, spectrum / np.max(spectrum), color=color, label=label)
        axes[1].plot(
            position, signal / np.max(np.abs(signal)), color=color, label=label
        )
        axes[2].plot(
            position,
            overlap / autocorrelation_at_zero(center, width),
            color=color,
            label=label,
        )

    for index, (axis, title, xlabel) in enumerate(
        zip(
            axes,
            (
                r"Fourier profile $\widehat f(\xi)$",
                r"Inverse transform $f(x)$",
                r"Autocorrelation $\Phi(x)$",
            ),
            (r"Frequency $\xi$", r"Position $x$", r"Lag $x$"),
            strict=True,
        )
    ):
        axis.set(title=title, xlabel=xlabel, ylabel="Normalized value")
        axis.axhline(0, color="#7b8794", linewidth=0.7)
        axis.grid(alpha=0.2)
        axis.legend(
            frameon=False, ncol=3, loc="upper left" if index == 0 else "upper right"
        )

    fig.suptitle(r"Gaussian test-function family, $\sigma=1$", fontsize=13)
    output = Path(__file__).resolve().parents[1] / "docs" / "figures" / "profiles.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
