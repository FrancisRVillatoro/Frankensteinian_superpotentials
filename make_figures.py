#!/usr/bin/env python3
"""Generate the figures used in the manuscript on Frankensteinian superpotentials."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def structural_decomposition():
    """Generate the smooth-versus-piecewise structural comparison."""
    phi = np.linspace(-1.08, 1.08, 1600)

    v_smooth = 0.5 * (1.0 - phi**2) ** 2
    phi_inf = 1.0 / np.sqrt(3.0)

    beta = 0.4
    abs_phi = np.abs(phi)
    v_tct = 0.5 * (
        (1.0 - abs_phi) ** 2
        - (1.0 / beta)
        * np.where(abs_phi <= beta, (beta - abs_phi) ** 2, 0.0)
    )
    v_tct[abs_phi > 1.0] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15))

    axes[0].plot(phi, v_smooth, label="smooth potential")
    axes[0].axvline(-phi_inf, linestyle="--", label="inflection locations")
    axes[0].axvline(phi_inf, linestyle="--")
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_ylim(-0.02, 0.64)
    axes[0].set_xlabel(r"$\phi$")
    axes[0].set_ylabel(r"$V(\phi)$")
    axes[0].set_title("(a) Smooth double well")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(phi, v_tct, label="tail-core-tail potential")
    axes[1].axvline(-beta, linestyle="--", label="exact sewing levels")
    axes[1].axvline(beta, linestyle="--")
    axes[1].set_xlim(-1.05, 1.05)
    axes[1].set_ylim(-0.02, 0.64)
    axes[1].set_xlabel(r"$\phi$")
    axes[1].set_title("(b) Piecewise model")
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_structural_decomposition_prd_v5.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "fig_structural_decomposition_prd_v5.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def bps_regularization():
    """Generate the loss-of-regularity plot for the singular deformation."""
    phi = np.linspace(-1.0, 1.0, 2401)
    eps_values = [1.0, 0.5, 0.2, 0.05]
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for eps, style in zip(eps_values, line_styles):
        q_phi = -2.0 * np.sign(phi) * np.abs(phi) ** eps
        q_phi[np.isclose(phi, 0.0)] = 0.0
        ax.plot(phi, q_phi, linestyle=style, label=rf"$\varepsilon={eps:g}$")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$Q_{\varepsilon,\phi}^{(1)}(\phi)$")
    ax.set_title("Approach to the tail-tail sewing")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_bps_regularization_v5.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "fig_bps_regularization_v5.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    structural_decomposition()
    bps_regularization()
    print(f"Figures written to {OUT}")
