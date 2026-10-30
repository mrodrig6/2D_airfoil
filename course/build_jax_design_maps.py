#!/usr/bin/env python3
"""Generate the coarse JAX-adjoint design maps used by the course practical.

The maps intentionally use a small grid: they are evidence for an in-class
interpretation exercise, not a high-fidelity design optimisation.  Each point
is a converged Euler solve followed by JAX adjoint evaluation of lift and drag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sensitivity import four_digit_case  # noqa: E402
from src.post.postprocess import scalar_field  # noqa: E402


OUT = Path(__file__).resolve().parent / "figures"
CAMBER = np.array([0.010, 0.025, 0.040])
ALPHA = np.array([0.5, 1.5, 2.5])
MESH = dict(n_surface=32, n_radial=24, far_field_radius=5.0, first_cell_height=0.02)
FLOW = dict(mach=0.80, tolerance=1e-6, max_iterations=40_000, print_interval=0)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cl = np.empty((len(ALPHA), len(CAMBER)))
    cd = np.empty_like(cl)
    max_mach = np.empty_like(cl)

    for i, alpha in enumerate(ALPHA):
        for j, camber in enumerate(CAMBER):
            print(f"m={camber:.3f}, alpha={alpha:.1f} deg", flush=True)
            case = four_digit_case(camber, 0.40, 0.12, alpha=alpha, **MESH, **FLOW)
            result = case.solve()
            # These values are evaluated through the same JAX objective used
            # by the adjoint; the returned gradients certify the JAX path.
            cl[i, j], _ = case.value_and_gradient("cl", ("camber", "alpha"))
            cd[i, j], _ = case.value_and_gradient("cd", ("camber", "alpha"))
            max_mach[i, j] = float(np.max(scalar_field(result.state, result.flow, "mach")))

    np.savez(OUT / "jax_design_sweep.npz", camber=CAMBER, alpha=ALPHA, cl=cl, cd=cd, max_mach=max_mach)
    x, y = np.meshgrid(100 * CAMBER, ALPHA)

    def draw(values: np.ndarray, label: str, filename: str, *, levels: int = 10, target: float | None = None) -> None:
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
        filled = ax.contourf(x, y, values, levels=levels, cmap="viridis")
        lines = ax.contour(x, y, values, colors="white", linewidths=0.8, levels=levels)
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
        if target is not None:
            ax.contour(x, y, values, levels=[target], colors="crimson", linewidths=2.2)
        fig.colorbar(filled, ax=ax, label=label)
        ax.set(xlabel="maximum camber, % chord", ylabel=r"incidence, $\alpha$ (deg)")
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)

    draw(cl, r"$C_L$", "jax_cl_map.png", target=0.30)
    draw(cd, r"$C_D$", "jax_cd_map.png")
    draw(max_mach, r"maximum Mach number", "jax_mach_map.png", target=1.30)


if __name__ == "__main__":
    main()
