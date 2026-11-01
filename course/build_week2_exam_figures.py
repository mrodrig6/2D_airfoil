#!/usr/bin/env python3
"""Generate new, branch-local figures for the Week 2 transonic practical.

Run from the repository root after installing the project's plotting and JAX
extras:

    python course/build_week2_exam_figures.py

The script deliberately does not reuse the repository's documented 0012
reference cases.  It studies three 12%-thick cambered candidates at a new
freestream condition, writes their field and surface-pressure figures, records
their coefficients, and creates a new JAX-adjoint camber/incidence design map.

Outputs are written to ``course/week2_figures/``.  Commit that directory to the
course branch after review; the practical TeX can then be pointed at the exact
figures you choose to assess.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Discretization, FlowConditions, SolverOptions, build_case, naca4  # noqa: E402
from src.post.plotting import plot_field, plot_surface_pressure  # noqa: E402
from src.post.postprocess import forces, scalar_field  # noqa: E402
from src.sensitivity import four_digit_case  # noqa: E402


OUT = Path(__file__).resolve().parent / "week2_figures"
MACH = 0.82
CL_TARGET = 0.75
MAX_MACH_LIMIT = 1.60
MESH = Discretization(n_surface=128, n_radial=48, far_field_radius=18.0)
OPTIONS = SolverOptions(tolerance=1e-6, max_iterations=80_000, print_interval=0)
WINDOW = (-0.35, 1.35, -0.50, 0.50)

# New teaching cases: fixed 12% thickness, differing camber and incidence.
CANDIDATES = (
    ("A", "4412", 2.25),
    ("B", "2412", 1.75),
    ("C", "1412", 1.50),
)


def candidate_figures() -> None:
    rows = []
    for label, section, alpha in CANDIDATES:
        case = build_case(naca4(section, MESH.n_surface), MESH)
        flow = FlowConditions(mach=MACH, alpha_deg=alpha)
        result = case.solve(flow, OPTIONS)
        if not result.converged:
            raise RuntimeError(f"Candidate {label} did not converge: {result.message}")
        load = forces(result)
        plot_field(result, OUT / f"candidate_{label}_mach.png", window=WINDOW)
        plot_surface_pressure(result, OUT / f"candidate_{label}_cp.png")
        rows.append((label, section, alpha, load.cl, load.cd, float(np.max(scalar_field(result.state, flow, "mach")))))

    with (OUT / "candidate_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("candidate", "section", "alpha_deg", "cl", "cd", "max_mach"))
        writer.writerows(rows)


def jax_design_maps() -> None:
    # Coarse, deliberately transparent design-region study, not an optimiser.
    camber = np.array([0.012, 0.025, 0.038])
    alpha = np.array([0.75, 1.50, 2.25])
    cl = np.empty((len(alpha), len(camber)))
    cd = np.empty_like(cl)
    max_mach = np.empty_like(cl)
    mesh = dict(n_surface=32, n_radial=24, far_field_radius=5.0, first_cell_height=0.02)

    for i, incidence in enumerate(alpha):
        for j, m in enumerate(camber):
            case = four_digit_case(m, 0.40, 0.12, mach=MACH, alpha=incidence,
                                   tolerance=1e-6, max_iterations=40_000,
                                   print_interval=0, **mesh)
            result = case.solve()
            cl[i, j], _ = case.value_and_gradient("cl", ("camber", "alpha"))
            cd[i, j], _ = case.value_and_gradient("cd", ("camber", "alpha"))
            max_mach[i, j] = float(np.max(scalar_field(result.state, result.flow, "mach")))

    np.savez(OUT / "jax_design_sweep.npz", camber=camber, alpha=alpha,
             cl=cl, cd=cd, max_mach=max_mach)
    x, y = np.meshgrid(100 * camber, alpha)
    for values, label, filename, threshold in (
        (cl, r"$C_L$", "jax_cl_map.png", CL_TARGET),
        (cd, r"$C_D$", "jax_cd_map.png", None),
        (max_mach, r"maximum Mach number", "jax_mach_map.png", MAX_MACH_LIMIT),
    ):
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
        color = ax.contourf(x, y, values, levels=10, cmap="viridis")
        lines = ax.contour(x, y, values, colors="white", linewidths=.8, levels=10)
        ax.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
        if threshold is not None:
            ax.contour(x, y, values, levels=[threshold], colors="crimson", linewidths=2.2)
        fig.colorbar(color, ax=ax, label=label)
        ax.set(xlabel="maximum camber, % chord", ylabel=r"incidence, $\alpha$ (deg)")
        fig.savefig(OUT / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    candidate_figures()
    jax_design_maps()


if __name__ == "__main__":
    main()
