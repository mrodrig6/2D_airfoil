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
from matplotlib import patheffects

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

# These ranges deliberately extend beyond the acceptable design region.  A useful
# screening map must show the lift requirement, the shock limit, and designs
# rejected by each; otherwise it becomes a "pick the upper-right corner" task.
SWEEP_CAMBER = np.linspace(0.012, 0.050, 6)
SWEEP_ALPHA = np.linspace(0.50, 3.25, 6)
SWEEP_MESH = dict(n_surface=64, n_radial=32, far_field_radius=10.0, first_cell_height=0.01)

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
        plot_field(result, OUT / f"candidate_{label}_mach.png", window=WINDOW, cmap="cividis")
        plot_surface_pressure(result, OUT / f"candidate_{label}_cp.png")
        rows.append((label, section, alpha, load.cl, load.cd, float(np.max(scalar_field(result.state, flow, "mach")))))

    with (OUT / "candidate_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("candidate", "section", "alpha_deg", "cl", "cd", "max_mach"))
        writer.writerows(rows)


def _halo(artists) -> None:
    """Keep contours legible over both dark and light portions of cividis."""
    for artist in artists:
        artist.set_path_effects([
            patheffects.Stroke(linewidth=artist.get_linewidth() + 1.8, foreground="white"),
            patheffects.Normal(),
        ])


def jax_design_maps() -> None:
    # A screening study, not an optimiser: 36 independently solved points are
    # shown, with interpolation only used to draw readable contours.
    camber = SWEEP_CAMBER
    alpha = SWEEP_ALPHA
    cl = np.empty((len(alpha), len(camber)))
    cd = np.empty_like(cl)
    max_mach = np.empty_like(cl)
    mesh = SWEEP_MESH

    for i, incidence in enumerate(alpha):
        for j, m in enumerate(camber):
            case = four_digit_case(m, 0.40, 0.12, mach=MACH, alpha=incidence,
                                   tolerance=1e-6, max_iterations=80_000,
                                   print_interval=0, **mesh)
            result = case.solve()
            cl[i, j], _ = case.value_and_gradient("cl", ("camber", "alpha"))
            cd[i, j], _ = case.value_and_gradient("cd", ("camber", "alpha"))
            max_mach[i, j] = float(np.max(scalar_field(result.state, result.flow, "mach")))

    np.savez(OUT / "jax_design_sweep.npz", camber=camber, alpha=alpha,
             cl=cl, cd=cd, max_mach=max_mach)
    x, y = np.meshgrid(100 * camber, alpha)
    for values, label, filename, threshold, line_style, constraint_label in (
        (cl, r"$C_L$", "jax_cl_map.png", CL_TARGET, "solid", r"$C_L=0.75$ requirement"),
        (cd, r"$C_D$", "jax_cd_map.png", None, None, None),
        (max_mach, r"maximum Mach number", "jax_mach_map.png", MAX_MACH_LIMIT,
         (0, (5, 3)), r"$M_{\rm max}=1.60$ limit"),
    ):
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
        color = ax.contourf(x, y, values, levels=10, cmap="cividis")
        lines = ax.contour(x, y, values, colors="#1a1a1a", linewidths=.75, levels=10)
        _halo(lines.collections)
        contour_labels = ax.clabel(lines, inline=True, fontsize=8, fmt="%.3f")
        for text in contour_labels:
            text.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()])
        if threshold is not None:
            if not values.min() < threshold < values.max():
                raise RuntimeError(
                    f"{constraint_label} does not cross the sweep. Expand the range before using this map."
                )
            limit = ax.contour(x, y, values, levels=[threshold], colors="#000000",
                               linewidths=2.0, linestyles=line_style)
            _halo(limit.collections)
            limit_labels = ax.clabel(limit, fmt={threshold: constraint_label}, inline=True, fontsize=8)
            for text in limit_labels:
                text.set_path_effects([patheffects.Stroke(linewidth=2.8, foreground="white"), patheffects.Normal()])
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
