#!/usr/bin/env python3
"""Week 1 input: a subcritical payload-trim study before Week 2 shape design.

Run from the repository root:
    python course/build_week1_subcritical_figures.py

It writes a NACA 2412 geometry, a subcritical lift/drag polar, and pressure
plots below and near the payload-trim incidence.  Week 2 retains the payload
trim but changes geometry and checks a high-Mach dash condition.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Discretization, FlowConditions, SolverOptions, build_case, naca4  # noqa: E402
from src.post.plotting import plot_polar, plot_surface_pressure  # noqa: E402
from src.post.postprocess import forces  # noqa: E402


OUT = Path(__file__).resolve().parent / "week1_figures"
ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
MACH = 0.50
MESH = Discretization(n_surface=96, n_radial=40, far_field_radius=15.0)
OPTIONS = SolverOptions(tolerance=1e-6, max_iterations=60_000, print_interval=0)


def geometry_figure(case) -> None:
    p = case.airfoil.points
    fig, ax = plt.subplots(figsize=(6.4, 2.2), constrained_layout=True)
    ax.plot(p[:, 0], p[:, 1], color="#0072B2", lw=2.2)
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.12, 0.16), xlabel=r"$x/c$", ylabel=r"$y/c$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("NACA 2412: subcritical payload-trim study", loc="left")
    fig.savefig(OUT / "week1_geometry.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    case = build_case(naca4("2412", MESH.n_surface), MESH)
    geometry_figure(case)

    state, loads, solved = None, [], []
    results = {}
    for alpha in ANGLES:
        result = case.solve(FlowConditions(mach=MACH, alpha_deg=alpha), OPTIONS, state=state)
        if not result.converged:
            raise RuntimeError(f"alpha={alpha:g} failed: {result.message}")
        state = result.state
        results[alpha] = result
        loads.append(forces(result))
        solved.append(alpha)

    plot_polar(solved, loads, OUT / "week1_polar.png", title=r"NACA 2412 at $M_\infty=0.50$")
    plot_surface_pressure(results[2.0], OUT / "week1_cp_alpha2.png")
    # This result is already part of the incidence sweep; no extra CFD solve is
    # needed to show the preliminary payload-trim pressure distribution.
    plot_surface_pressure(results[4.0], OUT / "week1_cp_alpha4.png")


if __name__ == "__main__":
    main()
