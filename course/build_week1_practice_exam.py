#!/usr/bin/env python3
"""Build the Week 1 JAX lift-trim practice exam.

The airfoil geometry is fixed.  JAX supplies dCL/dalpha so the script can
update incidence toward the payload-lift target.  Week 2 uses the same idea
but applies JAX sensitivities to geometry and drag instead.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import Discretization, FlowConditions, SolverOptions, build_case, naca4  # noqa: E402
from src.post.postprocess import forces, surface_pressure  # noqa: E402
from src.sensitivity import four_digit_case  # noqa: E402


OUT = Path(__file__).resolve().parent / "week1_practice_figures"
MACH, TARGET_CL = 0.50, 0.75
ANGLES = np.array([0.0, 2.0, 4.0, 6.0])
MESH = Discretization(n_surface=64, n_radial=28, far_field_radius=12.0)
OPTIONS = SolverOptions(backend="numba", tolerance=1e-5, max_iterations=30_000, print_interval=0)


def cp_plot(result, path: Path, title: str) -> None:
    trace = surface_pressure(result)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    for mask, label, color, style in (
        (trace["upper"], "upper surface", "#202020", "-"),
        (~trace["upper"], "lower surface", "#6f6f6f", "--"),
    ):
        order = np.argsort(trace["x"][mask])
        ax.plot(trace["x"][mask][order], -trace["cp"][mask][order], color=color, linestyle=style,
                linewidth=2.0, label=label)
    ax.set(xlabel="$x/c$", ylabel="$-C_p$", xlim=(-0.02, 1.02), title=title)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def polar_plot(loads, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(ANGLES, [load.cl for load in loads], "o-", color="#303030", linewidth=2)
    axes[0].axhline(TARGET_CL, color="#777777", linestyle="--", linewidth=1.2, label=r"target $C_L=0.75$")
    axes[0].set(xlabel=r"incidence, $\alpha$ (deg)", ylabel=r"$C_L$", title="lift polar")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(ANGLES, [load.cd for load in loads], "s--", color="#606060", linewidth=2)
    axes[1].set(xlabel=r"incidence, $\alpha$ (deg)", ylabel=r"$C_D$", title="drag polar")
    for ax in axes:
        ax.grid(alpha=.22)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def jax_trim() -> tuple[float, list[tuple[int, float, float, float]]]:
    """Three transparent JAX sensitivity updates from an intentionally low alpha."""
    alpha = 1.0
    history = []
    for iteration in range(3):
        case = four_digit_case(0.02, 0.40, 0.12, mach=MACH, alpha=alpha,
                               n_surface=48, n_radial=22, far_field_radius=10.0,
                               first_cell_height=0.02, backend="numba",
                               tolerance=1e-5, max_iterations=25_000, print_interval=0)
        cl, gradient = case.value_and_gradient("cl", ("alpha",))
        slope = gradient["alpha"]
        history.append((iteration, alpha, cl, slope))
        alpha += float(np.clip((TARGET_CL - cl) / slope, -1.5, 1.5))
    return alpha, history


def trim_plot(history, path: Path) -> None:
    iteration = np.array([item[0] for item in history])
    cl = np.array([item[2] for item in history])
    alpha = np.array([item[1] for item in history])
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(iteration, cl, "o-", color="#303030", linewidth=2, label=r"calculated $C_L$")
    ax.axhline(TARGET_CL, color="#777777", linestyle="--", linewidth=1.2, label=r"target $C_L=0.75$")
    for i, a in enumerate(alpha):
        ax.annotate(rf"$\alpha={a:.2f}^\circ$", (iteration[i], cl[i]), xytext=(4, 7),
                    textcoords="offset points", fontsize=8)
    ax.set(xlabel="JAX trim step", ylabel=r"$C_L$", title="JAX uses $dC_L/d\alpha$ to update incidence")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.22)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("[week 1] solving the fixed-geometry polar", flush=True)
    case = build_case(naca4("2412", MESH.n_surface), MESH)
    state, results, loads = None, {}, []
    for alpha in ANGLES:
        result = case.solve(FlowConditions(mach=MACH, alpha_deg=float(alpha)), OPTIONS, state=state)
        if not result.converged:
            raise RuntimeError(f"alpha={alpha:g} did not converge: {result.message}")
        state, results[float(alpha)] = result.state, result
        loads.append(forces(result))
    polar_plot(loads, OUT / "practice_polar.png")
    print("[week 1] running three JAX lift-trim steps", flush=True)
    trimmed_alpha, history = jax_trim()
    trim_plot(history, OUT / "practice_jax_trim.png")
    trimmed = case.solve(FlowConditions(mach=MACH, alpha_deg=trimmed_alpha), OPTIONS, state=state)
    if not trimmed.converged:
        raise RuntimeError(f"trimmed case did not converge: {trimmed.message}")
    cp_plot(results[2.0], OUT / "practice_cp_alpha2.png", r"$\alpha=2^\circ$: below target lift")
    cp_plot(trimmed, OUT / "practice_cp_trim.png", rf"JAX-trimmed: $\alpha={trimmed_alpha:.2f}^\circ$")
    np.savetxt(OUT / "practice_trim_history.csv", np.asarray(history),
               delimiter=",", header="iteration,alpha_deg,cl,dcl_dalpha", comments="")
    print(f"[week 1] done; JAX trim predicts alpha={trimmed_alpha:.3f} deg", flush=True)


if __name__ == "__main__":
    main()
