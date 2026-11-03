#!/usr/bin/env python3
"""Build the Week 2 trimmed two-condition JAX optimization study.

Every geometry is trimmed in incidence to carry the required section lift.
The design variables are maximum camber ``m`` and its crest location ``p``;
incidence is an operating response, not a free design choice.  The study then
screens the trimmed geometry at a higher-Mach dash condition.

Run from the repository root after ``./fv2d.sh install``:

    python course/build_week2_trimmed_optimization.py

The outputs go to ``course/week2_trimmed_figures/``.  They are deliberately
not certification data: the study is a teaching artifact for interpreting a
JAX-adjoint optimization, a constraint boundary, and a robust recommendation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.post.plotting import plot_field, plot_surface_pressure  # noqa: E402
from src.post.postprocess import forces, scalar_field  # noqa: E402
from src.sensitivity import four_digit_case  # noqa: E402


OUT = Path(__file__).resolve().parent / "week2_trimmed_figures"
THICKNESS = 0.12
CRUISE_MACH, CRUISE_CL = 0.78, 0.75
DASH_MACH, DASH_CL = 0.84, 0.55
MACH_LIMIT = 1.60
# The pressure-integral moment convention used by this solver gives values of
# order -0.2 at the present dash condition.  This represents an allowable
# tail/trim-load limit for this teaching screen, rather than an arbitrary
# textbook coefficient of order -0.08.
CM_MIN = -0.300
# The robust choice reserves numerical/modeling margin rather than sitting on
# a nominal constraint boundary.
# Sixteen trimmed map points are enough to show intersecting constraint
# boundaries.  The optimizer histories, not a dense contour interpolation,
# are where this exercise uses the JAX adjoint.
GRID_CAMBER = np.linspace(0.012, 0.050, 4)
GRID_CREST = np.linspace(0.25, 0.55, 4)
BOUNDS = ((0.012, 0.050), (0.25, 0.55))
MESH = dict(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.015)
# The map is a screening study.  Use the compiled forward path and a tolerance
# adequate for separating design regions; the final selected cases can later
# be recomputed on a refined mesh.
SOLVER = dict(backend="numba", tolerance=1e-5, max_iterations=25_000, print_interval=0)
VARIABLES = ("camber", "camber_position", "alpha")
COURSE_CMAP = "course_viridis_light"


def register_course_colormap() -> None:
    """Register a light, colorblind-safe portion of viridis for course figures."""
    if COURSE_CMAP not in plt.colormaps:
        colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.90, 256))
        cmap = LinearSegmentedColormap.from_list(COURSE_CMAP, colors)
        plt.colormaps.register(cmap)


def elapsed(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes:d}m {seconds:02d}s" if minutes else f"{seconds:d}s"


@dataclass
class TrimmedPoint:
    camber: float
    crest: float
    alpha_cruise: float
    alpha_dash: float
    cd_cruise: float
    mach_dash: float
    cm_dash: float
    cruise_case: object
    dash_case: object


def trim(camber: float, crest: float, *, mach: float, target_cl: float, guess: float):
    """Trim alpha by forward-solve secant iterations.

    The map needs values, not derivatives.  Keeping the adjoint out of this
    inner loop makes the study practical to run while reserving JAX for the
    optimization traces that students interpret.
    """
    alpha = guess
    previous: tuple[float, float] | None = None
    for _ in range(7):
        case = four_digit_case(camber, crest, THICKNESS, mach=mach, alpha=alpha, **MESH, **SOLVER)
        cl = forces(case.solve()).cl
        error = cl - target_cl
        if abs(error) <= 2.0e-3:
            return alpha, case
        if previous is None:
            # A bounded first move establishes the secant slope cheaply.
            increment = 0.50 if error < 0.0 else -0.50
        else:
            old_alpha, old_error = previous
            slope = (error - old_error) / (alpha - old_alpha)
            if abs(slope) < 1.0e-5:
                raise RuntimeError("trim secant slope is too small")
            increment = float(np.clip(-error / slope, -0.75, 0.75))
        previous = (alpha, error)
        alpha += increment
    raise RuntimeError(
        f"trim did not reach CL={target_cl:.3f} at m={camber:.4f}, p={crest:.3f}; last alpha={alpha:.3f} deg"
    )


def evaluate(
    camber: float, crest: float, *, gradient: bool = False, cruise_guess: float = 3.0, dash_guess: float = 2.0
) -> tuple[TrimmedPoint, np.ndarray | None]:
    """Trim one geometry at both conditions and optionally return dCD/d(m,p)|CL."""
    alpha_cruise, cruise = trim(camber, crest, mach=CRUISE_MACH, target_cl=CRUISE_CL, guess=cruise_guess)
    alpha_dash, dash = trim(camber, crest, mach=DASH_MACH, target_cl=DASH_CL, guess=dash_guess)
    cruise_result, dash_result = cruise.solve(), dash.solve()
    cruise_load, dash_load = forces(cruise_result), forces(dash_result)
    point = TrimmedPoint(
        camber=camber,
        crest=crest,
        alpha_cruise=alpha_cruise,
        alpha_dash=alpha_dash,
        cd_cruise=cruise_load.cd,
        mach_dash=float(np.max(scalar_field(dash_result.state, dash_result.flow, "mach"))),
        cm_dash=dash_load.cm,
        cruise_case=cruise,
        dash_case=dash,
    )
    if not gradient:
        return point, None

    # Implicit differentiation of the trim condition: dCD/dx at constant CL.
    _, dcd = cruise.value_and_gradient("cd", VARIABLES)
    _, dcl = cruise.value_and_gradient("cl", VARIABLES)
    derivative = np.array(
        [dcd[name] - dcd["alpha"] * dcl[name] / dcl["alpha"] for name in VARIABLES[:2]]
    )
    return point, derivative


def draw_map(x, y, values, *, label: str, filename: str, constraints=(), paths=()):
    fig, ax = plt.subplots(figsize=(6.0, 4.1), constrained_layout=True)
    filled = ax.contourf(x, y, values, levels=10, cmap=COURSE_CMAP)
    lines = ax.contour(x, y, values, levels=8, colors="#202020", linewidths=0.65)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3f")
    for constraint, style, text in constraints:
        line = ax.contour(x, y, constraint[0], levels=[constraint[1]], colors="black", linewidths=2.0,
                          linestyles=style)
        ax.clabel(line, fmt={constraint[1]: text}, inline=True, fontsize=8)
    for name, path, color in paths:
        ax.plot(100 * path[:, 0], path[:, 1], color=color, marker="o", markersize=2.8,
                linewidth=1.2, label=name)
    if paths:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.colorbar(filled, ax=ax, label=label)
    ax.set(xlabel="maximum camber, % chord", ylabel="camber-crest location, $p/c$")
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def optimise(starts: tuple[tuple[float, float], ...]) -> list[tuple[str, TrimmedPoint, list[np.ndarray]]]:
    """Draw short projected JAX-gradient descent traces after cruise trim.

    A line-search optimizer repeatedly samples a noisy shock-containing
    discrete objective and can spend dozens of expensive evaluations trying to
    satisfy a mathematical line-search condition.  Fixed, scaled steps make a
    transparent teaching trace and bound the JAX work deterministically.
    """
    answers = []
    span = np.array([BOUNDS[0][1] - BOUNDS[0][0], BOUNDS[1][1] - BOUNDS[1][0]])
    lower = np.array([BOUNDS[0][0], BOUNDS[1][0]])
    upper = np.array([BOUNDS[0][1], BOUNDS[1][1]])
    for run, start in enumerate(starts, start=1):
        history: list[np.ndarray] = []
        x = np.asarray(start, dtype=float)
        for iteration in range(5):
            point, gradient = evaluate(float(x[0]), float(x[1]), gradient=True)
            history.append(x.copy())
            print(
                f"[optimisation {run}] step {iteration + 1:02d}/5: m={100*x[0]:.2f}% p={x[1]:.3f} "
                f"CDcruise={point.cd_cruise:.4f} Mdash={point.mach_dash:.3f} Cmdash={point.cm_dash:.3f}",
                flush=True,
            )
            # Normalize units before stepping: m is O(10^-2), p is O(10^-1).
            scaled_gradient = gradient * span
            norm = np.linalg.norm(scaled_gradient)
            if norm < 1.0e-12:
                break
            x = np.clip(x - 0.18 * span * scaled_gradient / norm, lower, upper)
        point, _ = evaluate(float(x[0]), float(x[1]))
        history.append(x.copy())
        answers.append((f"JAX start {run}", point, history))
    return answers


def balanced_recommendation(feasible: list[TrimmedPoint]) -> TrimmedPoint:
    """Choose margin balance first and drag second from computed feasible points.

    Hard-coded ``robust`` margins are unjustified until the map exists.  This
    score instead normalizes each positive nominal margin by the largest
    available margin in this grid, maximizes the smaller one, then chooses the
    lower-drag member of the best-balanced band.
    """
    mach_scale = max(MACH_LIMIT - point.mach_dash for point in feasible)
    cm_scale = max(point.cm_dash - CM_MIN for point in feasible)
    if mach_scale <= 0.0 or cm_scale <= 0.0:
        raise RuntimeError("nominal feasible points have no positive constraint margin")

    def score(point: TrimmedPoint) -> float:
        return min((MACH_LIMIT - point.mach_dash) / mach_scale, (point.cm_dash - CM_MIN) / cm_scale)

    best = max(score(point) for point in feasible)
    balanced = [point for point in feasible if score(point) >= 0.85 * best]
    return min(balanced, key=lambda point: point.cd_cruise)


def write_candidates(rows: list[tuple[str, TrimmedPoint]], robust_point: TrimmedPoint) -> None:
    with (OUT / "candidate_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("candidate", "camber_percent", "crest_x_over_c", "alpha_cruise_deg", "alpha_dash_deg",
                         "cd_cruise", "max_mach_dash", "cm_dash", "nominal_feasible", "balanced_recommendation"))
        for name, point in rows:
            nominal = point.mach_dash <= MACH_LIMIT and point.cm_dash >= CM_MIN
            writer.writerow((name, 100 * point.camber, point.crest, point.alpha_cruise, point.alpha_dash,
                             point.cd_cruise, point.mach_dash, point.cm_dash, nominal, point is robust_point))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    register_course_colormap()
    started = perf_counter()
    print(f"Writing trimmed two-condition study to {OUT}", flush=True)
    print("[grid] each geometry is forward-trimmed to cruise and dash lift targets", flush=True)

    shape = (len(GRID_CREST), len(GRID_CAMBER))
    cd = np.empty(shape)
    mach = np.empty(shape)
    cm = np.empty(shape)
    alpha = np.empty(shape)
    points: list[TrimmedPoint] = []
    total = cd.size
    for i, crest in enumerate(GRID_CREST):
        # The first (lowest-camber) point needs the greatest trimmed incidence.
        # Subsequent points use the previous solution as a warm start.
        cruise_guess, dash_guess = 3.0, 2.0
        for j, camber in enumerate(GRID_CAMBER):
            number = i * len(GRID_CAMBER) + j + 1
            print(f"[grid {number:02d}/{total}] m={100*camber:.2f}% p={crest:.3f} ...", end=" ", flush=True)
            point, _ = evaluate(float(camber), float(crest), cruise_guess=cruise_guess, dash_guess=dash_guess)
            cruise_guess, dash_guess = point.alpha_cruise, point.alpha_dash
            points.append(point)
            cd[i, j], mach[i, j], cm[i, j], alpha[i, j] = (
                point.cd_cruise, point.mach_dash, point.cm_dash, point.alpha_cruise
            )
            print(f"CD={point.cd_cruise:.4f}, M={point.mach_dash:.3f}, Cm={point.cm_dash:.3f}", flush=True)

    # Persist the costly grid before optimization, so a later optimizer issue
    # never discards the data needed to diagnose or plot the design space.
    np.savez(OUT / "trimmed_design_sweep.npz", camber=GRID_CAMBER, crest=GRID_CREST, cd=cd,
             mach_dash=mach, cm_dash=cm, alpha_cruise=alpha)
    runs = optimise(((0.020, 0.30), (0.040, 0.50)))
    nominal_grid = min(points, key=lambda point: point.cd_cruise)
    feasible = [point for point in points if point.mach_dash <= MACH_LIMIT and point.cm_dash >= CM_MIN]
    if not feasible:
        raise RuntimeError("no nominally feasible grid point: revise the design bounds or nominal constraints")
    robust_grid = balanced_recommendation(feasible)
    candidate_rows = [("unconstrained grid minimum", nominal_grid), ("robust grid recommendation", robust_grid)]
    candidate_rows.extend((name, point) for name, point, _ in runs)
    write_candidates(candidate_rows, robust_grid)

    x, y = np.meshgrid(100 * GRID_CAMBER, GRID_CREST)
    paths = [(name, np.asarray(history), color) for (name, _, history), color in zip(runs, ("#0072B2", "#D55E00"))]
    draw_map(x, y, alpha, label=r"trimmed cruise $\alpha$ (deg)", filename="trim_alpha_map.png")
    draw_map(
        x, y, cd, label=r"cruise $C_D$", filename="trimmed_cd_map.png",
        constraints=(((mach, MACH_LIMIT), "--", r"dash $M_{max}=1.60$"),
                     ((cm, CM_MIN), ":", r"dash $C_m=-0.300$")),
        paths=paths,
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    scatter = ax.scatter(cd.ravel(), mach.ravel(), c=cm.ravel(), cmap=COURSE_CMAP, s=55, edgecolors="black", linewidths=.35)
    ax.axhline(MACH_LIMIT, color="black", linestyle="--", linewidth=1.6, label=r"$M_{max}=1.60$")
    ax.axvline(robust_grid.cd_cruise, color="#555555", linestyle=":", linewidth=1.0)
    for name, point in candidate_rows[:2]:
        ax.plot(point.cd_cruise, point.mach_dash, marker="*", color="black", markersize=11)
        ax.annotate(name.replace(" grid", ""), (point.cd_cruise, point.mach_dash), xytext=(5, 5), textcoords="offset points", fontsize=7)
    fig.colorbar(scatter, ax=ax, label=r"dash $C_m$")
    ax.set(xlabel=r"cruise $C_D$", ylabel=r"dash $M_{max}$")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "dash_tradeoff.png", dpi=220)
    plt.close(fig)

    for label, point in (("nominal", nominal_grid), ("robust", robust_grid)):
        result = point.dash_case.solve()
        plot_field(result, OUT / f"{label}_dash_mach.png", cmap=COURSE_CMAP, window=(-0.25, 1.25, -0.35, 0.35))
        plot_surface_pressure(result, OUT / f"{label}_dash_cp.png")

    print(f"[done] wrote maps, candidate CSV, and dash fields in {elapsed(perf_counter() - started)}", flush=True)


if __name__ == "__main__":
    main()
