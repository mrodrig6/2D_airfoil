#!/usr/bin/env python3
"""Inverse design driven by the adjoint: which camber line gives cl = 0.6?

A two-variable shape optimisation at fixed incidence and fixed thickness.  Each
design iteration costs one flow solve and one adjoint solve, and the optimiser is
handed the **exact** gradient rather than a difference of two solves.

The objective is a target lift plus a regulariser:

    J(m, p) = (cl(m, p) - 0.6)^2 + 0.5 * ((p - 0.4) / 0.4)^2

The second term is not decoration.  A target lift in two variables is
underdetermined -- many camber lines give the same cl -- and an optimiser handed
an underdetermined problem wanders along the valley floor until it runs out of
iterations.  The regulariser picks the member of that family closest to the
familiar crest position, which is a design decision made in the open rather than
an accident of where the search started.

Needs JAX and SciPy:  ./fv2d.sh install   (or  pip install -e ".[all]")

    python examples/04_optimise.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensitivity import four_digit_case  # noqa: E402

MESH = dict(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
FLOW = dict(mach=0.5, alpha=0.0, tolerance=1e-9, max_iterations=60_000, print_interval=0)

THICKNESS = 0.12
TARGET_CL = 0.60
CREST = 0.4  # the crest position the regulariser pulls towards
WEIGHT = 0.5
VARIABLES = ("camber", "camber_position")
BOUNDS = [(0.005, 0.08), (0.15, 0.60)]
START = np.array([0.02, 0.30])  # a NACA 2312: deliberately not the answer

history: list[tuple[np.ndarray, float, float]] = []


def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
    """``J`` and ``dJ/d(m, p)``, from one flow solve and one adjoint solve."""
    camber, position = float(x[0]), float(x[1])
    # Built from the parameters as *numbers*, not from a code: inside a design
    # loop there is no code to write, and "0012" cannot be perturbed by 1e-5.
    case = four_digit_case(camber, position, THICKNESS, **MESH, **FLOW)
    cl, gradient = case.value_and_gradient("cl", VARIABLES)
    offset = (position - CREST) / CREST
    J = (cl - TARGET_CL) ** 2 + WEIGHT * offset**2
    dJ = np.array(
        [
            2.0 * (cl - TARGET_CL) * gradient["camber"],
            2.0 * (cl - TARGET_CL) * gradient["camber_position"] + 2.0 * WEIGHT * offset / CREST,
        ]
    )
    history.append((x.copy(), cl, J))
    print(
        f"  {len(history):2d}   m = {camber:.5f}   p = {position:.4f}   "
        f"cl = {cl:+.5f}   J = {J:.3e}"
    )
    return J, dJ


def main() -> None:
    # `thickness` is not a design variable here, so the sections compared all
    # have the same 12% thickness and the lift difference is the camber line's.
    print(f"target cl = {TARGET_CL} at alpha = {FLOW['alpha']} deg, M = {FLOW['mach']}")
    print(f"design variables: {', '.join(VARIABLES)}, thickness fixed at 12%\n")

    t0 = time.perf_counter()
    found = minimize(
        objective,
        START,
        jac=True,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 20, "ftol": 1e-12, "gtol": 1e-8},
    )
    seconds = time.perf_counter() - t0

    camber, position = found.x
    _, cl, _ = history[int(np.argmin([h[2] for h in history]))]
    print(f"\n{found.message}")
    print(
        f"\ncamber {camber:.5f}  crest {position:.4f}  ->  "
        f"NACA {round(100 * camber)}{round(10 * position)}12 to the nearest digits"
    )
    print(f"cl = {cl:+.5f} against a target of {TARGET_CL:+.5f}")
    print(f"\n{len(history)} flow solves and {len(history)} adjoint solves in {seconds:.0f} s")
    print(
        "\nOne solve per design iteration is the result worth noticing.  A\n"
        "finite-difference optimiser needs one per variable *per iteration* on top\n"
        "of that, and the gap is the whole reason adjoints exist.\n"
        "\nTwo warnings before this is used on anything real.  The lift here is\n"
        "inviscid and first order, so the camber it asks for is larger than a real\n"
        "section would need -- docs/verification.md has the size of that error.  And\n"
        "the objective is lift alone: in transonic flow the interesting objective is\n"
        "wave drag, where the shock makes the discrete objective only piecewise\n"
        "differentiable and a gradient method needs more care than this."
    )


if __name__ == "__main__":
    main()
