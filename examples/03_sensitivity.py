#!/usr/bin/env python3
"""Exact design gradients, and the cost argument for taking them this way.

Five derivatives of the lift -- two flow variables and three shape variables --
from **one** adjoint solve, checked against the finite differences they replace.
The timing printed at the end is the whole point: the adjoint's cost does not
grow with the number of design variables, and the finite difference's does.

Needs JAX:  ./fv2d.sh install   (or  pip install -e ".[jax]")

    python examples/03_sensitivity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensitivity import (  # noqa: E402
    DESIGN_VARIABLES,
    differentiable_case,
    finite_difference_gradient,
)

MESH = dict(n_surface=64, n_radial=32, far_field_radius=10.0, first_cell_height=0.02)


def main() -> None:
    # Subcritical on purpose.  With a shock the discrete objective is only
    # piecewise differentiable in the design variables -- the shock jumps from
    # one cell to the next rather than sliding -- so a finite difference at a
    # transonic point measures that jump and disagrees for a legitimate reason.
    # The gradient is still well defined and still what an optimiser wants; it
    # just cannot be *checked* that way.  docs/workflows.md says more.
    case = differentiable_case(
        "2412",
        **MESH,
        mach=0.5,
        alpha=2.0,
        tolerance=1e-10,
        max_iterations=60_000,
        print_interval=0,
    )
    print(f"NACA 2412 at M = 0.5, 2 deg, {case.case.metrics.n_elements} cells")
    print(f"design variables: {', '.join(DESIGN_VARIABLES)}\n")

    # Converged first and outside both timings, because both columns below
    # start from the same converged state and neither should be charged for it.
    result = case.solve()
    print(f"forward solve      {result.seconds:6.1f} s, {result.iterations} iterations\n")

    t0 = time.perf_counter()
    cl, gradient = case.value_and_gradient("cl")
    adjoint_seconds = time.perf_counter() - t0
    print(f"cl = {cl:+.6f}, and its five derivatives:\n")

    t0 = time.perf_counter()
    approx = finite_difference_gradient(case, "cl", DESIGN_VARIABLES, step=1e-4)
    fd_seconds = time.perf_counter() - t0

    print(f"{'d(cl)/d':<18}{'adjoint':>17}{'finite diff':>17}{'rel err':>11}")
    for key in DESIGN_VARIABLES:
        error = abs(gradient[key] - approx[key]) / max(abs(approx[key]), 1e-12)
        print(f"{key:<18}{gradient[key]:+17.8e}{approx[key]:+17.8e}{error:11.2e}")

    # The honest version of the cost argument.  On a mesh this small the
    # adjoint is *not* dramatically cheaper -- the forward solver is compiled
    # Numba and the adjoint march runs in JAX, which is several times slower per
    # sweep.  What the adjoint has is the scaling: its cost is flat in the number
    # of design variables and the finite difference's is linear in it.
    n = len(DESIGN_VARIABLES)
    print(f"\nadjoint            {adjoint_seconds:6.1f} s for all {n}, and flat in that number")
    print(
        f"finite difference  {fd_seconds:6.1f} s for {n}, about {fd_seconds / n:.1f} s for each one"
    )
    print(
        f"                   so this mesh breaks even at roughly "
        f"{adjoint_seconds / (fd_seconds / n):.0f} variables"
    )
    print(
        "\nThat break-even is the number to keep in mind: a handful of variables\n"
        "does not need an adjoint, and a twenty-variable shape cannot do without\n"
        "one.  Note also what the adjoint column is paying for: JAX has to compile\n"
        "the traced residual before it can march, which is several of those\n"
        "seconds and happens exactly once however long the march then runs.\n"
        "\nThree of the five move the mesh; those take the hybrid route -- exact\n"
        "adjoint, mesh term finite-differenced at a frozen state, no extra solve.\n"
        "That is why they cost almost nothing here: look at how little the last\n"
        "three add to the adjoint column."
    )


if __name__ == "__main__":
    main()
