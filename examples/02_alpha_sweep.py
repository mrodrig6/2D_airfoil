#!/usr/bin/env python3
"""A lift curve from **one** mesh, and the slope it should have.

The point of the example is the cost.  The angle of attack enters this
formulation through the free-stream direction alone -- the mesh does not rotate
with incidence -- so a sweep needs one mesh, not one per point.  Each solve also
starts from the previous angle's converged state, which is most of the iterations
saved again.

    python examples/02_alpha_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import Discretization, FlowConditions, SolverOptions, build_case, naca4  # noqa: E402
from src.post.postprocess import forces  # noqa: E402

ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
MACH = 0.5  # subcritical, so the curve is the clean linear one


def main() -> None:
    mesh = Discretization(n_surface=128, n_radial=48, far_field_radius=20.0)
    case = build_case(naca4("2412", mesh.n_surface), mesh)
    options = SolverOptions(tolerance=1e-7, max_iterations=60_000, print_interval=0)

    print(f"NACA 2412, M = {MACH}, {case.metrics.n_elements} cells, one mesh for all of it\n")
    print(f"{'alpha':>7}  {'cl':>10}  {'cd':>10}  {'cm(c/4)':>10}  {'its':>7}  {'s':>6}")

    state, total, rows = None, 0.0, []
    for alpha in ANGLES:
        result = case.solve(FlowConditions(mach=MACH, alpha_deg=alpha), options, state=state)
        if not result.converged:
            print(f"{alpha:7.2f}  did not converge: {result.message}")
            continue
        # warm start: the next angle begins where this one finished
        state = result.state
        load = forces(result)
        total += result.seconds
        rows.append((alpha, load.cl))
        print(
            f"{alpha:7.2f}  {load.cl:+10.5f}  {load.cd:+10.5f}  {load.cm:+10.5f}"
            f"  {result.iterations:7d}  {result.seconds:6.1f}"
        )

    print(f"\n{len(rows)} points in {total:.1f} s on one mesh")

    # The slope is the check: thin-aerofoil theory gives 2*pi per radian, and
    # the Prandtl-Glauert factor stretches it by 1/sqrt(1 - M^2) in subsonic
    # compressible flow.  First order on a mesh this coarse will undershoot it.
    alpha, cl = np.array(rows).T
    linear = np.abs(alpha) <= 4.0
    slope = np.polyfit(np.radians(alpha[linear]), cl[linear], 1)[0]
    theory = 2.0 * np.pi / np.sqrt(1.0 - MACH**2)
    print(f"\nlift slope  measured {slope:.3f} /rad   thin aerofoil + Prandtl-Glauert {theory:.3f}")
    print(f"            {100 * slope / theory:.0f}% of it, which is the first-order mesh error")

    zero = np.interp(0.0, cl, alpha)
    print(f"zero-lift incidence {zero:+.2f} deg, negative because the section is cambered")


if __name__ == "__main__":
    main()
