#!/usr/bin/env python3
"""One operating point, start to finish: section, mesh, solve, loads.

The transonic NACA 0012 case the legacy MATLAB ran -- M = 0.8 at 1.25 degrees,
which puts a shock a little past mid-chord on the upper surface.

    python examples/01_solve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import solve_airfoil  # noqa: E402
from src.post.postprocess import entropy_error, shock_location  # noqa: E402


def main() -> None:
    result = solve_airfoil(
        "0012",
        mach=0.8,
        alpha=1.25,
        n_surface=192,
        n_radial=64,
        tolerance=1e-6,
        print_interval=1000,
    )
    print()
    print(result.report())

    # The report already carries both of these; pulled out here because they
    # are the two numbers this case is judged on.  The shock position is what
    # it is known for, and the entropy error is how much of the drag is the
    # scheme's rather than the flow's.
    print(f"\nshock x/c = {shock_location(result):.3f} against a published 0.62-0.65")
    print(f"entropy error = {entropy_error(result):.3e}")
    print(
        "\nThe lift is low and the drag high against experiment because the\n"
        "reconstruction is piecewise constant -- first order.  docs/verification.md\n"
        "measures exactly what that costs and how fast it goes away with the mesh."
    )


if __name__ == "__main__":
    main()
