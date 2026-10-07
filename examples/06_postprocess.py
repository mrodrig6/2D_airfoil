#!/usr/bin/env python3
"""Every figure this code draws about a *result*, from one script.

Three questions, three figures:

* **Where is the load on the section?** The surface pressure distribution,
  drawn as $-c_p$ so suction is up, with the sonic line marked -- on a
  transonic case the crossing back through it *is* the shock.
* **What does the field look like?** A Mach-number contour with the sonic line
  in white, so the supersonic pocket and the shock that closes it are visible
  rather than inferred.
* **How do the loads vary with incidence?** Lift, drag and moment against
  angle of attack at a fixed Mach number, from one mesh.

    python examples/06_postprocess.py [output-directory]

Needs matplotlib:  ./fv2d.sh install   (or  pip install -e ".[plot]")
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (  # noqa: E402
    Discretization,
    FlowConditions,
    SolverOptions,
    build_case,
    naca4,
)
from src.plotting import plot_field, plot_polar, plot_surface_pressure  # noqa: E402
from src.postprocess import forces, shock_location  # noqa: E402

SECTION = "0012"
TRANSONIC = FlowConditions(mach=0.8, alpha_deg=1.25)
SWEEP_MACH = 0.5  # subcritical, so the polar is the clean one
ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
OPTIONS = SolverOptions(tolerance=1e-6, max_iterations=60_000, print_interval=0)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    out.mkdir(parents=True, exist_ok=True)

    # One case, reused for both figures below: the mesh does not depend on the
    # flow, and at this resolution it is not free.
    # The same mesh docs/verification.md quotes its transonic numbers from, so
    # the figure and the table are the same run.
    mesh = Discretization(n_surface=192, n_radial=64, far_field_radius=20.0)
    case = build_case(naca4(SECTION, mesh.n_surface), mesh)
    print(f"{case.airfoil.name}, {case.metrics.n_elements} cells\n")

    # -- the transonic point: surface pressure, and the field ---------------
    print(f"solving M = {TRANSONIC.mach:g} at {TRANSONIC.alpha_deg:g} deg ...")
    result = case.solve(TRANSONIC, OPTIONS)
    print(f"  {result.summary()}")
    print(f"  {forces(result).summary()}")
    print(f"  shock at x/c = {shock_location(result):.3f}")

    print(f"  wrote {plot_surface_pressure(result, out / 'cp_transonic.png')}")
    print(f"  wrote {plot_field(result, out / 'mach_transonic.png', scalar='mach')}")
    # Entropy is conserved along streamlines in smooth flow, so this field is
    # the scheme's own error everywhere except across the shock, where it is
    # physical.  Worth a look next to the Mach field.
    print(f"  wrote {plot_field(result, out / 'entropy_transonic.png', scalar='entropy')}")

    # -- the polar: loads against incidence, one mesh -----------------------
    print(f"\nsweeping {len(ANGLES)} angles at M = {SWEEP_MACH:g} ...")
    print(f"{'alpha':>7}  {'cl':>10}  {'cd':>10}  {'cm(c/4)':>10}")
    state, loads, solved = None, [], []
    for alpha in ANGLES:
        swept = case.solve(FlowConditions(mach=SWEEP_MACH, alpha_deg=alpha), OPTIONS, state=state)
        if not swept.converged:
            print(f"{alpha:7.2f}  did not converge: {swept.message}")
            continue
        state = swept.state  # warm start, from a converged field only
        load = forces(swept)
        loads.append(load)
        solved.append(alpha)
        print(f"{alpha:7.2f}  {load.cl:+10.5f}  {load.cd:+10.5f}  {load.cm:+10.5f}")

    title = f"{case.airfoil.name} at $M_\\infty = {SWEEP_MACH:g}$"
    print(f"\n  wrote {plot_polar(solved, loads, out / 'polar.png', title=title)}")

    print(
        "\nReading the polar: lift is straight over the range a thin-aerofoil\n"
        "theory would claim it for, and drag is a parabola about the incidence of\n"
        "minimum drag.  Both are inviscid -- there is no skin friction in these\n"
        "equations -- so the drag here is numerical plus, above the critical Mach\n"
        "number, wave drag.  docs/verification.md measures which is which."
    )


if __name__ == "__main__":
    main()
