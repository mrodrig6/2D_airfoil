#!/usr/bin/env python3
"""Every figure this code draws about a *result*, from one script.

Three questions, and the figures that answer them:

* **Where is the load on the section?** The surface pressure distribution,
  drawn as $-c_p$ so suction is up, with the sonic line marked -- on a
  transonic case the crossing back through it *is* a shock.
* **What does the field look like?** Mach number with the sonic line in white,
  so the supersonic pocket and the shock that closes it are visible rather
  than inferred, and entropy, which shows where the scheme's own error is made.
* **How do the loads vary with incidence?** Lift, drag and moment against
  angle of attack at a fixed Mach number, from one mesh.

Two operating points, because they answer different questions.  ``transonic``
is the legacy case and the one `docs/verification.md` tabulates.  ``shock`` is
0.05 higher in free-stream Mach number and is the clearer picture: supersonic over most of
*both* surfaces, each pocket closed by its own shock.

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
from src.post.plotting import plot_field, plot_polar, plot_surface_pressure  # noqa: E402
from src.post.postprocess import forces, shock_location  # noqa: E402

SECTION = "0012"

#: ``(tag, flow)``.  The tag names the files, so the figures say which case
#: they came from without anyone having to remember.
POINTS = (
    ("transonic", FlowConditions(mach=0.8, alpha_deg=1.25)),
    ("shock", FlowConditions(mach=0.85, alpha_deg=1.0)),
)

SWEEP_MACH = 0.5  # subcritical, so the polar is the clean one
ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
OPTIONS = SolverOptions(tolerance=1e-6, max_iterations=80_000, print_interval=0)

#: Framed on the section rather than on the domain.  At far-field scale the
#: airfoil is a few pixels across and the shock is invisible.
WINDOW = (-0.4, 1.4, -0.6, 0.6)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    out.mkdir(parents=True, exist_ok=True)

    # One case for everything below: the mesh does not depend on the flow, and
    # at this resolution it is not free.  It is also the mesh
    # docs/verification.md quotes its transonic numbers from, so the figures
    # and that table are the same run rather than two runs that agree.
    mesh = Discretization(n_surface=192, n_radial=64, far_field_radius=20.0)
    case = build_case(naca4(SECTION, mesh.n_surface), mesh)
    print(f"{case.airfoil.name}, {case.metrics.n_elements} cells\n")

    for tag, flow in POINTS:
        print(f"solving M = {flow.mach:g} at {flow.alpha_deg:g} deg  [{tag}] ...")
        result = case.solve(flow, OPTIONS)
        print(f"  {result.summary()}")
        print(f"  {forces(result).summary()}")
        print(f"  upper-surface shock at x/c = {shock_location(result):.3f}")

        print(f"  wrote {plot_surface_pressure(result, out / f'cp_{tag}.png')}")
        print(f"  wrote {plot_field(result, out / f'mach_{tag}.png', window=WINDOW)}")
        # Entropy is conserved along streamlines in smooth flow, so this field
        # is the scheme's own error everywhere except across the shock, where it
        # is physical.  Worth a look next to the Mach field.
        print(
            f"  wrote "
            f"{plot_field(result, out / f'entropy_{tag}.png', scalar='entropy', window=WINDOW)}"
        )
        print()

    # -- the polar: loads against incidence, one mesh -----------------------
    print(f"sweeping {len(ANGLES)} angles at M = {SWEEP_MACH:g} ...")
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
        "number, wave drag.  docs/verification.md measures which is which.\n"
        "\nThe field figures are smooth-shaded, which interpolates between cell\n"
        "values.  Pass shading='flat' to plot_field, or shading=flat to\n"
        "'./fv2d.sh solve', to see the piecewise-constant values the solver\n"
        "actually holds -- and the mesh along with them."
    )


if __name__ == "__main__":
    main()
