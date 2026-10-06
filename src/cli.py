r"""The command line: ``python -m src <command>``, or ``./fv2d.sh <command>``.

Option names match the Python API's keyword names, so there is one vocabulary to
learn rather than two: ``--n-surface`` is ``n_surface``, ``--far-field-radius``
is ``far_field_radius``.  ``fv2d.sh`` translates ``name=value`` into those flags,
which is why the launcher and this module never disagree about what an option is
called.

Three commands exist so far -- ``check``, ``geometry`` and ``mesh``.  ``solve``
and ``sweep`` arrive with the solver.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .config import TOPOLOGIES, Discretization, FlowConditions
from .geometry import CLUSTERINGS, load_airfoil, write_dat
from .mesh import CGrid, build_mesh
from .meshio import read_mesh, write_su2
from .metrics import build_metrics


# --------------------------------------------------------------------------
# shared option groups
# --------------------------------------------------------------------------
def _add_geometry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--airfoil",
        default="0012",
        help="a NACA four-digit code, or a path to a Selig/Lednicer .dat file "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--n-surface",
        type=int,
        default=Discretization.n_surface,
        help="points around the section, even (default: %(default)s)",
    )
    parser.add_argument(
        "--clustering",
        choices=CLUSTERINGS,
        default=Discretization.clustering,
        help="surface point distribution (default: %(default)s)",
    )


def _add_mesh_options(parser: argparse.ArgumentParser) -> None:
    d = Discretization
    parser.add_argument(
        "--topology",
        choices=TOPOLOGIES,
        default=d.topology,
        help="'ogrid' wraps a loop round the section; 'cgrid' cuts along the wake "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--n-wake",
        type=int,
        default=d.n_wake,
        help="cgrid: cells along each side of the wake cut (default: %(default)s)",
    )
    parser.add_argument(
        "--wake-length",
        type=float,
        default=d.wake_length,
        help="cgrid: outflow plane, in chords (default: %(default)s)",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=d.n_radial,
        help="cells from the wall to the far field (default: %(default)s)",
    )
    parser.add_argument(
        "--far-field-radius",
        type=float,
        default=d.far_field_radius,
        help="far-field radius in chords (default: %(default)s)",
    )
    parser.add_argument(
        "--first-cell-height",
        type=float,
        default=d.first_cell_height,
        help="first cell height in chords (default: %(default)s)",
    )
    parser.add_argument(
        "--growth-limit",
        type=float,
        default=d.growth_limit,
        help="largest radial growth ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--orthogonality",
        type=float,
        default=d.orthogonality,
        help="wall-normal to radial blend exponent (default: %(default)s)",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="read a .su2 or .msh mesh instead of generating an O-grid",
    )


def _discretization(args: argparse.Namespace) -> Discretization:
    return Discretization(
        topology=args.topology,
        n_wake=args.n_wake,
        wake_length=args.wake_length,
        n_surface=args.n_surface,
        n_radial=args.n_radial,
        far_field_radius=args.far_field_radius,
        first_cell_height=args.first_cell_height,
        growth_limit=args.growth_limit,
        orthogonality=args.orthogonality,
        clustering=args.clustering,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Report the environment: what is installed and what that enables."""
    print(f"python     {sys.version.split()[0]}  ({sys.executable})")
    print(f"fvairfoil  {__version__}  ({Path(__file__).parent})")
    for name, what in (
        ("numpy", "required"),
        ("numba", "the fast solver"),
        ("jax", "gradients and sensitivity"),
        ("matplotlib", "figures"),
        ("scipy", "optimisation, and reading legacy .mat files"),
    ):
        found = importlib.util.find_spec(name) is not None
        print(f"  {'yes' if found else 'NO '}  {name:12s} {what}")
    return 0


def cmd_geometry(args: argparse.Namespace) -> int:
    """Inspect a section.  No mesh, no flow."""
    airfoil = load_airfoil(args.airfoil, args.n_surface, clustering=args.clustering)
    print(airfoil.summary())
    x, thickness, camber = airfoil.thickness_and_camber()
    print(
        f"  max thickness {thickness.max():.5f} at x/c = {x[np.argmax(thickness)]:.4f}\n"
        f"  max camber    {camber[np.argmax(np.abs(camber))]:+.5f} at "
        f"x/c = {x[np.argmax(np.abs(camber))]:.4f}\n"
        f"  enclosed area {airfoil.area:.6f}"
    )
    if args.dat:
        print(f"  wrote {write_dat(airfoil, args.dat)}")
    if args.figure:
        from .plotting import plot_airfoil

        print(f"  wrote {plot_airfoil(airfoil, args.figure, normals=args.normals)}")
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    """Generate (or read) a mesh and report its quality."""
    if args.mesh is not None:
        mesh = read_mesh(args.mesh)
    else:
        airfoil = load_airfoil(args.airfoil, args.n_surface, clustering=args.clustering)
        print(airfoil.summary())
        mesh, grid = build_mesh(airfoil, _discretization(args))
        heights = grid.first_cell_heights()
        print(f"  first cell height {heights.min():.3e} to {heights.max():.3e} chords")
        if isinstance(grid, CGrid):
            print(f"  first wake cell   {grid.first_wake_cell():.3e} chords")
    metrics = build_metrics(mesh)
    print(mesh.summary())
    print(f"  {metrics.summary()}")
    area = metrics.area
    print(f"  cell area {area.min():.3e} to {area.max():.3e} (ratio {area.max() / area.min():.3e})")
    if args.su2:
        print(f"  wrote {write_su2(mesh, args.su2)}")
    if args.figure:
        from .plotting import plot_mesh

        window = None if args.window == "all" else tuple(float(v) for v in args.window.split(","))
        if window is not None and len(window) != 4:
            raise SystemExit("--window needs four comma-separated numbers, or 'all'")
        print(f"  wrote {plot_mesh(mesh, args.figure, window=window)}")
    return 0


def cmd_flow(args: argparse.Namespace) -> int:
    """Print the non-dimensional free-stream state for an operating point."""
    flow = FlowConditions(mach=args.mach, alpha_deg=args.alpha, gamma=args.gamma)
    u, v = flow.velocity
    print(
        f"M = {flow.mach:g}, alpha = {flow.alpha_deg:g} deg, gamma = {flow.gamma:g}\n"
        f"  density          {flow.density:.10f}\n"
        f"  velocity         ({u:.10f}, {v:.10f})\n"
        f"  pressure         {flow.pressure:.10f}   = 1/(gamma M^2)\n"
        f"  sound speed      {flow.sound_speed:.10f}   = 1/M\n"
        f"  temperature      {flow.temperature:.10f}   with R = gamma - 1\n"
        f"  dynamic pressure {flow.dynamic_pressure:.10f}   always 1/2\n"
        f"  total pressure   {flow.total_pressure:.10f}\n"
        f"  state (rho, rho u, rho v, rho E)\n"
        f"        {np.array2string(flow.state, precision=10)}"
    )
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fvairfoil",
        description="2D finite-volume Euler solver for transonic airfoil studies",
    )
    parser.add_argument("--version", action="version", version=f"fvairfoil {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="report the Python and which extras are installed").set_defaults(
        run=cmd_check
    )

    geometry = sub.add_parser("geometry", help="inspect a section; no mesh, no flow")
    _add_geometry_options(geometry)
    geometry.add_argument("--figure", type=Path, default=None, help="write a PNG here")
    geometry.add_argument(
        "--normals", action="store_true", help="draw the outward normals on the figure"
    )
    geometry.add_argument(
        "--dat", type=Path, default=None, help="write the section as a Selig .dat file here"
    )
    geometry.set_defaults(run=cmd_geometry)

    mesh = sub.add_parser("mesh", help="generate or read a mesh and report its quality")
    _add_geometry_options(mesh)
    _add_mesh_options(mesh)
    mesh.add_argument("--figure", type=Path, default=None, help="write a PNG here")
    mesh.add_argument(
        "--window",
        default="-0.6,1.6,-0.7,0.7",
        help="figure extent 'xmin,xmax,ymin,ymax', or 'all' (default: %(default)s)",
    )
    mesh.add_argument("--su2", type=Path, default=None, help="write the mesh as SU2 here")
    mesh.set_defaults(run=cmd_mesh)

    flow = sub.add_parser("flow", help="print the non-dimensional free-stream state")
    flow.add_argument("--mach", type=float, default=FlowConditions.mach)
    flow.add_argument(
        "--alpha",
        type=float,
        default=FlowConditions.alpha_deg,
        help="angle of attack in degrees (default: %(default)s)",
    )
    flow.add_argument("--gamma", type=float, default=FlowConditions.gamma)
    flow.set_defaults(run=cmd_flow)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"fvairfoil: {exc}", file=sys.stderr)
        return 2
