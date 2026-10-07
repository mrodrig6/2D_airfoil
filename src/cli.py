r"""The command line: ``python -m src <command>``, or ``./fv2d.sh <command>``.

Option names match the Python API's keyword names, so there is one vocabulary to
learn rather than two: ``--n-surface`` is ``n_surface``, ``--far-field-radius``
is ``far_field_radius``.  ``fv2d.sh`` translates ``name=value`` into those flags,
which is why the launcher and this module never disagree about what an option is
called.

The commands are ``check``, ``geometry``, ``mesh``, ``flow``, ``solve``,
``sweep``, ``sensitivity`` and ``bench``.  ``./fv2d.sh help`` lists them with
examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .api import build_case
from .config import (
    BACKENDS,
    FLUXES,
    SCHEMES,
    TOPOLOGIES,
    Discretization,
    FlowConditions,
    SolverOptions,
)
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
        ("numba", "the compiled backend"),
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


def _add_flow_options(parser: argparse.ArgumentParser) -> None:
    f = FlowConditions
    parser.add_argument("--mach", type=float, default=f.mach)
    parser.add_argument(
        "--alpha",
        type=float,
        default=f.alpha_deg,
        help="angle of attack in degrees (default: %(default)s)",
    )
    parser.add_argument("--gamma", type=float, default=f.gamma)
    parser.add_argument(
        "--flux",
        choices=FLUXES,
        default=f.flux,
        help="the approximate Riemann solver (default: %(default)s)",
    )
    parser.add_argument(
        "--entropy-fix",
        type=float,
        default=f.entropy_fix,
        help="roe only: Harten-Hyman delta (default: %(default)s)",
    )


def _add_solver_options(parser: argparse.ArgumentParser) -> None:
    o = SolverOptions
    parser.add_argument("--cfl", type=float, default=o.cfl)
    parser.add_argument("--tolerance", type=float, default=o.tolerance)
    parser.add_argument("--max-iterations", type=int, default=o.max_iterations)
    parser.add_argument("--scheme", choices=SCHEMES, default=o.scheme)
    parser.add_argument("--print-interval", type=int, default=o.print_interval)


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve one operating point and report the loads."""
    from .postprocess import report

    case = (
        build_case(
            args.airfoil,
            _discretization(args),
            mesh_file=args.mesh,
            n_surface=args.n_surface,
            clustering=args.clustering,
        )
        if args.mesh is None
        else build_case(args.airfoil, mesh_file=args.mesh)
    )
    print(case.summary())
    flow = FlowConditions(
        mach=args.mach,
        alpha_deg=args.alpha,
        gamma=args.gamma,
        flux=args.flux,
        entropy_fix=args.entropy_fix,
    )
    options = SolverOptions(
        cfl=args.cfl,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        scheme=args.scheme,
        print_interval=args.print_interval,
    )
    result = case.solve(flow, options)
    print(report(result))

    if args.cp:
        from .plotting import plot_surface_pressure

        print(f"  wrote {plot_surface_pressure(result, args.cp)}")
    if args.figure:
        from .plotting import plot_field

        print(f"  wrote {plot_field(result, args.figure, scalar=args.scalar)}")
    if args.convergence:
        from .plotting import plot_convergence

        print(f"  wrote {plot_convergence(result, args.convergence)}")
    return 0 if result.converged else 1


def cmd_bench(args: argparse.Namespace) -> int:
    """Time the backends against each other on the same mesh and state."""
    import time

    import numpy as np

    from .backends import available_backends, get_backend
    from .solver import initial_state

    case = build_case(
        args.airfoil, _discretization(args), n_surface=args.n_surface, clustering=args.clustering
    )
    flow = FlowConditions(mach=args.mach, alpha_deg=args.alpha, flux=args.flux)
    options = SolverOptions(cfl=args.cfl)
    print(case.mesh.summary())
    U0 = initial_state(case.metrics, flow)

    timings = {}
    for name in available_backends():
        backend = get_backend(name, case.metrics, flow, options)
        # the compiled backend is given more steps because it needs them to be
        # measurable at all, and fewer would be dominated by the call overhead
        steps = args.steps if name != "numpy" else max(args.steps // 10, 20)
        if hasattr(backend, "warm_up"):
            print(f"  {name}: compiled in {backend.warm_up():.1f} s (cached on disk)")
        started = time.perf_counter()
        backend.run(np.array(U0), steps, 1.0)
        timings[name] = (steps, time.perf_counter() - started)

    # always relative to the reference backend, so the number means "times
    # faster than the readable implementation" whichever backends are present
    baseline = timings["numpy"][0] / timings["numpy"][1]
    print(f"\n{'backend':10s} {'steps':>8s} {'seconds':>9s} {'it/s':>10s} {'vs numpy':>10s}")
    for name, (steps, elapsed) in timings.items():
        rate = steps / elapsed
        print(f"{name:10s} {steps:8d} {elapsed:9.2f} {rate:10.0f} {rate / baseline:9.1f}x")
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


def _angles(text: str) -> list[float]:
    """``-4,-2,0,2`` or ``-4:8:2`` -- a list, or a ``start:stop:step`` range.

    Both forms exist because both are natural: a few named angles are a list,
    and a polar is a range.  The range is inclusive of ``stop`` when the step
    divides the interval, which is what anyone writing ``-4:8:2`` means.
    """
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) != 3 or parts[2] == 0.0:
            raise ValueError(f"expected start:stop:step with a non-zero step, got {text!r}")
        start, stop, step = parts
        n = int(math.floor((stop - start) / step + 1e-9))
        if n < 0:
            raise ValueError(f"the step in {text!r} points away from {stop:g}")
        return [start + i * step for i in range(n + 1)]
    return [float(p) for p in text.split(",") if p.strip()]


def cmd_sweep(args: argparse.Namespace) -> int:
    """Solve a range of incidences on **one** mesh, warm-starting each point.

    The two savings are the point of the command.  Incidence enters through the
    free-stream direction alone, so the mesh and every metric are built once;
    and each solve starts from the previous angle's converged field, which is
    already most of the way to the answer.
    """
    from .postprocess import forces

    angles = _angles(args.alphas)
    if not angles:
        raise ValueError("no angles to sweep; pass --alphas")

    case = (
        build_case(
            args.airfoil,
            _discretization(args),
            n_surface=args.n_surface,
            clustering=args.clustering,
        )
        if args.mesh is None
        else build_case(args.airfoil, mesh_file=args.mesh)
    )
    print(case.summary())
    options = SolverOptions(
        cfl=args.cfl,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        scheme=args.scheme,
        print_interval=args.print_interval,
        backend=args.backend,
    )

    # A deck that sets `alpha` is a normal thing to sweep from, and silently
    # dropping it would look like the deck had been applied.
    if args.alpha != FlowConditions.alpha_deg:
        print(f"note: a sweep takes --alphas, so alpha={args.alpha:g} is not used")

    print(f"\nM = {args.mach:g}, {len(angles)} angles, one mesh\n")
    print(f"{'alpha':>8}  {'cl':>10}  {'cd':>10}  {'cm(c/4)':>10}  {'its':>7}  {'s':>6}")
    state, loads, solved, seconds, stalled = None, [], [], 0.0, 0
    for alpha in angles:
        flow = FlowConditions(
            mach=args.mach,
            alpha_deg=alpha,
            gamma=args.gamma,
            flux=args.flux,
            entropy_fix=args.entropy_fix,
        )
        result = case.solve(flow, options, state=state)
        seconds += result.seconds
        if not result.converged:
            print(f"{alpha:8.2f}  did not converge: {result.message}")
            stalled += 1
            continue
        # warm start, but only from a converged field: seeding the next angle
        # with a diverging one spreads the failure down the sweep
        state = result.state
        load = forces(result)
        loads.append(load)
        solved.append(alpha)
        print(
            f"{alpha:8.2f}  {load.cl:+10.5f}  {load.cd:+10.5f}  {load.cm:+10.5f}"
            f"  {result.iterations:7d}  {result.seconds:6.1f}"
        )

    print(f"\n{len(solved)} of {len(angles)} converged, {seconds:.1f} s in total")
    if args.figure and loads:
        from .plotting import plot_polar

        title = f"{case.airfoil.name} at M = {args.mach:g}"
        print(f"  wrote {plot_polar(solved, loads, args.figure, title=title)}")
    return 1 if stalled else 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Exact design gradients by the discrete adjoint.

    One adjoint solve, however many design variables are asked for.
    ``--check`` adds the finite differences it replaces, which costs two full
    flow solves **per variable** and is a verification tool rather than a
    workflow.
    """
    try:
        from .sensitivity import OBJECTIVES, check_gradient, differentiable_case
    except ImportError as exc:  # pragma: no cover - needs jax absent
        raise ValueError(f"the adjoint needs JAX installed: pip install jax ({exc})") from exc

    if args.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {args.objective!r}; choose from {OBJECTIVES}")

    case = build_case(
        args.airfoil,
        _discretization(args),
        n_surface=args.n_surface,
        clustering=args.clustering,
    )
    dc = differentiable_case(
        case=case,
        mach=args.mach,
        alpha=args.alpha,
        gamma=args.gamma,
        flux=args.flux,
        entropy_fix=args.entropy_fix,
        cfl=args.cfl,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        print_interval=args.print_interval,
    )
    # build_case took the section, so the four-digit parameters have to be
    # recovered from what was asked for rather than from the Airfoil
    from .sensitivity import four_digit_parameters

    dc.shape = four_digit_parameters(args.airfoil)
    print(case.summary())

    variables = tuple(v.strip() for v in args.variables.split(",")) if args.variables else None
    result = dc.solve()
    print(f"\n{result.summary()}")

    if args.check:
        rows = check_gradient(dc, args.objective, variables, step=args.step)
        print(f"\n{'d(' + args.objective + ')/d':<20}{'adjoint':>17}{'finite diff':>17}{'rel':>10}")
        for name, row in rows.items():
            print(
                f"{name:<20}{row['adjoint']:+17.8e}{row['finite_difference']:+17.8e}"
                f"{row['relative_error']:10.1e}"
            )
        print(
            "\nThe two differ legitimately where the flow has a shock: it jumps from\n"
            "one cell to the next rather than sliding, so the discrete objective is\n"
            "only piecewise differentiable and a small step measures that jump."
        )
    else:
        value, gradient = dc.value_and_gradient(args.objective, variables)
        print(f"\n{args.objective} = {value:+.8f}\n")
        for name, derivative in gradient.items():
            unit = " per degree" if name == "alpha" else ""
            print(f"  d({args.objective})/d {name:<16}{derivative:+.8e}{unit}")
        print(f"\n  one adjoint solve for all {len(gradient)} of them")
    return 0


def cmd_adapt(args: argparse.Namespace) -> int:
    """Solve, flag the worst cells, refine, and solve again.

    ``legacy/main.m``'s outer loop, with the same Mach-jump indicator and the
    same 5% refinement fraction.
    """
    from .adapt import adaptive_solve

    flow = FlowConditions(
        mach=args.mach,
        alpha_deg=args.alpha,
        gamma=args.gamma,
        flux=args.flux,
        entropy_fix=args.entropy_fix,
    )
    options = SolverOptions(
        cfl=args.cfl,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        scheme=args.scheme,
        print_interval=args.print_interval,
        backend=args.backend,
    )
    history = adaptive_solve(
        args.airfoil,
        _discretization(args),
        flow,
        options,
        cycles=args.cycles,
        fraction=args.fraction,
    )
    print(f"\n{history.summary()}")
    print(history.mesh.summary())

    if args.figure:
        from .plotting import plot_mesh

        window = None if args.window == "all" else tuple(float(v) for v in args.window.split(","))
        if window is not None and len(window) != 4:
            raise SystemExit("--window needs four comma-separated numbers, or 'all'")
        print(f"  wrote {plot_mesh(history.mesh, args.figure, window=window)}")
    if args.history:
        from .plotting import plot_adaptation

        print(f"  wrote {plot_adaptation(history, args.history)}")
    return 1 if history.stalled else 0


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

    solve = sub.add_parser("solve", help="solve one operating point")
    _add_geometry_options(solve)
    _add_mesh_options(solve)
    _add_flow_options(solve)
    _add_solver_options(solve)
    solve.add_argument(
        "--backend",
        choices=("auto", *BACKENDS),
        default=SolverOptions.backend,
        help="execution strategy (default: %(default)s)",
    )
    solve.add_argument(
        "--batch",
        type=int,
        default=SolverOptions.batch,
        help="compiled-backend steps per call (default: %(default)s)",
    )
    solve.add_argument(
        "--cp", type=Path, default=None, help="write the surface pressure distribution here"
    )
    solve.add_argument("--figure", type=Path, default=None, help="write a field PNG here")
    solve.add_argument("--scalar", default="mach", help="which field (default: %(default)s)")
    solve.add_argument(
        "--convergence", type=Path, default=None, help="write the residual history here"
    )
    solve.set_defaults(run=cmd_solve)

    sweep = sub.add_parser("sweep", help="solve a range of incidences on one mesh")
    _add_geometry_options(sweep)
    _add_mesh_options(sweep)
    _add_flow_options(sweep)
    _add_solver_options(sweep)
    sweep.add_argument(
        "--alphas",
        default="-4:8:2",
        help="angles in degrees: a list '-4,0,4' or a range 'start:stop:step' "
        "(default: %(default)s).  This replaces --alpha, which a sweep ignores",
    )
    sweep.add_argument(
        "--backend",
        choices=("auto", *BACKENDS),
        default=SolverOptions.backend,
        help="execution strategy (default: %(default)s)",
    )
    sweep.add_argument("--figure", type=Path, default=None, help="write the three load curves here")
    sweep.set_defaults(run=cmd_sweep)

    sensitivity = sub.add_parser(
        "sensitivity", help="exact design gradients by the discrete adjoint (needs JAX)"
    )
    _add_geometry_options(sensitivity)
    _add_mesh_options(sensitivity)
    _add_flow_options(sensitivity)
    _add_solver_options(sensitivity)
    sensitivity.add_argument(
        "--objective",
        default="cl",
        help="cl, cd, cm or efficiency (default: %(default)s)",
    )
    sensitivity.add_argument(
        "--variables",
        default=None,
        help="comma-separated; default is everything the case offers",
    )
    sensitivity.add_argument(
        "--check",
        action="store_true",
        help="also take the finite differences the adjoint replaces -- two full "
        "solves per variable",
    )
    sensitivity.add_argument(
        "--step", type=float, default=1e-4, help="finite-difference step (default: %(default)s)"
    )
    sensitivity.set_defaults(run=cmd_sensitivity)

    adapt = sub.add_parser("adapt", help="solve, refine where the error is, and solve again")
    _add_geometry_options(adapt)
    _add_mesh_options(adapt)
    _add_flow_options(adapt)
    _add_solver_options(adapt)
    adapt.add_argument(
        "--cycles", type=int, default=5, help="solve-refine cycles (default: %(default)s)"
    )
    adapt.add_argument(
        "--fraction",
        type=float,
        default=0.05,
        help="fraction of cells refined per cycle (default: %(default)s)",
    )
    adapt.add_argument(
        "--backend",
        choices=("auto", *BACKENDS),
        default=SolverOptions.backend,
        help="execution strategy (default: %(default)s)",
    )
    adapt.add_argument("--figure", type=Path, default=None, help="write the final mesh here")
    adapt.add_argument(
        "--window",
        default="-0.6,1.6,-0.7,0.7",
        help="figure extent 'xmin,xmax,ymin,ymax', or 'all' (default: %(default)s)",
    )
    adapt.add_argument(
        "--history", type=Path, default=None, help="write the loads against cell count here"
    )
    adapt.set_defaults(run=cmd_adapt)

    bench = sub.add_parser("bench", help="time the backends against each other")
    _add_geometry_options(bench)
    _add_mesh_options(bench)
    _add_flow_options(bench)
    bench.add_argument("--cfl", type=float, default=SolverOptions.cfl)
    bench.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="compiled-backend steps to time (default: %(default)s)",
    )
    bench.set_defaults(run=cmd_bench)

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
