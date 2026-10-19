r"""**fvairfoil** -- a 2D finite-volume Euler solver for transonic airfoil studies.

A cell-centred finite-volume solver for the two-dimensional Euler equations
around an airfoil, built so that students *use* it rather than write it: change
the section, sweep the angle of attack, take sensitivities, explain the trends.

It is a Python port of a MATLAB teaching code, which lives unchanged in
``legacy/``.  ``docs/theory.md`` records the formulation; ``legacy/README.md``
records what the port changed and why.

Quick start
-----------
>>> from src import solve_airfoil                       # doctest: +SKIP
>>> result = solve_airfoil("0012", mach=0.8, alpha=1.25)
>>> print(result.report())

Read an externally designed section -- from XFOIL, or from anywhere:

>>> from src import load_airfoil                        # doctest: +SKIP
>>> section = load_airfoil("my_design.dat", n_points=192)

Layout
------
``src/hierarchy.md`` draws the dependency tree; the subpackages are

=============================  ==============================================
:mod:`~src.config`             the free stream, the gas, and the mesh
                               resolution; the non-dimensionalisation is
                               documented here
:mod:`~src.geometry`           sections and meshes: the NACA four- and
                               five-digit families, coordinates read from a
                               file, O-grid and C-grid generation,
                               triangulation, and meshes built elsewhere
                               (SU2, Gmsh)
:mod:`~src.physics`            the equations: thermodynamics, the Roe, HLLC
                               and AUSM+-up approximate Riemann solvers, and
                               the boundary fluxes
:mod:`~src.spatial`            the right-hand side: areas, face lengths and
                               normals, the vectorised finite-volume
                               residual, and adaptive mesh refinement
:mod:`~src.time`               how far to step and when to stop, and the
                               pseudo-time march to steady state
:mod:`~src.backends`           how the residual gets evaluated: vectorised
                               NumPy, compiled Numba, or traceable JAX
:mod:`~src.post`               lift, drag, moment, surface pressure, fields,
                               and matplotlib figures
:mod:`~src.api`                :func:`~src.api.solve_airfoil`, the one call
:mod:`~src.sensitivity`        exact design gradients by the discrete adjoint
=============================  ==============================================

Take a design gradient -- one adjoint solve, however many design variables:

>>> from src.sensitivity import differentiable_case     # doctest: +SKIP
>>> case = differentiable_case("2412", mach=0.5, alpha=2.0)
>>> cl, gradient = case.value_and_gradient("cl")

``docs/workflows.md`` covers sweeps, sensitivities and shape optimisation, and
``examples/`` has each of them as a script that runs with no arguments.
``docs/lab.md`` is the exercise set built on all of it.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import Case, build_case, solve_airfoil
from .config import (
    BACKENDS,
    FLUXES,
    SCHEMES,
    TOPOLOGIES,
    Discretization,
    FlowConditions,
    SolverOptions,
)
from .geometry.mesh import (
    BoundaryTag,
    CGrid,
    MeshTopology,
    OGrid,
    build_mesh,
    generate_cgrid,
    generate_ogrid,
    radial_distribution,
    triangulate,
)
from .geometry.meshio import read_mesh, write_su2
from .geometry.section import (
    CLUSTERINGS,
    Airfoil,
    load_airfoil,
    naca,
    naca4,
    naca5,
    read_dat,
    resample,
    surface_normals,
    write_dat,
)
from .physics import SOLVERS, interface_flux
from .post.postprocess import (
    Forces,
    entropy_error,
    forces,
    report,
    scalar_field,
    shock_location,
    surface_pressure,
)
from .spatial.assembly import residual
from .spatial.metrics import Metrics, build_metrics
from .time.march import SolveResult, solve_steady

__all__ = [
    "__version__",
    # configuration
    "FlowConditions",
    "Discretization",
    "BACKENDS",
    "FLUXES",
    "SCHEMES",
    "TOPOLOGIES",
    "SolverOptions",
    "CLUSTERINGS",
    # geometry
    "Airfoil",
    "naca",
    "naca4",
    "naca5",
    "load_airfoil",
    "read_dat",
    "write_dat",
    "resample",
    "surface_normals",
    # mesh
    "BoundaryTag",
    "CGrid",
    "MeshTopology",
    "OGrid",
    "build_mesh",
    "generate_cgrid",
    "generate_ogrid",
    "triangulate",
    "radial_distribution",
    "read_mesh",
    "write_su2",
    # metrics
    "Metrics",
    "build_metrics",
    # solving
    "Case",
    "build_case",
    "solve_airfoil",
    "solve_steady",
    "SolveResult",
    "residual",
    "SOLVERS",
    "interface_flux",
    # results
    "Forces",
    "forces",
    "report",
    "entropy_error",
    "scalar_field",
    "shock_location",
    "surface_pressure",
    "plotting",
]


def __getattr__(name: str):
    """Import the optional-dependency modules lazily.

    ``plotting`` needs matplotlib.  Deferring it keeps ``import src`` working --
    and fast -- when matplotlib is not installed.

    ``importlib`` rather than ``from .post import plotting``: the ``from`` form
    asks the parent package for the attribute, which lands straight back in
    this function and recurses until the stack runs out.  Caching the module in
    ``globals()`` also means this runs once.
    """
    if name == "plotting":
        import importlib

        globals()[name] = importlib.import_module(f"{__name__}.post.plotting")
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
