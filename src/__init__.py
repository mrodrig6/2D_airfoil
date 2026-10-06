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
=========================  ==================================================
:mod:`~src.config`         the free stream, the gas, and the mesh resolution;
                           the non-dimensionalisation is documented here
:mod:`~src.geometry`       the NACA four- and five-digit families, and
                           coordinates read from a file
:mod:`~src.mesh`           O-grid and C-grid generation, triangulation,
                           connectivity
:mod:`~src.meshio`         meshes generated elsewhere (SU2, Gmsh)
:mod:`~src.metrics`        areas, face lengths and normals
:mod:`~src.physics`        Euler fluxes, four approximate Riemann solvers,
                           boundary conditions
:mod:`~src.assembly`       the vectorised finite-volume residual
:mod:`~src.solver`         the pseudo-time march to steady state
:mod:`~src.postprocess`    lift, drag, moment, surface pressure, fields
:mod:`~src.api`            :func:`~src.api.solve_airfoil`, the one call
:mod:`~src.plotting`       matplotlib figures
=========================  ==================================================

The fast backends and the adjoint -- ``backends`` and ``sensitivity`` -- are
not here yet; see ``docs/roadmap.md`` for what lands when.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import Case, build_case, solve_airfoil
from .assembly import residual
from .config import (
    BACKENDS,
    FLUXES,
    SCHEMES,
    TOPOLOGIES,
    Discretization,
    FlowConditions,
    SolverOptions,
)
from .geometry import (
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
from .mesh import (
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
from .meshio import read_mesh, write_su2
from .metrics import Metrics, build_metrics
from .physics import SOLVERS, interface_flux
from .postprocess import (
    Forces,
    entropy_error,
    forces,
    report,
    scalar_field,
    shock_location,
    surface_pressure,
)
from .solver import SolveResult, solve_steady

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
    """
    if name == "plotting":
        from . import plotting

        return plotting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
