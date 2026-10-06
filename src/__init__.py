r"""**fvairfoil** -- a 2D finite-volume Euler solver for transonic airfoil studies.

A cell-centred finite-volume solver for the two-dimensional Euler equations
around an airfoil, built so that students *use* it rather than write it: change
the section, sweep the angle of attack, take sensitivities, explain the trends.

It is a Python port of a MATLAB teaching code, which lives unchanged in
``legacy/``.  ``docs/theory.md`` records the formulation; ``legacy/README.md``
records what the port changed and why.

Quick start
-----------
>>> from src import naca4, build_mesh                   # doctest: +SKIP
>>> mesh, grid = build_mesh(naca4("0012"))
>>> print(mesh.summary())

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
:mod:`~src.plotting`       matplotlib figures
=========================  ==================================================

The modules the solver itself needs -- ``physics``, ``assembly``, ``solver``,
``postprocess``, ``backends``, ``sensitivity`` -- are not here yet.  This is the
first of five increments, and it stops at a meshed geometry on purpose: see
``docs/roadmap.md`` for what lands when.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import BACKENDS, FLUXES, TOPOLOGIES, Discretization, FlowConditions
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

__all__ = [
    "__version__",
    # configuration
    "FlowConditions",
    "Discretization",
    "BACKENDS",
    "FLUXES",
    "TOPOLOGIES",
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
