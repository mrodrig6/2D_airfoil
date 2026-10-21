r"""The high-level interface: one call from a section to a converged solution.

This is the layer students use::

    from src import solve_airfoil

    result = solve_airfoil(airfoil="0012", mach=0.8, alpha=1.25)
    print(result.report())

Everything below it stays available -- :func:`build_case` hands back the mesh
and metrics so a sweep can reuse them, which is what makes an angle-of-attack
sweep cost one mesh rather than one per point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import Discretization, FlowConditions, SolverOptions
from .geometry import Airfoil, load_airfoil
from .mesh import CGrid, MeshTopology, OGrid, build_mesh
from .meshio import read_mesh
from .metrics import Metrics, build_metrics
from .postprocess import Forces, forces, report
from .solver import SolveResult, solve_steady

_GEOMETRY_KEYS = ("airfoil", "n_surface", "clustering")
_MESH_KEYS = (
    "topology",
    "n_radial",
    "n_wake",
    "wake_length",
    "far_field_radius",
    "far_field_centre",
    "first_cell_height",
    "growth_limit",
    "orthogonality",
)
_FLOW_KEYS = ("mach", "alpha", "gamma", "flux", "entropy_fix")
_SOLVER_KEYS = (
    "backend",
    "batch",
    "cfl",
    "tolerance",
    "max_iterations",
    "scheme",
    "print_interval",
    "divergence_factor",
)


@dataclass
class Case:
    """A prepared problem: the section, its mesh and its metrics, ready to solve.

    Build one with :func:`build_case` and reuse it across a sweep that changes
    only the *flow*.  The mesh and every metric stay valid -- the angle of
    attack enters through the free-stream direction alone and the mesh does not
    rotate with it -- which removes the dominant setup cost from each point.
    """

    airfoil: Airfoil
    discretization: Discretization
    mesh: MeshTopology
    metrics: Metrics
    grid: OGrid | CGrid | None = None

    def solve(
        self,
        flow: FlowConditions | None = None,
        options: SolverOptions | None = None,
        *,
        state: np.ndarray | None = None,
    ) -> SolveResult:
        return solve_steady(self.metrics, flow, options, state=state)

    def summary(self) -> str:
        return f"{self.airfoil.summary()}\n{self.mesh.summary()}\n{self.metrics.summary()}"


def _split(overrides: dict[str, Any]) -> tuple[dict, dict, dict, dict]:
    """Sort loose keyword arguments into the four configuration objects."""
    geometry, mesh, flow, solver = {}, {}, {}, {}
    for key, value in overrides.items():
        if key in _GEOMETRY_KEYS:
            geometry[key] = value
        elif key in _MESH_KEYS:
            mesh[key] = value
        elif key in _FLOW_KEYS:
            flow["alpha_deg" if key == "alpha" else key] = value
        elif key in _SOLVER_KEYS:
            solver[key] = value
        else:
            known = sorted(_GEOMETRY_KEYS + _MESH_KEYS + _FLOW_KEYS + _SOLVER_KEYS)
            raise TypeError(f"unknown option {key!r}; known options are {known}")
    return geometry, mesh, flow, solver


def build_case(
    airfoil: str | Airfoil = "0012",
    discretization: Discretization | None = None,
    *,
    mesh_file: str | Path | None = None,
    **overrides: Any,
) -> Case:
    """Resolve a section, mesh it, and compute the metrics.

    ``mesh_file`` reads a mesh from SU2 or Gmsh instead of generating one; the
    section is then used only for reporting, since the mesh already fixes the
    geometry.
    """
    geometry, mesh_keys, flow_keys, solver_keys = _split(overrides)
    if flow_keys or solver_keys:
        raise TypeError(
            f"build_case takes geometry and mesh options only; "
            f"{sorted(flow_keys | solver_keys.keys())} belong to the solve"
        )
    disc = replace(
        discretization or Discretization(),
        **{k: v for k, v in mesh_keys.items()},
        **{k: v for k, v in geometry.items() if k in ("n_surface", "clustering")},
    )
    section = load_airfoil(
        geometry.get("airfoil", airfoil), disc.n_surface, clustering=disc.clustering
    )
    if mesh_file is not None:
        mesh, grid = read_mesh(mesh_file), None
    else:
        mesh, grid = build_mesh(section, disc)
    return Case(
        airfoil=section, discretization=disc, mesh=mesh, metrics=build_metrics(mesh), grid=grid
    )


def solve_airfoil(
    airfoil: str | Airfoil = "0012",
    *,
    case: Case | None = None,
    mesh_file: str | Path | None = None,
    state: np.ndarray | None = None,
    **overrides: Any,
) -> SolveResult:
    r"""Take a section and an operating point to a converged solution.

    >>> from src import solve_airfoil                       # doctest: +SKIP
    >>> result = solve_airfoil("0012", mach=0.8, alpha=1.25)
    >>> print(result.report())

    Supported geometry, flow and solver overrides are routed by this module.
    The high-level flow spelling is ``alpha``; ``alpha_deg`` belongs to
    :class:`~src.config.FlowConditions` and is not a high-level keyword.
    ``alpha`` is the one alias, for ``alpha_deg``, because writing the units
    into the name of something you type constantly is a poor trade.

    Pass ``case`` to reuse a mesh, and ``state`` to warm-start from a previous
    solution -- together they are what make a sweep cheap.
    """
    geometry, mesh_keys, flow_keys, solver_keys = _split(overrides)
    if case is None:
        case = build_case(airfoil, mesh_file=mesh_file, **geometry, **mesh_keys)
    elif geometry or mesh_keys:
        raise TypeError(
            f"{sorted(geometry | mesh_keys.keys())} would change the mesh, but a "
            f"prepared case was given; build a new case instead"
        )
    return case.solve(FlowConditions(**flow_keys), SolverOptions(**solver_keys), state=state)


# attach the reporting helpers so a result is self-describing
SolveResult.report = report  # type: ignore[attr-defined]
SolveResult.forces = forces  # type: ignore[attr-defined]

__all__ = ["Case", "Forces", "build_case", "solve_airfoil"]
