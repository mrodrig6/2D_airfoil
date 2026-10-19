r"""Adaptive mesh refinement, driven by the legacy's Mach-jump indicator.

What this is for
----------------
The scheme is first order, so resolution is the only lever on accuracy, and a
uniformly finer mesh spends most of its cells where nothing happens.  Adaptation
puts them where the solution is changing -- at the shock, at the leading edge,
in the first few cells off the wall -- and leaves the far field alone.

The indicator is the legacy's
------------------------------
``legacy/post_calc.m`` flags elements by the **jump in Mach number** across
their faces, weighted by face length:

.. math::
    \varepsilon_e = \sum_{f \in \text{interior}} |M_e - M_{e'}|\,\ell_f
                  \;+\; \sum_{f \in \text{wall}} |M_n|\,\ell_f

reproduced here exactly, including the second term.  That term is a *consistency*
measure rather than a jump: the wall condition says the normal velocity is zero,
so whatever normal Mach number survives at a wall face is error, and it flags the
leading edge and the trailing edge without anyone naming them.  The top
``fraction`` of elements by :math:`\varepsilon_e` are refined, exactly as
``f = 0.05`` did there.

How the refinement works, and the one trap in it
------------------------------------------------
Each flagged triangle is cut into four by joining its edge midpoints -- a **red**
refinement, which is similar to the parent and so cannot degenerate however many
times it is applied.  A neighbour that was not flagged then has a **hanging
node** in the middle of one of its edges, and the solver has no concept of one:
every face must have exactly one element on each side.  The hanging node is
removed by bisecting that neighbour from its opposite vertex -- a **green**
closure.

Green triangles are the trap.  They are *not* similar to their parent, and
bisecting a green triangle again, and again, drives the smallest angle to zero;
the cells become slivers, the local time step collapses with them, and the march
slows to a stop long before anyone notices the mesh is bad.  The classic fix is
to undo the green closure before refining again.

This module does something equivalent and simpler to be sure of: it keeps the
**tree** of red subdivisions and treats green closure as a *rendering* step,
recomputed from scratch every cycle.  A green triangle is therefore never
refined, because it is never a leaf of the tree -- it does not exist between
cycles at all.  The tree also enforces **one-irregularity**: two triangles
sharing an edge differ by at most one level, so a face is split at most once and
a single bisection always closes it.

New boundary nodes land on the geometry, not on the chord
----------------------------------------------------------
The midpoint of a wall face lies on the straight line between two surface points,
which is *inside* the airfoil.  Leaving it there means the section never gets
rounder however fine the mesh becomes, and the leading edge -- where the error
indicator is loudest -- converges to the wrong shape.  Every new wall node is
therefore projected onto the section, and every new far-field node onto the
far-field circle.  The legacy got this free from the PDE Toolbox, which refined
against the geometry rather than against the mesh; :class:`BoundaryGeometry` is
that geometry, kept explicitly.

What the state does across a refinement
-----------------------------------------
Each child inherits its parent's cell average.  For a piecewise-constant scheme
that is the exact projection of the old solution onto the new mesh -- there is no
better answer available from the data -- and it is a warm start rather than a
result: the next cycle still marches to convergence from it, and typically needs
a fraction of the iterations the first one did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Discretization, FlowConditions, SolverOptions
from ..geometry.mesh import BoundaryTag, MeshTopology, build_edges, enforce_ccw
from ..geometry.section import Airfoil, naca, resample
from ..physics import FLOOR
from ..post.postprocess import Forces, forces
from ..time.march import SolveResult, solve_steady
from .metrics import Metrics, build_metrics

#: Fraction of elements refined per cycle, as ``legacy/post_calc.m`` had it.
REFINE_FRACTION = 0.05

#: How densely the surface is sampled before a new wall node is projected onto
#: it, as a multiple of the mesh's own surface points and as an absolute floor.
#: The projection can only be as good as the curve projected onto, and a coarse
#: mesh asking for eight times its own 48 points would get a thin one.
PROJECTION_REFINEMENT = 8
PROJECTION_MINIMUM = 1024


# --------------------------------------------------------------------------
# the indicator
# --------------------------------------------------------------------------
def mach_jump_indicator(result: SolveResult) -> np.ndarray:
    r"""``legacy/post_calc.m``'s error indicator, element by element.

    Returns :math:`\varepsilon_e`, which has the dimensions of a length: it is a
    Mach-number jump integrated over the element's boundary, not a density.
    Two elements of very different size are therefore **not** compared fairly --
    a large cell can out-score a small one simply by having longer faces.  That
    is the legacy's choice and it is kept, because it is also self-limiting:
    refining a cell halves its faces, so it stops being flagged once it is small
    enough, which is the behaviour an indicator needs.
    """
    metrics = result.metrics
    gamma = result.flow.gamma
    U = result.state

    rho = np.maximum(U[:, 0], FLOOR)
    u, v = U[:, 1] / rho, U[:, 2] / rho
    q2 = u * u + v * v
    p = np.maximum((gamma - 1.0) * (U[:, 3] - 0.5 * rho * q2), FLOOR)
    mach = np.sqrt(q2 / (gamma * p / rho))

    error = np.zeros(metrics.n_elements)
    left, right = metrics.interior_left, metrics.interior_right
    jump = np.abs(mach[left] - mach[right]) * metrics.interior_length
    np.add.at(error, left, jump)
    np.add.at(error, right, jump)

    # The wall term.  The pressure here uses the *tangential* kinetic energy
    # only -- the same state the wall flux is built from -- so it is the wall
    # condition's own view of the cell, and the leftover normal Mach number is
    # the extent to which the solution disobeys it.
    s = metrics.tag_slice(BoundaryTag.WALL)
    element = metrics.boundary_element[s]
    nx, ny = metrics.boundary_normal[s, 0], metrics.boundary_normal[s, 1]
    length = metrics.boundary_length[s]

    vn = u[element] * nx + v[element] * ny
    qt2 = (u[element] - vn * nx) ** 2 + (v[element] - vn * ny) ** 2
    p_wall = np.maximum((gamma - 1.0) * (U[element, 3] - 0.5 * rho[element] * qt2), FLOOR)
    a_wall = np.sqrt(gamma * p_wall / rho[element])
    np.add.at(error, element, np.abs(vn / a_wall) * length)
    return error


def flag_fraction(error: np.ndarray, fraction: float = REFINE_FRACTION) -> np.ndarray:
    """The indices of the worst ``fraction`` of elements, largest error first.

    A fixed fraction rather than a threshold, which is what the legacy did and
    is the right default for a teaching code: the cost of a cycle is then known
    in advance -- each one grows the mesh by a predictable amount -- where a
    threshold can refine everything or nothing depending on how the first solve
    happened to come out.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = max(1, int(round(fraction * error.size)))
    return np.argsort(error, kind="stable")[::-1][:n]


# --------------------------------------------------------------------------
# where a new boundary node belongs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryGeometry:
    """The curves the boundary nodes live on, so refinement can stay on them.

    Without this, refining a wall face puts the new node on the chord between
    two surface points -- inside the body -- and the section never converges to
    its own shape.
    """

    surface: np.ndarray | None = None  #: a dense polyline of the section
    far_field_radius: float | None = None
    far_field_centre: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_case(cls, airfoil: Airfoil, discretization: Discretization) -> BoundaryGeometry:
        """Build one from a case's section and mesh settings.

        The section is re-generated analytically when its name is a NACA code,
        because that is the actual curve the mesh is approximating.  For a
        section read from a file there is no analytic curve to recover, so the
        polyline is densified by the same Hermite interpolation
        :func:`~src.geometry.resample` uses elsewhere -- which is the best
        statement available of what lies between two given points.
        """
        n = max(PROJECTION_REFINEMENT * airfoil.n_points, PROJECTION_MINIMUM)
        try:
            dense = naca(airfoil.name, n if n % 2 == 0 else n + 1)
        except ValueError:
            dense = resample(airfoil, n if n % 2 == 0 else n + 1)
        return cls(
            surface=np.vstack([dense.points, dense.points[:1]]),
            far_field_radius=discretization.far_field_radius,
            far_field_centre=tuple(discretization.far_field_centre),
        )

    def project(self, points: np.ndarray, tag: int) -> np.ndarray:
        """Move ``points`` onto the curve carrying ``tag``; unknown tags stay put."""
        if tag == int(BoundaryTag.WALL) and self.surface is not None:
            return _project_to_polyline(points, self.surface)
        if tag == int(BoundaryTag.FARFIELD) and self.far_field_radius is not None:
            centre = np.asarray(self.far_field_centre, dtype=float)
            offset = points - centre
            radius = np.hypot(offset[:, 0], offset[:, 1])
            scale = self.far_field_radius / np.maximum(radius, FLOOR)
            return centre + offset * scale[:, None]
        return points


def _project_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """The nearest point of a polyline, for each query point.

    Brute force over every segment.  There are a few hundred new nodes against a
    few thousand segments per cycle, which is milliseconds; an acceleration
    structure here would be code to maintain in exchange for nothing visible.
    """
    a = polyline[:-1]
    d = polyline[1:] - a
    length2 = np.maximum(np.einsum("ij,ij->i", d, d), FLOOR)

    offset = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("qsj,sj->qs", offset, d) / length2, 0.0, 1.0)
    foot = a[None, :, :] + t[..., None] * d[None, :, :]
    gap = np.einsum("qsj,qsj->qs", foot - points[:, None, :], foot - points[:, None, :])
    return foot[np.arange(points.shape[0]), np.argmin(gap, axis=1)]


# --------------------------------------------------------------------------
# the refinement tree
# --------------------------------------------------------------------------
class RefinementTree:
    """A hierarchy of red subdivisions over a base mesh.

    The tree holds only **red** refinements, which are similar to their parent
    and so never degenerate.  Green closure is not stored: :meth:`build`
    recomputes it from the current leaves every time, so a green triangle cannot
    be refined, which is the failure mode this design exists to make impossible.
    """

    def __init__(self, mesh: MeshTopology, geometry: BoundaryGeometry | None = None):
        self.geometry = geometry or BoundaryGeometry()
        # Lists, not arrays: both grow one node at a time, and `np.append`
        # copies the whole thing on every call.
        self.nodes = [tuple(p) for p in mesh.node_coords]
        self.node_tag = [0] * mesh.n_nodes
        for tag in (BoundaryTag.WALL, BoundaryTag.FARFIELD):
            for node in mesh.boundary_nodes(tag):
                self.node_tag[int(node)] = int(tag)

        self.triangles = [tuple(int(v) for v in e) for e in mesh.elements]
        self.children: list[tuple[int, int, int, int] | None] = [None] * len(self.triangles)
        #: Maintained as the tree grows.  Rebuilding it on demand instead makes
        #: :func:`transfer_state` an O(n^2) walk, which on a few thousand cells
        #: is by far the slowest thing in an adaptive run -- slower than the
        #: flow solves it sits between.
        self.parent = [-1] * len(self.triangles)
        self.level = [0] * len(self.triangles)
        self._midpoint: dict[tuple[int, int], int] = {}
        self._source = mesh.source

    # -- leaves -----------------------------------------------------------
    @property
    def leaves(self) -> list[int]:
        """Tree elements with no children, in tree order."""
        return [i for i, kids in enumerate(self.children) if kids is None]

    # -- the geometry of a split ------------------------------------------
    def _edge_tag(self, a: int, b: int) -> int:
        """A new node is on a boundary only when **both** its parents are.

        The one-sided test would be wrong in a way that is easy to miss: an
        interior edge running from a wall node out into the field would have its
        midpoint tagged as wall, and that midpoint would then be dragged onto
        the section -- tangling the mesh near the surface.
        """
        ta, tb = self.node_tag[a], self.node_tag[b]
        return ta if ta == tb else 0

    def _split(self, a: int, b: int) -> int:
        """The node halfway along edge ``(a, b)``, creating it once."""
        key = (a, b) if a < b else (b, a)
        found = self._midpoint.get(key)
        if found is not None:
            return found
        tag = self._edge_tag(a, b)
        point = 0.5 * (np.asarray(self.nodes[a]) + np.asarray(self.nodes[b]))
        if tag:
            point = self.geometry.project(point[None, :], tag)[0]
        self._midpoint[key] = len(self.nodes)
        self.nodes.append((float(point[0]), float(point[1])))
        self.node_tag.append(tag)
        return self._midpoint[key]

    def _is_split(self, a: int, b: int) -> bool:
        return ((a, b) if a < b else (b, a)) in self._midpoint

    # -- refinement --------------------------------------------------------
    def refine(self, leaves) -> None:
        """Red-subdivide each named leaf into four similar children."""
        for leaf in leaves:
            if self.children[leaf] is not None:
                continue
            a, b, c = self.triangles[leaf]
            ab, bc, ca = self._split(a, b), self._split(b, c), self._split(c, a)
            # Three corner children and the medial triangle in the middle.  All
            # four are similar to the parent and inherit its winding, so no
            # orientation test is needed here or afterwards.
            kids = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            first = len(self.triangles)
            self.triangles.extend(kids)
            self.children.extend([None] * 4)
            self.parent.extend([leaf] * 4)
            self.level.extend([self.level[leaf] + 1] * 4)
            self.children[leaf] = (first, first + 1, first + 2, first + 3)

    def close(self) -> None:
        """Make the tree one-irregular, and red-refine anything over-split.

        Two rules, applied until nothing changes:

        * a leaf with **two or three** split edges becomes red -- bisecting it
          would need two cuts, and two cuts of a triangle is a red refinement
          done badly;
        * a leaf whose edge is split **twice over** -- its midpoint is itself a
          midpoint of something finer -- is two levels coarser than its
          neighbour, so it is refined to bring the difference back to one.

        No adjacency structure is needed for either: whether an edge is split is
        a question about the midpoint registry, which already knows.
        """
        while True:
            again = []
            for leaf in self.leaves:
                a, b, c = self.triangles[leaf]
                edges = ((a, b), (b, c), (c, a))
                split = [e for e in edges if self._is_split(*e)]
                if len(split) >= 2:
                    again.append(leaf)
                    continue
                if any(
                    self._is_split(x, self._midpoint[(x, y) if x < y else (y, x)])
                    or self._is_split(self._midpoint[(x, y) if x < y else (y, x)], y)
                    for x, y in split
                ):
                    again.append(leaf)
            if not again:
                return
            self.refine(again)

    # -- rendering ---------------------------------------------------------
    def build(self) -> tuple[MeshTopology, np.ndarray]:
        """The conforming mesh, and the tree leaf each of its elements came from.

        Green closure happens **here**, on the way out, and is thrown away
        afterwards.  The second return value is what carries the solution across
        a refinement: element ``i`` of the new mesh belongs to tree leaf
        ``leaf_of[i]``, and a leaf knows its ancestors.
        """
        self.close()
        elements, leaf_of = [], []
        for leaf in self.leaves:
            a, b, c = self.triangles[leaf]
            rotations = ((a, b, c), (b, c, a), (c, a, b))
            for x, y, z in rotations:
                if self._is_split(x, y):
                    m = self._midpoint[(x, y) if x < y else (y, x)]
                    elements.extend([(x, m, z), (m, y, z)])
                    leaf_of.extend([leaf, leaf])
                    break
            else:
                elements.append((a, b, c))
                leaf_of.append(leaf)

        coords = np.asarray(self.nodes, dtype=float)
        connectivity = enforce_ccw(coords, np.asarray(elements, dtype=np.int64))
        interior, boundary, tags, slices = build_edges(
            coords, connectivity, np.asarray(self.node_tag, dtype=np.int64)
        )
        mesh = MeshTopology(
            node_coords=coords,
            elements=connectivity,
            interior_edges=interior,
            boundary_edges=boundary,
            boundary_tags=tags,
            shape=None,
            source=f"adapted {self._source}",
            _tag_slices=slices,
        )
        return mesh, np.asarray(leaf_of, dtype=np.int64)

    def ancestors(self, leaf: int):
        """``leaf`` and every tree element above it, nearest first."""
        while leaf >= 0:
            yield leaf
            leaf = self.parent[leaf]


def transfer_state(
    state: np.ndarray, tree: RefinementTree, old_leaf_of: np.ndarray, new_leaf_of: np.ndarray
) -> np.ndarray:
    """Carry a solution onto the refined mesh: every child takes its parent's value.

    The exact :math:`L^2` projection for a piecewise-constant representation, and
    the only honest one -- a finer mesh holds no information the coarse solution
    did not have.  It is a starting point, not an answer: the next solve still
    converges from it.
    """
    value = {}
    for element, leaf in enumerate(old_leaf_of):
        value.setdefault(int(leaf), state[element])

    out = np.empty((new_leaf_of.size, state.shape[1]))
    for element, leaf in enumerate(new_leaf_of):
        for ancestor in tree.ancestors(int(leaf)):
            if ancestor in value:
                out[element] = value[ancestor]
                break
        else:  # pragma: no cover - every leaf descends from a base element
            raise RuntimeError(f"element {element} descends from no solved cell")
    return out


# --------------------------------------------------------------------------
# the solve-adapt loop
# --------------------------------------------------------------------------
@dataclass
class AdaptCycle:
    """One solve on one mesh, and what it cost."""

    cycle: int
    n_elements: int
    loads: Forces
    result: SolveResult = field(repr=False)
    refined: int = 0

    @property
    def row(self) -> str:
        return (
            f"{self.cycle:>5}  {self.n_elements:>8}  {self.loads.cl:+10.5f}  "
            f"{self.loads.cd:+10.5f}  {self.loads.cm:+10.5f}  "
            f"{self.result.iterations:>7}  {self.result.seconds:>6.1f}"
        )


@dataclass
class AdaptHistory:
    """Every cycle of an adaptive run, and the final mesh."""

    cycles: list[AdaptCycle]
    mesh: MeshTopology
    metrics: Metrics = field(repr=False)
    stalled: bool = False

    @property
    def final(self) -> AdaptCycle:
        return self.cycles[-1]

    def table(self) -> str:
        head = (
            f"{'cycle':>5}  {'cells':>8}  {'cl':>10}  {'cd':>10}  "
            f"{'cm(c/4)':>10}  {'its':>7}  {'s':>6}"
        )
        return "\n".join([head, *(c.row for c in self.cycles)])

    def summary(self) -> str:
        first, last = self.cycles[0], self.cycles[-1]
        growth = last.n_elements / first.n_elements
        seconds = sum(c.result.seconds for c in self.cycles)
        moved = abs(last.loads.cl - self.cycles[-2].loads.cl) if len(self.cycles) > 1 else math.nan
        return (
            f"{len(self.cycles)} cycles, {first.n_elements} -> {last.n_elements} cells "
            f"({growth:.1f}x) in {seconds:.1f} s; cl moved {moved:+.5f} over the last cycle"
        )


def adaptive_solve(
    airfoil: str | Airfoil = "0012",
    discretization: Discretization | None = None,
    flow: FlowConditions | None = None,
    options: SolverOptions | None = None,
    *,
    cycles: int = 5,
    fraction: float = REFINE_FRACTION,
    verbose: bool = True,
) -> AdaptHistory:
    """Solve, flag, refine, and solve again -- ``legacy/main.m``'s outer loop.

    Each cycle after the first starts from the previous solution carried onto the
    new mesh, which is most of the march already done.

    **The iteration count still climbs**, and steeply, because each cycle makes
    the smallest cell smaller and an explicit march is paced by it: on the
    transonic case the first cycle converges in 5 000 iterations and the sixth in
    56 000, for cells that are 50 times smaller.  Adaptation buys accuracy per
    cell, not accuracy per second, and the two are not the same claim.  If the
    march runs out of iterations the loop **stops and returns what it has**, with
    :attr:`AdaptHistory.stalled` set, rather than refining a mesh from a solution
    that never converged.
    """
    from ..api import build_case

    case = build_case(airfoil, discretization)
    flow = flow or FlowConditions()
    options = options or SolverOptions(print_interval=0)

    tree = RefinementTree(case.mesh, BoundaryGeometry.from_case(case.airfoil, case.discretization))
    leaf_of = np.arange(case.mesh.n_elements, dtype=np.int64)
    mesh, metrics, state = case.mesh, case.metrics, None

    history: list[AdaptCycle] = []
    for cycle in range(cycles):
        result = solve_steady(metrics, flow, options, state=state)
        entry = AdaptCycle(
            cycle=cycle, n_elements=metrics.n_elements, loads=forces(result), result=result
        )
        history.append(entry)
        if verbose:
            if cycle == 0:
                print(
                    f"{'cycle':>5}  {'cells':>8}  {'cl':>10}  {'cd':>10}  "
                    f"{'cm(c/4)':>10}  {'its':>7}  {'s':>6}"
                )
            print(entry.row)
        if not result.converged:
            # Stop, but keep everything up to here.  Refining on would flag the
            # march's transients rather than the flow, and the cycles already
            # done are perfectly good -- throwing them away with an exception
            # would punish the user for a limit they can simply raise.
            if verbose:
                print(
                    f"\nstopping at cycle {cycle}: {result.message}.  A deeply refined "
                    f"mesh has much smaller cells, so the march needs many more "
                    f"iterations; raise max_iterations, or adapt fewer times."
                )
            return AdaptHistory(cycles=history, mesh=mesh, metrics=metrics, stalled=True)
        if cycle == cycles - 1:
            break

        flagged = flag_fraction(mach_jump_indicator(result), fraction)
        entry.refined = int(flagged.size)
        tree.refine(leaf_of[flagged])
        mesh, new_leaf_of = tree.build()
        state = transfer_state(result.state, tree, leaf_of, new_leaf_of)
        metrics, leaf_of = build_metrics(mesh), new_leaf_of

    return AdaptHistory(cycles=history, mesh=mesh, metrics=metrics)


__all__ = [
    "PROJECTION_MINIMUM",
    "PROJECTION_REFINEMENT",
    "REFINE_FRACTION",
    "AdaptCycle",
    "AdaptHistory",
    "BoundaryGeometry",
    "RefinementTree",
    "adaptive_solve",
    "flag_fraction",
    "mach_jump_indicator",
    "transfer_state",
]
