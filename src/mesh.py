r"""Mesh generation: a body-fitted O-grid around the airfoil, split into triangles.

Why this file exists at all
--------------------------
The MATLAB code it replaces **contained no mesher**.  ``legacy/initialmesh.m`` is
four lines that hand a decomposed geometry matrix to the MATLAB PDE Toolbox's
``initmesh``, and ``legacy/adaptmesh.m`` does the same with ``refinemesh``.
Neither has a Python equivalent, so the mesh is the one genuinely new part of
this port, and it is worth saying what was chosen and why.

The choice: an algebraic O-grid
-------------------------------
``n_surface`` points around the airfoil by ``n_radial`` cells out to a circular
far field, each quadrilateral cell split into two triangles.  Against an
unstructured Delaunay mesher this buys four things that matter here:

1. **It needs no dependency** and fits in one readable file.
2. **Array shapes are fixed and known in advance**, which is what Numba and JAX
   want -- a mesh whose element count depends on the geometry forces a
   recompile per design point.
3. **Node positions are a smooth, closed-form function of the surface points**,
   so a shape derivative flows through the mesh.  This is the part that makes the
   adjoint possible at all: an unstructured remesh is not differentiable, and a
   gradient taken across one is noise.
4. **Cells hug the wall**, where the pressure that makes up the lift is read.

The cost is that flagged isotropic refinement is not natural on a structured
grid, so resolution is set by ``n_surface`` and ``n_radial`` rather than by an
error indicator.  That is a deliberate trade, and the adaptive path stays open:
see ``docs/theory.md``.

How a node is placed
--------------------
For surface point :math:`\vec{A}_i` with outward unit normal :math:`\hat{n}_i`,
the matching far-field point is

.. math::
    \vec{B}_i = \vec{C} + R\,(\cos\varphi_i,\ \sin\varphi_i), \qquad
    \varphi_i = \varphi_0 + \frac{2\pi i}{N},

where :math:`\varphi_0` is the direction of the trailing edge seen from the
far-field centre :math:`\vec{C}`.  The angle runs *uniformly* while the surface
points are clustered, so grid lines fan out from the dense leading edge to an
evenly graded far field, and :math:`\varphi_i` is monotone in :math:`i` by
construction -- grid lines cannot cross at the outer boundary.

Each node then sits at

.. math::
    \vec{X}_{ij} = \vec{A}_i + L_i f_j \hat{D}_{ij}, \qquad
    L_i = \lVert \vec{B}_i - \vec{A}_i \rVert, \qquad
    \hat{D}_{ij} = \frac{(1 - w_j)\hat{n}_i + w_j \hat{u}_i}
                        {\lVert (1 - w_j)\hat{n}_i + w_j \hat{u}_i \rVert},

with :math:`\hat{u}_i` the unit vector from :math:`\vec{A}_i` to
:math:`\vec{B}_i`, :math:`f_j` the geometric radial distribution (:math:`f_0 =
0`, :math:`f_{n} = 1`) and :math:`w_j = (j/n)^{\kappa}` the marching-direction
blend.

The two ends are exact rather than approximate, which is the point of writing it
this way: :math:`w_0 = 0` gives :math:`\hat{D} = \hat{n}`, so the first cell is
**wall-orthogonal**; :math:`w_n = 1` gives :math:`\hat{D} = \hat{u}` and
:math:`f_n = 1`, so the last node lands **exactly on the circle**.  Because
:math:`\kappa \ge 1` the blend also leaves the wall with zero slope, so there is
no kink in the first few layers.

Centring :math:`\vec{C}` at mid-chord is what keeps this well conditioned: there
:math:`\hat{n}_i` and :math:`\hat{u}_i` very nearly agree at the leading edge,
at the trailing edge and over most of the surface of a slender section, so the
blend turns the marching direction through a small angle rather than a large one.
A fold is still possible for an extreme request, so :func:`MeshTopology.validate`
checks every triangle's area and says what to change.

Boundary tags
-------------
Two tags, taken from the structured indices rather than from a geometric test,
so there is no tolerance to tune: :attr:`BoundaryTag.WALL` at :math:`j = 0` and
:attr:`BoundaryTag.FARFIELD` at :math:`j = n_{radial}`.  The legacy code wrote
these as ``-1`` and ``-2``; nothing but ``legacy/`` uses those values now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import Discretization
from .geometry import Airfoil, signed_area, surface_normals

#: Largest aspect ratio allowed for a first cell whose height is not set by a
#: wall.  Only the C-grid's wake cut is affected; see :func:`generate_cgrid`.
FIRST_CELL_ASPECT_CAP = 20.0


class BoundaryTag(IntEnum):
    """Which boundary condition a boundary edge gets.

    The legacy MATLAB used ``-1`` for the airfoil and ``-2`` for the far field.
    """

    WALL = 1
    FARFIELD = 2


# --------------------------------------------------------------------------
# The radial distribution
# --------------------------------------------------------------------------
def shortest_reachable_first_cell(n_cells: int, total_length: float, growth_limit: float) -> float:
    r"""The smallest first cell that still spans ``total_length`` in ``n_cells``.

    Inverting the geometric sum at the maximum allowed ratio:

    .. math::
        h_{1,\min} = L\,\frac{r_{\max} - 1}{r_{\max}^{\,n} - 1} .

    :func:`radial_distribution` *raises* when it is handed something shorter,
    because there the first cell is what the user asked for and quietly
    changing it would be changing the mesh behind their back.  The wake cut is
    the opposite case: its first cell is **derived** from the trailing-edge
    panel, so there is nothing of the user's to override, and clamping it is
    the right answer rather than a failure.
    """
    return total_length * (growth_limit - 1.0) / (growth_limit**n_cells - 1.0)


def radial_profile(
    n_cells: int,
    first_height: np.ndarray,
    total_length: np.ndarray,
    growth_limit: float = 1.25,
) -> np.ndarray:
    """:func:`radial_distribution` for many stations at once, one row each.

    The marching distance and the first cell both vary along the inner path, so
    each station solves its own growth ratio.  Same bisection as the scalar
    version, run on arrays.
    """
    h1 = np.asarray(first_height, dtype=float) / np.asarray(total_length, dtype=float)
    if np.any(h1 <= 0.0):
        raise ValueError("every first cell height must be positive")
    uniform = h1 * n_cells >= 1.0

    def span(r):
        return np.where(uniform, 1.0, h1 * (r**n_cells - 1.0) / (r - 1.0))

    reach = span(np.full_like(h1, growth_limit))
    if np.any(reach < 1.0):
        worst = int(np.argmin(reach))
        raise ValueError(
            f"cannot reach the outer boundary at station {worst} of {h1.size}: "
            f"{n_cells} cells starting at {first_height[worst]:g} and growing by at "
            f"most {growth_limit:g} span only "
            f"{reach[worst] * total_length[worst]:.3g} of {total_length[worst]:g}.  "
            f"Raise n_radial, raise first_cell_height, raise growth_limit, or shrink "
            f"the domain."
        )

    lo = np.full_like(h1, 1.0 + 1e-12)
    hi = np.full_like(h1, growth_limit)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        short = span(mid) < 1.0
        lo = np.where(short, mid, lo)
        hi = np.where(short, hi, mid)
    r = 0.5 * (lo + hi)

    step = h1[:, None] * r[:, None] ** np.arange(n_cells)[None, :]
    step = np.where(uniform[:, None], 1.0 / n_cells, step)
    f = np.concatenate([np.zeros((h1.size, 1)), np.cumsum(step, axis=1)], axis=1)
    return f / f[:, -1:]


def radial_distribution(
    n_cells: int,
    first_height: float,
    total_length: float,
    growth_limit: float = 1.25,
) -> np.ndarray:
    r"""Normalised radial node positions: ``n_cells + 1`` values from 0 to 1.

    Spacing grows geometrically, :math:`h_k = h_1 r^k`, with the ratio
    :math:`r` solved so the layers exactly span the distance:

    .. math::
        h_1 \frac{r^{n} - 1}{r - 1} = L .

    The left-hand side increases with :math:`r`, so one bisection on
    :math:`(1, r_{\max}]` finds it.  If even :math:`r_{\max}` cannot span
    :math:`L` this raises, rather than returning a grid that stops short of the
    far field -- which would otherwise show up much later as a solver that will
    not converge.
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    h1 = first_height / total_length
    if h1 <= 0.0:
        raise ValueError(f"first_height must be positive, got {first_height}")
    if h1 * n_cells >= 1.0:
        # the requested first cell already spans the domain: uniform is the most
        # the geometry allows, and growing from there would overshoot
        return np.linspace(0.0, 1.0, n_cells + 1)

    def span(r: float) -> float:
        return h1 * (r**n_cells - 1.0) / (r - 1.0)

    if span(growth_limit) < 1.0:
        raise ValueError(
            f"cannot reach the far field: {n_cells} cells starting at "
            f"{first_height:g} and growing by at most {growth_limit:g} span only "
            f"{span(growth_limit) * total_length:.3g} of {total_length:g}.  "
            f"Raise n_radial, raise first_cell_height, raise growth_limit, or "
            f"shrink far_field_radius."
        )
    lo, hi = 1.0 + 1e-12, growth_limit
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if span(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)

    f = np.concatenate([[0.0], np.cumsum(h1 * r ** np.arange(n_cells))])
    return f / f[-1]  # pin the outer boundary at exactly 1


# --------------------------------------------------------------------------
# The structured grid
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OGrid:
    """Structured node coordinates, shaped ``(n_surface, n_radial + 1, 2)``.

    Index ``i`` runs counter-clockwise around the airfoil from the trailing edge
    and is periodic; index ``j`` runs outward from the wall.
    """

    coords: np.ndarray
    airfoil: Airfoil
    discretization: Discretization

    @property
    def n_surface(self) -> int:
        return self.coords.shape[0]

    @property
    def n_radial(self) -> int:
        return self.coords.shape[1] - 1

    @property
    def wall(self) -> np.ndarray:
        return self.coords[:, 0, :]

    @property
    def far_field(self) -> np.ndarray:
        return self.coords[:, -1, :]

    def node_index(self, i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray | int:
        """Flat node number of structured index ``(i, j)``; ``i`` wraps."""
        return np.asarray(j) * self.n_surface + np.asarray(i) % self.n_surface

    def first_cell_heights(self) -> np.ndarray:
        """Actual height of the first cell at every surface station.

        ``Discretization.first_cell_height`` is applied relative to the
        far-field radius, but the distance from the surface to the circle varies
        a little around the section, so the realised heights vary with it.  This
        is what to look at when a wall-resolution claim has to be exact.
        """
        return np.hypot(*(self.coords[:, 1, :] - self.coords[:, 0, :]).T)


def march(
    inner: np.ndarray,
    outer: np.ndarray,
    n_hat: np.ndarray,
    n_cells: int,
    first_height: np.ndarray | float,
    growth_limit: float,
    orthogonality: float,
) -> np.ndarray:
    r"""Place the nodes between an inner path and an outer one.

    The shared construction behind both topologies, and the reason either can be
    swapped for the other without touching anything downstream.  See the module
    docstring for the formula; ``f`` is the normalised radial distribution and
    ``orthogonality`` the exponent :math:`\kappa` of the marching-direction
    blend.  The two ends are pinned exactly: the first layer is the inner path
    and the last is the outer one.
    """
    span = outer - inner
    L = np.hypot(span[:, 0], span[:, 1])
    if np.any(L <= 0.0):
        raise ValueError("the outer boundary touches the inner one somewhere")
    u_hat = span / L[:, None]

    f = radial_profile(n_cells, np.broadcast_to(first_height, L.shape), L, growth_limit)
    w = (np.arange(n_cells + 1) / n_cells) ** orthogonality
    direction = (1.0 - w)[None, :, None] * n_hat[:, None, :] + w[None, :, None] * u_hat[:, None, :]
    norm = np.hypot(direction[..., 0], direction[..., 1])
    if np.any(norm < 1e-12):
        raise ValueError(
            "the wall normal and the outward direction are opposed somewhere on the "
            "inner path; move far_field_centre inside the section"
        )
    direction /= norm[..., None]

    coords = inner[:, None, :] + (L[:, None] * f)[..., None] * direction
    coords[:, 0, :] = inner
    coords[:, -1, :] = outer
    return np.ascontiguousarray(coords)


def generate_ogrid(airfoil: Airfoil, disc: Discretization) -> OGrid:
    """Build the O-grid.  See the module docstring for the construction."""
    A = airfoil.points
    n_hat = surface_normals(A)
    N = A.shape[0]

    centre = np.asarray(disc.far_field_centre, dtype=float)
    radius = float(disc.far_field_radius)
    offset = A[0] - centre
    if np.hypot(*offset) < 1e-12:
        raise ValueError("far_field_centre coincides with the trailing edge")
    phi0 = math.atan2(offset[1], offset[0])
    phi = phi0 + 2.0 * math.pi * np.arange(N) / N
    B = centre + radius * np.stack([np.cos(phi), np.sin(phi)], axis=-1)

    coords = march(
        A,
        B,
        n_hat,
        disc.n_radial,
        disc.first_cell_height,
        disc.growth_limit,
        disc.orthogonality,
    )
    return OGrid(coords=coords, airfoil=airfoil, discretization=disc)


# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshTopology:
    """A conforming triangular mesh with tagged boundaries.

    The solver core sees only this, never the O-grid, so an externally generated
    mesh (see :mod:`src.meshio`) is interchangeable with a generated one.

    Attributes
    ----------
    node_coords
        ``(n_nodes, 2)``.
    elements
        ``(n_elements, 3)`` node numbers, wound counter-clockwise.
    interior_edges
        ``(n_interior, 4)`` as ``[left element, right element, node a, node b]``.
        The edge runs from ``a`` to ``b`` in the *left* element's own
        counter-clockwise order, so the normal :math:`(\\Delta y, -\\Delta x)/\\ell`
        points from the left element into the right one.
    boundary_edges
        ``(n_boundary, 3)`` as ``[element, node a, node b]``, grouped by tag.
        The same orientation rule makes that normal point **out of the domain**.
    boundary_tags
        ``(n_boundary,)`` of :class:`BoundaryTag`.
    shape
        ``(n_surface, n_radial)`` if this came from an O-grid, else ``None``.
    """

    node_coords: np.ndarray
    elements: np.ndarray
    interior_edges: np.ndarray
    boundary_edges: np.ndarray
    boundary_tags: np.ndarray
    shape: tuple[int, int] | None = None
    source: str = "ogrid"
    _tag_slices: dict[int, slice] = field(default_factory=dict, repr=False)

    @property
    def n_nodes(self) -> int:
        return self.node_coords.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_interior_edges(self) -> int:
        return self.interior_edges.shape[0]

    @property
    def n_boundary_edges(self) -> int:
        return self.boundary_edges.shape[0]

    def tag_slice(self, tag: BoundaryTag) -> slice:
        """The contiguous run of boundary edges carrying ``tag``."""
        return self._tag_slices.get(int(tag), slice(0, 0))

    def boundary_nodes(self, tag: BoundaryTag) -> np.ndarray:
        """Unique node numbers on one boundary."""
        return np.unique(self.boundary_edges[self.tag_slice(tag), 1:])

    def signed_areas(self) -> np.ndarray:
        """Signed area of every triangle; all positive once wound counter-clockwise."""
        p = self.node_coords[self.elements]
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    def validate(self) -> None:
        """Raise if the mesh is not a usable conforming triangulation.

        Checks the three failures that actually happen: a folded or zero-area
        cell (the generator was pushed too hard), a node that no element uses,
        and a boundary that does not close.
        """
        area = self.signed_areas()
        bad = np.flatnonzero(area <= 0.0)
        if bad.size:
            worst = bad[np.argmin(area[bad])]
            centre = self.node_coords[self.elements[worst]].mean(axis=0)
            raise ValueError(
                f"{bad.size} of {self.n_elements} cells have non-positive area; the "
                f"worst is element {worst} at ({centre[0]:.4f}, {centre[1]:.4f}) with "
                f"area {area[worst]:.3e}.  The grid has folded: lower "
                f"'orthogonality', raise 'n_surface', or raise 'far_field_radius'."
            )
        used = np.unique(self.elements)
        if used.size != self.n_nodes:
            raise ValueError(
                f"{self.n_nodes - used.size} of {self.n_nodes} nodes belong to no element"
            )
        for tag in BoundaryTag:
            edges = self.boundary_edges[self.tag_slice(tag), 1:]
            if edges.size == 0:
                continue
            starts, ends = np.sort(edges[:, 0]), np.sort(edges[:, 1])
            if not np.array_equal(starts, ends):
                raise ValueError(f"the {tag.name} boundary is not a closed loop")

    def n_faces_on(self, tag: BoundaryTag) -> int:
        """How many boundary faces carry ``tag``."""
        s = self.tag_slice(tag)
        return s.stop - s.start

    def summary(self) -> str:
        shape = f"{self.shape[0]}x{self.shape[1]} {self.source}" if self.shape else self.source
        return (
            f"{shape}: {self.n_elements} triangles, {self.n_nodes} nodes, "
            f"{self.n_interior_edges} interior edges, "
            f"{self.n_boundary_edges} boundary edges "
            f"({self.n_faces_on(BoundaryTag.WALL)} wall, "
            f"{self.n_faces_on(BoundaryTag.FARFIELD)} far field)"
        )


def enforce_ccw(node_coords: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Rewind any clockwise triangle, the vectorised ``legacy/counter_clock_sort.m``.

    The generator here produces the right winding already; this exists for meshes
    read from a file, where the winding is whatever the writer chose.
    """
    elements = np.ascontiguousarray(elements, dtype=np.int64)
    p = node_coords[elements]
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (
        p[:, 2, 0] - p[:, 0, 0]
    )
    flip = cross < 0.0
    elements[flip] = elements[flip][:, [0, 2, 1]]
    return elements


def build_edges(
    node_coords: np.ndarray,
    elements: np.ndarray,
    node_tags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, slice]]:
    r"""Find every edge and say which elements touch it.

    This replaces the sparse-hash loop of ``legacy/meshconnect.m`` with a sort.
    Local edge :math:`k` of a counter-clockwise triangle is the one *opposite*
    node :math:`k`, running from node :math:`k+1` to node :math:`k+2`, so its
    outward normal needs no sign test.

    ``node_tags`` carries a :class:`BoundaryTag` per node, or 0 for an interior
    node; a boundary edge takes the tag its two nodes agree on.
    """
    half = np.stack(
        [elements[:, [1, 2]], elements[:, [2, 0]], elements[:, [0, 1]]], axis=1
    ).reshape(-1, 2)
    key = np.sort(half, axis=1)

    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    if counts.max() > 2:
        raise ValueError("an edge is shared by more than two elements; the mesh is not manifold")

    order = np.argsort(inverse, kind="stable")
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])

    inner = np.flatnonzero(counts == 2)
    first, second = order[start[inner]], order[start[inner] + 1]
    interior = np.column_stack([first // 3, second // 3, half[first, 0], half[first, 1]])

    outer = order[start[np.flatnonzero(counts == 1)]]
    tags = node_tags[half[outer, 0]]
    disagree = tags != node_tags[half[outer, 1]]
    if np.any(disagree):
        raise ValueError(
            f"{int(disagree.sum())} boundary edges join nodes with different tags; "
            f"a boundary edge must lie wholly on one boundary"
        )
    if np.any(tags == 0):
        raise ValueError("a boundary edge has untagged nodes, so it gets no boundary condition")

    # group by tag so a boundary condition is a contiguous slice
    regroup = np.argsort(tags, kind="stable")
    outer, tags = outer[regroup], tags[regroup]
    boundary = np.column_stack([outer // 3, half[outer, 0], half[outer, 1]])

    slices: dict[int, slice] = {}
    for tag in np.unique(tags):
        hit = np.flatnonzero(tags == tag)
        slices[int(tag)] = slice(int(hit[0]), int(hit[-1]) + 1)

    return (
        np.ascontiguousarray(interior, dtype=np.int64),
        np.ascontiguousarray(boundary, dtype=np.int64),
        np.ascontiguousarray(tags, dtype=np.int64),
        slices,
    )


@dataclass(frozen=True)
class CGrid:
    """A C-grid: the structured nodes, plus the map that collapses the wake cut.

    Index ``i`` runs along one open path -- in from the downstream outflow plane
    along the upper side of the wake cut, around the section from the trailing
    edge over the upper surface to the leading edge and back along the lower,
    then out along the lower side of the cut.  Index ``j`` runs outward.  Unlike
    the O-grid's, ``i`` does **not** wrap.

    ``node_ids`` is what makes the cut work.  At ``j = 0`` the two sides of the
    cut are the same physical points, so they are given the **same node number**;
    the cells above and below the cut then share those faces and are ordinary
    interior neighbours.  The cut needs no boundary condition and appears
    nowhere in the solver.  Above ``j = 0`` the two sides separate and every
    node is its own.
    """

    coords: np.ndarray
    node_ids: np.ndarray
    airfoil: Airfoil
    discretization: Discretization

    @property
    def n_columns(self) -> int:
        """Quadrilateral columns: one fewer than the number of ``i`` stations."""
        return self.coords.shape[0] - 1

    @property
    def n_radial(self) -> int:
        return self.coords.shape[1] - 1

    @property
    def n_wake(self) -> int:
        return self.discretization.n_wake

    @property
    def wall_slice(self) -> slice:
        """The ``i`` stations lying on the section, trailing edge to trailing edge."""
        return slice(self.n_wake, self.n_wake + self.airfoil.n_points + 1)

    @property
    def wall(self) -> np.ndarray:
        return self.coords[self.wall_slice, 0, :]

    def first_cell_heights(self) -> np.ndarray:
        """Height of the first cell at every station on the section."""
        on_wall = self.coords[self.wall_slice]
        return np.hypot(*(on_wall[:, 1, :] - on_wall[:, 0, :]).T)

    def first_wake_cell(self) -> float:
        """Streamwise length of the first cell aft of the trailing edge."""
        i = self.n_wake
        return float(np.hypot(*(self.coords[i - 1, 0, :] - self.coords[i, 0, :])))


def _wake_cut(trailing_edge: np.ndarray, disc: Discretization, first: float) -> np.ndarray:
    """Points along one side of the wake cut, trailing edge first.

    The cut runs straight aft at the trailing edge's own height, with the same
    geometric stretching the radial direction uses, started at the length of the
    surface panel next to the trailing edge -- so the first wake cell continues
    the surface spacing instead of jumping away from it.
    """
    length = disc.wake_length - trailing_edge[0]
    if length <= 0.0:
        raise ValueError(
            f"wake_length {disc.wake_length} is not aft of the trailing edge at "
            f"x = {trailing_edge[0]:.4f}"
        )
    # Cosine clustering makes the last surface panel very short -- 2.7e-4 chords
    # at the default resolution -- and stretching from there to the outflow
    # plane would need a growth ratio far past the limit.  The surface spacing
    # is set by the curvature of the section, which the wake does not have, so
    # the floor here is not a compromise: it is the spacing the wake needs.
    floor = 1.001 * shortest_reachable_first_cell(disc.n_wake, length, disc.growth_limit)
    tau = radial_distribution(
        disc.n_wake, min(max(first, floor), 0.5 * length), length, disc.growth_limit
    )
    return trailing_edge + np.stack([length * tau, np.zeros_like(tau)], axis=-1)


def fan_monotonically(n_hat: np.ndarray) -> np.ndarray:
    r"""Force a marching-direction field to rotate one way along an open path.

    Two neighbouring stations' marching lines cross if the direction field turns
    *back* on itself, and the C-grid's does, at the trailing edge, for a reason
    that is geometric rather than a bug.  The wake cut's own normal is straight
    up; the upper surface leaves a sharp trailing edge sloping **aft** -- about
    :math:`8.3^\circ` for a NACA 0012 -- so its normal tilts aft too, while the
    stations march forward.  The angle therefore dips by those few degrees at
    the junction and climbs back.  The two stations either side of it are one
    short wake panel apart, so a 4-degree dip closes that gap after only 0.05
    chords of a 20-chord march, and the cells there invert.

    No choice of the blend exponent fixes it -- it was measured across the whole
    range -- and neither does smoothing the normals, because the dip is real
    rather than noise.  What does fix it is requiring the fan to be monotone:
    the running maximum of the unwrapped angle, taken in whichever sense the
    path turns.

    The cost is bounded and small.  Only the stations inside the dip move, which
    is the rear of each surface between the trailing edge and the point of
    maximum thickness, and the largest correction is the trailing-edge slope
    itself.  Forward of maximum thickness -- the whole leading-edge region, where
    orthogonality earns its keep -- the field is already monotone and is left
    exactly alone.
    """
    theta = np.unwrap(np.arctan2(n_hat[:, 1], n_hat[:, 0]))
    sense = 1.0 if theta[-1] >= theta[0] else -1.0
    turned = sense * theta
    # Anchored at *both* ends, not just the start.  A running maximum alone
    # flattens the dip at the first trailing edge but carries the second one
    # outward with it, which leans the far end of the wake cut by the same few
    # degrees and bows the outflow plane off vertical.  Clipping to the two end
    # values keeps both legs exactly perpendicular to it and flattens the two
    # dips symmetrically.
    turned = np.clip(np.maximum.accumulate(turned), turned[0], turned[-1])
    return np.stack([np.cos(sense * turned), np.sin(sense * turned)], axis=-1)


def generate_cgrid(airfoil: Airfoil, disc: Discretization) -> CGrid:
    r"""Build the C-grid.

    The inner path is assembled from three pieces and the outer from three
    matching ones, so the correspondence between them is fixed by index rather
    than found by a search:

    ======================  ====================================  =========================
    inner                   runs                                  outer
    ======================  ====================================  =========================
    upper side of the cut   outflow plane to the trailing edge    straight leg at :math:`y = +R`
    the section             trailing edge, round, and back        semicircle of radius :math:`R`
    lower side of the cut   trailing edge to the outflow plane    straight leg at :math:`y = -R`
    ======================  ====================================  =========================

    The semicircle is swept **uniformly in angle**, for the reason the O-grid's
    far field is, and the two straight legs reuse the cut's own stretching, so
    radial lines in the wake stay close to parallel instead of fanning.

    The trailing edge is the one station appearing twice -- once ending the
    upper cut, once starting the lower -- and its two radial lines run to
    opposite ends of the semicircle.  That is the cut opening up, and it is
    exactly what an O-grid cannot do: there, every line out of the trailing edge
    has to fan into the same wedge.
    """
    loop = airfoil.points
    trailing_edge = loop[0]
    first_panel = float(np.hypot(*(loop[1] - loop[0])))

    centre = np.asarray(disc.far_field_centre, dtype=float)
    radius = float(disc.far_field_radius)
    if disc.wake_length <= centre[0]:
        raise ValueError(
            f"wake_length {disc.wake_length} must be aft of far_field_centre "
            f"x = {centre[0]}, or the outer boundary turns back on itself"
        )

    cut = _wake_cut(trailing_edge, disc, first_panel)  # trailing edge -> outflow
    inner = np.vstack([cut[::-1][:-1], loop, loop[:1], cut[1:]])

    # The outer boundary: two straight legs at y = +/- R joined by a semicircle.
    #
    # The legs get their *own* stretching, started at the arc's spacing where
    # they meet it.  Reusing the wake cut's distribution here is the obvious
    # thing and it is wrong: the cut clusters hard at the trailing edge, so the
    # outer spacing would fall to 0.0033 right where the arc's is 0.327, and a
    # hundredfold jump between neighbouring columns puts a sliver of 0.03
    # degrees at the far field.  Measured: it took the worst cell in the mesh
    # from 0.96 to 0.9995.  Starting the leg at the arc's spacing instead keeps
    # the two sides of the junction within a factor of two.
    leg_length = disc.wake_length - centre[0]
    arc_spacing = np.pi * radius / loop.shape[0]
    tau = radial_distribution(
        disc.n_wake, min(arc_spacing, 0.5 * leg_length), leg_length, disc.growth_limit
    )
    leg_x = centre[0] + leg_length * tau  # centre -> outflow
    angle = np.linspace(0.5 * np.pi, 1.5 * np.pi, loop.shape[0] + 1)
    arc = centre + radius * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    outer = np.vstack(
        [
            np.stack([leg_x[::-1][:-1], np.full(disc.n_wake, radius)], axis=-1),
            arc,
            np.stack([leg_x[1:], np.full(disc.n_wake, -radius)], axis=-1),
        ]
    )
    if inner.shape != outer.shape:
        raise AssertionError(f"inner {inner.shape} and outer {outer.shape} must match")

    # The first cell is a *wall* height, and the wake cut has no wall.  Sharing
    # one value puts a 0.002-tall cell under a station 3.8 chords long far
    # downstream -- aspect ratio 1900, which triangulates into 0.03-degree
    # slivers and, worse, collapses the local time step through the hydraulic
    # diameter across the whole wake.  Capping the aspect ratio of the first
    # cell leaves every station on the section untouched (their panels are far
    # shorter than the cap allows) and opens the wake out to match its own
    # streamwise spacing.
    spacing = np.hypot(*np.diff(inner, axis=0).T)
    spacing = np.concatenate([spacing[:1], np.minimum(spacing[1:], spacing[:-1]), spacing[-1:]])
    first = np.maximum(disc.first_cell_height, spacing / FIRST_CELL_ASPECT_CAP)

    n_hat = fan_monotonically(surface_normals(inner, closed=False))
    coords = march(inner, outer, n_hat, disc.n_radial, first, disc.growth_limit, disc.orthogonality)

    ni, na, nw = inner.shape[0], loop.shape[0], disc.n_wake
    # at j = 0 the lower side of the cut is the upper side read backwards, so it
    # reuses those node numbers; above j = 0 the two sides separate
    station = np.arange(ni)
    shared = station >= nw + na
    base = np.where(shared, ni - 1 - station, station)
    if not np.allclose(coords[shared, 0, :], coords[base[shared], 0, :]):
        raise AssertionError("the two sides of the wake cut are not coincident")
    node_ids = np.empty((ni, disc.n_radial + 1), dtype=np.int64)
    node_ids[:, 0] = base
    node_ids[:, 1:] = (nw + na) + np.arange(disc.n_radial * ni).reshape(disc.n_radial, ni).T

    return CGrid(
        coords=coords,
        node_ids=np.ascontiguousarray(node_ids),
        airfoil=airfoil,
        discretization=disc,
    )


def _triangulate_structured(
    coords: np.ndarray,
    node_ids: np.ndarray,
    node_tags: np.ndarray,
    *,
    periodic: bool,
    shape: tuple[int, int] | None,
) -> MeshTopology:
    """Split a structured node block into triangles and build its connectivity.

    Shared by both topologies; ``periodic`` is the only thing that differs, and
    it is what makes the O-grid's index wrap and the C-grid's stop at the
    outflow plane.
    """
    ni, nj = node_ids.shape
    columns = ni if periodic else ni - 1
    i = np.arange(columns)[:, None]
    j = np.arange(nj - 1)[None, :]
    nxt = (i + 1) % ni if periodic else i + 1

    q0, q1 = node_ids[i, j], node_ids[i, j + 1]
    q2, q3 = node_ids[nxt, j + 1], node_ids[nxt, j]

    forward = (i + j) % 2 == 0
    tri_a = np.where(forward[..., None], np.stack([q0, q1, q2], -1), np.stack([q0, q1, q3], -1))
    tri_b = np.where(forward[..., None], np.stack([q0, q2, q3], -1), np.stack([q1, q2, q3], -1))
    elements = np.concatenate([tri_a.reshape(-1, 3), tri_b.reshape(-1, 3)]).astype(np.int64)

    node_coords = np.zeros((int(node_ids.max()) + 1, 2))
    node_coords[node_ids.ravel()] = coords.reshape(-1, 2)

    interior, boundary, tags, slices = build_edges(node_coords, elements, node_tags)
    mesh = MeshTopology(
        node_coords=np.ascontiguousarray(node_coords),
        elements=elements,
        interior_edges=interior,
        boundary_edges=boundary,
        boundary_tags=tags,
        shape=shape,
        source="O-grid" if periodic else "C-grid",
        _tag_slices=slices,
    )
    mesh.validate()
    return mesh


def triangulate_cgrid(grid: CGrid) -> MeshTopology:
    """Triangulate a C-grid: three sets of boundary faces, but only two conditions.

    The section is the wall.  The outer C **and** the two outflow faces at the
    downstream plane are both tagged ``FARFIELD``, because the characteristic
    far-field condition decides inflow from outflow face by face from the local
    normal velocity -- which is what a characteristic condition is for, so a
    separate outflow tag would be a second name for the same thing.

    The wake cut is not a boundary at all: its nodes are shared, so each of its
    faces has an element on either side and never reaches a boundary condition.
    """
    ni = grid.coords.shape[0]
    node_tags = np.zeros(int(grid.node_ids.max()) + 1, dtype=np.int64)
    node_tags[grid.node_ids[:, -1]] = BoundaryTag.FARFIELD
    node_tags[grid.node_ids[0, :]] = BoundaryTag.FARFIELD
    node_tags[grid.node_ids[ni - 1, :]] = BoundaryTag.FARFIELD
    node_tags[grid.node_ids[grid.wall_slice, 0]] = BoundaryTag.WALL
    return _triangulate_structured(
        grid.coords,
        grid.node_ids,
        node_tags,
        periodic=False,
        shape=(grid.airfoil.n_points, grid.n_radial),
    )


def triangulate(grid: OGrid) -> MeshTopology:
    """Split every quadrilateral cell of the O-grid into two triangles.

    The quadrilateral ``(i, j) -> (i, j+1) -> (i+1, j+1) -> (i+1, j)`` is
    counter-clockwise, and the diagonal alternates with the parity of ``i + j``
    so the triangulation has no preferred direction -- a single fixed diagonal
    biases a shock that happens to run along it.
    """
    ns, nr = grid.n_surface, grid.n_radial
    station = np.arange(ns)
    node_ids = np.stack([grid.node_index(station, j) for j in range(nr + 1)], axis=-1)

    node_tags = np.zeros(ns * (nr + 1), dtype=np.int64)
    node_tags[grid.node_index(station, 0)] = BoundaryTag.WALL
    node_tags[grid.node_index(station, nr)] = BoundaryTag.FARFIELD

    return _triangulate_structured(grid.coords, node_ids, node_tags, periodic=True, shape=(ns, nr))


def build_mesh(
    airfoil: Airfoil, disc: Discretization | None = None
) -> tuple[MeshTopology, OGrid | CGrid]:
    """Generate and triangulate in one call, for whichever topology is configured.

    Returns the mesh and the structured grid behind it.  The mesh is all the
    solver ever sees, which is what lets the topology be switched without
    anything downstream knowing it changed.
    """
    disc = disc or Discretization()
    if disc.topology == "cgrid":
        cgrid = generate_cgrid(airfoil, disc)
        return triangulate_cgrid(cgrid), cgrid
    ogrid = generate_ogrid(airfoil, disc)
    return triangulate(ogrid), ogrid


__all__ = [
    "BoundaryTag",
    "CGrid",
    "MeshTopology",
    "OGrid",
    "build_edges",
    "build_mesh",
    "enforce_ccw",
    "fan_monotonically",
    "generate_cgrid",
    "generate_ogrid",
    "march",
    "radial_distribution",
    "radial_profile",
    "shortest_reachable_first_cell",
    "signed_area",
    "triangulate",
    "triangulate_cgrid",
]
