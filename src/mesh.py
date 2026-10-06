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


class BoundaryTag(IntEnum):
    """Which boundary condition a boundary edge gets.

    The legacy MATLAB used ``-1`` for the airfoil and ``-2`` for the far field.
    """

    WALL = 1
    FARFIELD = 2


# --------------------------------------------------------------------------
# The radial distribution
# --------------------------------------------------------------------------
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

    span = B - A
    L = np.hypot(span[:, 0], span[:, 1])
    if np.any(L <= 0.0):
        raise ValueError("the far-field circle passes through the airfoil surface")
    u_hat = span / L[:, None]

    f = radial_distribution(disc.n_radial, disc.first_cell_height, radius, disc.growth_limit)
    w = (np.arange(disc.n_radial + 1) / disc.n_radial) ** disc.orthogonality

    direction = (1.0 - w)[None, :, None] * n_hat[:, None, :] + w[None, :, None] * u_hat[:, None, :]
    norm = np.hypot(direction[..., 0], direction[..., 1])
    if np.any(norm < 1e-12):
        raise ValueError(
            "the wall normal and the radial direction are opposed somewhere on the "
            "surface; move far_field_centre inside the section"
        )
    direction /= norm[..., None]

    coords = A[:, None, :] + (L[:, None] * f[None, :])[..., None] * direction
    # pin both boundaries exactly, so a wall node is the surface point it came
    # from and a far-field node is on the circle to round-off
    coords[:, 0, :] = A
    coords[:, -1, :] = B
    return OGrid(coords=np.ascontiguousarray(coords), airfoil=airfoil, discretization=disc)


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
        shape = f"{self.shape[0]}x{self.shape[1]} O-grid" if self.shape else self.source
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


def triangulate(grid: OGrid) -> MeshTopology:
    """Split every quadrilateral cell of the O-grid into two triangles.

    The quadrilateral ``(i, j) -> (i, j+1) -> (i+1, j+1) -> (i+1, j)`` is
    counter-clockwise, and the diagonal alternates with the parity of ``i + j``
    so the triangulation has no preferred direction -- a single fixed diagonal
    biases a shock that happens to run along it.
    """
    ns, nr = grid.n_surface, grid.n_radial
    i = np.arange(ns)[:, None]
    j = np.arange(nr)[None, :]

    q0 = grid.node_index(i, j)
    q1 = grid.node_index(i, j + 1)
    q2 = grid.node_index(i + 1, j + 1)
    q3 = grid.node_index(i + 1, j)

    forward = ((i + j) % 2 == 0) | np.zeros_like(q0, dtype=bool)
    tri_a = np.where(forward[..., None], np.stack([q0, q1, q2], -1), np.stack([q0, q1, q3], -1))
    tri_b = np.where(forward[..., None], np.stack([q0, q2, q3], -1), np.stack([q1, q2, q3], -1))
    elements = np.concatenate([tri_a.reshape(-1, 3), tri_b.reshape(-1, 3)], axis=0).astype(np.int64)

    node_coords = np.ascontiguousarray(
        grid.coords.transpose(1, 0, 2).reshape(-1, 2)
    )  # node_index(i, j) = j * ns + i

    node_tags = np.zeros(node_coords.shape[0], dtype=np.int64)
    node_tags[grid.node_index(np.arange(ns), 0)] = BoundaryTag.WALL
    node_tags[grid.node_index(np.arange(ns), nr)] = BoundaryTag.FARFIELD

    interior, boundary, tags, slices = build_edges(node_coords, elements, node_tags)
    mesh = MeshTopology(
        node_coords=node_coords,
        elements=elements,
        interior_edges=interior,
        boundary_edges=boundary,
        boundary_tags=tags,
        shape=(ns, nr),
        _tag_slices=slices,
    )
    mesh.validate()
    return mesh


def build_mesh(airfoil: Airfoil, disc: Discretization | None = None) -> tuple[MeshTopology, OGrid]:
    """Generate and triangulate in one call.  Returns the mesh and the O-grid."""
    disc = disc or Discretization()
    grid = generate_ogrid(airfoil, disc)
    return triangulate(grid), grid


__all__ = [
    "BoundaryTag",
    "MeshTopology",
    "OGrid",
    "build_edges",
    "build_mesh",
    "enforce_ccw",
    "generate_ogrid",
    "radial_distribution",
    "signed_area",
    "triangulate",
]
