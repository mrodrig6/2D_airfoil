r"""Geometric quantities the finite-volume scheme needs, computed once per mesh.

This is the vectorised replacement for ``legacy/meshconnect.m``, which built the
same arrays with a sparse hash table and a loop over elements.  Nothing here
depends on the flow, so it is computed once and reused across every angle of
attack, Mach number and flux in a sweep -- which is the whole reason it is a
separate object rather than something the residual recomputes.

What the solver consumes
------------------------
A cell-centred finite-volume update of :math:`\vec{U}` over element :math:`e`
with area :math:`A_e` is

.. math::
    A_e \frac{d\vec{U}_e}{dt}
    + \sum_{\text{faces } f} \hat{\vec{F}}_f \cdot \hat{n}_f\, \ell_f = 0 ,

so each face needs exactly three numbers -- a length, a unit normal, and which
elements sit on either side -- and each element needs its area.  That is all
this module produces.

The normal convention, stated once
----------------------------------
Every triangle is wound counter-clockwise, and the edge opposite local node
:math:`k` runs from node :math:`k+1` to node :math:`k+2`.  For that traversal,

.. math::
    \hat{n} = \frac{(\Delta y,\ -\Delta x)}{\ell},
    \qquad (\Delta x, \Delta y) = \vec{p}_{k+2} - \vec{p}_{k+1},

is the **outward** normal of the element, with no sign test anywhere.  Two
consequences follow and both are checked by the test suite:

* An **interior** face's normal points from its ``left`` element into its
  ``right`` element, so the residual takes ``+F`` on the left and ``-F`` on the
  right exactly as ``legacy/residual_calc.m`` did.
* A **boundary** face's normal points **out of the flow domain** -- which at the
  wall means *into* the airfoil, and at the far field means away from it.

The discrete Gauss identity :math:`\sum_f \hat{n}_f \ell_f = \vec{0}` over the
faces of any closed cell is what makes the scheme conservative and a uniform
flow exact.  :meth:`Metrics.closure_error` measures it, and it is the single most
useful check that a mesh and these metrics agree.

Length scales
-------------
The local time step needs a length per element.  The legacy code used the
hydraulic diameter :math:`d_h = 2A/P`, and that is kept here both because it is
the right scale for a triangle of any aspect ratio and because keeping it makes
the ported solver comparable to the MATLAB it came from.  On the high-aspect
cells next to the wall :math:`d_h` is close to twice the cell's thin dimension,
which is what the acoustic CFL condition actually cares about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.mesh import BoundaryTag, MeshTopology, signed_areas


def element_geometry(
    node_coords: np.ndarray, elements: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Areas and centroids of every triangle.

    The area is :func:`src.geometry.mesh.signed_areas`, not a second copy of
    the cross product: the sign convention is the mesh's to define.
    """
    return signed_areas(node_coords, elements), node_coords[elements].mean(axis=1)


def _face_geometry(
    node_coords: np.ndarray, node_a: np.ndarray, node_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Length and outward unit normal of faces given as node pairs."""
    d = node_coords[node_b] - node_coords[node_a]
    length = np.hypot(d[:, 0], d[:, 1])
    if np.any(length <= 0.0):
        raise ValueError("a face has zero length, so it has no normal")
    return length, np.stack([d[:, 1] / length, -d[:, 0] / length], axis=-1)


@dataclass(frozen=True)
class Metrics:
    """Everything geometric the residual needs, for one fixed mesh."""

    mesh: MeshTopology
    area: np.ndarray
    centroid: np.ndarray
    perimeter: np.ndarray
    hydraulic_diameter: np.ndarray
    interior_left: np.ndarray
    interior_right: np.ndarray
    interior_length: np.ndarray
    interior_normal: np.ndarray
    interior_midpoint: np.ndarray
    boundary_element: np.ndarray
    boundary_length: np.ndarray
    boundary_normal: np.ndarray
    boundary_midpoint: np.ndarray
    boundary_tag: np.ndarray

    @property
    def n_elements(self) -> int:
        return self.area.shape[0]

    @property
    def n_interior_faces(self) -> int:
        return self.interior_length.shape[0]

    @property
    def n_boundary_faces(self) -> int:
        return self.boundary_length.shape[0]

    def tag_slice(self, tag: BoundaryTag) -> slice:
        return self.mesh.tag_slice(tag)

    def closure_error(self) -> float:
        r"""Largest :math:`\lvert \sum_f \hat{n}_f \ell_f \rvert` over all elements.

        Zero to round-off on any valid mesh.  A non-zero value means the faces
        and the elements disagree about the mesh -- a missing face, a normal with
        the wrong sign, or a left/right pair swapped -- and it will show up in the
        solver as a uniform free stream that does not stay uniform.
        """
        total = np.zeros((self.n_elements, 2))
        contribution = self.interior_normal * self.interior_length[:, None]
        np.add.at(total, self.interior_left, contribution)
        np.add.at(total, self.interior_right, -contribution)
        np.add.at(
            total, self.boundary_element, self.boundary_normal * self.boundary_length[:, None]
        )
        return float(np.abs(total).max())

    def wall_tangent(self) -> np.ndarray:
        """Unit tangent of every wall face, pointing along the stored surface order."""
        s = self.tag_slice(BoundaryTag.WALL)
        n = self.boundary_normal[s]
        return np.stack([-n[:, 1], n[:, 0]], axis=-1)

    def reference_area(self) -> float:
        """Total flow area, for reporting and for the mesh test."""
        return float(self.area.sum())

    def summary(self) -> str:
        h = self.hydraulic_diameter
        return (
            f"{self.n_elements} cells, area {self.reference_area():.6f}, "
            f"hydraulic diameter {h.min():.3e} to {h.max():.3e}, "
            f"closure error {self.closure_error():.2e}"
        )


def build_metrics(mesh: MeshTopology) -> Metrics:
    """Compute the metrics of one mesh.  Validates the Gauss identity before returning."""
    area, centroid = element_geometry(mesh.node_coords, mesh.elements)

    interior_length, interior_normal = _face_geometry(
        mesh.node_coords, mesh.interior_edges[:, 2], mesh.interior_edges[:, 3]
    )
    boundary_length, boundary_normal = _face_geometry(
        mesh.node_coords, mesh.boundary_edges[:, 1], mesh.boundary_edges[:, 2]
    )
    interior_midpoint = 0.5 * (
        mesh.node_coords[mesh.interior_edges[:, 2]] + mesh.node_coords[mesh.interior_edges[:, 3]]
    )
    midpoint = 0.5 * (
        mesh.node_coords[mesh.boundary_edges[:, 1]] + mesh.node_coords[mesh.boundary_edges[:, 2]]
    )

    perimeter = np.zeros(mesh.n_elements)
    np.add.at(perimeter, mesh.interior_edges[:, 0], interior_length)
    np.add.at(perimeter, mesh.interior_edges[:, 1], interior_length)
    np.add.at(perimeter, mesh.boundary_edges[:, 0], boundary_length)

    metrics = Metrics(
        mesh=mesh,
        area=area,
        centroid=centroid,
        perimeter=perimeter,
        hydraulic_diameter=2.0 * area / perimeter,
        interior_left=np.ascontiguousarray(mesh.interior_edges[:, 0]),
        interior_right=np.ascontiguousarray(mesh.interior_edges[:, 1]),
        interior_length=interior_length,
        interior_normal=interior_normal,
        interior_midpoint=interior_midpoint,
        boundary_element=np.ascontiguousarray(mesh.boundary_edges[:, 0]),
        boundary_length=boundary_length,
        boundary_normal=boundary_normal,
        boundary_midpoint=midpoint,
        boundary_tag=np.ascontiguousarray(mesh.boundary_tags),
    )

    scale = float(np.sqrt(area.max()))
    error = metrics.closure_error()
    if error > 1e-10 * scale:
        raise ValueError(
            f"the discrete Gauss identity fails by {error:.3e} (scale {scale:.3e}); "
            f"the faces and the elements disagree about this mesh"
        )
    return metrics


__all__ = ["Metrics", "build_metrics", "element_geometry"]
