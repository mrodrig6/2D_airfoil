"""The O-grid, its triangulation, and the metrics the solver reads off it.

Three properties carry the whole finite-volume scheme, and each has a test here:
the mesh is a valid conforming triangulation, every face normal points the way
the residual assumes, and the discrete Gauss identity closes to round-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import Discretization, build_mesh, build_metrics, naca4
from src.mesh import BoundaryTag, enforce_ccw, radial_distribution


# -- the radial distribution ------------------------------------------------
def test_radial_distribution_spans_exactly_and_starts_where_asked():
    f = radial_distribution(64, 2e-3, 20.0)
    assert f.shape == (65,)
    assert f[0] == 0.0 and f[-1] == 1.0
    assert f[1] * 20.0 == pytest.approx(2e-3, rel=1e-9)
    ratio = np.diff(f)[1:] / np.diff(f)[:-1]
    assert np.allclose(ratio, ratio[0])  # geometric, by construction
    assert 1.0 < ratio[0] < 1.25


def test_a_radial_request_that_cannot_reach_the_far_field_says_so():
    with pytest.raises(ValueError, match="cannot reach the far field"):
        radial_distribution(8, 1e-4, 50.0, growth_limit=1.2)


# -- the grid ---------------------------------------------------------------
def test_the_grid_pins_both_boundaries_exactly(tiny_grid, section, tiny):
    assert np.allclose(tiny_grid.wall, section.points)
    centre = np.asarray(tiny.far_field_centre)
    r = np.hypot(*(tiny_grid.far_field - centre).T)
    assert np.allclose(r, tiny.far_field_radius, rtol=0, atol=1e-12)


def test_the_first_cell_is_wall_orthogonal(tiny_grid, section):
    """The blend starts at the surface normal, which is what the wall flux wants."""
    from src.geometry import surface_normals

    step = tiny_grid.coords[:, 1, :] - tiny_grid.coords[:, 0, :]
    step /= np.hypot(*step.T)[:, None]
    alignment = np.einsum("ij,ij->i", step, surface_normals(section.points))
    assert alignment.min() > 0.9999


def test_the_requested_first_cell_height_is_delivered(tiny_grid, tiny):
    heights = tiny_grid.first_cell_heights()
    assert np.allclose(heights, tiny.first_cell_height, rtol=0.05)


# -- the triangulation ------------------------------------------------------
def test_the_mesh_is_a_valid_conforming_triangulation(tiny_mesh, tiny):
    tiny_mesh.validate()  # areas positive, every node used, boundaries closed
    assert tiny_mesh.n_elements == tiny.n_elements
    assert tiny_mesh.n_nodes == tiny.n_nodes
    assert np.all(tiny_mesh.signed_areas() > 0.0)
    # Euler's formula for a triangulated annulus: 3E = 2*interior + boundary
    assert 3 * tiny_mesh.n_elements == 2 * tiny_mesh.n_interior_edges + tiny_mesh.n_boundary_edges


def test_each_boundary_has_one_face_per_surface_station(tiny_mesh, tiny):
    for tag in BoundaryTag:
        assert tiny_mesh.n_faces_on(tag) == tiny.n_surface
    assert len(tiny_mesh.boundary_nodes(BoundaryTag.WALL)) == tiny.n_surface


def test_enforce_ccw_rewinds_without_changing_the_mesh(tiny_mesh):
    flipped = tiny_mesh.elements.copy()
    flipped[::3] = flipped[::3][:, [0, 2, 1]]
    repaired = enforce_ccw(tiny_mesh.node_coords, flipped)
    assert set(map(frozenset, repaired)) == set(map(frozenset, tiny_mesh.elements))
    p = tiny_mesh.node_coords[repaired]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    assert np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0.0)


def test_the_triangulation_diagonal_alternates(tiny_mesh):
    """A single fixed diagonal biases any feature that lines up with it."""
    halves = np.split(tiny_mesh.elements, 2)
    assert not np.array_equal(halves[0][0], halves[0][1])


# -- the metrics ------------------------------------------------------------
def test_the_discrete_gauss_identity_closes(tiny_metrics):
    assert tiny_metrics.closure_error() < 1e-12


def test_the_total_area_is_the_polygon_it_actually_meshes(tiny_metrics, section, tiny):
    """Not pi R^2: the far field is an n-gon inscribed in the circle."""
    n, r = tiny.n_surface, tiny.far_field_radius
    polygon = 0.5 * n * r * r * np.sin(2 * np.pi / n)
    assert tiny_metrics.reference_area() == pytest.approx(polygon - section.area, rel=1e-12)


def test_an_interior_normal_points_from_its_left_element_to_its_right(tiny_metrics):
    centroid = tiny_metrics.centroid
    offset = centroid[tiny_metrics.interior_right] - centroid[tiny_metrics.interior_left]
    assert np.all(np.einsum("ij,ij->i", tiny_metrics.interior_normal, offset) > 0.0)


def test_a_boundary_normal_points_out_of_the_flow_domain(tiny_metrics, tiny):
    """At the wall that means into the airfoil; at the far field, away from it."""
    outward = tiny_metrics.boundary_midpoint - tiny_metrics.centroid[tiny_metrics.boundary_element]
    assert np.all(np.einsum("ij,ij->i", tiny_metrics.boundary_normal, outward) > 0.0)

    centre = np.asarray(tiny.far_field_centre)
    for tag, sign in ((BoundaryTag.WALL, -1.0), (BoundaryTag.FARFIELD, +1.0)):
        s = tiny_metrics.tag_slice(tag)
        radial = tiny_metrics.boundary_midpoint[s] - centre
        assert np.all(sign * np.einsum("ij,ij->i", tiny_metrics.boundary_normal[s], radial) > 0.0)


def test_the_wall_faces_add_up_to_the_section_perimeter(tiny_metrics, section):
    s = tiny_metrics.tag_slice(BoundaryTag.WALL)
    loop = np.vstack([section.points, section.points[:1]])
    assert tiny_metrics.boundary_length[s].sum() == pytest.approx(
        np.hypot(*np.diff(loop, axis=0).T).sum()
    )


def test_hydraulic_diameter_matches_the_legacy_definition(tiny_metrics):
    """``legacy/meshconnect.m`` used 2A/P, and the local time step still does."""
    assert np.allclose(
        tiny_metrics.hydraulic_diameter, 2.0 * tiny_metrics.area / tiny_metrics.perimeter
    )


# -- the production mesh ----------------------------------------------------
@pytest.mark.slow
def test_the_default_mesh_builds_and_is_valid():
    mesh, grid = build_mesh(naca4("0012", Discretization.n_surface), Discretization())
    metrics = build_metrics(mesh)
    assert mesh.n_elements == Discretization().n_elements
    assert metrics.closure_error() < 1e-12
    assert metrics.area.min() > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("code,n_surface", [("0012", 128), ("2412", 160), ("4415", 96)])
def test_cambered_and_thick_sections_mesh_without_folding(code, n_surface):
    disc = Discretization(
        n_surface=n_surface, n_radial=40, far_field_radius=15.0, first_cell_height=5e-3
    )
    mesh, _ = build_mesh(naca4(code, n_surface), disc)
    assert np.all(mesh.signed_areas() > 0.0)
    assert build_metrics(mesh).closure_error() < 1e-12
