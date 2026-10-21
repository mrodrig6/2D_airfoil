"""Reading a mesh someone else generated.

The round trips matter because they are how an externally meshed geometry gets
in; the rejections matter more, because a boundary that silently receives no
condition still runs and merely gives the wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import build_metrics, read_mesh, write_su2
from src.geometry.mesh import BoundaryTag
from src.geometry.meshio import read_gmsh, read_su2


def write_gmsh22(mesh, path, *, clockwise=False):
    """Write a Gmsh 2.2 ASCII file, optionally with the triangles wound backwards."""
    out = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        "2",
        '1 1 "airfoil"',
        '1 2 "farfield"',
        "$EndPhysicalNames",
        "$Nodes",
        str(mesh.n_nodes),
    ]
    out += [f"{i + 1} {x:.16e} {y:.16e} 0.0" for i, (x, y) in enumerate(mesh.node_coords)]
    out += ["$EndNodes"]
    rows, n = [], 0
    for tag, group in ((BoundaryTag.WALL, 1), (BoundaryTag.FARFIELD, 2)):
        for _, a, b in mesh.boundary_edges[mesh.tag_slice(tag)]:
            n += 1
            rows.append(f"{n} 1 2 {group} {group} {a + 1} {b + 1}")
    for tri in mesh.elements:
        n += 1
        order = tri[::-1] if clockwise else tri
        rows.append(f"{n} 2 2 3 3 {order[0] + 1} {order[1] + 1} {order[2] + 1}")
    out += ["$Elements", str(n)] + rows + ["$EndElements"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("clockwise", [False, True])
def test_a_gmsh_mesh_reads_back_identical_whichever_way_it_is_wound(
    tiny_mesh, tiny_metrics, tmp_path, clockwise
):
    mesh = read_gmsh(write_gmsh22(tiny_mesh, tmp_path / "m.msh", clockwise=clockwise))
    assert mesh.n_elements == tiny_mesh.n_elements
    assert np.allclose(mesh.node_coords, tiny_mesh.node_coords)
    assert np.all(mesh.signed_areas() > 0.0)
    metrics = build_metrics(mesh)
    assert metrics.reference_area() == pytest.approx(tiny_metrics.reference_area())
    assert metrics.closure_error() < 1e-12


def test_an_su2_mesh_round_trips(tiny_mesh, tiny_metrics, tmp_path):
    mesh = read_mesh(write_su2(tiny_mesh, tmp_path / "m.su2"))
    assert np.allclose(mesh.node_coords, tiny_mesh.node_coords)
    for tag in BoundaryTag:
        assert mesh.tag_slice(tag) == tiny_mesh.tag_slice(tag)
    assert build_metrics(mesh).reference_area() == pytest.approx(tiny_metrics.reference_area())


def test_an_unmappable_marker_is_refused_and_names_what_it_found(tiny_mesh, tmp_path):
    path = write_su2(tiny_mesh, tmp_path / "m.su2")
    path.write_text(
        path.read_text(encoding="utf-8").replace("MARKER_TAG= farfield", "MARKER_TAG= mystery"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mystery"):
        read_su2(path)


def test_a_mesh_missing_a_boundary_is_refused(tiny_mesh, tmp_path):
    path = write_su2(tiny_mesh, tmp_path / "m.su2")
    text = path.read_text(encoding="utf-8").replace("MARKER_TAG= farfield", "MARKER_TAG= airfoil")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="FARFIELD"):
        read_su2(path)


def test_gmsh_4_says_how_to_re_export(tmp_path):
    path = tmp_path / "v4.msh"
    path.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="format msh2"):
        read_gmsh(path)


def test_a_three_dimensional_or_non_triangular_mesh_is_refused(tmp_path):
    (tmp_path / "a.su2").write_text("NDIME= 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two-dimensional"):
        read_su2(tmp_path / "a.su2")
    (tmp_path / "b.su2").write_text("NDIME= 2\nNELEM= 1\n9 0 1 2 3 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a triangle"):
        read_su2(tmp_path / "b.su2")


def test_an_unknown_extension_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown mesh extension"):
        read_mesh(tmp_path / "m.vtk")
