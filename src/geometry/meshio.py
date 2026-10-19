r"""Read a triangular mesh someone else generated.

The solver core takes a :class:`~src.geometry.mesh.MeshTopology` and does not care where
it came from, so an externally generated mesh is a first-class input alongside
the built-in O-grid.  That matters for two cases this code expects:

* an airfoil meshed around a geometry from XFOIL with a real mesher, when the
  O-grid's structured resolution is not what a study needs;
* a mesh from another course or another solver, used to check that a result does
  not depend on this code's own grid generator.

Two formats, and why only two
-----------------------------
**SU2** (``.su2``) and **Gmsh 2.2 ASCII** (``.msh``).  Both are plain text, both
are what the common meshers export, and between them they cover anything a
student is likely to have.  Gmsh's own 4.x format replaces the flat element list
with nested entity blocks; rather than carry a second parser for it, this reader
says what to run:

.. code-block:: text

    gmsh -2 -format msh2 airfoil.geo -o airfoil.msh

Boundary markers
----------------
A mesh file names its boundaries; this code has exactly two
(:class:`~src.geometry.mesh.BoundaryTag`), so the names have to be mapped onto them.  The
defaults in :data:`WALL_NAMES` and :data:`FARFIELD_NAMES` cover the usual
spellings, matched case-insensitively.  An unrecognised marker is an **error
naming every marker in the file**, not a silently dropped boundary -- a boundary
that quietly receives no condition is the single worst way for this to go wrong,
because the solver still runs and the answer is merely incorrect.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .mesh import BoundaryTag, MeshTopology, build_edges, enforce_ccw

#: Marker names mapped to :attr:`~src.geometry.mesh.BoundaryTag.WALL`.
WALL_NAMES = ("airfoil", "wall", "body", "surface", "profile", "solid")

#: Marker names mapped to :attr:`~src.geometry.mesh.BoundaryTag.FARFIELD`.
FARFIELD_NAMES = ("farfield", "far_field", "far-field", "outer", "freestream", "external")


def _resolve_tag(
    name: str, wall_names: Iterable[str], farfield_names: Iterable[str]
) -> BoundaryTag | None:
    key = name.strip().strip('"').lower()
    if key in {w.lower() for w in wall_names}:
        return BoundaryTag.WALL
    if key in {f.lower() for f in farfield_names}:
        return BoundaryTag.FARFIELD
    return None


def _assemble(
    coords: np.ndarray,
    triangles: np.ndarray,
    lines: dict[str, np.ndarray],
    source: str,
    wall_names: Iterable[str],
    farfield_names: Iterable[str],
) -> MeshTopology:
    """Turn parsed pieces into a validated :class:`MeshTopology`."""
    if triangles.size == 0:
        raise ValueError(f"{source}: no triangles found")
    if not lines:
        raise ValueError(f"{source}: no boundary markers found, so no boundary conditions")

    resolved = {name: _resolve_tag(name, wall_names, farfield_names) for name in lines}
    unknown = sorted(name for name, tag in resolved.items() if tag is None)
    if unknown:
        raise ValueError(
            f"{source}: cannot map boundary marker(s) {unknown} onto a boundary "
            f"condition.  Markers in this file: {sorted(lines)}.  Rename them, or "
            f"pass wall_names=/farfield_names= naming them explicitly."
        )
    missing = {t.name for t in BoundaryTag} - {t.name for t in resolved.values()}
    if missing:
        raise ValueError(f"{source}: no marker maps to {sorted(missing)}")

    node_tags = np.zeros(coords.shape[0], dtype=np.int64)
    for name, segments in lines.items():
        node_tags[segments.ravel()] = int(resolved[name])

    # drop nodes no triangle uses -- meshers routinely leave geometry points behind
    used = np.unique(triangles)
    if used.size != coords.shape[0]:
        renumber = np.full(coords.shape[0], -1, dtype=np.int64)
        renumber[used] = np.arange(used.size)
        coords, node_tags = coords[used], node_tags[used]
        triangles = renumber[triangles]
        if np.any(triangles < 0):
            raise ValueError(f"{source}: a triangle refers to a node that does not exist")

    triangles = enforce_ccw(coords, triangles)
    interior, boundary, tags, slices = build_edges(coords, triangles, node_tags)
    mesh = MeshTopology(
        node_coords=np.ascontiguousarray(coords),
        elements=triangles,
        interior_edges=interior,
        boundary_edges=boundary,
        boundary_tags=tags,
        shape=None,
        source=source,
        _tag_slices=slices,
    )
    mesh.validate()
    return mesh


def read_su2(
    path: str | Path,
    *,
    wall_names: Iterable[str] = WALL_NAMES,
    farfield_names: Iterable[str] = FARFIELD_NAMES,
) -> MeshTopology:
    """Read a 2D triangular SU2 mesh.

    Element type codes are VTK's: 5 is a triangle, 3 a line.  Anything else in a
    2D mesh (a quadrilateral, type 9) is rejected rather than split, because
    splitting it would silently change the mesh the user asked for.
    """
    path = Path(path)
    tokens = [line.split("%")[0].strip() for line in path.read_text().splitlines() if line.strip()]
    coords: list[list[float]] = []
    triangles: list[list[int]] = []
    lines: dict[str, list[list[int]]] = {}
    marker: str | None = None

    index = 0
    while index < len(tokens):
        line = tokens[index]
        index += 1
        if not line:
            continue
        head, _, value = (part.strip() for part in line.partition("="))
        key = head.upper()
        if key == "NDIME":
            if int(value) != 2:
                raise ValueError(f"{path}: NDIME={value}; this solver is two-dimensional")
        elif key == "NELEM":
            for _ in range(int(value)):
                field = tokens[index].split()
                index += 1
                code = int(field[0])
                if code != 5:
                    raise ValueError(
                        f"{path}: element type {code} is not a triangle; this solver "
                        f"needs a purely triangular mesh"
                    )
                triangles.append([int(v) for v in field[1:4]])
        elif key == "NPOIN":
            for _ in range(int(value.split()[0])):
                field = tokens[index].split()
                index += 1
                coords.append([float(field[0]), float(field[1])])
        elif key == "MARKER_TAG":
            marker = value
            lines.setdefault(marker, [])
        elif key == "MARKER_ELEMS":
            if marker is None:
                raise ValueError(f"{path}: MARKER_ELEMS before any MARKER_TAG")
            for _ in range(int(value)):
                field = tokens[index].split()
                index += 1
                lines[marker].append([int(field[1]), int(field[2])])

    return _assemble(
        np.array(coords, dtype=float),
        np.array(triangles, dtype=np.int64),
        {k: np.array(v, dtype=np.int64) for k, v in lines.items() if v},
        f"SU2 {path.name}",
        wall_names,
        farfield_names,
    )


def read_gmsh(
    path: str | Path,
    *,
    wall_names: Iterable[str] = WALL_NAMES,
    farfield_names: Iterable[str] = FARFIELD_NAMES,
) -> MeshTopology:
    """Read a 2D triangular mesh in Gmsh 2.2 ASCII format.

    Element lines are ``id type n_tags tag... node...``, where type 2 is a
    three-node triangle and type 1 a two-node line.  The first tag is the
    physical group, which ``$PhysicalNames`` gives a name.
    """
    path = Path(path)
    text = path.read_text().splitlines()

    version = ""
    for i, line in enumerate(text):
        if line.strip() == "$MeshFormat":
            version = text[i + 1].split()[0]
            break
    if not version:
        raise ValueError(f"{path}: no $MeshFormat section; is this a Gmsh file?")
    if not version.startswith("2"):
        raise ValueError(
            f"{path}: Gmsh format {version} is not supported, only 2.2 ASCII.  "
            f"Re-export with:  gmsh -2 -format msh2 <input> -o {path.name}"
        )

    def section(name: str) -> list[str]:
        try:
            start = text.index(f"${name}") + 1
            stop = text.index(f"$End{name}")
        except ValueError:
            return []
        return [line for line in text[start:stop] if line.strip()]

    names: dict[int, str] = {}
    for line in section("PhysicalNames")[1:]:
        field = line.split(None, 2)
        if len(field) == 3:
            names[int(field[1])] = field[2].strip().strip('"')

    node_lines = section("Nodes")[1:]
    node_id: dict[int, int] = {}
    coords = np.empty((len(node_lines), 2))
    for row, line in enumerate(node_lines):
        field = line.split()
        node_id[int(field[0])] = row
        coords[row] = (float(field[1]), float(field[2]))

    triangles: list[list[int]] = []
    lines: dict[str, list[list[int]]] = {}
    for line in section("Elements")[1:]:
        field = [int(v) for v in line.split()]
        kind, n_tags = field[1], field[2]
        tags, nodes = field[3 : 3 + n_tags], field[3 + n_tags :]
        if kind == 2:
            triangles.append([node_id[n] for n in nodes[:3]])
        elif kind == 1:
            group = tags[0] if tags else 0
            label = names.get(group, f"physical {group}")
            lines.setdefault(label, []).append([node_id[n] for n in nodes[:2]])
        elif kind == 15:
            continue  # a single node, which meshers emit for geometry points
        else:
            raise ValueError(
                f"{path}: element type {kind} is neither a triangle nor a boundary line; "
                f"this solver needs a purely triangular mesh"
            )

    return _assemble(
        coords,
        np.array(triangles, dtype=np.int64),
        {k: np.array(v, dtype=np.int64) for k, v in lines.items()},
        f"Gmsh {path.name}",
        wall_names,
        farfield_names,
    )


def read_mesh(
    path: str | Path,
    *,
    wall_names: Iterable[str] = WALL_NAMES,
    farfield_names: Iterable[str] = FARFIELD_NAMES,
) -> MeshTopology:
    """Read a mesh, choosing the parser by file extension."""
    path = Path(path)
    reader = {".su2": read_su2, ".msh": read_gmsh}.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"{path}: unknown mesh extension {path.suffix!r}; expected .su2 or .msh")
    return reader(path, wall_names=wall_names, farfield_names=farfield_names)


def write_su2(mesh: MeshTopology, path: str | Path) -> Path:
    """Write the mesh as SU2, which is also how the reader gets tested."""
    path = Path(path)
    out = ["NDIME= 2", f"NELEM= {mesh.n_elements}"]
    out += [f"5 {a} {b} {c} {i}" for i, (a, b, c) in enumerate(mesh.elements)]
    out.append(f"NPOIN= {mesh.n_nodes}")
    out += [f"{x:.16e} {y:.16e} {i}" for i, (x, y) in enumerate(mesh.node_coords)]
    out.append(f"NMARK= {len(BoundaryTag)}")
    for tag, name in ((BoundaryTag.WALL, "airfoil"), (BoundaryTag.FARFIELD, "farfield")):
        edges = mesh.boundary_edges[mesh.tag_slice(tag)]
        out.append(f"MARKER_TAG= {name}")
        out.append(f"MARKER_ELEMS= {len(edges)}")
        out += [f"3 {a} {b}" for _, a, b in edges]
    path.write_text("\n".join(out) + "\n")
    return path


__all__ = [
    "FARFIELD_NAMES",
    "WALL_NAMES",
    "read_gmsh",
    "read_mesh",
    "read_su2",
    "write_su2",
]
