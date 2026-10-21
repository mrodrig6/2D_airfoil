# Importing meshes

Read triangular SU2 or Gmsh 2.2 ASCII meshes. Boundary markers must map to wall and far-field conditions.

## Using a mesh from elsewhere

```bash
# Create a sample SU2 file, then read it through the external-mesh path.
./fv2d.sh mesh @grid su2=external.su2
./fv2d.sh mesh mesh=external.su2
```

```python
from src import Discretization, build_case, read_mesh, write_su2

resolution = Discretization(
    n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02
)
case = build_case("0012", resolution)
write_su2(case.mesh, "external.su2")
mesh = read_mesh("external.su2")
print(mesh.summary())
```

SU2 (`.su2`) and **Gmsh 2.2 ASCII** (`.msh`), triangles only. For Gmsh's own
4.x format, re-export:

If Gmsh and your own `airfoil.geo` are available, the export command is
`gmsh -2 -format msh2 airfoil.geo -o airfoil.msh`. Neither Gmsh nor that external geometry file is required for the SU2 round trip above. This optional external-tool recipe was not run here.

The file's boundary markers have to be mapped onto this code's two boundary
conditions. The usual spellings are recognised case-insensitively —
`airfoil`, `wall`, `body`, `surface`, `profile`, `solid` become the wall;
`farfield`, `far_field`, `outer`, `freestream`, `external` become the far field
— and anything else is an **error naming every marker in the file**:

```
SU2 m.su2: cannot map boundary marker(s) ['mystery'] onto a boundary
condition.  Markers in this file: ['airfoil', 'mystery'].  Rename them, or
pass wall_names=/farfield_names= naming them explicitly.
```

Silently dropping an unrecognised boundary is the worst available failure here,
because the solver still runs and the answer is merely wrong. Pass
`wall_names=`/`farfield_names=` to name yours. Triangle winding is fixed on
read, so a clockwise mesh is fine.
