# Library usage

Public entry points, prepared cases, result objects and short runnable examples.

## From Python

No install is needed; run from the repository root, or put it on `PYTHONPATH`.

```python
from src import Discretization, build_mesh, build_metrics, naca4

section = naca4("0012", 192)
print(section.summary())
# NACA 0012: 192 points, chord 1.000000, max thickness 0.12001, ... sharp

mesh, grid = build_mesh(section, Discretization(n_surface=192, n_radial=64))
metrics = build_metrics(mesh)
print(mesh.summary())
# 192x64 O-grid: 24576 triangles, 12480 nodes, 36672 interior edges, ...
print(metrics.summary())
# 24576 cells, area 1256.331089, hydraulic diameter 1.259e-04 to 5.254e-01, ...
```

`build_mesh` returns both the triangulated `MeshTopology` the solver will take
and the structured `OGrid` it came from. The O-grid is worth keeping: it is what
knows the $(i, j)$ layout, and `grid.first_cell_heights()` reports the wall
resolution actually delivered.
