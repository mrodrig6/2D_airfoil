# Generating meshes

Generate an O- or C-grid and inspect its quality before solving. The builder refuses radial distributions or cells that fail its geometric checks.

## Meshing it

```python
from src import Discretization, build_case

resolution = Discretization(
    n_surface=192,  # points around the section
    n_radial=64,  # cells from the wall to the far field
    far_field_radius=20.0,  # chords
    far_field_centre=(0.5, 0.0),
    first_cell_height=2.0e-3,  # chords
    growth_limit=1.25,  # largest cell-to-cell radial growth
    orthogonality=2.0,  # wall-normal to radial blend exponent
    clustering="cosine",
    n_wake=32,  # cgrid only: cells along each side of the cut
    wake_length=20.0,  # cgrid only: the outflow plane, in chords
)
case = build_case("0012", resolution)
print(case.mesh.summary())
print(case.metrics.summary())
```

`n_surface` and `n_radial` set the cost: the mesh has `2 * n_surface * n_radial`
triangles, so the default is 24 576. Doubling both quadruples it.

**If the mesh builder refuses**, it is almost always the radial distribution,
and the message names the four numbers that would fix it:

```
cannot reach the far field: 24 cells starting at 0.005 and growing by at
most 1.25 span only 4.22 of 15.  Raise n_radial, raise first_cell_height,
raise growth_limit, or shrink far_field_radius.
```

This is deliberate. A grid that silently stopped short of the far field would
show up much later as a solver that will not converge, which is a far more
expensive way to find out.

**`orthogonality`** trades wall-orthogonality against far-field smoothness.
Raising it keeps cells square against the surface further out; lowering it
toward 0 marches radially from the first layer. 2 suits every section tried. If
a thick or strongly cambered section folds the grid, `validate()` says so and
names the cell — lower `orthogonality` or raise `n_surface`.

### Reading the quality report

```
192x64 O-grid: 24576 triangles, 12480 nodes, 36672 interior edges, 384 boundary
  24576 cells, area 1256.331089, hydraulic diameter 1.259e-04 to 5.254e-01,
  closure error 4.44e-16
  cell area 1.055e-06 to 2.642e+00 (ratio 2.504e+06)
```

**Closure error** is the one to look at. It is
$\max_e \lvert \sum_f \hat n_f \ell_f \rvert$, the discrete Gauss identity that
makes the scheme conservative and a uniform flow exact. It must be at round-off;
`build_metrics` refuses to return a mesh where it is not.

The **cell area ratio** is large by design — six orders of magnitude between the
sliver at the trailing edge and a far-field cell is normal for an O-grid at this
stretching, and it is what the local time step exists to handle.
