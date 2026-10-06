# Using the code

Everything available today, from both the launcher and Python. The solver
arrives in the next increment ([`roadmap.md`](roadmap.md)); what is here is the
geometry and the mesh, which is what it will be built on.

## Contents

- [The launcher](#the-launcher)
- [Decks: a case in a file](#decks-a-case-in-a-file)
- [From Python](#from-python)
- [Describing a section](#describing-a-section)
- [Using a section from XFOIL](#using-a-section-from-xfoil)
- [Meshing it](#meshing-it)
- [Using a mesh from elsewhere](#using-a-mesh-from-elsewhere)
- [Units: the code is non-dimensional](#units-the-code-is-non-dimensional)
- [Checking it still works](#checking-it-still-works)

---

## The launcher

`./fv2d.sh` needs nothing set up first. It finds a Python, puts the repository
on the import path if the package is not installed, and runs from any working
directory.

```bash
./fv2d.sh help                 # everything below, in one screen
./fv2d.sh check                # the Python, and which extras are installed
./fv2d.sh list                 # the decks '@' expands
```

| command | what it does |
|---|---|
| `check` | report the Python and which optional packages are present |
| `geometry` | inspect a section — thickness, camber, area; optionally a figure or a `.dat` |
| `mesh` | generate (or read) a mesh and report its quality; optionally a figure or a `.su2` |
| `flow` | print the non-dimensional free-stream state for an operating point |
| `test` | run the test suite |
| `verify` | tests, lint, format and spelling |
| `install` | `pip install -e .[all]`, which is optional |
| `docs` | print where the documentation is |

Arguments are `name=value`, using the **same names as the Python API**, so there
is one vocabulary rather than two:

```bash
./fv2d.sh geometry airfoil=2412 n_surface=160 figure=section.png normals=yes
./fv2d.sh mesh airfoil=0012 n_surface=192 n_radial=64 figure=mesh.png
./fv2d.sh flow mach=0.8 alpha=1.25
```

A few short aliases exist: `naca=airfoil`, `ns=n_surface`, `nr=n_radial`,
`R=far_field_radius`, `h1=first_cell_height`, `aoa=alpha`. Ordinary `--flags`
pass through, so anything `python -m src --help` documents still works.

**`M` is deliberately not an alias.** In a compressible-flow code `M` is the
Mach number and nothing else may claim that letter.

## Decks: a case in a file

A `@name` argument is replaced by the lines of `cases/name.fv`, so a case you
run often lives in a file instead of in your shell history. Later arguments win,
which makes a deck a starting point rather than a fixed case:

```bash
./fv2d.sh mesh @transonic                  # exactly the lines in that deck
./fv2d.sh mesh @transonic n_radial=96      # the deck, with one value changed
./fv2d.sh mesh @coarse naca=2412
```

A deck holds exactly what you would have typed — `#` starts a comment, blank
lines are ignored, and a deck may name another deck to build on it
(`cases/fine.fv` is `@transonic` with two values raised). There is no schema to
keep in step with the code, because it is argument expansion and nothing more.

The consequence is that a deck works with any command that accepts the keys in
it. Handing `mesh` a deck carrying keys it does not take produces
`unrecognized arguments` naming them, which is the intended behaviour — it is
what a typo needs too.

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

## Describing a section

```python
from src import load_airfoil, naca4, read_dat, resample, write_dat
```

| | |
|---|---|
| `naca4("2412", 192)` | generate a four-digit section directly at 192 points |
| `read_dat("s1223.dat")` | read a Selig or Lednicer file, told apart automatically |
| `load_airfoil(spec, n)` | either of the above, chosen by what `spec` looks like |
| `resample(section, 256)` | redistribute an existing section |
| `write_dat(section, path)` | write Selig, which is what XFOIL reads back |

An `Airfoil` reports what it is:

```python
>>> section = naca4("2412", 160)
>>> section.max_thickness, section.max_camber, section.area
(0.12005, 0.02000, 0.081756)
>>> section.is_sharp
True
>>> upper, lower = section.sides()      # both run leading edge -> trailing edge
```

`closed_trailing_edge=False` selects the original open four-digit polynomial.
The section then carries a **second** point at the trailing edge — the other
corner of the base — so it comes back with `n_points + 1`.
[`theory.md`](theory.md#the-naca-four-digit-family) says why closed is the
default.

### Design variables

| option | meaning | default |
|---|---|---|
| `airfoil` | a four-digit code, or a path to a `.dat` file | `0012` |
| `n_surface` | points around the section; must be even | 192 |
| `clustering` | `cosine` (clusters at both edges) or `uniform` | `cosine` |

## Using a section from XFOIL

XFOIL's `SAVE` writes Selig format, which this reads directly:

```
XFOIL   c> LOAD my_design.dat
XFOIL   c> PANE
XFOIL   c> SAVE my_design_paneled.dat
```

```bash
./fv2d.sh geometry airfoil=my_design_paneled.dat figure=check.png normals=yes
```

Three things worth knowing:

* **Winding does not matter.** Either format, wound either way, is normalised to
  this code's convention on read. Draw the normals (`normals=yes`) if you want
  to see it: they point away from the section.
* **It is rescaled** to unit chord with the leading edge at the origin. A
  section given in millimetres comes out the same shape.
* **An open trailing edge is kept open.** If your file has a base, the mesh
  wraps it. Close it in XFOIL first if you would rather it did not.

## Meshing it

```python
Discretization(
    n_surface=192,  # points around the section
    n_radial=64,  # cells from the wall to the far field
    far_field_radius=20.0,  # chords
    far_field_centre=(0.5, 0.0),
    first_cell_height=2.0e-3,  # chords
    growth_limit=1.25,  # largest cell-to-cell radial growth
    orthogonality=2.0,  # wall-normal to radial blend exponent
    clustering="cosine",
)
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

## Using a mesh from elsewhere

```bash
./fv2d.sh mesh mesh=external.su2
```

```python
from src import read_mesh

mesh = read_mesh("external.msh")
```

SU2 (`.su2`) and **Gmsh 2.2 ASCII** (`.msh`), triangles only. For Gmsh's own
4.x format, re-export:

```bash
gmsh -2 -format msh2 airfoil.geo -o airfoil.msh
```

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

## Units: the code is non-dimensional

There is no SI anywhere. $\rho_\infty = |\vec v_\infty| = c = 1$ and
$R = \gamma - 1$, which fixes everything else:

```bash
$ ./fv2d.sh flow mach=0.8 alpha=1.25
M = 0.8, alpha = 1.25 deg, gamma = 1.4
  density          1.0000000000
  velocity         (0.9997620271, 0.0218148850)
  pressure         1.1160714286   = 1/(gamma M^2)
  sound speed      1.2500000000   = 1/M
  temperature      2.7901785714   with R = gamma - 1
  dynamic pressure 0.5000000000   always 1/2
  total pressure   1.7012723321
```

The two that surprise people:

* **Pressure is not 1.** Fixing density and speed means the Mach number is
  carried by the pressure, $p_\infty = 1/(\gamma M_\infty^2)$.
* **Dynamic pressure is exactly 1/2**, at every operating point. A force
  coefficient is therefore twice the force per unit span, and $c_\ell$ at
  $M = 0.5$ compares directly with $c_\ell$ at $M = 0.8$.

Angle of attack enters in **one place only** — the direction of
$\vec v_\infty$. The mesh does not rotate with it, so an $\alpha$ sweep reuses
one mesh and one set of metrics.

→ [`theory.md`](theory.md#non-dimensionalisation) for the derivation.

## Checking it still works

There is no CI. You check, on your own machine:

```bash
./fv2d.sh verify              # tests, lint, format, spelling
./fv2d.sh verify --fast       # same, minus the full-size meshes
./fv2d.sh test tests/test_mesh.py
./fv2d.sh test -k normal      # one test by name
```

`pytest --fast` and `pytest --slow` select disjoint sets and work directly too.
Each `verify` step is skipped with a note rather than failing if its tool is not
installed — a missing linter is not a broken mesher — and the exit code is
non-zero if anything that did run failed.
