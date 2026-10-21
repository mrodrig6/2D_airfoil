# Airfoil geometry

Generate a NACA section or read coordinates from a file. The geometry API normalizes winding and chord before meshing.

## Describing a section

```python
from src import load_airfoil, naca4, read_dat, resample, write_dat
```

| | |
|---|---|
| `naca("2412", 192)` | generate a NACA section, four- or five-digit, at 192 points |
| `naca4("2412", 192)` / `naca5("23012", 192)` | either family explicitly |
| `read_dat("s1223.dat")` | read a Selig or Lednicer file, told apart automatically |
| `load_airfoil(spec, n)` | either of the above, chosen by what `spec` looks like |
| `resample(section, 256)` | redistribute an existing section |
| `write_dat(section, path)` | write Selig, which is what XFOIL reads back |

An `Airfoil` reports what it is:

Continue in the same Python session:

```python
>>> section = naca4("2412", 160)
>>> section.max_thickness, section.max_camber, section.area
(0.12004984931063187, 0.01999996307060152, 0.08175629208417193)
>>> section.is_sharp
True
>>> upper, lower = section.sides()      # both run leading edge -> trailing edge
```

### The five-digit family

`LPSTT`: design lift $0.15L$, camber crest at $P/20$, plain (`S=0`) or reflexed
(`S=1`), $TT$% thick. `naca5("23012")` is the canonical one.

The tabulated camber positions are the only ones that exist — `P` of 1–5 plain,
2–5 reflexed — and anything else is **refused rather than interpolated**, since
an interpolated five-digit section is not a NACA section:

```
NACA 26012: camber position 6 is not tabulated for a plain five-digit camber
line.  Available: [1, 2, 3, 4, 5], putting the crest at
[0.05, 0.1, 0.15, 0.2, 0.25] chord.
```

The six-series (`63-012`) and modified four-digit (`0012-64`) sections have no
closed form here. Read those from a coordinate file — every airfoil database
publishes them.

`closed_trailing_edge=False` selects the original open four-digit polynomial.
The section then carries a **second** point at the trailing edge — the other
corner of the base — so it comes back with `n_points + 1`.
[airfoil geometry](../theory/airfoil-geometry.md#the-naca-four-digit-family) says why closed is the
default.

### Design variables

| option | meaning | default |
|---|---|---|
| `airfoil` | a four- or five-digit NACA code, or a path to a `.dat` file | `0012` |
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
