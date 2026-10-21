# MATLAB provenance

The MATLAB files record the source of the Python port. This page explains their geometry, dependencies and the differences introduced by the port.

## What is here

| file | what |
|---|---|
| `main.m` | the driver: three adaptive iterations, each marching to a residual threshold |
| `initialmesh.m`, `adaptmesh.m` | **MATLAB PDE Toolbox**: `initmesh` and `refinemesh` |
| `convertmesh.m`, `meshconnect.m`, `counter_clock_sort.m` | mesh conversion, edge connectivity, element winding |
| `initialstate.m` | the free stream: $M = 0.8$, $\alpha = 1.25^\circ$, non-dimensional |
| `residual_calc.m` | the finite-volume residual, the Roe flux and the inviscid wall condition |
| `roe_test.m` | the Roe flux on its own, for checking |
| `post_calc.m` | lift, drag, the pressure coefficient, and the mesh-refinement indicator |
| `post_process.m`, `plotmesh.m`, `plotsolution.m` | figures |
| `nacageom.mat` | the geometry: a MATLAB decomposed geometry matrix |

## What the geometry actually is

Worth writing down, because the `.mat` is opaque until you decode it:

* **NACA 0012**, chord 1, as 200 line segments — and specifically the
  **closed-trailing-edge** thickness polynomial, $a_4 = 0.1036$. The points
  match that to $4\times10^{-17}$ and the original open form only to
  $1.3\times10^{-3}$, which is why `src/geometry/section.py` defaults to closed.
* A far field of **four circular arcs**: radius 50, centred on $(0.5, 0)$.
* The segments are **not stored in loop order**, which is why the test compares
  against the thickness polynomial rather than against a point list.
* The surface runs trailing edge → lower → leading edge → upper, which is
  clockwise. The port stores counter-clockwise; see
  [`../docs/theory.md`](../theory/airfoil-geometry.md#the-storage-convention) for why.

With `initialstate.m`'s $M_\infty = 0.8$ and $\alpha = 1.25^\circ$, this is the
standard transonic NACA 0012 case, with a shock on the upper surface.

## What the port changed, and why

Most of the MATLAB transfers directly. These did not.

### The mesher had to be written from scratch

`initialmesh.m` and `adaptmesh.m` are the PDE Toolbox. There is no Python
equivalent inside a reasonable dependency budget, so the mesh is new code: an
algebraic O-grid, generated in `src/geometry/mesh.py`. This is the single largest
difference between the two codes, and
[`../docs/theory.md`](../theory/euler.md) gives the construction
and what it trades away.

Two topologies are available — an O-grid and a C-grid — and the solver sees
only the triangles either produces, so the choice is one option rather than a
fork in the code.

The flagged refinement of `adaptmesh.m` goes with it. The Mach-jump indicator from `post_calc.m` is now implemented in `src/adapt.py`. The initial O- or C-grid uses `n_surface` and `n_radial`; adaptive cycles refine its triangulation.

Worth noting what the legacy's mesh did and did not have, since it is the
obvious comparison. `initmesh` produces a roughly isotropic unstructured
triangulation, so it has no trailing-edge fan — but it has no wall clustering
either, and the legacy never resolved the surface to anything like the
$2\times10^{-3}$ chords used here. Its answer to resolution was *adaptation*:
three iterations refining 5% of cells on the Mach jump. The trade here is
different rather than strictly better — wall resolution and fixed array shapes,
which Numba and JAX need, in exchange for structure.

### The force integral was wrong

`post_calc.m` computed

```matlab
D = -sum(p_b.*sign(n(:,1)).*s_edge)*sin(alpha);
L =  sum(p_b.*sign(n(:,2)).*s_edge)*cos(alpha);
```

This is incorrect in two independent ways:

1. **`sign(n_x)` and `sign(n_y)` where the formula needs `n_x` and `n_y`.** Each
   panel then contributes its full edge length instead of its projected area, so
   the sum is not $\oint p\,\hat n\,ds$.
2. **Two separate integrals scaled by $\sin\alpha$ and $\cos\alpha$**, where the
   correct operation is a rotation of the force vector through $\alpha$. The two
   agree only when one component vanishes.

With the flow-domain outward normal, the port computes $\vec F = \oint p \hat n\,ds$ and rotates it.
[`../docs/theory.md`](../theory/euler.md) has the
formula. The legacy expression stays reproducible for comparison.

### The far-field condition

`residual_calc.m` imposes the full free-stream state at the far field and lets
the Roe solver sort it out. That reflects outgoing waves, which is why the
domain had to be 50 chords across. The port uses a characteristic condition
built from the one-dimensional Riemann invariants, which admits a much smaller
domain.

### The convergence criterion

`main.m` marches until `norm(R, Inf) > 1e-2` is false — an **absolute**
$L_\infty$ residual, never normalised. That is both loose and incomparable
between meshes, since refining the mesh changes the residual's scale. The port
uses a component-weighted RMS residual scaled by its initial value, so histories
from different meshes, fluxes and operating points can be put on one plot. This
matters more than it sounds: the Riemann-flux comparison in increment 5 is
entirely a comparison of residual histories.

### Smaller things

* `initialstate.m` computes `c_inf` and never uses it.
* `post_calc.m` plots $(p_\infty - p_b)/q_\infty$, which is $-c_p$, against an
  axis labelled $-c_p$ — self-consistent, but the port defines
  $c_p = (p - p_\infty)/q_\infty$ and negates it when plotting.
* Boundary tags were `-1` (airfoil) and `-2` (far field); the port uses a named
  `BoundaryTag` enum.
* `plotsolution.m` draws the Mach field with `jet`. The port refuses the
  non-monotonic colour maps by name — see
  [`../docs/README.md`](../running/plotting.md#color-maps).

## Running the MATLAB

It needs MATLAB with the **PDE Toolbox** (for `initmesh`, `refinemesh` and
`pdeplot`). From this directory:

```matlab
main
```

It writes `cp`, `cl`, `cd`, `residual` and per-iteration `mesh`/`mach` figures
into the working directory.
