# Mesh generation

The airfoil mesh generator builds a body-fitted O- or C-grid and splits it into triangles. These algorithms control wall spacing, topology and mesh quality.

## 5. Mesh generation

Implemented in `src/geometry/mesh.py`.

### Why this is new code

**The MATLAB contained no mesher.** `legacy/initialmesh.m` is four lines handing
a decomposed geometry matrix to the PDE Toolbox's `initmesh`, and
`legacy/adaptmesh.m` does the same with `refinemesh`. Neither has a Python
equivalent inside a reasonable dependency budget, so the mesh had to be written
from scratch and is the one genuinely new part of this port.

### The choice, and what it buys

An **algebraic O-grid**: $N$ points around the section by $n_r$ cells out to a
circular far field, each quadrilateral split into two triangles. Against an
unstructured Delaunay mesher this buys four things:

1. **No dependency**, and it fits in one readable file.
2. **Fixed, known array shapes.** A mesh whose element count depends on the
   geometry forces a recompile per design point under Numba and a retrace under
   JAX.
3. **Node positions are a smooth closed-form function of the surface points**,
   so a shape derivative flows through the mesh. This is the property that makes
   the adjoint possible at all: an unstructured remesh is not differentiable,
   and a gradient taken across one is noise.
4. **Cells hug the wall**, which is where the pressure that makes up the lift is
   read.

### The construction

For surface point $\vec{A}_i$ with outward unit normal $\hat{n}_i$, the matching
far-field point is

```math
\vec{B}_i = \vec{C} + R\left(\cos\varphi_i,\ \sin\varphi_i\right),
\qquad \varphi_i = \varphi_0 + \frac{2\pi i}{N},
```

where $\vec C$ is the far-field centre and $\varphi_0$ the direction of the
trailing edge seen from it. The angle runs **uniformly** while the surface points
are clustered, so grid lines fan out from the dense leading edge to an evenly
graded far field — and because $\varphi_i$ is monotone in $i$ by construction,
grid lines cannot cross at the outer boundary.

Each node then sits at

```math
\vec{X}_{ij} = \vec{A}_i + L_i f_j\, \hat{D}_{ij},
\qquad L_i = \lVert \vec{B}_i - \vec{A}_i \rVert,
\qquad \hat{u}_i = \frac{\vec{B}_i - \vec{A}_i}{L_i},
```

```math
\hat{D}_{ij} = \frac{(1 - w_j)\,\hat{n}_i + w_j\,\hat{u}_i}
                    {\lVert (1 - w_j)\,\hat{n}_i + w_j\,\hat{u}_i \rVert},
\qquad w_j = \left(\frac{j}{n_r}\right)^{\kappa} .
```

The marching direction blends from the **wall normal** to the **radial**
direction, and the two ends are exact rather than approximate — which is the
point of writing it this way:

| | $w$ | $f$ | $\hat D$ | result |
|---|---|---|---|---|
| $j = 0$ | 0 | 0 | $\hat n_i$ | the node *is* the surface point, and the first cell is **wall-orthogonal** |
| $j = n_r$ | 1 | 1 | $\hat u_i$ | the node lands **exactly on the circle** |

Because $\kappa \ge 1$ the blend also leaves the wall with zero slope, so there
is no kink in the first few layers. $\kappa$ is `orthogonality`, default 2.

Centring $\vec C$ at mid-chord is what keeps this well conditioned: there
$\hat n_i$ and $\hat u_i$ very nearly agree at the leading edge, at the trailing
edge and over most of a slender section, so the blend turns the marching
direction through a small angle rather than a large one. A fold is still
possible for an extreme request, and `MeshTopology.validate` checks every
triangle's area and says what to change rather than returning a broken mesh.

### The radial distribution

Spacing grows geometrically, $h_k = h_1 r^k$, with the ratio solved so the
layers exactly span the distance:

```math
h_1\,\frac{r^{n_r} - 1}{r - 1} = L .
```

The left side increases with $r$, so one bisection on $(1, r_{\max}]$ finds it.
If even $r_{\max}$ cannot span $L$, the builder **raises and names the four
numbers that would fix it** rather than returning a grid that stops short — a
mesh that quietly fails to reach the far field shows up much later as a solver
that will not converge, which is a far more expensive way to find out.

`first_cell_height` is applied relative to the far-field radius, so the realised
height varies by a few percent around the section as $L_i$ does.
`OGrid.first_cell_heights()` reports the actual values.

### Triangulation and connectivity

The quadrilateral $(i,j) \to (i,j{+}1) \to (i{+}1,j{+}1) \to (i{+}1,j)$ is
counter-clockwise, and splits into two triangles along a diagonal that
**alternates with the parity of $i + j$** — a single fixed diagonal biases any
feature that happens to line up with it.

Connectivity is built by sorting rather than by the sparse hash table of
`legacy/meshconnect.m`. Local edge $k$ of a counter-clockwise triangle is the
one *opposite* node $k$, running from node $k{+}1$ to node $k{+}2$; for that
traversal

```math
\hat{n} = \frac{(\Delta y,\ -\Delta x)}{\ell},
\qquad (\Delta x, \Delta y) = \vec{p}_{k+2} - \vec{p}_{k+1},
```

is the element's outward normal with no sign test. Two consequences follow, and
`tests/test_mesh.py` checks both: an **interior** face's normal points from its
left element into its right one, and a **boundary** face's normal points out of
the flow domain — at the wall, into the airfoil.

The identity that matters most is discrete Gauss,

```math
\sum_{f \in \partial e} \hat{n}_f\, \ell_f = \vec{0}
\qquad \text{for every cell } e,
```

which is what makes the scheme conservative and a uniform flow exact.
`Metrics.closure_error()` measures it; it is $4\times10^{-16}$ on the meshes
here, and `build_metrics` refuses to return if it is not at round-off.

![The default 192x64 O-grid, framed on the section](../figures/mesh.png)

### The C-grid

An O-grid's radial lines all radiate from the trailing edge, so cells behind the
section fan out *across* the wake rather than running along it. The C-grid
exists to remove that: it **cuts along the wake**, running from a downstream
outflow plane up one side of the cut, around the section, and back down the
other side.

![The C-grid near the trailing edge: cells run along the wake, not out of it](../figures/cgrid.png)

The cut itself is **not a boundary**. At $j = 0$ its two sides are the same
physical points and are given the same node number, so the cells above and
below it are ordinary interior neighbours sharing a face. Nothing in the solver
knows the cut is there. Above $j = 0$ the two sides separate, and the two
radial lines leaving the trailing edge run to opposite ends of the outer
boundary — that is the cut opening up, and it is what an O-grid cannot do.

Three things about it were found by measurement rather than by design, and each
is worth recording because each looked like something else first.

**The marching fan has to be monotone.** Marching along the surface normal fails
here, and not for a tunable reason. The wake cut's own normal is straight up;
the upper surface leaves a sharp trailing edge sloping **aft** — $8.3^\circ$ for
a NACA 0012 — so its normal tilts aft too, while the stations march forward. The
direction field therefore *dips* by a few degrees at the junction and climbs
back. The two stations either side are one short wake panel apart, so a
$4^\circ$ dip closes that gap after 0.05 chords of a 20-chord march and the
cells invert. Sweeping the blend exponent over its whole range changed the fold
count from 122 to 220 and never to zero; smoothing the normals does not help
either, because the dip is real rather than noise. What works is requiring the
fan to turn one way — the running maximum of the unwrapped angle, **anchored at
both ends**. Anchoring only the start flattens the first dip but carries the
second outward with it, which leans the far end of the cut and bows the outflow
plane off vertical. The cost is bounded: only the rear of each surface moves,
between the trailing edge and the point of maximum thickness, and the largest
correction is the trailing-edge slope itself. The leading-edge region, where
orthogonality earns its keep, is left exactly alone.

**The outer legs need their own spacing.** Reusing the wake cut's distribution
for the straight legs is the obvious thing and it is wrong: the cut clusters
hard at the trailing edge, so the outer spacing falls to 0.0033 right where the
arc's is 0.327, and a hundredfold jump between neighbouring columns leaves a
$0.03^\circ$ sliver at the far field. Starting each leg at the arc's spacing
keeps the two sides of that junction within a factor of two.

**The first cell is a wall height, and the cut has no wall.** Sharing one value
puts a 0.002-tall cell under a station 3.8 chords long far downstream — aspect
ratio 1900, which triangulates into $0.03^\circ$ slivers and, worse, collapses
the local time step through the hydraulic diameter across the whole wake.
Capping the first cell's aspect ratio leaves every station on the section
untouched, since their panels are far shorter than the cap allows, and opens
the wake out to match its own streamwise spacing.

### Choosing a topology

Measured on a NACA 0012 at $192 \times 64$, equiangle skewness (0 is
equilateral, 1 a degenerate sliver):

| | cells | worst skew | min angle | cells in the wake | wake skew (median) |
|---|---|---|---|---|---|
| **O-grid** | 24 576 | 0.961 | $2.3^\circ$ | 1 936 | 0.846 |
| **C-grid** | 32 768 | 0.974 | $1.5^\circ$ | 3 242 | 0.797 |

The C-grid costs a third more cells and puts two thirds more of them in the
wake, at better quality there; the O-grid is smaller and marginally better in
its worst cell. **Which one is right is a question about the answer, not about
the mesh**, The solver now supports both; the O-grid remains the default. Compare loads and mesh quality for the operating point of interest.

### What a structured mesh costs

* **Resolution is uniform in index, not in error.** A shock is resolved by
  whatever `n_surface` and `n_radial` happen to provide there.
  [adaptation](../running/adaptation.md) lifts this, at the price the bullet predicted:
  an adapted mesh is no longer structured, the $(i, j)$ layout is gone, and only
  the triangulation remains.
* **The far-field boundary is a polygon**, not a circle. The O-grid's area is
  $\tfrac12 N R^2 \sin(2\pi/N)$, which is what the area test compares against —
  $\pi R^2$ would be wrong by $0.02\%$ at $N = 192$.
* **Skewness is the wrong metric to read alone.** A right isoceles triangle,
  which is what splitting a square gives, already scores 0.25, and one from a
  2:1 rectangle scores 0.56. The numbers above are comparable with each other
  and not with an unstructured mesher's.
