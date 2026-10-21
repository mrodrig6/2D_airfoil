# Adaptive algorithm

Adaptive refinement uses a Mach-jump indicator and conforming red–green refinement. The tree and boundary projection control cell shape and geometric error.

## The indicator is the legacy's

`legacy/post_calc.m` scores each element by the jump in Mach number across its
faces, weighted by face length:

```math
\varepsilon_e = \sum_{f \in \text{interior}} |M_e - M_{e'}|\,\ell_f
\;+\; \sum_{f \in \text{wall}} |M_n|\,\ell_f
```

`src/adapt.py` reproduces it exactly, including the second term, and refines the
worst 5% of elements per cycle — the same `f = 0.05`.

**The second term is not a jump**, and it is the more interesting half. The wall
condition says the normal velocity vanishes, so whatever normal Mach number
survives at a wall face is the solution disobeying its own boundary condition.
That term flags the leading edge and the trailing edge without anyone naming
them, which is why the refinement in the figures is not only at the shock.

**It is a quantity with the dimensions of a length, not a density.** Two cells of
very different size are therefore not compared fairly: a big cell can out-score a
small one simply by having longer faces. That is the legacy's choice and it is
kept, partly because it is self-limiting — refining a cell halves its faces, so
it stops being flagged once it is small enough — and partly because it turns out
to matter for the result below.

## Red, green, and the trap

A flagged triangle is cut into four by joining its edge midpoints: a **red**
refinement. The four children are similar to the parent, so red refinement can be
applied forever without changing the shape of anything.

The neighbour across a refined edge now has a **hanging node** in the middle of
one of its edges, and the solver has no concept of one — every face must have
exactly one element on each side, or the flux is not conservative. The hanging
node is removed by bisecting that neighbour from the opposite vertex: a **green**
closure.

**Green triangles are the trap.** They are not similar to their parent — the
bisection splits the angle at the opposite vertex — so bisecting a green triangle
again, and again, drives the smallest angle towards zero. The cells become
slivers, the local time step collapses with them, and the march slows to a halt
long after anyone has stopped suspecting the mesh.

This implementation makes that impossible rather than guarding against it. It
stores only the **tree of red subdivisions**; green closure is a rendering step,
recomputed from the current leaves every time a mesh is built and thrown away
afterwards. A green triangle is therefore never refined, because it never exists
between cycles. The tree also enforces **one-irregularity** — two triangles
sharing an edge differ by at most one level — so a face is split at most once and
a single bisection always closes it.

Measured over eight cycles on the transonic case, the smallest angle in the mesh
goes 4.89° → 4.56° → … → 4.07°: it pays for the first green closure once and then
stops falling. A naive implementation halves it every cycle.

## New boundary nodes go on the geometry

The midpoint of a wall face lies on the straight line between two surface points,
which is *inside* the airfoil. Leave it there and the section never gets rounder
however fine the mesh becomes — the mesh converges to a polygon, not to the
airfoil, and the leading edge, where the indicator is loudest, converges to the
wrong shape.

Every new wall node is therefore projected onto the section, and every new
far-field node onto the far-field circle. The section used for the projection is
regenerated analytically when the name is a NACA code, and densified by the same
Hermite interpolation `resample` uses when it is not — for a section read from a
file there is no analytic curve to recover.

The test for this is global rather than per-node: the area enclosed by the
refined wall polygon must be **larger** than the coarse one and no larger than
the section's true area. On a 48-point NACA 0012: 0.081473 coarse, 0.081647
refined, 0.081706 true.

## Carrying the solution across

Each child inherits its parent's cell average. For a piecewise-constant scheme
that is the exact projection onto the new mesh — a finer mesh holds no
information the coarse solution did not have — and it means each cycle starts
most of the way to its answer.

It saves less than it first appears. The second cycle converges in 2 600
iterations against the first's 5 075, but the sixth needs 55 900, because each
cycle makes the smallest cell smaller and an explicit march is paced by its
smallest cell. **Adaptation buys accuracy per cell, not accuracy per second**,
and those are different claims.
