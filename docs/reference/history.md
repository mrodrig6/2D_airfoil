# Development history

The port was completed in five increments. This history records what each increment added and what remains deliberately deferred.

## 1. Scaffold, geometry, mesh — done

The MATLAB moved to [`../legacy/`](../../legacy) unchanged. At the end of this first increment, the Python package described a section and meshed it:

* `src/config.py` — the free stream, the gas, the mesh resolution, and the
  non-dimensionalisation
* `src/geometry/section.py` — the NACA four- and five-digit families; Selig and
  Lednicer coordinate files; resampling
* `src/geometry/mesh.py` — the O-grid and the C-grid, their triangulation, and
  connectivity
* `src/geometry/meshio.py` — SU2 and Gmsh 2.2 meshes from elsewhere
* `src/spatial/metrics.py` — areas, face lengths and normals
* `src/post/plotting.py`, `src/cli.py`, `fv2d.sh`

At that historical point no solver had been added: the geometry and the mesh are what the
solver is built *on*, and they are worth getting exactly right while they are
the only thing in the repository.

## 2. The solver — done

`physics.py` with the approximate Riemann solvers, the inviscid wall and
characteristic far-field conditions, the vectorised residual, the
local-time-step march, and the corrected force integral. Reconstruction is
piecewise constant, exactly the legacy scheme.

`solve_airfoil("0012", mach=0.8, alpha=1.25)` converges and puts the shock at
$x/c = 0.629$ against a published 0.62–0.65, with a peak Mach number of 1.341
against ~1.35. Force coefficients are not yet converged — the scheme is first
order, and [tests](../validation/tests.md) measures what that costs.

Three things came out of it that were not planned:

* **The far-field default is confirmed by measurement**: $c_\ell$ moves 1.7%
  between 8 and 50 chords, and 20 is within 0.05% of 50.
* **The flux comparison got its answer**, earlier than planned, SLAU2 was removed. AUSM⁺-up was initially removed and later restored after correcting its implementation. → [flux study](../validation/flux-study.md)

## 3. Speed — done

The whole march fused into one `nopython` kernel — residual, local time step and
the update — called once per few hundred iterations so Python sees almost
nothing. Assembly in gather form, each face flux computed from both sides, so
`prange` is race-free without atomics. **23× the vectorised reference** on the
production mesh: a converged transonic solve in 16 s rather than six minutes,
and the answer does not depend on which backend ran.
→ [backends](../method/backends.md)

### 3b. The reconstruction, deliberately deferred

A genuinely higher-order reconstruction was to be decided here rather than
assumed, and the decision was to **not do it yet**. The O-grid is structured, so
dimensional WENO along grid lines is the real candidate, and it wants the
quadrilateral grid rather than the triangulated one — an architectural choice,
not a flag. WENO3 is the cheap version of it and can be added later without
disturbing anything now in place, which is why it waits.

What it is worth is measured rather than assumed: projecting the convergence
data in [tests](../validation/tests.md), going from first to third order is
**90× fewer cells** for a drag of 0.001, while third to fifth is only 2×. The
first step is the one that matters.

## 4. JAX, the adjoint, and design sensitivity — done

The same vectorised code under `jax.numpy` — not a third implementation: both
`assembly.py` and `physics.py` take an `xp` array module, and the only thing
that genuinely differs between NumPy and JAX is the scatter. The march is fused
by `lax.fori_loop`, and float64 is turned on at import, because JAX defaults to
single precision and would cap the residual at about $10^{-7}$ silently.

`sensitivity.py` computes exact design gradients by the **discrete adjoint** —
the implicit function theorem on $R(\vec U, \vec a) = 0$, with both Jacobian
actions from `jax.vjp`, rather than taping thousands of iterations. The adjoint
system is marched in pseudo-time with the forward solver's own local time step,
which is legitimate because the two operators share a spectrum.

Design variables are the angle of attack, the Mach number, and the four-digit
$m$, $p$ and $t$ taken continuously. Gradients agree with finite differences to
seven or eight significant figures at a shock-free point; the shock's effect on
differentiability is documented rather than hidden.

Two things came out differently from the plan:

* **The shape derivatives take a hybrid route.** Tracing the mesh generator
  would mean a differentiable marcher, bisection included, and a traced copy of
  every metric — a large refactor for a small term. The adjoint stays exact and
  the mesh term is finite-differenced at a *frozen state*, which costs two
  residual evaluations and no flow solve.
  → [adjoint](../theory/adjoint.md#the-shape-variables-and-the-one-compromise)
* **Hicks–Henne bumps are not implemented.** The four-digit parameters were
  enough to demonstrate and verify the shape route, and a richer
  parameterisation is a geometry question rather than an adjoint one — it plugs
  into the same `_shape_partials` without touching the adjoint.

And one measurement worth having: on a 4096-cell mesh the adjoint is *not* the
cheaper route for five variables — it breaks even at about nine, because the
forward solver is compiled Numba while the adjoint march runs in JAX. The
scaling is what the adjoint buys, not the constant.
→ [backends](../method/backends.md#what-it-costs)

## 5. Adaptation, and the lab material — done

**Adaptive refinement** came last, because conforming refinement trades away the
fixed array shapes that make increments 3 and 4 fast — an adapted mesh is no
longer structured, so the O-grid's $(i, j)$ layout is gone and only the
triangulation remains. The legacy Mach-jump indicator from `legacy/post_calc.m`
drives it unchanged, including its wall term.

The implementation keeps a **tree of red subdivisions** and recomputes green
closure every cycle rather than storing it, so a green triangle is never refined
— the classic way an adaptive code quietly fills itself with slivers. New
boundary nodes are projected back onto the section, without which refining the
wall only adds points to its chords.
→ [adaptation](../running/adaptation.md)

**The result is not the textbook one**, and it is the most interesting thing in
this increment: against uniform refinement at equal cell count, adaptation wins
on drag and loses on lift. [adaptation](../validation/adaptation.md#does-it-pay) has the
measurement and the reason, which is that the indicator is a jump rather than an
error estimate and nothing in it knows which functional is being computed.

**[lab guide](../running/lab-guide.md)** is the lab guide: seven exercises, each with the number
this code actually gives, so a student can tell a result from a mistake.

The examples landed early, with increment 4 — [`../examples/`](../../examples) and
[sweeps](../running/sweeps.md) — because a gradient nobody can run is not a
feature.

## What was left out, on purpose

* **A higher-order reconstruction** (increment 3b). Deferred by decision, not by
  omission; it is the single most valuable thing still undone, and
  [tests](../validation/tests.md) measures what it would buy.
* **Hicks–Henne bumps.** The four-digit parameters were enough to demonstrate
  and verify the shape route; a richer parameterisation plugs into the same
  `_shape_partials` without touching the adjoint.
* **A goal-oriented (adjoint-weighted) error indicator.** The pieces are all
  here — `sensitivity.py` already solves the adjoint this would weight by — and
  [adaptation](../validation/adaptation.md#what-would-change-the-answer) says what it
  would change.
