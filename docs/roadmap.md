# Roadmap

The port is being done in five increments. Each one is a reviewable change that
leaves the repository working, rather than one drop at the end.

| | Increment | Status |
|---|---|---|
| **1** | **Scaffold, geometry, mesh** | **done** |
| 1b | Five-digit sections, and the C-grid topology | **done** |
| **2** | **The solver** | **done** |
| **3** | **Speed: the Numba backend** | **done** |
| 3b | A higher-order reconstruction | deferred, see below |
| **4** | **JAX, the adjoint, and design sensitivity** | **done** |
| 5 | Adaptive refinement, and the lab material | next |

The Riemann-flux study that increment 5 was to carry out
[happened early](riemann_fluxes.md), in increment 2, because the answer was
needed before anything was built on top of a flux.

## 1. Scaffold, geometry, mesh — done

The MATLAB moved to [`../legacy/`](../legacy/) unchanged. The Python package
describes a section and meshes it, and nothing more:

* `src/config.py` — the free stream, the gas, the mesh resolution, and the
  non-dimensionalisation
* `src/geometry.py` — the NACA four- and five-digit families; Selig and
  Lednicer coordinate files; resampling
* `src/mesh.py` — the O-grid and the C-grid, their triangulation, and
  connectivity
* `src/meshio.py` — SU2 and Gmsh 2.2 meshes from elsewhere
* `src/metrics.py` — areas, face lengths and normals
* `src/plotting.py`, `src/cli.py`, `fv2d.sh`

Nothing solves anything yet, on purpose: the geometry and the mesh are what the
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
order, and [`verification.md`](verification.md) measures what that costs.

Three things came out of it that were not planned:

* **The far-field default is confirmed by measurement**: $c_\ell$ moves 1.7%
  between 8 and 50 chords, and 20 is within 0.05% of 50.
* **The flux comparison got its answer**, earlier than planned, and two of the
  four schemes were removed as a result. → [`riemann_fluxes.md`](riemann_fluxes.md)

## 3. Speed — done

The whole march fused into one `nopython` kernel — residual, local time step and
the update — called once per few hundred iterations so Python sees almost
nothing. Assembly in gather form, each face flux computed from both sides, so
`prange` is race-free without atomics. **23× the vectorised reference** on the
production mesh: a converged transonic solve in 16 s rather than six minutes,
and the answer does not depend on which backend ran.
→ [`performance.md`](performance.md)

### 3b. The reconstruction, deliberately deferred

A genuinely higher-order reconstruction was to be decided here rather than
assumed, and the decision was to **not do it yet**. The O-grid is structured, so
dimensional WENO along grid lines is the real candidate, and it wants the
quadrilateral grid rather than the triangulated one — an architectural choice,
not a flag. WENO3 is the cheap version of it and can be added later without
disturbing anything now in place, which is why it waits.

What it is worth is measured rather than assumed: projecting the convergence
data in [`verification.md`](verification.md), going from first to third order is
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
  → [`workflows.md`](workflows.md#the-shape-variables-and-the-one-compromise)
* **Hicks–Henne bumps are not implemented.** The four-digit parameters were
  enough to demonstrate and verify the shape route, and a richer
  parameterisation is a geometry question rather than an adjoint one — it plugs
  into the same `_shape_partials` without touching the adjoint.

And one measurement worth having: on a 4096-cell mesh the adjoint is *not* the
cheaper route for five variables — it breaks even at about nine, because the
forward solver is compiled Numba while the adjoint march runs in JAX. The
scaling is what the adjoint buys, not the constant.
→ [`workflows.md`](workflows.md#what-it-costs)

## 5. Adaptation, and the lab material

**Adaptive refinement** last, because conforming red–green refinement trades
away the fixed array shapes that make increments 3 and 4 fast. The legacy
Mach-jump indicator from `legacy/post_calc.m` is what drives it.

The examples landed early, with increment 4 — [`../examples/`](../examples/)
and [`workflows.md`](workflows.md) — because a gradient nobody can run is not a
feature. What remains for increment 5 is the refinement, the lab guide and the
remaining figures.
