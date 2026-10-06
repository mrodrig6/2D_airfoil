# Roadmap

The port is being done in five increments. Each one is a reviewable change that
leaves the repository working, rather than one drop at the end.

| | Increment | Status |
|---|---|---|
| **1** | **Scaffold, geometry, mesh** | **done** |
| 1b | Five-digit sections, and the C-grid topology | **done** |
| **2** | **The solver** | **done** |
| 3 | Speed: the Numba backend, and a higher-order reconstruction | next |
| 4 | JAX, the adjoint, and angle-of-attack sensitivity | |
| 5 | The Riemann-flux study, adaptive refinement, and the lab material | |

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

`physics.py` with all four approximate Riemann solvers, the inviscid wall and
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
* **The AUSM⁺-up default changed.** Liou's $M_{co} \sim M_\infty$ diverges here;
  the measured window is 0.30–0.45 and does not move with the free stream.
* **The flux comparison already has its answer**, earlier than planned, and it
  is the opposite of what the nozzle expected.
  → [`riemann_fluxes.md`](riemann_fluxes.md)

## 3. Speed, and the reconstruction decision

The whole march fused into one `nopython` kernel — residual, local time step and
the Runge–Kutta update — called once per few hundred iterations so Python sees
almost nothing. Assembly in gather form, each face flux computed from both
sides, so `prange` is race-free without atomics.

A genuinely higher-order reconstruction is decided here rather than assumed. The
O-grid is structured, so dimensional WENO along grid lines is the real
candidate, and it wants the quadrilateral grid rather than the triangulated one
— which is an architectural choice, not a flag.

## 4. JAX, the adjoint, and angle-of-attack sensitivity

The same vectorised code under `jax.numpy`, with the march fused by
`lax.fori_loop`, and `sensitivity.py` computing exact design gradients by the
**discrete adjoint** — the implicit function theorem on $R(\vec U, \vec a) = 0$,
with both Jacobian actions from `jax.vjp`, rather than taping thousands of
iterations.

Design variables are the angle of attack, the Mach number and the shape, the
last through smooth parameterisations (four-digit $m$, $p$, $t$, and Hicks–Henne
bumps). Gradients are verified against finite differences at a shock-free point,
and the shock's effect on differentiability is documented rather than hidden.

## 5. The study, adaptation, and the lab material

**The Riemann-flux measurement** the companion nozzle code asked for: that
project found Roe converged and the AUSM family did not, but its domain is
shock-free by construction, so its conclusion explicitly does not transfer. This
code has a shock on the body. Each flux is measured on the same operating point
— residual history, converged $c_\ell$ and $c_d$, the shock position, wall time
— and the result becomes `riemann_fluxes.md`.

**Adaptive refinement** last, because conforming red–green refinement trades
away the fixed array shapes that make increments 3 and 4 fast. The legacy
Mach-jump indicator from `legacy/post_calc.m` is what drives it.

Then the examples, the lab guide and the figures.
