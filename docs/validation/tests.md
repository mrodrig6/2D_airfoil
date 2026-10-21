# Test-suite claims

The existing suite checks the claims listed here. Slow tests run converged flow solves; optional backend tests require the matching extra.

## What is checked automatically

`./fv2d.sh verify` runs the whole suite. The fast half takes about 15 seconds
and the slow half about a minute; neither needs a network or a service.

| Property | Where |
|---|---|
| Every flux is consistent, conservative, fully upwind when supersonic, and survives a near-vacuum | `tests/test_physics.py` |
| **AUSM⁺-up's split polynomials satisfy their two exact identities**, match their closed forms, and stay finite at zero Mach | `tests/test_physics.py` |
| AUSM⁺-up moves mass across a pressure jump from rest, and carries a moving contact at constant pressure | `tests/test_physics.py` |
| The far field reproduces the free stream exactly, at any angle of attack | `tests/test_physics.py` |
| **A uniform free stream stays uniform**, for every flux | `tests/test_solver.py` |
| The local time step equals the legacy's `CFL * d_h / s` | `tests/test_solver.py` |
| A symmetric section at zero incidence makes exactly zero lift | `tests/test_solver.py` |
| Lift is antisymmetric in angle of attack | `tests/test_solver.py` |
| The transonic case puts a shock on the upper surface | `tests/test_solver.py` |
| The discrete Gauss identity closes to round-off | `tests/test_mesh.py` |
| NACA 0012 reproduces `legacy/nacageom.mat` | `tests/test_geometry.py` |
| The compiled and the traced backends reproduce the vectorised residual | `tests/test_backends.py` |
| **The adjoint gradient matches a finite difference**, for a flow variable and a shape variable | `tests/test_sensitivity.py` |
| The traced free stream and the traced load integral equal the ones the solver reports | `tests/test_sensitivity.py` |
| A refined mesh still closes the discrete Gauss identity, and every leaf has at most one split edge | `tests/test_adapt.py` |
| **Repeated refinement does not produce slivers** — the smallest angle falls once and then holds | `tests/test_adapt.py` |
| Refining the wall makes the section rounder, towards its true area and not past it | `tests/test_adapt.py` |

## Free-stream preservation

The single most informative check, and the one worth running first after any
change to the fluxes, the metrics or the assembly.

A uniform free stream must give **exactly zero** residual in every cell that
does not touch the wall. Each interior face returns the exact flux by
consistency, so the cell sum collapses to $\mathbf{F}_\infty \cdot \sum_f
\hat{n}_f \ell_f$, which the discrete Gauss identity makes zero; the far-field
condition reproduces the free stream, so those faces behave the same. Only wall
cells differ, because a uniform flow does not satisfy a wall condition — which
is the entire reason the solution is not uniform.

Measured at $10^{-11}$ relative, for the currently shipped fluxes at $\alpha = 0$ and
$\alpha = 2.5^\circ$. Nothing else catches a sign error in a face normal or a
left/right pair swapped: both leave a uniform flow visibly decaying, and both
would otherwise show up only as a slightly wrong answer.

## Running the suite

```bash
# Run all scientific and interface assertions.
./fv2d.sh test
# Run fast tests only.
./fv2d.sh test --fast
# Run slow solves and gradient checks only.
./fv2d.sh test --slow
```

Install `[all,dev]` to include Numba and JAX coverage. A missing optional backend produces skips; `--fast` deselects slow tests. Neither outcome is a pass for the omitted checks. CI requires numerical backends to import before the full run.

## Claims and implementation

- Refinement has to leave a mesh the solver can still use. [test_adapt.py](../../tests/test_adapt.py).
- The one call students make, and the option plumbing behind it. [test_api.py](../../tests/test_api.py).
- The compiled backend must agree with the vectorised one, exactly enough. [test_backends.py](../../tests/test_backends.py).
- The non-dimensionalisation. [test_flow.py](../../tests/test_flow.py).
- The airfoil surface: the NACA family, coordinate files, and the winding. [test_geometry.py](../../tests/test_geometry.py).
- The O-grid, its triangulation, and the metrics the solver reads off it. [test_mesh.py](../../tests/test_mesh.py).
- Reading a mesh someone else generated. [test_meshio.py](../../tests/test_meshio.py).
- The fluxes and the boundary conditions. [test_physics.py](../../tests/test_physics.py).
- The project's own consistency: the command line runs, and the documents link. [test_project.py](../../tests/test_project.py).
- The adjoint gradient, and the three places it could silently be wrong. [test_sensitivity.py](../../tests/test_sensitivity.py).
- Assembly, the march, and the loads that come out of it. [test_solver.py](../../tests/test_solver.py).
