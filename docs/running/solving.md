# Solving a case

Choose a geometry, operating point and resolution, then check the returned convergence status. A finished process does not imply every study point converged.

## Solving

```bash
./fv2d.sh solve @transonic
./fv2d.sh solve @subcritical cp=cp.png figure=mach.png convergence=history.png
./fv2d.sh solve airfoil=23012 mach=0.7 alpha=3 flux=hllc
```

```python
from src import solve_airfoil

result = solve_airfoil("0012", mach=0.8, alpha=1.25)
assert result.converged, result.message
print(result.report())
load = result.forces()
print(load.cl, load.cd, load.cm)
```

Reuse a mesh across an angle-of-attack sweep, and warm-start each point from
the last. The mesh does not rotate with $\alpha$ — it enters only through the
free-stream direction — so this is exact, not an approximation:

Continue in the same Python session:

```python
from src import build_case, FlowConditions, SolverOptions, forces

case = build_case("0012", n_surface=192, n_radial=64)
options, state = SolverOptions(print_interval=0, max_iterations=60_000), None
for alpha in [0.0, 1.0, 2.0, 3.0, 4.0]:
    result = case.solve(FlowConditions(mach=0.5, alpha_deg=alpha), options, state=state)
    assert result.converged, result.message
    state = result.state  # warm start the next point
    print(alpha, forces(result).cl)
```

A run that does not converge **returns** rather than raises — a stalled march
is a result worth looking at, and for the flux comparison it is *the* result.
`result.converged` says which happened and `result.message` says why; the
`solve` command exits non-zero.

See [reading results](output.md) for the coefficient and residual definitions.
