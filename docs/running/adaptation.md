# Adaptive refinement

Solve/refine loop, controls, returned history, stalled-cycle behavior and practical costs.

## Using it

```bash
./fv2d.sh adapt @transonic cycles=6 figure=adapted.png history=loads.png
./fv2d.sh adapt @transonic cycles=8 fraction=0.10 max_iterations=200000
```

```python
from src import Discretization, FlowConditions, SolverOptions
from src.adapt import adaptive_solve

history = adaptive_solve(
    "0012",
    Discretization(n_surface=64, n_radial=24, far_field_radius=10.0, first_cell_height=0.02),
    FlowConditions(mach=0.8, alpha_deg=1.25),
    SolverOptions(tolerance=1e-6, max_iterations=80_000, print_interval=0),
    cycles=6,
)
print(history.table())
print(history.final.loads.summary())
```

If the march runs out of iterations, the loop **stops and returns what it has**
with `history.stalled` set, rather than refining a mesh from a solution that
never converged — the cycles already done are good, and the limit is one option
away from being raised.

→ [`../examples/05_adapt.py`](../../examples/05_adapt.py) runs both sides of the
table above. [lab guide](lab-guide.md#lab-6--spending-cells-where-they-matter) is the
exercise built on it.
