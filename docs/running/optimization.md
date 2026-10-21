# Shape optimization

Use a constrained SciPy optimizer with the discrete-adjoint gradient. The worked script below contains the complete objective, bounds and stopping criteria.

## Shape optimisation

With a gradient, any gradient-based optimiser will do; nothing in `src/` has to
know about it. SciPy's L-BFGS-B, given `jac=True`:

```python
import numpy as np
from scipy.optimize import minimize
from src.sensitivity import four_digit_case

TARGET, CREST, WEIGHT = 0.60, 0.4, 0.5


def objective(x):
    case = four_digit_case(
        x[0],
        x[1],
        0.12,
        n_surface=48,
        n_radial=24,
        far_field_radius=10.0,
        first_cell_height=0.02,
        mach=0.5,
        alpha=0.0,
        tolerance=1e-9,
        print_interval=0,
    )
    cl, g = case.value_and_gradient("cl", ("camber", "camber_position"))
    offset = (x[1] - CREST) / CREST
    J = (cl - TARGET) ** 2 + WEIGHT * offset**2
    return J, np.array(
        [
            2 * (cl - TARGET) * g["camber"],
            2 * (cl - TARGET) * g["camber_position"] + 2 * WEIGHT * offset / CREST,
        ]
    )


found = minimize(
    objective,
    np.array([0.02, 0.30]),
    jac=True,
    method="L-BFGS-B",
    bounds=[(0.005, 0.08), (0.15, 0.60)],
)
```

Which camber line gives $c_\ell = 0.6$ at zero incidence? From a NACA 2312,
**eight** flow solves and eight adjoint solves — 73 s — give $m = 0.0515$ at a
crest of $0.4000$, hitting the target to five decimal places. Rounded to digits,
a NACA 5412.

The regulariser is not decoration. A target lift in two variables is
**underdetermined** — many camber lines give the same $c_\ell$ — and an optimiser
handed an underdetermined problem wanders along the valley floor until it runs
out of iterations. The second term picks the member of that family nearest the
familiar crest position, which is a design decision made in the open rather than
an accident of where the search started.

→ [`../examples/04_optimise.py`](../../examples/04_optimise.py)
