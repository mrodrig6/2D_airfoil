# Sensitivity analysis

Compute derivatives of a converged discrete solution with JAX. Check the forward convergence and gradient before using it in a design loop.

## Taking one

```python
from src.sensitivity import differentiable_case

case = differentiable_case(
    "2412",
    n_surface=64,
    n_radial=32,
    far_field_radius=10.0,
    first_cell_height=0.02,
    mach=0.5,
    alpha=2.0,
    tolerance=1e-10,
    print_interval=0,
)
cl, gradient = case.value_and_gradient("cl")
```

```
cl = +0.485223
d(cl)/d alpha            = +1.185470e-01   per degree
d(cl)/d mach             = +2.429333e-01
d(cl)/d camber           = +1.217831e+01
d(cl)/d camber_position  = +1.735151e-01
d(cl)/d thickness        = +1.527750e-01
```

| | |
|---|---|
| objectives | `cl`, `cd`, `cm`, `efficiency` (= $c_\ell/c_d$) |
| flow variables | `alpha` (degrees), `mach` |
| shape variables | `camber`, `camber_position`, `thickness` — the four-digit $m$, $p$, $t$ |

`value_and_gradient("cd", ("alpha",))` restricts the variables; `case.variables`
reports what is available, which is the flow pair alone when the section did not
come from a four-digit code.

Inside a design loop there is no code to write, so build the case from the
parameters as numbers:

Continue in the same Python session:

```python
from src.sensitivity import four_digit_case

design_case = four_digit_case(0.0213, 0.37, 0.12, mach=0.5, alpha=1.0)
```

This matters more than it looks. `"0012"` cannot be perturbed by $10^{-5}$, and
recovering $(m, p, t)$ by *sampling* a surface is not accurate enough to
differentiate: a four-digit camber line is flat near its crest, so a fit returns
$p = 0.441$ for a NACA 2412 whose true crest is at $0.4$. The parameters are
therefore parsed from the digits when there is a code, and shape derivatives are
simply **unavailable** for a section read from a file — inventing parameters for
it would be worse than saying so.

## Checking a gradient

Continue in the same Python session:

```python
from src.sensitivity import check_gradient

for name, row in check_gradient(case, "cl", ("alpha", "thickness")).items():
    print(name, row["adjoint"], row["finite_difference"], row["relative_error"])
```

The [gradient comparison](../validation/gradients.md) owns the measured reference table. Run the check at the same operating point and step size before comparing digits.

→ [`../examples/03_sensitivity.py`](../../examples/03_sensitivity.py)

## When the gradient is not to be trusted

The gradient is of the **discrete** problem. It matches a finite difference of
*this solver's* output, not of the exact Euler equations — which is precisely
what a gradient-based optimiser needs, since the optimiser is minimising this
solver's output too. Three things to watch:

**A shock breaks differentiability.** The shock jumps from one cell to the next
rather than sliding, so the discrete objective is only piecewise differentiable
in the design variables. The gradient stays well defined and useful, but a
finite-difference check at a transonic point disagrees at the level of that jump,
and a step smaller than one cell's worth of shock movement measures noise rather
than a derivative. This is why the verification above is done subcritically, and
why `check_gradient` reports instead of asserting.

**The march must have converged.** A gradient taken from a stalled state is the
derivative of nothing in particular. `DifferentiableCase.solve` raises rather
than hand one back, and the forward tolerance should be tighter than for
reporting loads — $10^{-9}$ or $10^{-10}$, not $10^{-6}$.

**The flow is inviscid and first order.** The drag in particular is dominated by
the scheme at these resolutions: [numerical drag](../validation/numerical-drag.md) measures
0.0131 of spurious drag at zero incidence on the finest mesh there. A gradient of
`cd` is therefore a real derivative of a quantity that is partly numerical, and
`efficiency` ($c_\ell/c_d$) inherits that. Lift and moment are on much firmer
ground, which is why the examples optimise lift.
