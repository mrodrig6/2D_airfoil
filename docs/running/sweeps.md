# Parameter sweeps

Solve a sequence of design or operating points and record failures as well as converged results. Reuse a case or warm-start when the API permits it.

## Sweeping the angle of attack

Incidence enters this formulation through the free-stream **direction** alone.
The mesh does not rotate with it, so a sweep costs one mesh — not one per point —
and that is the single largest saving available here. Build the case once and
reuse it:

```python
from src import Discretization, FlowConditions, SolverOptions, build_case, naca4
from src.post.postprocess import forces

mesh = Discretization(n_surface=128, n_radial=48, far_field_radius=20.0)
case = build_case(naca4("2412", mesh.n_surface), mesh)
options = SolverOptions(tolerance=1e-7, print_interval=0)

state = None
for alpha in (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0):
    result = case.solve(FlowConditions(mach=0.5, alpha_deg=alpha), options, state=state)
    state = result.state  # warm start: the next angle begins here
    print(alpha, forces(result).cl)
```

Two things are doing the work. `case` holds the mesh and every metric, so no
geometry is rebuilt. And `state=state` starts each solve from the previous
angle's converged field, which is a flow already close to the answer.

Measured on the mesh above — 12 288 cells, a NACA 2412 at $M = 0.5$ — seven
angles converge to $10^{-7}$ in **30 s in total**, about 4.3 s each.

![Lift, drag and quarter-chord moment of a NACA 0012 against angle of attack at Mach 0.5, as three panels](../figures/polar.png)

*Three panels rather than three curves on one axis: $c_\ell$, $c_d$ and $c_m$
differ by an order of magnitude, and a shared axis flattens two of them into the
baseline. From the launcher, `./fv2d.sh sweep @subcritical alphas=-4:8:2
figure=polar.png`; from Python, `src.plotting.plot_polar`.*

The curve is also the first check worth making on a new mesh. Thin-aerofoil
theory with the Prandtl–Glauert factor gives a lift slope of
$2\pi/\sqrt{1 - M^2} = 7.26$ per radian at $M = 0.5$; the fit over
$|\alpha| \le 4^\circ$ gives **7.12**, or 98% of it, and the zero-lift incidence
comes out at $-2.09^\circ$, negative because the section is cambered.

→ [`../examples/02_alpha_sweep.py`](../../examples/02_alpha_sweep.py) for the
numbers, [`../examples/06_postprocess.py`](../../examples/06_postprocess.py) for
the figure.
