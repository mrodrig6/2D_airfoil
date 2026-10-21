# Time stepping and convergence

The solver marches in pseudo-time to a steady state with local steps. The scheme, convergence norm and stability restrictions below determine when it stops.

## Time integration

Explicit, marched to steady state with a **local** time step: each cell advances
at its own stable rate, which is legitimate because only the fixed point is
wanted. With the legacy's hydraulic diameter $d_h = 2A/P$ as the length scale,

```math
\Delta t_e = \mathrm{CFL}\,\frac{d_{h,e}}{s_e},
\qquad s_e = \frac{1}{P_e}\sum_{f \in \partial e}
\left(|v_n| + a\right)_f \ell_f ,
```

with $(|v_n| + a)_f$ the maximum signal speed the face's Riemann solver reports.
This is `legacy/main.m`'s $\mathrm{CFL}\,d_h/s$ with $d_h = 2A/P$ substituted;
the perimeter cancels, which removes a quantity that otherwise has to be kept
in step with the faces for no benefit.

Convergence is measured by the component-weighted RMS of $d\vec U/dt$, scaled by
the value it takes for a **free-stream start at the same operating point**. The
scaling is what makes one tolerance mean the same thing on every mesh — and the
reference is deliberately not the run's *own* first iteration, which would make
a warm-started run chase the same reduction factor from a smaller start and so
save nothing.

## Supported scheme and checks

Forward Euler (`scheme="euler"`) is the only supported scheme on all three backends. The shared step is defined in `src/backends/base.py`; Numba supplies an equivalent buffered path. All backends check convergence every `batch` steps, with a default of 25.

`SolverOptions.cfl=None` resolves to 0.9, using the measured Euler stability limit of 1.15 and a margin of 0.78, rounded to two decimals. An explicit positive CFL overrides the recommendation. These measurements do not guarantee convergence for every geometry or flux. The adjoint also uses the Euler recommendation when CFL is omitted.

SSP-RK3 and RK4 were removed. The [backend measurements](backends.md#one-scheme-and-the-measurement-that-settled-it) record the branch's comparison; neither scheme is an available configuration option. MUSCL reconstruction remains reverted, and the implemented spatial scheme is first order.
