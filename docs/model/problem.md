# Problem and assumptions

The solver computes steady, inviscid, two-dimensional flow of a calorically perfect gas. This page states the domain, assumptions and limits.

## 1. Scope and notation

The flow is **steady, two-dimensional, inviscid and compressible**, around a
single airfoil section of unit chord at an angle of attack $\alpha$ in a uniform
free stream of Mach number $M_\infty$. The gas is calorically perfect with
constant $\gamma = 1.4$.

| symbol | meaning |
|---|---|
| $\rho$ | density |
| $\vec{v} = (u, v)$ | velocity |
| $p$ | static pressure |
| $E$ | total energy per unit mass |
| $H = E + p/\rho$ | total enthalpy per unit mass |
| $a = \sqrt{\gamma p / \rho}$ | speed of sound |
| $\gamma$ | ratio of specific heats, 1.4 |
| $c$ | chord, the length scale |
| $\hat{n} = (n_x, n_y)$ | unit normal, outward from the cell |
| $\alpha$ | angle of attack, from the chord line, positive nose-up |

A subscript $\infty$ is a free-stream value. The conservative state vector is

```math
\vec{U} = \left(\rho,\ \rho u,\ \rho v,\ \rho E\right)^T .
```

### A note on what "transonic" costs

At $M_\infty = 0.8$ and $\alpha = 1.25^\circ$ — the operating point the legacy
MATLAB used, and the standard NACA 0012 test case — the flow accelerates past
$M = 1$ over the upper surface and returns to subsonic through a **normal shock**
at roughly mid-chord. Two things follow, and both shape the rest of this
document:

* The problem is **mixed elliptic–hyperbolic**, with the character changing
  across a surface whose position is part of the answer. No change of variables
  removes this; the scheme has to handle both.
* The shock sits **on the body**, where the pressure that makes up the lift is
  read. The accuracy of the force coefficients is therefore inseparable from how
  the shock is captured, which is why the choice of interface flux is a first-
  class question here and not a detail.

## Assumptions and constraints

Every result from this code will be conditional on the following. None is a bug
or a gap to be filled later; they are the modelling choices that define what the
solver *is*.

| | Assumption | Consequence |
|---|---|---|
| 1 | **Inviscid.** The Euler equations — no viscosity, no boundary layer | No skin friction, no separation. Transonic drag here is **wave drag only**, and comparing it to a measured drag is comparing two different quantities |
| 2 | **Calorically perfect gas**, constant γ = 1.4 | Fine for the whole transonic range this code targets |
| 3 | **Steady** | Marched in pseudo-time to a fixed point. Buffet and any other unsteady transonic phenomenon is outside the model, not merely unresolved |
| 4 | **Two-dimensional**, unit span | A section, not a wing. No induced drag, and no three-dimensional relief of the shock |
| 5 | **First-order accurate.** Piecewise-constant reconstruction, as the legacy had | Lift is within a few per cent; **drag is not usable** — the spurious drag at a point where it must be zero is 0.013 against a published transonic wave drag of 0.022. Measured |
| 6 | **Sharp trailing edge, no Kutta condition imposed** | Circulation is set by the discretisation at the trailing edge rather than by an explicit condition — which is standard for an Euler solver, and is the reason trailing-edge resolution affects lift |
| 7 | **Non-dimensional throughout** | No SI anywhere. Compare runs with `cl` and `cd`, not raw forces |
| 8 | **The initial mesh is structured** | The initial O- or C-grid uses `n_surface` and `n_radial`. Optional adaptive refinement then changes the triangulation using a Mach-jump indicator. See `docs/theory.md` |

---

See [compatibility](../reference/compatibility.md) for implementation restrictions.
