# Boundary conditions

Boundary conditions impose only the information carried into the domain. This page describes the wall and characteristic branches implemented by the solver.

## 7. Boundary conditions

Two boundaries, matching `src.geometry.mesh.BoundaryTag`.

**Inviscid wall** ($\hat n$ pointing into the section). No flow through the
surface, so the normal velocity vanishes and the flux carries pressure only:

```math
\hat{\vec{F}}_{\text{wall}} = \left(0,\ p_b n_x,\ p_b n_y,\ 0\right)^T,
\qquad
p_b = (\gamma - 1)\left(\rho E - \tfrac12 \rho\,
      \lvert \vec v - (\vec v \cdot \hat n)\hat n \rvert^2\right),
```

the pressure of the interior state with its normal velocity removed. This is
`legacy/residual_calc.m`'s `inviscid_wall_bc`, unchanged.

**Characteristic far field.** The legacy imposed the full free-stream state and
let the Riemann solver sort it out, which is reflective and is why its domain had
to be 50 chords across. This port instead imposes only what the incoming
characteristics carry, using the one-dimensional Riemann invariants normal to the
boundary,

```math
R^{\pm} = v_n \pm \frac{2a}{\gamma - 1},
```

taking $R^+$ from the interior and $R^-$ from the free stream on a subsonic
inflow face and the reverse on outflow, with entropy and the tangential velocity
taken from whichever side the flow comes from. A non-reflecting boundary admits a
much smaller domain, which is free speed. How much smaller was then measured
rather than asserted: $c_\ell$ moves 1.7% between 8 and 50 chords, and the
20-chord default is within 0.05% of 50.
→ [far field](../validation/far-field.md#how-far-away-the-far-field-has-to-be)
