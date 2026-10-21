# Legacy force integral

Correct pressure-force derivation, old MATLAB formula and measured discrepancy.

## The legacy force formula, measured

`legacy/post_calc.m` integrates the surface pressure with `sign(n_x)` and
`sign(n_y)` in place of the normal components, and applies $\sin\alpha$ and
$\cos\alpha$ to two separate integrals rather than rotating the force vector.
[euler](../theory/euler.md) has the derivation. The size of
the error, at $M_\infty = 0.5$, $\alpha = 2^\circ$:

| | $c_\ell$ | $c_d$ |
|---|---|---|
| correct | +0.2464 | +0.0236 |
| legacy formula | +0.2567 | +0.1465 |
| error | +4% | **+520%** |

Lift survives because the error partly cancels between the two surfaces. Drag
does not: it is six times too large, which would make any drag-based conclusion
from the legacy code meaningless. `legacy_forces` reproduces the old expression
so this stays checkable rather than remembered.

## This corrects a defect in the legacy code

`legacy/post_calc.m` computed

```matlab
D = -sum(p_b.*sign(n(:,1)).*s_edge)*sin(alpha);
L =  sum(p_b.*sign(n(:,2)).*s_edge)*cos(alpha);
```

which is wrong in two independent ways:

1. It uses `sign(n_x)` and `sign(n_y)` where the formula needs $n_x$ and $n_y$.
   Every panel then contributes its **full length** rather than its projected
   area, so the integral is not $\oint p\,\hat n\,ds$ at all.
2. It scales two separately accumulated integrals by $\sin\alpha$ and
   $\cos\alpha$, where the correct operation is a **rotation of the force
   vector**. These coincide only when one of the two components vanishes.

The port implements the formula above, and `legacy_forces` reproduces the old
expression so the difference stays checkable. Measured at $M_\infty = 0.5$,
$\alpha = 2^\circ$: lift is 4% out and **drag is 520% out**.
→ [legacy forces](legacy-forces.md#the-legacy-force-formula-measured)
