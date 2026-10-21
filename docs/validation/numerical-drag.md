# Numerical drag and pressure error

A shock-free inviscid reference has zero drag. The reported nonzero values and pressure overshoot measure numerical error at the documented resolutions.

## Spurious drag, and what first order costs

D'Alembert's paradox says an inviscid, shock-free, steady flow produces
**exactly zero drag**. Anything the code reports at such a point is purely
numerical, which makes it a direct measure of the scheme's error.

NACA 0012 at $M_\infty = 0.5$, $\alpha = 0$, where $c_d$ must be 0:

| mesh | cells | $c_d$ (spurious) | entropy error |
|---|---|---|---|
| 64 × 32 | 4 096 | 0.0290 | 2.20e−4 |
| 96 × 40 | 7 680 | 0.0207 | 1.53e−4 |
| 144 × 56 | 16 128 | 0.0155 | 1.12e−4 |
| 192 × 72 | 27 648 | 0.0131 | 9.32e−5 |

It falls roughly as $h$ — first order, which is exactly what a piecewise-constant
reconstruction gives and no better.

![Entropy rise around the transonic NACA 0012: a bright sheet along the whole wall, the shock, and the wake](../figures/entropy_transonic.png)

The entropy field says where that error is made. Entropy is conserved along
streamlines in smooth flow, so everything bright here except the shock and the
wake behind it is the scheme rather than the physics — and the brightest part is
the **sheet along the entire wall**, which is numerical dissipation at the
surface acting as a boundary layer nobody asked for. That sheet is the spurious
drag in the table above, and it is why the loads converge so slowly.

Two consequences worth taking seriously:

* **Transonic drag is not usable at these resolutions.** The spurious drag at
  $\alpha = 0$ is 0.013, against a published wave drag of about 0.022 for the
  transonic case. The number the code reports is the sum of the two.
* **Lift is far less affected**, because it comes from a pressure difference
  that the numerical dissipation damages symmetrically. $c_\ell$ is within a
  few per cent where $c_d$ is out by half.

This is the strongest argument for a higher-order reconstruction, which was
decided in increment 3 and deliberately deferred ([history](../reference/history.md)).
It is also why the lift tests carry tight tolerances and there is no drag test.
Adaptive refinement recovers part of it — [measured in
`adaptation.md`](adaptation.md#does-it-pay) — but only part, and at a cost in
iterations.

## Stagnation-point overshoot

A known first-order artifact, recorded because it looks alarming in a $c_p$
plot and is not a bug.

At the stagnation point the computed static pressure **exceeds the free-stream
total pressure**, which is physically impossible:

| mesh | $p_{\max}/p_{t\infty}$ | $c_p$ overshoot |
|---|---|---|
| 96 × 40 | 1.039 | +25% |
| 192 × 72 | 1.016 | +10% |

It halves with refinement, so it is discretisation error rather than a
formulation defect — and the wall condition is not the cause, since the raw
interior cell pressure overshoots by the same amount. The reason it looks so
much worse in $c_p$ than in $p$ is that $c_p$ subtracts a large baseline: a 4%
pressure error becomes a 25% $c_p$ error at a point where $c_p \approx 1$.
