# Gradient comparisons

Measured adjoint versus finite-difference values, tolerances and operating-point caveats.

## The adjoint gradient

The gradient is of the **discrete** problem, so the reference is a central
difference of this solver's own output — not of the exact Euler equations.
Measured on a 4096-cell O-grid around a NACA 2412 at $M = 0.5$ and
$2^\circ$, subcritical so that the objective is actually differentiable:

| $d c_\ell/d$ | adjoint | central difference | relative |
|---|---|---|---|
| `alpha` | $+1.18547016 \times 10^{-1}$ | $+1.18547020 \times 10^{-1}$ | $3 \times 10^{-8}$ |
| `mach` | $+2.42933313 \times 10^{-1}$ | $+2.42933500 \times 10^{-1}$ | $8 \times 10^{-7}$ |
| `camber` | $+1.21783132 \times 10^{1}$ | $+1.21783114 \times 10^{1}$ | $2 \times 10^{-7}$ |
| `camber_position` | $+1.73515107 \times 10^{-1}$ | $+1.73515127 \times 10^{-1}$ | $1 \times 10^{-7}$ |
| `thickness` | $+1.52774973 \times 10^{-1}$ | $+1.52774872 \times 10^{-1}$ | $7 \times 10^{-7}$ |

The two flow variables are exact adjoint derivatives; the three shape variables
take the hybrid route, with the mesh term finite-differenced at a frozen state,
which is why they sit an order of magnitude looser and still well inside what an
optimiser needs. Reproduce it with

```bash
./fv2d.sh sensitivity @subcritical naca=2412 ns=64 nr=32 R=10 h1=0.02 \
    tolerance=1e-10 max_iterations=60000 check=yes
```

**This will not agree at a transonic point**, and that is not a failure. The
shock jumps from one cell to the next rather than sliding, so the discrete
objective is only piecewise differentiable in the design variables and a small
step measures the jump rather than a derivative. →
[sensitivity](../running/sensitivity.md#when-the-gradient-is-not-to-be-trusted)
