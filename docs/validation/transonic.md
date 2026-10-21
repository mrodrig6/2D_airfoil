# Transonic NACA 0012

Measured shock/Mach/load values and an explicitly sourced or source-pending external comparison.

## The transonic case

The legacy operating point: NACA 0012, $M_\infty = 0.8$, $\alpha = 1.25^\circ$.
This is the standard inviscid test case, and it has a shock on the upper
surface.

![Surface pressure at the legacy operating point, with the upper-surface shock at x/c = 0.63](../figures/cp_transonic.png)

192 × 64 O-grid, far field 20 chords, Roe flux, converged to $10^{-6}$ in
17 925 iterations — the run that drew the figure above, reproduced by
`python examples/06_postprocess.py`:

| | this code | historical external values (source unavailable) |
|---|---|---|
| upper-surface shock | $x/c = 0.629$ | 0.62–0.65 |
| lower-surface shock | $x/c \approx 0.38$ | ~0.35–0.40 |
| peak Mach number | 1.341 | ~1.35 |
| $c_\ell$ | 0.304 | ~0.35 |
| $c_d$ | 0.0327 | ~0.022 |

**The shock structure is right and the force coefficients are not yet.** Both
discrepancies have the same cause and it is not a mystery: the scheme is first
order, and the next section measures what that costs.

![Mach field, with the sonic line drawn in white](../figures/mach_transonic.png)

Half a Mach number higher the picture is harder to miss. At $M_\infty = 0.85$,
$\alpha = 1^\circ$ — `figures/mach_shock.png` and `figures/cp_shock.png`, also
drawn by `examples/06_postprocess.py` — **both** surfaces go supersonic and each
pocket ends in its own shock, at $x/c = 0.865$ above and $0.66$ below, with a
peak Mach number of 1.412 and $c_\ell = 0.334$. It takes 33 500 iterations
against this case's 17 925, which is the usual price of a stronger shock on an
explicit march.

## External source unavailable

The original documentation quoted the external ranges above without a paper, dataset, mesh definition or version. The project owner does not currently have the source. Keep these as historical, unverified context, not as traceable validation or a claim of agreement to a stated number of digits.

The solver values remain reproducible with `examples/01_solve.py` and `examples/06_postprocess.py`. The current production solve gives a shock at `x/c=0.629`, peak Mach `1.3414`, lift `0.304415` and drag `0.032683`. The source limitation affects the comparison column, not whether the solver run converged.
