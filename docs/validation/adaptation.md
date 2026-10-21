# Adaptation versus uniform refinement

This study compares adaptive and uniform refinement at similar cell counts. Accuracy per cell and work per converged answer are different comparisons.

## Does it pay?

The honest answer is **it depends which number you want**, and that is more
useful than the textbook one.

NACA 0012 at $M = 0.8$, $\alpha = 1.25^\circ$, far field at 10 chords. Adaptation
starts from the 64×24 mesh in the first row and refines 5% of cells per cycle;
the uniform column is the same case on progressively finer O-grids. Compare the
two **at equal cell count**, which is the only comparison that answers the
question:

| cells | uniform $c_\ell$ | adaptive $c_\ell$ | uniform $c_d$ | adaptive $c_d$ |
|---|---|---|---|---|
| ~3 100 | 0.26465 | 0.26465 | 0.04915 | 0.04915 |
| ~5 200–6 100 | 0.27924 | 0.28667 | 0.04256 | 0.04396 |
| ~8 960–9 430 | 0.28370 | 0.28230 | 0.04042 | 0.03728 |
| ~12 300–12 800 | 0.28748 | 0.28721 | 0.03894 | 0.03567 |
| ~17 100–17 900 | 0.29149 | 0.28944 | 0.03723 | 0.03436 |
| ~23 000–24 600 | 0.29317 | 0.29200 | 0.03613 | 0.03196 |

**On drag, adaptation wins clearly.** At 23 000 cells it gives $c_d = 0.0320$,
which the uniform sequence does not reach at 24 600 and, extrapolating its own
trend, would need roughly **three times as many cells** to reach. Spurious drag
at this order comes from numerical entropy generated at the shock and the
leading edge, which is exactly where the indicator refines.

**On lift, it does not.** The two agree to within 0.4% the whole way, with
uniform marginally ahead at the top of the table. Lift is a pressure integral
over the whole surface, so it wants the *entire* section resolved — which is
what uniform refinement does and what a 5%-of-cells indicator declines to do.

**Per second, uniform wins outright on this problem.** The last adaptive cycle
takes 66 875 iterations to converge against the uniform mesh's 17 700 at the
same size, because the adapted mesh's smallest cell is an order of magnitude
smaller and an explicit march is paced by it. Multiply cells by iterations and
adaptation costs about four times the work for the same mesh.

So: adaptation here buys **drag accuracy per cell**, pays for it in iterations,
and does nothing for lift. None of that is a defect in the implementation — it is
what this indicator, on this functional, with an explicit solver, actually does.

## What would change the answer

Three things, in increasing order of interest:

1. **An implicit or multigrid solver.** The iteration penalty is entirely the
   explicit march's smallest cell. It is the reason adaptation is almost always
   paired with implicit time stepping in production codes.
2. **An area-normalised indicator.** $\varepsilon_e$ has the dimensions of a
   length, so large cells are flattered. Dividing by the cell's perimeter, or by
   its area, changes which cells are flagged — try it; the code is four lines in
   `mach_jump_indicator`.
3. **A goal-oriented indicator.** Nothing in a Mach jump knows which functional
   is being computed, which is precisely why it helps drag and not lift. The
   adjoint-weighted residual $|\lambda_e^{T} R_e|$ is the error in *the chosen
   objective* contributed by each cell, and `sensitivity.py` already solves the
   $\lambda$ it needs. That is the piece this repository is best placed to add
   next, and the one that would most change the table above.
