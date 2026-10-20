# Lab guide

Seven exercises that use the solver rather than write it. Each one is a question
with a measurable answer, a command that produces the evidence, and the number
this code actually gives — so you can tell a result from a mistake.

Everything runs on a laptop. The longest exercise is about ten minutes of
compute; most are under one.

## Contents

- [Before you start](#before-you-start)
- [Lab 1 — Does the mesh decide the answer?](#lab-1--does-the-mesh-decide-the-answer)
- [Lab 2 — The lift curve, against theory](#lab-2--the-lift-curve-against-theory)
- [Lab 3 — The transonic case](#lab-3--the-transonic-case)
- [Lab 4 — Drag that is not there](#lab-4--drag-that-is-not-there)
- [Lab 5 — Choosing a flux](#lab-5--choosing-a-flux)
- [Lab 6 — Spending cells where they matter](#lab-6--spending-cells-where-they-matter)
- [Lab 7 — Design: gradients, and what to do with one](#lab-7--design-gradients-and-what-to-do-with-one)
- [For the instructor](#for-the-instructor)

---

## Before you start

```bash
git clone https://github.com/mrodrig6/2D_airfoil.git
cd 2D_airfoil
./fv2d.sh check          # what you have; NO next to something is fine for labs 1-6
./fv2d.sh install        # adds Numba, JAX, matplotlib, SciPy
./fv2d.sh verify --fast  # 15 seconds; everything should pass before you trust a result
```

Labs 1–6 need NumPy alone, though Numba makes them roughly twenty times faster.
Lab 7 needs JAX, and its last part needs SciPy.

Two things to have read, or to read as you go:

* [`theory.md`](theory.md) — the equations, the non-dimensionalisation, the
  scheme. In particular: **there are no units here.**
  $\rho_\infty = |v_\infty| = c = 1$, so dynamic pressure is exactly $1/2$
  and the only meaningful outputs are coefficients.
* The [assumptions table](../README.md#assumptions-and-constraints) in the
  README. Every answer below is conditional on those eight lines; most of the
  surprises in these labs are one of them making itself felt.

Each lab ends with **hand in**: the smallest set of numbers or figures that
shows you did the experiment and understood it.

---

## Lab 1 — Does the mesh decide the answer?

**The question.** A computed lift coefficient is a property of the flow *and* of
the mesh. How much of what you are about to measure is the airfoil, and how much
is the grid?

```bash
./fv2d.sh solve naca=0012 ns=64  nr=24 R=10 h1=0.02 mach=0.5 alpha=2
./fv2d.sh solve naca=0012 ns=128 nr=48 R=10 h1=0.02 mach=0.5 alpha=2
./fv2d.sh solve naca=0012 ns=192 nr=64 R=10 h1=0.02 mach=0.5 alpha=2
```

Then move the outer boundary instead of the resolution:

```bash
./fv2d.sh solve naca=0012 ns=128 nr=48 R=8  h1=0.02 mach=0.5 alpha=2
./fv2d.sh solve naca=0012 ns=128 nr=48 R=20 h1=0.02 mach=0.5 alpha=2
./fv2d.sh solve naca=0012 ns=128 nr=48 R=50 h1=0.02 mach=0.5 alpha=2
```

**What to look for.** Two different convergences. Refining the mesh changes the
answer because the *scheme* is approximate; moving the far field changes it
because the *domain* is. Only one of them is about accuracy in the usual sense.

**Reference.** Lift moves about 1.7% between $R = 8$ and $R = 50$; at $R = 20$
it is within 0.05% of $R = 50$, which is why 25 is the default. Note that the
far-field error at $R = 8$ is **larger** than the mesh error between the two
finest grids — a tempting mesh study on a short domain measures mostly the
boundary.

**Hand in.** The two convergence tables, and one sentence on which error
dominates at $R = 8$. → [`verification.md`](verification.md#how-far-away-the-far-field-has-to-be)

---

## Lab 2 — The lift curve, against theory

**The question.** Thin-aerofoil theory says $c_\ell = 2\pi\alpha$, and the
Prandtl–Glauert factor stretches that to $2\pi/\sqrt{1 - M^2}$ in subsonic
compressible flow. Does an Euler solver agree, and where does it stop agreeing?

```bash
./fv2d.sh sweep naca=2412 ns=128 nr=48 mach=0.5 alphas=-4:8:2 figure=polar.png
```

**What to look for.** Fit the slope over $|\alpha| \le 4^\circ$ only — the point
of the exercise is to find where the linear theory stops. Read the zero-lift
incidence off the curve and ask what sets it.

**Reference.** The measured slope is 7.12 per radian against a theoretical 7.26
— 98% of it, the shortfall being first-order mesh error. Zero lift falls at
$-2.09^\circ$, which is camber: a symmetric NACA 0012 gives exactly zero lift at
zero incidence, and the test suite asserts it does.

**Push it.** Repeat at $M = 0.3$ and $M = 0.7$. Does the measured slope follow
$1/\sqrt{1 - M^2}$? Where does it stop, and is that the theory failing or the
mesh?

**Hand in.** The polar figure, the fitted slope at each Mach number, and the
Mach number at which Prandtl–Glauert visibly breaks down.

---

## Lab 3 — The transonic case

**The question.** At $M_\infty = 0.8$ the free stream is subsonic everywhere and
the flow over the upper surface is not. Where does it come back?

```bash
./fv2d.sh solve @transonic cp=cp.png figure=mach.png
```

**What to look for.** The $-c_p$ plot carries the sonic line $c_p^*$: everything
above it is supersonic, and where the upper-surface curve crosses back through
it is the shock. Compare that to the `shock_location` the report prints, which
finds the same thing from the Mach field.

**Reference.** On the 192×64 mesh: shock at $x/c = 0.629$ against a published
0.62–0.65, peak Mach 1.341 against about 1.35, $c_\ell = 0.304$ against about
0.35.

**The interesting part.** The shock position is right and the lift is 13% low.
Why does the same calculation get one of them and miss the other? (Lab 4.)

**Push it.** Find the **critical Mach number** — the free-stream Mach number at
which a sonic point first appears — by bisecting on `mach` at $\alpha = 0$ and
watching the peak Mach in the report. Then sweep upward and plot $c_d$ against
$M_\infty$ to find **drag divergence**.

**Hand in.** The $c_p$ figure with the shock marked, the measured shock position,
and your critical Mach number with the two solves that bracket it.

---

## Lab 4 — Drag that is not there

**The question.** At $\alpha = 0$ a symmetric section in subsonic inviscid flow
has exactly zero drag — d'Alembert's paradox, and it is a theorem, not an
approximation. What does this code give?

```bash
for nr in 24 32 48 64; do
    ./fv2d.sh solve naca=0012 ns=$((nr * 3)) nr=$nr mach=0.5 alpha=0
done
```

**What to look for.** A drag coefficient that is not zero, and that falls as the
mesh is refined. Fit $c_d \propto h^{p}$ to find $p$, taking $h \propto
N^{-1/2}$ in two dimensions.

**Reference.** 0.0290 down to 0.0131 over four meshes, with a fitted order of
**0.83** — first order, as the piecewise-constant reconstruction promises. For
scale: the published transonic wave drag for Lab 3 is 0.022, which is *smaller*
than the spurious drag on the coarsest mesh here. This is why the README says
drag from this code is not usable, and why the roadmap treats a higher-order
reconstruction as the single most valuable thing left undone.

**Push it.** Entropy is conserved along streamlines in smooth isentropic flow,
so the `entropy error` in the report is a pure measure of numerical dissipation.
Does it fall at the same rate as the spurious drag? Should it?

**Hand in.** The table, the fitted order, and an estimate — by extrapolating
your fit — of how many cells a drag accurate to 0.001 would need.

---

## Lab 5 — Choosing a flux

**The question.** The upwinding in a finite-volume scheme *is* the approximate
Riemann solver. Does the choice matter?

```bash
./fv2d.sh solve @transonic flux=roe    convergence=roe.png
./fv2d.sh solve @transonic flux=hllc   convergence=hllc.png
./fv2d.sh solve @transonic flux=ausmup convergence=ausmup.png
./fv2d.sh solve @transonic flux=roe    entropy_fix=0
```

**What to look for.** Roe and HLLC agree on the loads to about 2% and differ in
how they get there. Then the third run: Roe's linearisation admits an expansion
shock, a solution that satisfies the conservation laws and violates the second
law, and the Harten–Hyman entropy fix is what rules it out. Setting it to zero
is a *physics* failure, not a numerical one, and it shows up near the sonic line.

**Reference.** Roe and HLLC converge on every transonic case in the study, and
so does `ausmup` — try it as a third run. SLAU2 was implemented, measured and
removed: it stalls from a peak Mach of 0.924 with **zero supersonic cells**,
which disproved the explanation we first offered for it.

**Worth reading alongside this lab**, because it is the better lesson:
[`riemann_fluxes.md`](riemann_fluxes.md) carries two retractions, left visible
on purpose. One is that SLAU2 explanation. The other is larger — AUSM⁺-up was
removed on a measurement that turned out to be of a buggy implementation, and
the bug was a kind the test suite could not catch, because consistency
identities constrain the split polynomials and say nothing about the diffusion
coefficients. Ask yourself what test *would* have caught it before reading the
answer.

**Push it further.** Run `flux=ausmup` at $M_\infty = 0.3$ on a 144×56 mesh and
watch the residual. It does not diverge — it stalls at a fixed level and stays
there forever. What is the difference, and which one should worry you more?

**Hand in.** The two convergence histories on one pair of axes, the loads from
each flux, and what changed when you turned the entropy fix off.

---

## Lab 6 — Spending cells where they matter

**The question.** A uniformly finer mesh refines the far field, where nothing is
happening, as hard as it refines the shock. Is a cell spent where the error is
worth more than a cell spent everywhere?

```bash
./fv2d.sh adapt @transonic cycles=6 figure=adapted.png history=adapt.png
python examples/05_adapt.py          # the same thing, against uniform refinement
```

**What to look for.** First the mesh figure: where did the refinement go? The
indicator is the jump in Mach number across each face — `legacy/post_calc.m`'s,
unchanged — plus a wall term that measures how far the solution is from
satisfying the wall condition. Nobody told it where the shock was.

Then compare the two tables **at equal cell count**, not at equal row. That is
the only comparison that answers the question.

**Push it.** Change `fraction` from 0.05 to 0.20. Fewer, bigger cycles reach a
given cell count sooner — do they reach the same answer? Then try `cycles=10`
and watch the smallest angle in the mesh: this implementation regenerates its
green closure from a tree of red subdivisions every cycle *specifically* so that
repeated refinement cannot produce slivers, and you should be able to confirm
that the worst angle stops falling.

**Hand in.** The adapted mesh, the two tables, and a statement of the cell count
at which adaptation becomes worth its complexity on this case — including, if
that is what you find, that it does not.

---

## Lab 7 — Design: gradients, and what to do with one

**The question.** You can measure $dc_\ell/d\alpha$ by solving twice and
subtracting. Why would anyone build an adjoint?

```bash
./fv2d.sh sensitivity @subcritical naca=2412 ns=64 nr=32 R=10 h1=0.02 \
    tolerance=1e-10 max_iterations=60000 check=yes
python examples/03_sensitivity.py
python examples/04_optimise.py
```

**What to look for.** First that the adjoint and the finite difference agree —
they do, to seven or eight significant figures, and a gradient that cannot
survive that check is wrong. Then the cost: one adjoint solve gives *every*
derivative, while finite differences cost two solves each.

**Reference.** $dc_\ell/d\alpha = +1.18547016 \times 10^{-1}$ per degree against
$+1.18547020 \times 10^{-1}$ by central difference. On a 4096-cell mesh the
adjoint is **not** the cheaper route for five variables — it breaks even at about
nine — because the forward solver is compiled and the adjoint march is not. The
scaling is what you are buying, not the constant.

**The part to think hardest about.** Run the gradient check at the *transonic*
point instead:

```bash
./fv2d.sh sensitivity @transonic naca=2412 ns=64 nr=32 R=10 h1=0.02 \
    tolerance=1e-10 max_iterations=60000 check=yes
```

The two columns now disagree, and the adjoint is not the one that is wrong. The
shock sits in a cell; perturb the design slightly and it jumps to the next cell
rather than sliding. The discrete objective is a step function at that scale, so
a small finite-difference step measures the step and not a derivative.

**Hand in.** The subcritical comparison table, the transonic one, and an
explanation of which number you would give an optimiser and why.

---

## For the instructor

**There is no CI and no autograder.** `./fv2d.sh verify` is the check, it runs on
the student's own machine in about a minute, and it either passes or names what
failed. A student who changes the solver and breaks free-stream preservation
finds out immediately and locally.

**Every reference number above is reproducible** by the command printed with it,
on the committed code. Where a number came out differently from what was
expected, the documents say so and say what was measured instead — the SLAU2
retraction in [`riemann_fluxes.md`](riemann_fluxes.md) is the clearest example,
and is worth assigning on its own as a short reading about what to do when a
plausible explanation is refuted by the data.

**The MATLAB is in [`legacy/`](../legacy/)** and still readable. Asking students
to find the force integral in `legacy/post_calc.m` and compare it to
`src/post/postprocess.py` is a good exercise in its own right: the legacy formula uses
`sign(n_x)` and `sign(n_y)` in place of the normal components, which is a
projection onto the axes rather than an integral of the traction, and
[`verification.md`](verification.md#the-legacy-force-formula-measured) measures
what it costs.

**Suggested orders.** Labs 1–4 are a self-contained numerical-methods sequence.
Labs 3, 5 and 6 are a transonic-aerodynamics sequence. Lab 7 stands alone and is
the one to assign if the course is about design.
