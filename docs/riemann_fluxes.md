# The approximate Riemann solvers, measured

Four interface fluxes were implemented and measured. **Three ship and one was
removed.** This is the record of why.

> **2026-10-15: AUSM⁺-up is back, and the first verdict on it was wrong.**
> It was removed for needing a tuned cutoff Mach number. It turned out the
> implementation had a coefficient 7.5 times too large, and the tuning was
> compensating for that. With the coefficient right it needs no tuning at all.
> → [AUSM⁺-up: the first verdict was about a bug](#ausm-up-the-first-verdict-was-about-a-bug)

It exists because the companion nozzle code measured the same four, reached a
conclusion, and **said explicitly that the conclusion should not be carried
across** — so it was re-measured here rather than inherited.

> **The two removed schemes are no longer in `src/`.** They are not archived
> here either, and deliberately so: verified copies already exist in the
> companion project's `legacy/riemann_solvers.py`, and this code's were checked
> **bit-identical** to them (see below). So everything on this page can be
> reproduced from that file plus this one, without a second copy of dead code
> to keep in step with the solver.

> **Status.** The headline question — does each flux work on this problem — is
> answered: 60 solves over four fluxes, five operating points and three meshes,
> plus a Mach sweep that brackets the failure. What is *not* answered is the
> mechanism. See [Open questions](#open-questions).

## Contents

- [Short answer](#short-answer)
- [The question, and why it had to be asked again](#the-question-and-why-it-had-to-be-asked-again)
- [First: are the implementations right?](#first-are-the-implementations-right)
- [The result](#the-result)
- [Do the ones that converge agree?](#do-the-ones-that-converge-agree)
- [AUSM⁺-up: the first verdict was about a bug](#ausm-up-the-first-verdict-was-about-a-bug)
- [SLAU2 reproduces the nozzle's limit cycle exactly](#slau2-reproduces-the-nozzles-limit-cycle-exactly)
- [What this does and does not establish](#what-this-does-and-does-not-establish)
- [Open questions](#open-questions)
- [Recommendation](#recommendation)

---

## Short answer

| flux | works? | | shipped? |
|---|---|---|---|
| **`roe`** | **yes** | 15 / 15 | **yes**, the default |
| **`hllc`** | **yes** | 15 / 15 | **yes** |
| **`ausmup`** | **yes transonic**, limit-cycles at low Mach | 12 / 12 transonic; fails at $M_\infty \le 0.5$ on fine meshes | **yes**, with the caveat below |
| `slau2` | **no**, above a threshold | 7 / 15 — converges below peak Mach ≈ 0.91, fails above | **removed** |

Roe and HLLC are unconditional and both ship. Roe stays the default.

**AUSM⁺-up ships again**, re-ported from
[`mrodrig6/approximate_riemann_solver`](https://github.com/mrodrig6/approximate_riemann_solver),
whose coefficients come from the group's EMS solver where they are benchmarked.
It has **no tunable constant**, converges on every transonic case tried, and
costs about 12% more drag than Roe at the same resolution — it is the more
dissipative of the three. It does **not** converge at low free-stream Mach
number on a fine mesh, where it settles into a limit cycle; see below for what
is measured and what is still a hypothesis.

SLAU2 fails on everything transonic and was removed outright.

| case | peak Mach | `roe` | `hllc` | `ausmup` | `slau2` |
|---|---|---|---|---|---|
| $M$ 0.50, $\alpha$ 0° | 0.60 | yes | yes | yes | **yes** |
| $M$ 0.50, $\alpha$ 2° | 0.66 | yes | yes | yes | **yes** |
| $M$ 0.72, $\alpha$ 0° (64×32) | 0.94 | yes | yes | yes | **yes** |
| $M$ 0.72, $\alpha$ 0° (finer) | 0.95 | yes | yes | yes | **no** |
| $M$ 0.80, $\alpha$ 1.25° | 1.34 | yes | yes | yes | **no** |
| $M$ 0.80, $\alpha$ 3.0° | 1.44 | yes | yes | yes | **no** |

## The question, and why it had to be asked again

The nozzle code found Roe-with-entropy-fix and HLLC converged, while AUSM⁺-up
and SLAU2 reached nearly the right answer and then sat in a bounded limit cycle
whose residual never decayed. Six mechanisms were tested there and all six were
refuted; the cause was never identified.

Its `legacy/README.md` then said why that result might not generalise:

> The airfoil problem has a **shock on the body surface**. This nozzle solver
> does not… That difference is not a detail — it inverts which flux is the right
> choice… **So do not carry this project's conclusion across.** […] Carry the
> measurement and the reasoning; re-measure the conclusion.

The reasoning behind that expectation is sound and worth restating, because it
is what makes the result below interesting. Kitamura's 2016 assessment scores
twelve fluxes on the standard 1.5D carbuncle test — robustness at a **captured
strong shock**:

| flux | shock-robustness score | captured shock |
|---|---|---|
| **Roe (entropy fix)** | **0** | thin |
| Roe | 8 | thin |
| **HLLC** | **8** | thin |
| AUSM⁺-up | 16 | thin |
| van Leer FVS | 20 | broad |
| **SLAU2** | **20** | broad |

In the nozzle the ranking came out exactly inverted: the two that score worst
there converged, and the one that scores best did not. Since the nozzle's domain
is shock-free by construction, the property being scored was never exercised —
so on a problem that *does* have a surface shock, the inversion might reasonably
be expected to run the other way.

**It does not.** This code has the shock, and the ranking is the same as the
nozzle's.

## First: are the implementations right?

Before any conclusion about a *scheme*, the port has to be cleared of being the
problem. A subtly wrong SLAU2 that fails to converge would look exactly like a
scheme that fails to converge, and the conclusion would be worthless.

Two independent checks, and they are the reason the rest of this page can be
read as being about the schemes.

**Against the nozzle's reference implementations.** The companion project's
`legacy/riemann_solvers.py` carries all four fluxes, verified there. Run on 200
random state pairs spanning Mach 0.018 to 8.86, including 289 supersonic faces:

| flux | max relative difference | verdict |
|---|---|---|
| `roe` | 1.2e−16 | identical to round-off |
| `hllc` | 2.5e−16 | identical to round-off |
| `ausmup` | **0** | bit-identical |
| `slau2` | **0** | bit-identical |

**Against the properties a numerical flux must have.** Every solver is checked
for consistency $\hat{F}(U, U, \hat{n}) = F(U)\cdot\hat{n}$, conservation
$\hat{F}(L, R, \hat{n}) = -\hat{F}(R, L, -\hat{n})$, full upwinding in the
supersonic limit, and finiteness at a near-vacuum — all to $10^{-10}$ or better,
in `tests/test_physics.py`. The AUSM split polynomials are checked to sum to
the identities that make the scheme consistent.

**So the failures below are properties of the schemes**, reproduced
independently in two codebases, and not artefacts of this port.

## The result

NACA 0012, $M_\infty = 0.8$, $\alpha = 1.25^\circ$ — a shock at 63% chord on the
upper surface and a weaker one at 38% on the lower. 96 × 40 O-grid, 7 680 cells,
far field 15 chords, CFL 0.9, forward Euler, converged means a scaled residual
below $10^{-6}$.

| flux | converged | $c_\ell$ | $c_d$ | shock $x/c$ | iterations |
|---|---|---|---|---|---|
| **`roe`** | **yes** | +0.2831 | +0.0402 | 0.629 | 9 042 |
| **`hllc`** | **yes** | +0.2743 | +0.0402 | 0.629 | 8 989 |
| `ausmup` (as then implemented) | **no** — diverged | — | — | — | 195 |
| `slau2` | **no** — limit cycle | +0.1957 | +0.0423 | 0.533 | 30 000 (stopped) |

![Residual history by flux](figures/flux_convergence.png)

Roe and HLLC agree on the shock position to within one cell and on $c_\ell$ to
3%, which is the right order for two different schemes at first order. The two
AUSM-family schemes fail, in two different ways.

> **The `ausmup` row is of a broken implementation**, kept because the rest of
> the table and the figure come from the same run. The scheme as it now stands
> converges on this case in 8 275 iterations with
> $c_\ell = +0.2703$, $c_d = +0.0477$ — see
> [below](#ausm-up-the-first-verdict-was-about-a-bug).

## Do the ones that converge agree?

Self-consistency is not "all four give the same answer" — two of them do not
give an answer at all. It is whether the ones that **do** converge agree, and
whether that agreement **improves as the mesh is refined**. If it does, the
spread is discretisation error and the schemes are consistent with each other;
if it does not, something is wrong with one of them.

Roe, HLLC and AUSM⁺-up at $M_{co} = 0.4$, all converged to $10^{-6}$:

| | mesh | cells | `roe` | `hllc` | `ausmup` | $c_\ell$ spread | $c_d$ spread | shock spread |
|---|---|---|---|---|---|---|---|---|
| **subcritical** | 64 × 32 | 4 096 | 0.2406 | 0.2365 | 0.2350 | 2.37% | 17.9% | — |
| $M = 0.5$ | 96 × 40 | 7 680 | 0.2464 | 0.2429 | 0.2421 | 1.74% | 17.1% | — |
| $\alpha = 2^\circ$ | 128 × 48 | 12 288 | 0.2503 | 0.2480 | 0.2478 | **1.03%** | 16.7% | — |
| **transonic** | 64 × 32 | 4 096 | 0.2713 | 0.2624 | 0.2597 | 4.38% | 15.3% | 0.000 |
| $M = 0.8$ | 96 × 40 | 7 680 | 0.2835 | 0.2750 | 0.2713 | 4.42% | 12.0% | 0.032 |
| $\alpha = 1.25^\circ$ | 128 × 48 | 12 288 | 0.2916 | 0.2829 | 0.2808 | **3.78%** | 10.5% | 0.000 |

Three things to read off it.

**Lift agreement tightens with refinement** — 2.4% to 1.0% subcritical, 4.4% to
3.8% transonic — and all three march toward the same limit. Subcritically that
limit is checkable: thin-airfoil theory with the Prandtl–Glauert correction
gives 0.2532, and Roe reaches 0.2503 on the finest mesh. The spread is
discretisation error, behaving as discretisation error should.

**The shock lands in the same place.** All three agree to within one cell on two
of the three meshes, and to 0.03 chord on the third. Three schemes with
completely different derivations putting a shock in the same cell is a strong
statement that none of them is wrong.

**Drag agrees far less well, and that is expected rather than alarming.** At
first order most of the computed $c_d$ *is* the scheme's numerical dissipation
(see [`verification.md`](verification.md#spurious-drag-and-what-first-order-costs)),
and the three schemes differ precisely in how they dissipate. So drag is the
quantity where they must differ most. It is also shrinking — 17.9% to 16.7%
subcritical, 15.3% to 10.5% transonic — which is what it should do if the
difference is dissipation that is itself vanishing with $h$.

## AUSM⁺-up: the first verdict was about a bug

This section used to report that AUSM⁺-up converged only inside a narrow window
of its cutoff Mach number $M_{co}$ — 0.30 to 0.45 — and diverged at the
$M_{co} \sim M_\infty$ its author recommends. That measurement was real. The
conclusion drawn from it was wrong, and the reason is worth keeping.

### What was actually wrong

The scheme has two diffusion coefficients. The removed implementation had

```python
kp, ku, sigma = 0.25, 0.75, 1.0
p_half = ... - ku * p5p * p5m * (rhoL + rhoR) * (fa * a_half) * (vnR - vnL)
```

against the benchmarked values — $K_u = 0.2$ applied to
$\rho_{1/2} = (\rho_L + \rho_R)/2$. Writing 0.75 against the *sum* rather than
0.2 against the *mean* makes that term

$$\frac{0.75 \times 2}{0.2} = 7.5\ \text{times too large.}$$

$K_u$ sets the velocity diffusion in the interface pressure. Too large, it stops
being diffusive: the companion repository measures it amplifying round-off by
about 1.5 per step on a plain moving contact until the pressure error reaches
50%. Sweeping $M_{co}$ was partly compensating for that — $f_a$ multiplies the
same term — which is why a window existed at all, and why it sat where no
physical argument put it.

**Every consistency identity passed throughout.** $M^{+}_{(4)} + M^{-}_{(4)} = M$
and $P^{+}_{(5)} + P^{-}_{(5)} = 1$ hold for *any* $K_p$ and $K_u$, because they
constrain the split polynomials and the diffusion terms are not part of them.
Identity tests cannot catch a wrong diffusion coefficient; only a physical
invariant can. `test_ausm_carries_a_moving_contact_without_touching_the_pressure`
is now that guard.

### The current implementation

Ported from [`mrodrig6/approximate_riemann_solver`](https://github.com/mrodrig6/approximate_riemann_solver),
`riemann/fluxes.py`, whose coefficients come from EMS
(`src/riemann_solvers/ausm6.cpp`) where they are benchmarked:

| | value | tunable? |
|---|---|---|
| $\beta$ | 1/8 | no — fixed by smoothness at $\lvert M\rvert = 1$ |
| $\alpha$ | 3/16 | no — likewise |
| $K_p$ | 0.4 | yes, benchmarked |
| $K_u$ | 0.2, against $\rho_{1/2}$ | yes, benchmarked |
| $a_{1/2}$ | $(a_L + a_R)/2$ | — arithmetic mean, not Liou's critical speed |
| $f_a$ | **absent** | — no all-speed scaling, hence no $M_{co}$ |

There is no cutoff Mach number to get wrong. The objection that removed it is
gone.

### What it does on the airfoil

36 solves — four operating points, three meshes, three fluxes:

| case | peak Mach | `roe` | `hllc` | `ausmup` |
|---|---|---|---|---|
| $M$ 0.80, $\alpha$ 1.25° | 1.29–1.34 | 3/3 | 3/3 | **3/3** |
| $M$ 0.85, $\alpha$ 1.0° | 1.38–1.41 | 3/3 | 3/3 | **3/3** |
| $M$ 0.80, $\alpha$ 3.0° | 1.41–1.46 | 3/3 | 3/3 | **3/3** |
| $M$ 0.50, $\alpha$ 2.0° | 0.65–0.68 | 3/3 | 3/3 | **2/3** |

Everything transonic converges. Loads agree with Roe to about 1% on lift; drag
runs consistently higher, by 12–17% on the coarsest meshes and 11% at 144×56 —
AUSM⁺-up is the most dissipative of the three here, which is the honest cost of
its robustness.

| 144×56, $M$ 0.8, $\alpha$ 1.25° | $c_\ell$ | $c_d$ |
|---|---|---|
| `roe` | 0.28936 | 0.03782 |
| `hllc` | 0.28229 | 0.03773 |
| `ausmup` | 0.28421 | 0.04240 |

### Where it fails, and what is and is not established

The one failure is **not** transonic. It is the *subcritical* case on the
*finest* mesh — $M_\infty = 0.5$, $\alpha = 2°$, 144×56, peak Mach 0.68 — and it
is a limit cycle, not a divergence:

| iteration | scaled residual |
|---|---|
| 75 | $1.18 \times 10^{-1}$ ← the minimum |
| 18 000 | $2.715 \times 10^{-1}$ |
| 100 000 | $2.715 \times 10^{-1}$ |
| 200 000 | $2.715 \times 10^{-1}$ |

It oscillates between $2.7149 \times 10^{-1}$ and $2.7168 \times 10^{-1}$
indefinitely. The lift it sits at, 0.2489, is within 1.5% of Roe's converged
0.24525, so the *state* is close to right; it simply will not settle.

Dropping to $M_\infty = 0.3$ makes it worse and earlier: it fails at 96×40
(residual $4.4 \times 10^{-2}$) and at 144×56 ($9.0 \times 10^{-1}$), where Roe
converges on both.

**Established:** the failure is at low free-stream Mach number, gets worse as
the mesh is refined, and is a limit cycle rather than a blow-up.

**Hypothesis, not established:** this is the missing $f_a$. The all-speed
scaling is precisely the part of Liou 2006 that keeps the pressure–velocity
coupling correct as $M \to 0$, and it is the part EMS omits — reasonably, since
EMS and the 1-D reference both run at $O(1)$ Mach where it does not matter. An
airfoil has a stagnation point, so there is always a low-Mach region for it to
bite. **This has not been tested.** Restoring $f_a$ with the correct $K_u$ is
the obvious experiment and it has not been run; it would also reintroduce
$M_{co}$, which is a design decision rather than a bug fix, and one for the
owner of the scheme to make.

### What this means for using it

Use `ausmup` transonically, where it is solid and gives a genuinely independent
second opinion — it shares no machinery at all with Roe or HLLC, which decompose
the jump into waves where this splits the flux. Below about $M_\infty = 0.6$ on
a fine mesh, expect it to stall, and use Roe or HLLC.

## SLAU2 reproduces the nozzle's limit cycle exactly

SLAU2 is the interesting one, because it has **no tunable constant at all** —
every switch is built from the states themselves. There is nothing to get wrong,
which is precisely why the nozzle carried it.

It behaves here exactly as it did there: it reaches a plausible answer and then
stops improving. $c_\ell = 0.196$ against Roe's 0.283, the shock at 0.533 against
0.629, an entropy error of the same order as Roe's — close enough to be
recognisable, far enough to be useless — and a residual that sits near $10^{-1}$
for 30 000 iterations without decaying.

That is the same signature, on a domain with the shock the nozzle lacked. The
hypothesis that the shock was the missing ingredient is **not supported**.

### Where the residual lives, and what sets the threshold

The nozzle asked where the residual lives and found the failure *distributed* —
its ten worst cells carried 8.3% of the total, spread through the supersonic
section rather than piled at one feature. Repeating it here:

| region | share of cells | share of residual |
|---|---|---|
| Mach > 1.1 | 4.3% | **60.8%** |
| Mach 0.9 – 1.1 | 12.1% | 36.7% |
| Mach < 0.9 | 83.6% | **2.6%** |
| wall-adjacent cells | 1.2% | 5.6% |
| within 0.06 chord of the shock | 4.8% | 4.3% |

**It is neither the shock nor the wall.** The worst cells sit at
$x/c = 0.26$–$0.41$ just above the upper surface at Mach 1.25–1.29, upstream of
the shock. The ten worst carry 5.6%, matching the nozzle's 8.3%, so it is not a
handful of bad cells; but the worst 10% carry 99.9% against Roe's 68.6%, so it
is strongly regional.

Reading that table, it is tempting to conclude "the supersonic pocket". **That
is wrong, and a sweep in Mach number shows it.** At zero incidence on a fixed
96 × 40 mesh:

| $M_\infty$ | peak Mach | supersonic cells | `slau2` | `hllc` |
|---|---|---|---|---|
| 0.50 | 0.600 | 0 | yes | yes |
| 0.65 | 0.815 | 0 | yes | yes |
| 0.70 | **0.904** | **0** | **yes** | yes |
| 0.71 | **0.924** | **0** | **no** | yes |
| 0.72 | 0.946 | 0 | no | yes |
| 0.76 | 1.061 | 240 | no | yes |

The threshold sits between a peak Mach of 0.904 and 0.924, and **there is not a
single supersonic cell on either side of it**. SLAU2 fails on a flow that is
subsonic everywhere. The same point shows up in the mesh study above: at
$M_\infty = 0.72$ it converges on 64 × 32 (peak Mach 0.9367) and fails on
96 × 40 (0.9457) — the same operating point, the difference being only how
nearly sonic the finer mesh resolves the flow to be.

So the trigger is the **near-sonic region**, not a shock and not supersonic
flow. That the residual then concentrates where $M > 0.9$ is consistent with it;
it is not evidence for the pocket, because the failure begins before any pocket
exists.

The limit cycle is also far more violent here than in the nozzle: the state
keeps moving with a saturated amplitude of $1.7\times10^{-1}$ relative, against
the nozzle's $8\times10^{-4}$. Measured over 400 further steps after the march
was abandoned, the amplitude in the last 50 equals the amplitude in the first
50 to four digits — saturated, not decaying.

That the failure concentrates in the *supersonic* region, and that it is worse
on a case with a supersonic pocket than on one without, is the most specific
thing either project has about the mechanism. It is still not a demonstration,
and it is written down as [open question 3](#open-questions) rather than a
conclusion.

## What this does and does not establish

**Established.**

* Roe with the Harten–Hyman entropy fix and HLLC both converge on this problem
  and agree with each other.
* AUSM⁺-up, with the benchmarked coefficients and no tunable constant,
  converges on every transonic case tried and limit-cycles at low free-stream
  Mach number on a fine mesh.
* The earlier finding that AUSM⁺-up needs a tuned $M_{co}$ **does not hold**:
  it was an artefact of a $K_u$ 7.5 times too large.
* SLAU2 does not converge here, with no constant available to blame.
* The nozzle's ranking **transfers** to a problem with a surface shock, which
  that project explicitly expected it might not.

**Not established.**

* *Why*. The mechanism behind the AUSM-family limit cycle is no better
  understood than it was in the nozzle, and nothing here was designed to find
  it.
* **Whether the missing $f_a$ explains AUSM⁺-up's low-Mach limit cycle.** The
  hypothesis fits — $f_a$ is exactly the all-speed term, it is absent here, and
  the failure is at low Mach — but the experiment has not been run.
* Whether AUSM⁺-up's low-Mach threshold moves with the CFL or the section. Two
  Mach numbers, three meshes, one section.
* Whether a second-order reconstruction changes any of it. Everything here is
  first order; the AUSM family's published advantages are largely about
  *shock capturing*, which first order smears anyway.

**No remedy has been added to the solver.** Where a scheme fails it fails in
the open, with no extra damping, no scheme-specific CFL and no smoothing to
paper over it — which is what makes this page checkable rather than a story.
AUSM⁺-up ships with its low-Mach limit cycle intact and documented, not
patched.

## Open questions

For increment 5, written down now so the null results are not lost:

1. **Is the high-$M_{co}$ failure really the stagnation point?** Testable: run
   AUSM⁺-up at $M_{co} = 0.8$ on a case with no stagnation region in the
   interior, or compare the residual's spatial distribution against the
   low-Mach cells.
2. **Does the window depend on the mesh?** If 0.30–0.45 is really set by the
   smallest resolved Mach number, refining the leading edge should move it.
3. **Why near-sonic?** The threshold is a peak Mach of 0.91 and the flow there
   is subsonic everywhere, so nothing about shock capturing can explain it.
   SLAU2's switches — $g$, $\chi = (1-\hat M)^2$, the $\dot m$ split — are all
   functions of Mach number, and $\chi$ in particular goes to zero as
   $\hat M \to 1$, removing the pressure-difference term from the mass flux
   exactly there. That is a candidate with an obvious test: hold $\chi$ at a
   floor and see whether the threshold moves. It has not been tried.
4. **Does second order change the ranking?** The AUSM family's case rests on
   shock capturing. Increment 3 makes this answerable.

## Recommendation

**Use `roe` (the default) or `hllc`.** Both converge on every case tried at
every Mach number tried, they agree with each other, and HLLC has no
entropy-fix parameter to choose.

**Use `ausmup` transonically, as a second opinion.** It converges on every
transonic case tried, needs no tuning, and is built from a completely different
idea — a flux split rather than a wave decomposition — so when it agrees with
Roe that agreement means something. It costs about 12% more drag at the same
resolution, and it stalls below roughly $M_\infty = 0.6$ on a fine mesh. Check
that the run converged.

**`slau2` is not available.** It was removed: usable below a peak Mach of about
0.9 and not above, which rules out every transonic case on this geometry.

Two things are new against the nozzle's conclusion. The nozzle's ranking
**transfers** to a problem with a surface shock, which that project explicitly
doubted it would. And the failure that does remain is **not about the shock at
all** — SLAU2 begins failing in flow that is subsonic everywhere, and
AUSM⁺-up's limit cycle is at *low* Mach number, not high. Neither project would
have predicted either.
