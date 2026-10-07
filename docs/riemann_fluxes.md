# The approximate Riemann solvers, measured

Four interface fluxes were implemented and measured. **Two ship and two were
removed.** This is the record of why.

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
- [AUSM⁺-up has a narrow window, and it is not where Liou says](#ausm-up-has-a-narrow-window-and-it-is-not-where-liou-says)
- [SLAU2 reproduces the nozzle's limit cycle exactly](#slau2-reproduces-the-nozzles-limit-cycle-exactly)
- [What this does and does not establish](#what-this-does-and-does-not-establish)
- [Open questions](#open-questions)
- [Recommendation](#recommendation)

---

## Short answer

Four fluxes, five operating points, three meshes — 60 converged-or-not solves on
a NACA 0012.

| flux | works? | | shipped? |
|---|---|---|---|
| **`roe`** | **yes** | 15 / 15 | **yes**, the default |
| **`hllc`** | **yes** | 15 / 15 | **yes** |
| `ausmup` | yes, conditionally | 15 / 15 at $M_{co} = 0.4$; **diverges** at Liou's recommended $M_{co}\sim M_\infty$ | **removed** |
| `slau2` | **no**, above a threshold | 7 / 15 — converges below peak Mach ≈ 0.91, fails above | **removed** |

Roe and HLLC are unconditional and both ship.

AUSM⁺-up does converge on every case tried — but only inside a narrow window in
its one tunable constant, and **not where its author's own guidance puts that
window**. A scheme that works only if you already know the answer to a question
the literature gets wrong for this geometry is a trap in a teaching code, so it
was removed rather than shipped with a warning.

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
| `ausmup` | **no** — diverged | — | — | — | 195 |
| `slau2` | **no** — limit cycle | +0.1957 | +0.0423 | 0.533 | 30 000 (stopped) |

![Residual history by flux](figures/flux_convergence.png)

Roe and HLLC agree on the shock position to within one cell and on $c_\ell$ to
3%, which is the right order for two different schemes at first order. The two
AUSM-family schemes fail, in two different ways.

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

## AUSM⁺-up has a narrow window, and it is not where Liou says

The run above used $M_{co} = M_\infty = 0.8$, following Liou's own guidance.
That is what diverged, so the one tunable constant was swept.

| $M_{co}$ | transonic ($M_\infty = 0.8$) | subcritical ($M_\infty = 0.5$) |
|---|---|---|
| 0.02 – 0.20 | **diverges in < 35 iterations** | diverges |
| 0.30 | converged, $c_\ell$ = 0.2672 | converged, $c_\ell$ = 0.2384 |
| 0.35 | converged, 0.2694 | converged, 0.2406 |
| **0.40** | **converged, 0.2713** | **converged, 0.2424** |
| 0.45 | converged, 0.2729 | converged, 0.2439 |
| 0.50 | stalls at 9.5e−2 | stalls at 5.2e−1 |
| 0.80 | diverges at 195 iterations | diverges |

Three things follow, and the third is the surprise.

**The low-$M_{co}$ divergence is the documented trap.** $f_a = M_0(2 - M_0)$
appears as $K_p/f_a$ in the pressure-diffusion term, so driving $M_{co}$ towards
zero makes that coefficient blow up. The nozzle recorded the same failure.

**The high-$M_{co}$ failure is the opposite problem.** As $M_{co} \to 1$,
$f_a \to 1$ and the low-Mach enhancement switches off, recovering plain AUSM⁺ —
which has the well-known pressure–velocity decoupling as $M \to 0$. An airfoil
has a **stagnation point**, so there is always a region at $M \approx 0$ for that
to bite. A nozzle has no stagnation point in its interior, which is plausibly
why the nozzle never saw this end of the failure.

**The window does not move with the free stream.** It is 0.30–0.45 at
$M_\infty = 0.8$ *and* at $M_\infty = 0.5$. Liou's guidance is
$M_{co} \sim M_\infty$, and that would predict the window to shift by a factor
of 1.6 between the two. It does not shift at all. Whatever sets the window here,
it is not the free-stream Mach number — which is consistent with it being set at
the other end of the range, by the stagnation region that both cases share.

The shipped default is therefore **0.4**, measured, not 0.8 inherited.

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
* AUSM⁺-up converges only inside a narrow $M_{co}$ window, and that window is
  not predicted by Liou's guidance for this geometry.
* SLAU2 does not converge here, with no constant available to blame.
* The nozzle's ranking **transfers** to a problem with a surface shock, which
  that project explicitly expected it might not.

**Not established.**

* *Why*. The mechanism behind the AUSM-family limit cycle is no better
  understood than it was in the nozzle, and nothing here was designed to find
  it.
* Whether the window moves with the mesh, the CFL, or the section. One mesh,
  one section, two operating points.
* Whether a second-order reconstruction changes any of it. Everything here is
  first order; the AUSM family's published advantages are largely about
  *shock capturing*, which first order smears anyway.

**No remedy has been added to the solver.** The two schemes that fail are
shipped failing, with no extra damping, no scheme-specific CFL, and no smoothing
to paper over it — which is what makes this page checkable rather than a story.

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

**Use `roe` (the default) or `hllc`.** Both converge on every case tried, they
agree with each other, and HLLC has no entropy-fix parameter to choose.

**`ausmup` works**, at $M_{co}$ near 0.4, on all fifteen cases, and gives $c_\ell$
within 4% of Roe's. The window is narrow enough that it should not be used
without checking that the run converged — and it is not where its author's own
guidance points, which is the thing to remember about it.

**`slau2` is usable below a peak Mach of about 0.9** and not above, which rules
it out for any transonic case on this geometry. It is shipped for the
comparison, not for use.

That is the same practical conclusion the nozzle reached, for a problem with a
surface shock that the nozzle could not test. What is new is that the failure
is **not about the shock at all**: it begins in flow that is subsonic
everywhere, which neither project would have predicted.
