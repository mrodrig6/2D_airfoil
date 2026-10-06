# The approximate Riemann solvers, measured

Four interface fluxes ship, and this is the record of what each one actually
does on this problem. It exists because the companion nozzle code measured the
same four, reached a conclusion, and **said explicitly that the conclusion
should not be carried across** — so it was re-measured rather than inherited.

> **Status: increment 2.** Everything below is measured on one section at two
> operating points on one mesh. The ranking is already clear and the mechanism
> behind the AUSM-family failure is not. The remaining work — mesh dependence,
> the hypotheses in [Open questions](#open-questions), and the shock-resolution
> comparison — is increment 5. See [`roadmap.md`](roadmap.md).

## Contents

- [The question, and why it had to be asked again](#the-question-and-why-it-had-to-be-asked-again)
- [The result](#the-result)
- [AUSM⁺-up has a narrow window, and it is not where Liou says](#ausm-up-has-a-narrow-window-and-it-is-not-where-liou-says)
- [SLAU2 reproduces the nozzle's limit cycle exactly](#slau2-reproduces-the-nozzles-limit-cycle-exactly)
- [What this does and does not establish](#what-this-does-and-does-not-establish)
- [Open questions](#open-questions)
- [Recommendation](#recommendation)

---

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
3. **Does SLAU2's limit cycle localise?** In the nozzle it did not — the ten
   worst cells carried 8.3% of the residual. Worth repeating here, because a
   shock gives it somewhere obvious to localise, and if it still does not, that
   rules out the shock as the cause in both codes.
4. **Does second order change the ranking?** The AUSM family's case rests on
   shock capturing. Increment 3 makes this answerable.

## Recommendation

**Use `roe` (the default) or `hllc`.** They converge, they agree, and HLLC needs
no entropy-fix parameter.

`ausmup` is usable at $M_{co}$ near 0.4 and gives an answer within 4% of Roe's,
but the window is narrow enough that it should not be used without checking
convergence. `slau2` does not converge here and is shipped for the comparison,
not for use.

That is the same practical conclusion the nozzle reached. The difference is that
it is now a conclusion for a problem **with** a surface shock, which is the one
the nozzle could not test.
