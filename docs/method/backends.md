# Backends and performance

NumPy, Numba and JAX evaluate the same discrete model through different execution paths. The measurements below include their hardware and scheme restrictions.

## What it costs

One adjoint solve, whatever the number of design variables. That is the whole
claim, and it is worth being precise about what it buys *here*:

| | 64×32 mesh, 4096 cells |
|---|---|
| forward solve (Numba) | 2.2 s |
| adjoint solve + all five gradients (JAX) | ~27 s, flat in the number of variables |
| central finite difference, five variables | 14.4 s, about 2.9 s **each** |

In that historical timing, the adjoint broke even at around nine variables. This is not a fixed property of a 4096-cell mesh: the current captured example breaks even at about five variables on this environment. The reason is not the method: the forward solver is fused
Numba and the adjoint march runs in JAX, which is several times slower per sweep,
and a few of those seconds are JAX compiling the traced residual once.

The scaling is what matters. The adjoint column does not move as variables are
added; the finite-difference column grows linearly. A handful of variables does
not need an adjoint. A twenty-variable shape cannot do without one.

## The numbers

```bash
./fv2d.sh bench @grid steps=3000
```

Historical measurements from before the shared stepping refactor: NACA 0012, O-grid, Roe flux, one thread of control and four cores. These timings describe the earlier fused Numba march, not a benchmark of the current Python-orchestrated kernel loop:

| mesh | cells | numpy | numba | speedup |
|---|---|---|---|---|
| 64 × 32 | 4 096 | 450 it/s | 4 302 it/s | **9.6×** |
| 96 × 40 | 7 680 | 252 it/s | 2 759 it/s | **11.0×** |
| 144 × 56 | 16 128 | 130 it/s | 1 628 it/s | **12.5×** |
| 192 × 64 | 24 576 | 49 it/s | 1 128 it/s | **23.2×** |

The gap **widens** with the mesh, which is the giveaway about where NumPy's time
goes: the per-call overhead is fixed, so it should shrink in relative terms as
the arrays grow. It does not, because the larger meshes stop fitting in cache
and the vectorised path makes a dozen passes over memory per iteration where
the compiled one makes one.

What that is worth on a real run — the legacy operating point, 192 × 64,
converged to $10^{-6}$:

| | iterations | wall time |
|---|---|---|
| numpy | 17 909 | **383 s** |
| numba | 17 925 | **15.7 s** |

Same answer in that historical comparison: $c_\ell = +0.304414$ against $+0.304415$. The 16 extra iterations came from checking convergence at different intervals. The current driver batches both backends, so this table does not describe their current stopping behavior.

## What makes it fast

Not the flux evaluation. NumPy's inner loops are compiled C already, and
rewriting them in Numba on its own would buy very little. Three things did the
work, in descending order:

**The residual and stage operations are compiled.**
`NumbaBackend.run` now orchestrates each forward-Euler step from Python and
calls the compiled residual, rate, timestep, norm and stage kernels. It keeps
one residual buffer, one rate buffer and two state buffers, plus the wave and
timestep arrays, across the whole batch. The shared `Backend` defines the
step contract; the Numba path performs the equivalent operations in those
buffers. `batch` controls convergence-check frequency on every backend, not
the number of iterations inside one fused Numba call. Use `./fv2d.sh bench`
to measure this implementation on the current machine.

**One pass over memory instead of a dozen.** The vectorised residual builds
intermediate arrays for the primitives, the left and right states, the Roe
averages, the eigenvalues, the flux — each a full sweep through memory. The
compiled kernel carries all of that in registers, one face at a time, and
touches each element's state once.

**No four-element arrays.** Every function in the kernel module takes and
returns loose floats rather than `(..., 4)` arrays. Allocating a tiny array per
face per iteration is most of what a naive compiled port spends its time on;
returning a tuple of floats costs nothing, because Numba unboxes it.

## Gather, not scatter

The one structural difference between the two backends, and the only place
where they are not line-for-line the same arithmetic.

The vectorised residual computes each interior flux **once** and adds it to one
cell while subtracting it from the other. That is a scatter, and under `prange`
two threads can write to the same cell at the same time. The usual fixes are
atomics, which are slow, or colouring the mesh so no two adjacent cells share a
thread, which is a lot of machinery.

The compiled kernel does neither. Every element loops over **its own** three
faces and computes each flux itself, so an interior flux is evaluated **twice**,
once from each side. Every write is to the element the thread owns, nothing
needs synchronising, and the loop is embarrassingly parallel.

That is twice the flux arithmetic for a result that is still far faster — and it
is only legal because the flux is **conservative**:

$$\hat{\mathbf{F}}(L, R, \hat{n}) = -\hat{\mathbf{F}}(R, L, -\hat{n})$$

exactly, so the two evaluations cannot disagree and conservation survives. That
identity is checked for all three shipped fluxes in `tests/test_physics.py`, which makes
it a load-bearing test rather than a nicety: if it ever failed, the compiled
backend would silently stop conserving.

## What it cost

**A compile on first use**, about 20 seconds, cached to disk afterwards so it is
paid once per machine rather than once per run. `./fv2d.sh bench` reports it
separately instead of amortising it into the first batch.

**Convergence is tested every `batch` steps**, not every step, so a run can
overshoot the tolerance by up to 25 iterations. At 1 100 it/s that is 0.02
seconds. The test suite pins that batching changes only where the march stops,
not what it converges to.

**Every backend runs every scheme.** The time steppers live once, in
`src/backends/base.py`, and all three backends inherit them, so the scheme no
longer constrains the backend choice and the three cannot drift apart
numerically.

## One scheme, and the measurement that settled it

`euler` — forward Euler, one residual evaluation per step — is the only
scheme. The explicit stability limit in `cfl`, bisected to about 3% over four
cases, is **1.15**, and the default `cfl` of 0.9 sits a fifth below it.

SSP-RK3 was implemented here, measured, and removed. It is worth recording
what it actually bought, because the intuition that a higher-order integrator
is better is wrong for this problem:

| scheme | stability limit | iterations | residual evals | wall |
|---|---|---|---|---|
| `euler` | 1.15 | 26 750 | 26 750 | 44.7 s |
| SSP-RK3 | 1.51 | 20 375 | 61 125 | 95.7 s |

NACA 0012 at M 0.8, α 1.25°, 192 × 64, tolerance 1e-8, one Numba process.
SSP-RK3 buys **1.31× the step for 3× the work**, so it reaches the same steady
state **2.14× slower** — and it is the same steady state: the two agree to
$2\times10^{-8}$ on $c_\ell$ and $3\times10^{-9}$ on $c_d$, with the same
shock position.

There is no accuracy to recover. The fixed point does not know how it was
reached, and the local time step gave up time accuracy before any of this by
putting every cell on its own clock. The one real argument for an SSP scheme
is that a positivity limiter's guarantee rests on each stage being a convex
combination of forward-Euler steps — but forward Euler *is* such a stage, so
that argument covers it directly. SSP-RK3 inherits the property rather than
owning it.

For a teaching code that students run repeatedly, 2.14× was the wrong price
for nothing, so it was taken out.

Classical `rk4` was removed for the same reason, earlier and more easily: a
fourth residual per step, no time accuracy to gain, and no strong-stability
property to justify the cost.

## The JAX backend

It exists to be **differentiated**, not to be fast. `src/sensitivity.py` needs
the residual as a function it can take a vector–Jacobian product of, and that is
the whole reason for it.

It is also not a third implementation of the physics. `src/spatial/assembly.py` and
`src/physics/fluxes.py` take an `xp` array module, so the JAX backend runs the identical
source under `jax.numpy`; the only thing that genuinely differs between NumPy and
JAX is the scatter, and `assembly.accumulate` isolates that in four lines. A
third copy of the Euler fluxes is precisely what a project this size cannot keep
in step.

For a plain march, Numba uses compiled scalar kernels while JAX builds arrays. The adjoint march is accordingly several
times slower per sweep than the forward march it mirrors, which is why
[backends](backends.md#what-it-costs) puts the adjoint's break-even
against finite differences at around nine design variables rather than two.

**Float64 is forced at import.** JAX defaults to single precision, which is wrong
here and silently so: a residual converging to $10^{-6}$ of its initial value
needs more than seven digits to mean anything, and a finite-difference gradient
check needs far more. `enable_double_precision()` runs when the module is
imported, before any array exists.

## Choosing a backend

```bash
./fv2d.sh solve @transonic                  # auto: compiled if available
./fv2d.sh solve @transonic backend=numpy    # the readable reference
./fv2d.sh check                             # says which are installed
```

`auto` is the default and takes the fastest installed backend; it never picks
`jax`, because nothing about a plain march is better there. Reach for `numpy` in
exactly two situations: when a result looks wrong and you want the
simplest possible implementation to check it against, and when you are changing
the physics — write it in `src/physics/fluxes.py` first, confirm it there, then port it
to the kernels and let `tests/test_backends.py` tell you the two agree.
