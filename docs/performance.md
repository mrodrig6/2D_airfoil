# Backends and performance

Three backends compute the same residual by different routes. The answer does
not depend on which one ran — they agree to $10^{-13}$ on the residual and to six
significant figures on $c_\ell$ — so this page is mostly about speed. The third,
`jax`, is **not** here for speed at all; see [below](#the-jax-backend).

## Contents

- [The numbers](#the-numbers)
- [What makes it fast](#what-makes-it-fast)
- [Gather, not scatter](#gather-not-scatter)
- [What it cost](#what-it-cost)
- [The JAX backend](#the-jax-backend)
- [Choosing a backend](#choosing-a-backend)

---

## The numbers

```bash
./fv2d.sh bench @grid steps=3000
```

NACA 0012, O-grid, Roe flux, one thread of control and four cores:

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

Same answer: $c_\ell = +0.304414$ against $+0.304415$. The 16 extra iterations
are the batch granularity, below.

## What makes it fast

Not the flux evaluation. NumPy's inner loops are compiled C already, and
rewriting them in Numba on its own would buy very little. Three things did the
work, in descending order:

**The march does not return to Python.**
`src/backends/_numba_kernels.march` runs `batch` iterations — 25 by default —
inside one compiled call. The interpreter sees one call per 25 steps instead of
roughly a dozen array operations per step, and nothing is allocated in between.
This is most of the speedup, and it is why `batch` cannot be 1 without giving
the gain straight back.

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
identity is checked for all four fluxes in `tests/test_physics.py`, which makes
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

**Forward Euler only.** The multi-stage schemes (`ssprk3`, `rk4`) are vectorised
only, and `backend='auto'` quietly gives NumPy when one of them is asked for
rather than refusing. They exist for comparison; the local-time-step march to a
steady state does not need them.

## The JAX backend

It exists to be **differentiated**, not to be fast. `src/sensitivity.py` needs
the residual as a function it can take a vector–Jacobian product of, and that is
the whole reason for it.

It is also not a third implementation of the physics. `src/assembly.py` and
`src/physics.py` take an `xp` array module, so the JAX backend runs the identical
source under `jax.numpy`; the only thing that genuinely differs between NumPy and
JAX is the scatter, and `assembly.accumulate` isolates that in four lines. A
third copy of the Euler fluxes is precisely what a project this size cannot keep
in step.

For a plain march, Numba is and will stay faster: it compiles to one fused loop
over scalars where this builds arrays. The adjoint march is accordingly several
times slower per sweep than the forward march it mirrors, which is why
[`workflows.md`](workflows.md#what-it-costs) puts the adjoint's break-even
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
the physics — write it in `src/physics.py` first, confirm it there, then port it
to the kernels and let `tests/test_backends.py` tell you the two agree.
