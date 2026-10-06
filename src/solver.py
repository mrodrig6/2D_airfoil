r"""The pseudo-time march to steady state.

The steady Euler equations are mixed elliptic-hyperbolic, so they are not
marched in physical time but in a **pseudo-time** whose only purpose is to
reach the fixed point :math:`R(\vec{U}) = 0`.  Each cell therefore takes its
own step -- see :func:`~src.assembly.local_time_step` -- which on a mesh whose
cells span six orders of magnitude in area is the difference between a run that
finishes and one that does not.

Convergence
-----------
The measure is the component-weighted RMS of :math:`d\vec U/dt`, **scaled by
its own value at the first iteration**.  That scaling is the part that matters:
``legacy/main.m`` stopped on an absolute :math:`\lVert R \rVert_\infty < 10^{-2}`,
which is neither comparable between meshes -- refining changes the residual's
scale -- nor between operating points.  A scaled residual means one tolerance
is the same statement everywhere, which is what makes the flux comparison in
``docs/riemann_fluxes.md`` a comparison rather than four separate anecdotes.

The reference is deliberately **not** this run's own first iteration.  That
would make a warm-started run chase the same reduction *factor* from a smaller
starting point, so it would converge to a tighter absolute level and save no
iterations -- defeating the purpose of warm starting a sweep.

A run that does not converge **returns** rather than raises.  A stalled march
is a result worth looking at, and for the flux study it is *the* result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import assembly as asm
from .config import FlowConditions, SolverOptions
from .metrics import Metrics


@dataclass
class SolveResult:
    """A converged (or stalled) state, and everything needed to interpret it."""

    state: np.ndarray
    metrics: Metrics
    flow: FlowConditions
    options: SolverOptions
    converged: bool
    iterations: int
    residual: float
    history: np.ndarray = field(repr=False)
    seconds: float = 0.0
    message: str = ""
    backend: str = "numpy"

    @property
    def mesh(self):
        return self.metrics.mesh

    def summary(self) -> str:
        verdict = "converged" if self.converged else f"NOT converged ({self.message})"
        return (
            f"{verdict} after {self.iterations} iterations in {self.seconds:.2f} s "
            f"({self.iterations / max(self.seconds, 1e-9):.0f} it/s on {self.backend}), "
            f"scaled residual {self.residual:.3e}"
        )


def _step(U, metrics, flow, opts, xp=np):
    """One pseudo-time step.  Returns ``(U_new, scaled_rate, wave)``."""
    out = asm.residual(U, metrics, flow, xp=xp)
    dt = asm.local_time_step(out.wave, metrics, opts.cfl, xp=xp)
    rate = asm.residual_norm(out.R, metrics, flow, xp=xp)
    area = metrics.area[:, None]
    step = (dt / metrics.area)[:, None]

    if opts.scheme == "euler":
        return U - step * out.R, rate
    if opts.scheme == "ssprk3":
        u1 = U - step * out.R
        u2 = 0.75 * U + 0.25 * (u1 - step * asm.residual(u1, metrics, flow, xp=xp).R)
        u3 = (1.0 / 3.0) * U + (2.0 / 3.0) * (u2 - step * asm.residual(u2, metrics, flow, xp=xp).R)
        return u3, rate
    # classical RK4
    k1 = out.R / area
    k2 = asm.residual(U - 0.5 * dt[:, None] * k1, metrics, flow, xp=xp).R / area
    k3 = asm.residual(U - 0.5 * dt[:, None] * k2, metrics, flow, xp=xp).R / area
    k4 = asm.residual(U - dt[:, None] * k3, metrics, flow, xp=xp).R / area
    return U - (dt[:, None] / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), rate


def resolve_backend(options: SolverOptions) -> str:
    """Turn ``backend='auto'`` into a name, preferring the compiled one.

    The multi-stage schemes are vectorised only, so asking for ``auto`` with
    ``scheme='rk4'`` gives NumPy rather than an error.
    """
    from .backends import available_backends

    if options.backend != "auto":
        return options.backend
    if options.scheme != "euler":
        return "numpy"
    return available_backends()[0]


def initial_state(metrics: Metrics, flow: FlowConditions) -> np.ndarray:
    """The free stream everywhere, which is what ``legacy/initialstate.m`` did.

    For an external flow this is a good start: it is exact over most of the
    domain and wrong only near the body, so the transient is local.
    """
    return np.broadcast_to(flow.state, (metrics.n_elements, 4)).copy()


def solve_steady(
    metrics: Metrics,
    flow: FlowConditions | None = None,
    options: SolverOptions | None = None,
    *,
    state: np.ndarray | None = None,
) -> SolveResult:
    """March to steady state and report what happened.

    Pass ``state`` to warm-start from a previous solution -- which is what makes
    an angle-of-attack sweep cost far less than one solve per point, since the
    mesh and the metrics are unchanged and only the free-stream direction moves.
    """
    flow = flow or FlowConditions()
    opts = options or SolverOptions()
    U = initial_state(metrics, flow) if state is None else np.array(state, dtype=float)

    # The reference is the residual of the **free-stream** start, always, even
    # when warm starting.  Scaling by this run's own first iteration instead
    # looks equivalent and is not: a warm start begins with a small residual
    # and would then have to cut it by the same *factor* as a cold one, so it
    # would converge to a far tighter absolute level and save no iterations at
    # all.  Measured against a fixed reference, the tolerance means one thing
    # regardless of where the march was started -- which is what makes warm
    # starting worth doing in a sweep.  For a cold start the two are identical.
    reference = max(
        asm.residual_norm(
            asm.residual(initial_state(metrics, flow), metrics, flow).R, metrics, flow
        ),
        asm.ph.FLOOR,
    )

    backend_name = resolve_backend(opts)
    compiled = None
    if backend_name != "numpy":
        from .backends import get_backend

        compiled = get_backend(backend_name, metrics, flow, opts)

    history: list[float] = []
    message = ""
    converged = False
    iteration = 0
    started = time.perf_counter()

    with np.errstate(over="ignore", invalid="ignore"):
        while iteration < opts.max_iterations:
            if compiled is None:
                U, rate = _step(U, metrics, flow, opts)
                scaled = rate / reference
                iteration += 1
                history.append(scaled)
            else:
                # the compiled backend runs a batch without returning to Python,
                # which is the whole reason it is faster; convergence is
                # therefore tested every `batch` steps rather than every step
                steps = min(opts.batch, opts.max_iterations - iteration)
                U, scaled = compiled.run(U, steps, reference)
                iteration += steps
                history.extend([scaled] * steps)

            if not np.isfinite(scaled):
                message = f"the residual became {scaled} at iteration {iteration}"
                break
            if scaled > opts.divergence_factor:
                message = (
                    f"the residual grew to {scaled:.3e} times its initial value, so the "
                    f"march has diverged; lower cfl, or try flux='hllc'"
                )
                break
            if opts.print_interval and iteration % opts.print_interval < opts.batch:
                print(f"  iteration {iteration:6d}   residual {scaled:.6e}")
            if scaled < opts.tolerance:
                converged = True
                break
        else:
            message = f"reached max_iterations ({opts.max_iterations}) without converging"

    return SolveResult(
        state=U,
        metrics=metrics,
        flow=flow,
        options=opts,
        converged=converged,
        iterations=len(history),
        residual=history[-1] if history else float("nan"),
        history=np.asarray(history),
        seconds=time.perf_counter() - started,
        message=message,
        backend=backend_name,
    )


__all__ = ["SolveResult", "initial_state", "solve_steady"]
