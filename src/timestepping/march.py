r"""The pseudo-time march to steady state.

The steady Euler equations are mixed elliptic-hyperbolic, so they are not
marched in physical time but in a **pseudo-time** whose only purpose is to
reach the fixed point :math:`R(\vec{U}) = 0`.  Each cell therefore takes its
own step -- see :func:`~src.timestepping.step.local_time_step` -- which on a mesh whose
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

from ..config import FlowConditions, SolverOptions
from ..spatial import assembly as asm
from ..spatial.metrics import Metrics
from . import step as ts


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

    def report(self) -> str:
        """The full report -- convergence, free stream, loads, field extrema.

        A method as well as :func:`src.post.postprocess.report` because this is what
        the quick start shows, and a student should not have to know which
        module a formatter lives in.  Imported here rather than at module scope:
        post-processing reads a :class:`SolveResult`, so the dependency only
        runs one way at import time.
        """
        from ..post.postprocess import report as _report

        return _report(self)


def resolve_backend(options: SolverOptions, flow: FlowConditions | None = None) -> str:
    """Turn ``backend='auto'`` into a name, preferring the compiled one.

    Every backend implements the scheme, so the choice does not depend on
    that.  It does depend on the reconstruction: MUSCL lives only in the
    shared ``xp`` source, so ``auto`` gives NumPy when it is asked for.
    """
    from ..backends import available_backends

    if options.backend != "auto":
        return options.backend
    if flow is not None and flow.reconstruction != "constant":
        # The compiled kernels restate the physics for `nopython` mode and do
        # not restate the reconstruction, so MUSCL exists only in the shared
        # `xp` source that NumPy and JAX run.  Picking NumPy here is not a
        # silent downgrade -- the compiled backend refuses outright -- it is
        # `auto` meaning what it says.
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
    if flow.reconstruction != "constant":
        raise NotImplementedError(
            "reconstruction='muscl' is not ready to march.  The reconstruction "
            "itself is verified -- the gradients are exact on linear data and "
            "both limiters keep the face states physical -- but the pseudo-time "
            "march diverges on the transonic case at every cfl tried, after a "
            "few thousand iterations of falling residual.  The leading suspect "
            "is that the reconstruction is done in conserved rather than "
            "primitive variables.  Until that is settled this refuses rather "
            "than hands back a diverged field; see src/spatial/reconstruct.py."
        )
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
        ts.residual_norm(
            asm.residual(initial_state(metrics, flow), metrics, flow).R, metrics, flow
        ),
        asm.ph.FLOOR,
    )

    from ..backends import get_backend

    backend_name = resolve_backend(opts, flow)
    backend = get_backend(backend_name, metrics, flow, opts)

    history: list[float] = []
    message = ""
    converged = False
    iteration = 0
    started = time.perf_counter()

    with np.errstate(over="ignore", invalid="ignore"):
        while iteration < opts.max_iterations:
            # Every backend runs a chunk without returning here, which is the
            # whole reason the compiled ones are faster; convergence is
            # therefore tested every `batch` steps rather than every step, for
            # all of them alike.
            steps = min(opts.batch, opts.max_iterations - iteration)
            U, rate = backend.run(U, steps, opts.scheme)
            scaled = rate / reference
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
