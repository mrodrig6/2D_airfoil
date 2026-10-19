r"""The JAX backend: the same vectorised code, differentiable.

This is deliberately **not** a third implementation.  :mod:`src.spatial.assembly` and
:mod:`src.physics` both take an ``xp`` array module, so this backend runs the
identical source under ``jax.numpy`` -- the only thing that genuinely differs
between NumPy and JAX is the scatter, and :func:`src.spatial.assembly.accumulate`
isolates that in four lines.  A third copy of the Euler fluxes is exactly what
a project like this cannot afford to keep in step.

Why it exists
-------------
Not for speed.  The Numba backend is faster for a plain march and always will
be, because it compiles to one fused loop over scalars where this one builds
arrays.  JAX is here because it **differentiates**: the whole point is
:mod:`src.sensitivity`, which needs the residual as a function it can take a
vector-Jacobian product of.

The march is fused with :func:`jax.lax.fori_loop` rather than a Python loop, so
the trace is one compiled program regardless of how many steps it runs.  That
matters less for wall time than it does for memory: an unrolled thousand-step
loop is a thousand copies of the residual graph.

Sixty-four bits, not thirty-two
-------------------------------
JAX defaults to single precision, which is wrong here and silently so.  A
residual converging to :math:`10^{-6}` of its initial value needs more than
seven digits to be meaningful, and a finite-difference gradient check needs
far more than that.  :func:`enable_double_precision` sets the flag, and it must
run before the first array is made -- importing this module does it.
"""

from __future__ import annotations

import numpy as np

from ..config import FlowConditions, SolverOptions
from ..spatial import assembly as asm
from ..spatial.metrics import Metrics
from ..timestepping import step as ts
from .base import Backend


def enable_double_precision() -> None:
    """Turn on float64, which JAX leaves off by default.

    Must happen before any array is created, so this module calls it at import.
    Single precision would cap the residual at roughly :math:`10^{-7}` and make
    every gradient check meaningless.
    """
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)


enable_double_precision()


class JaxBackend(Backend):
    """The vectorised assembly under ``jax.numpy``, with the march fused."""

    name = "jax"

    def __init__(self, metrics: Metrics, flow: FlowConditions, options: SolverOptions):
        super().__init__(metrics, flow, options)
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._area = jnp.asarray(metrics.area)[:, None]
        self._weight = jnp.asarray(self.component_weight)
        # one compiled program per scheme, built on first use
        self._compiled: dict[str, object] = {}

    # -- the five primitives, in jnp so the steppers above trace ----------
    def residual(self, U):
        out = asm.residual(U, self.metrics, self.flow, xp=self._jnp)
        return out.R, out.wave

    def inverse_mass(self, R):
        return R / self._area

    def time_step(self, wave):
        dt = ts.local_time_step(wave, self.metrics, self.cfl, xp=self._jnp)
        return dt[:, None]

    def norm(self, A):
        """A traced scalar, not a ``float``.

        The inherited steppers call this inside ``jax.jit``, where coercing to
        a Python float would end the trace.  :meth:`run` converts once, at the
        end, outside the compiled region.
        """
        return self._jnp.sqrt(self._jnp.mean((A * self._weight) ** 2))

    def asarray(self, A):
        return self._jnp.asarray(A)

    def to_numpy(self, A) -> np.ndarray:
        return np.asarray(A)

    # -- the march, as one compiled program --------------------------------
    def run(self, U, n_steps: int, scheme: str):
        """Advance ``n_steps`` inside one ``fori_loop``.

        Overridden rather than inherited because the gain here is not avoiding
        Python: it is that the trace stays *one* program however many steps it
        runs.  A Python loop would unroll into one copy of the residual graph
        per step, which is a memory problem long before it is a speed one.
        """
        jnp = self._jnp
        step = self._compiled.get(scheme)
        if step is None:
            step = self._jax.jit(self.make_step(scheme))
            self._compiled[scheme] = step

        def body(_, carry):
            U_i, _res = carry
            return step(U_i)

        final, res = self._jax.lax.fori_loop(
            0, int(n_steps), body, (jnp.asarray(U), jnp.asarray(0.0))
        )
        return np.asarray(final), float(res)

    # -- what the adjoint needs --------------------------------------------
    def residual_fn(self):
        """The residual as a pure function of the state, for ``jax.vjp``.

        :mod:`src.sensitivity` differentiates *this*, which is why the adjoint
        follows the scheme automatically: change the flux or the reconstruction
        and the gradient changes with it, with nothing to re-derive by hand.
        """
        return lambda U: asm.residual(U, self.metrics, self.flow, xp=self._jnp).R
