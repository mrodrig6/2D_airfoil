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
from ..time import stepping as ts
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
        if options.scheme != "euler":
            raise ValueError(
                f"the jax backend implements forward Euler only, not {options.scheme!r}"
            )
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        self._area = jnp.asarray(metrics.area)
        self._weight = jnp.asarray(ts.component_weights(flow))
        self._step = jax.jit(self._one_step)

    # -- the pieces, written once and shared with the other backends -------
    def _residual(self, U):
        return asm.residual(U, self.metrics, self.flow, xp=self._jnp)

    def _one_step(self, U):
        out = self._residual(U)
        dt = ts.local_time_step(out.wave, self.metrics, self.options.cfl, xp=self._jnp)
        rate = out.R / self._area[:, None]
        norm = self._jnp.sqrt(self._jnp.mean((rate * self._weight) ** 2))
        return U - dt[:, None] * rate, norm

    # -- the Backend interface ---------------------------------------------
    def residual(self, U):
        out = self._residual(self._jnp.asarray(U))
        return np.asarray(out.R), np.asarray(out.wave)

    def run(self, U, n_steps: int, scale: float):
        """Advance ``n_steps`` inside one compiled ``fori_loop``."""
        jnp = self._jnp
        state = jnp.asarray(U)

        def body(_, carry):
            U_i, _norm = carry
            return self._step(U_i)

        final, norm = self._jax.lax.fori_loop(0, int(n_steps), body, (state, jnp.asarray(0.0)))
        return np.asarray(final), float(norm) / scale

    # -- what the adjoint needs --------------------------------------------
    def residual_fn(self):
        """The residual as a pure function of the state, for ``jax.vjp``.

        :mod:`src.sensitivity` differentiates *this*, which is why the adjoint
        follows the scheme automatically: change the flux or the reconstruction
        and the gradient changes with it, with nothing to re-derive by hand.
        """
        return lambda U: self._residual(U).R
