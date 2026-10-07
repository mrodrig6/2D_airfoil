"""The vectorised reference backend.

Thin, on purpose: it is :mod:`src.assembly` with a backend's interface around
it.  Everything else is checked against this, so it is the one place where
being obviously correct matters more than being fast.
"""

from __future__ import annotations

from .. import assembly as asm
from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def residual(self, U):
        out = asm.residual(U, self.metrics, self.flow)
        return out.R, out.wave

    def run(self, U, n_steps: int, scale: float):
        metrics, flow, opts = self.metrics, self.flow, self.options
        area = metrics.area[:, None]
        norm = float("nan")
        for _ in range(n_steps):
            R, wave = self.residual(U)
            dt = asm.local_time_step(wave, metrics, opts.cfl)
            norm = asm.residual_norm(R, metrics, flow) / scale
            U = U - (dt[:, None] / area) * R
        return U, norm
