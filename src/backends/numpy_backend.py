"""The vectorised reference backend.

Thin, on purpose: it is :mod:`src.spatial.assembly` with a backend's interface
around it, and it inherits every time stepper from :class:`~src.backends.base.Backend`
rather than writing its own.  Everything else is checked against this, so it is
the one place where being obviously correct matters more than being fast.
"""

from __future__ import annotations

import numpy as np

from ..spatial import assembly as asm
from ..timestepping import step as ts
from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def residual(self, U):
        out = asm.residual(U, self.metrics, self.flow)
        return out.R, out.wave

    def inverse_mass(self, R):
        return R / self.metrics.area[:, None]

    def time_step(self, wave):
        return ts.local_time_step(wave, self.metrics, self.cfl)[:, None]

    def norm(self, A) -> float:
        return float(np.sqrt(np.mean((A * self.component_weight) ** 2)))
