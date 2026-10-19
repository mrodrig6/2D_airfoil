"""The compiled backend: the march fused into one ``nopython`` call.

What makes it fast is not that the flux evaluation is compiled -- NumPy's is
compiled too.  It is that the **loop** is: :func:`~src.backends._numba_kernels.march`
runs hundreds of iterations without returning to Python, so the interpreter
sees one call instead of a dozen array operations per step, and no temporary
arrays are allocated in between.

The trade is a compile on first use.  It is paid once per process and cached on
disk, and :attr:`NumbaBackend.compile_seconds` reports it so a benchmark can
exclude it honestly rather than quietly amortising it.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import FLUXES, FlowConditions, SolverOptions
from ..spatial import assembly as asm
from ..spatial.metrics import Metrics
from ..timestepping import step as ts
from . import _numba_kernels as kernels
from .base import Backend, element_faces


class NumbaBackend(Backend):
    """Compiled kernels over a gather layout.  See :mod:`._numba_kernels`."""

    name = "numba"

    def __init__(self, metrics: Metrics, flow: FlowConditions, options: SolverOptions):
        super().__init__(metrics, flow, options)
        if options.scheme != "euler":
            raise ValueError(
                f"the numba backend implements forward Euler only, not {options.scheme!r}; "
                f"use backend='numpy' for the multi-stage schemes"
            )
        faces = element_faces(metrics)
        self.neighbour = faces.neighbour
        self.length = np.ascontiguousarray(faces.length)
        self.normal = faces.normal
        self.area = np.ascontiguousarray(metrics.area)
        self.U_inf = np.ascontiguousarray(flow.state)
        self.weight = np.ascontiguousarray(ts.component_weights(flow))
        self.kind = FLUXES.index(flow.flux)
        self.entropy_fix = float(flow.entropy_fix)
        self.compile_seconds = 0.0

    def _args(self):
        return (
            self.neighbour,
            self.length,
            self.normal,
            self.U_inf,
            self.flow.gamma,
            self.kind,
            self.entropy_fix,
        )

    def residual(self, U):
        U = np.ascontiguousarray(U, dtype=float)
        R = np.empty_like(U)
        wave = np.empty(U.shape[0])
        kernels.residual_gather(U, *self._args(), R, wave)
        return R, wave

    def run(self, U, n_steps: int, scale: float):
        """Advance ``n_steps`` inside one compiled call."""
        U = np.ascontiguousarray(U, dtype=float)
        started = time.perf_counter()
        norm = kernels.march(
            U,
            self.neighbour,
            self.length,
            self.normal,
            self.U_inf,
            self.area,
            self.flow.gamma,
            self.kind,
            self.entropy_fix,
            self.options.cfl,
            self.weight,
            scale,
            n_steps,
        )
        if not self.compile_seconds:
            # the first call includes the compile; record it so a benchmark can
            # report it separately instead of hiding it in the first batch
            self.compile_seconds = time.perf_counter() - started
        return U, float(norm)

    def warm_up(self) -> float:
        """Compile the kernels now, on a single throwaway step.  Returns the seconds."""
        started = time.perf_counter()
        self.run(np.ascontiguousarray(asm.np.zeros((self.area.size, 4)) + self.U_inf), 1, 1.0)
        self.compile_seconds = time.perf_counter() - started
        return self.compile_seconds
