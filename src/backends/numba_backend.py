"""Compiled residual and Euler kernels over reusable scratch arrays.

The Python batch loop calls compiled kernels for the residual, rate, timestep,
norm and stage update. The scratch arrays survive across steps, so the state
is copied in once and out once per batch without allocating per-stage arrays.
The shared :class:`~src.backends.base.Backend` defines the equivalent Euler
step used to check this buffered path against the other backends.

Compilation is paid on first use and cached on disk. Call
:meth:`NumbaBackend.warm_up` before benchmarking to report that cost separately.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import FLUXES, SCHEMES, FlowConditions, SolverOptions
from ..spatial.metrics import Metrics
from ..timestepping import step as ts
from . import _numba_kernels as kernels
from .base import Backend, element_faces


class NumbaBackend(Backend):
    """Compiled kernels over a gather layout.  See :mod:`._numba_kernels`."""

    name = "numba"

    def __init__(self, metrics: Metrics, flow: FlowConditions, options: SolverOptions):
        super().__init__(metrics, flow, options)
        faces = element_faces(metrics)
        self.neighbour = faces.neighbour
        self.length = np.ascontiguousarray(faces.length)
        self.normal = faces.normal
        self.area = np.ascontiguousarray(metrics.area)
        self.U_inf = np.ascontiguousarray(flow.state)
        self.weight = np.ascontiguousarray(self.component_weight)
        self.kind = FLUXES.index(flow.flux)
        self.entropy_fix = float(flow.entropy_fix)
        self.compile_seconds = 0.0

        # Scratch space, allocated once per backend rather than once per step.
        # A chunk of several hundred steps then performs two array copies in
        # total instead of a handful of fresh allocations per stage, which is
        # what makes the staged orchestration below cost what a fused loop
        # costs.
        n = metrics.n_elements
        self._R = np.empty((n, 4))
        self._wave = np.empty(n)
        self._dt = np.empty(n)
        self._F = np.empty((n, 4))  # one forward-Euler rate
        self._S = np.empty((2, n, 4))  # current and next states

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

    # -- the five primitives ----------------------------------------------
    def residual(self, U):
        U = np.ascontiguousarray(U, dtype=float)
        R = np.empty_like(U)
        wave = np.empty(U.shape[0])
        kernels.residual_gather(U, *self._args(), R, wave)
        return R, wave

    def inverse_mass(self, R):
        return R / self.area[:, None]

    def time_step(self, wave):
        return ts.local_time_step(wave, self.metrics, self.cfl)[:, None]

    def norm(self, A) -> float:
        return float(kernels.rms_weighted(np.ascontiguousarray(A, dtype=float), self.weight))

    # -- the march, staged through the scratch buffers ---------------------
    def _rate_into(self, U, F):
        """``F = -R(U)/area``, and leave the wave-speed sum in ``self._wave``."""
        kernels.residual_gather(U, *self._args(), self._R, self._wave)
        kernels.rate_into(self._R, self.area, F)

    def _limit_into(self, U) -> None:
        """Apply the limiter in place.

        The fused paths below cannot call :meth:`Backend.limit`, which returns
        a new array and would allocate once per stage, so they call this
        instead.  It must stay equivalent to ``limit``: the reconstruction is
        piecewise constant today, so both are no-ops, and whichever one gains
        a body has to give it to the other.
        """
        return

    def _euler_into(self, U, out) -> float:
        """One forward-Euler step: ``U`` in, ``out`` written, norm returned."""
        F = self._F
        self._rate_into(U, F)
        kernels.dt_into(self._wave, self.area, self.cfl, self._dt)
        res = float(kernels.rms_weighted(F, self.weight))
        kernels.stage(U, F, 1.0, self._dt, out)
        self._limit_into(out)
        return res

    def run(self, U, n_steps: int, scheme: str):
        """Advance ``n_steps`` through the preallocated buffers.

        Overridden rather than inherited so that the state never leaves the
        scratch arrays: ``U`` is copied in once and out once, whatever the
        chunk length, and no stage allocates.
        """
        self._check_scheme(scheme)
        if scheme != "euler":
            raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
        step = self._euler_into

        cur, nxt = self._S[0], self._S[1]
        cur[...] = np.ascontiguousarray(U, dtype=float)
        started = time.perf_counter()
        res = float("nan")
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(n_steps):
                res = step(cur, nxt)
                cur, nxt = nxt, cur
        if not self.compile_seconds:
            # the first call includes the compile; record it so a benchmark can
            # report it separately instead of hiding it in the first batch
            self.compile_seconds = time.perf_counter() - started
        return cur.copy(), res

    def warm_up(self, scheme: str = "euler") -> float:
        """Compile the kernels now, on a single throwaway step.  Returns the seconds."""
        started = time.perf_counter()
        U = np.broadcast_to(self.U_inf, (self.area.size, 4)).copy()
        self.run(U, 1, scheme)
        self.compile_seconds = time.perf_counter() - started
        return self.compile_seconds
