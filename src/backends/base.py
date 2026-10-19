r"""What a backend is, the shared time steppers, and the per-element face data.

A backend supplies five primitives -- ``residual``, ``inverse_mass``, ``limit``,
``time_step`` and ``norm`` -- and **inherits** the time integrators.  Keeping
the schemes in one place means the three backends cannot drift apart
numerically, which is what makes cross-backend agreement testable to round-off.
It is also why adding a scheme is one method here rather than three.

A backend owns no physics -- the equations live in :mod:`src.physics` and the
assembly in :mod:`src.spatial.assembly` -- only the question of *how* those get
evaluated.

Schemes
-------
``euler``
    Forward Euler, one residual evaluation per step.  What
    ``legacy/main.m`` implements, and the cheapest route to steady state on
    this problem: see ``docs/performance.md`` for the measurement.

``ssprk3``
    Three-stage strong-stability-preserving.  Each stage is a convex
    combination of forward-Euler updates, which is exactly the structure the
    Zhang-Shu positivity limiter assumes, so ``ssprk3`` is the scheme a
    positivity guarantee can be built on and Euler is not.  It costs three
    residual evaluations per step.

The mass matrix of a finite-volume scheme
-----------------------------------------
:meth:`Backend.inverse_mass` looks like an odd thing for a cell-centred code to
have.  It is there because it is the honest name: discretising with one
piecewise-constant basis function per cell gives a mass matrix that is diagonal
with the cell area on it, so :math:`M^{-1}R` is :math:`R/A`.  The companion
discontinuous-Galerkin code carries the same method with a dense elemental
block behind it, and at :math:`p = 0` its block *is* this division -- which is
the sense in which these two solvers run the same scheme.

The gather layout
-----------------
:func:`element_faces` turns the edge lists in :class:`~src.spatial.metrics.Metrics`
into a per-element view: for each element, its three faces with the neighbour
on the other side, the face length, and the outward normal.  The vectorised
assembly does not need this; the compiled one does, because it is what lets
each thread own an element and write only to that element.  See
:mod:`src.backends._numba_kernels` for why that matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from ..config import SCHEMES, FlowConditions, SolverOptions
from ..geometry.mesh import BoundaryTag
from ..spatial.metrics import Metrics
from ..timestepping import step as ts

#: Neighbour codes for a face with no element on the far side.
WALL, FARFIELD = -1, -2


class ElementFaces(NamedTuple):
    """Per-element face data, shaped for a gather."""

    neighbour: np.ndarray  #: ``(n_elem, 3)``; an element index, or WALL/FARFIELD
    length: np.ndarray  #: ``(n_elem, 3)``
    normal: np.ndarray  #: ``(n_elem, 3, 2)``, outward from *this* element


def element_faces(metrics: Metrics) -> ElementFaces:
    """Rebuild the connectivity as three faces per element.

    An interior face appears **twice** -- once for each side, with the normal
    negated the second time -- which is exactly the duplication the gather
    trades for being lock-free.
    """
    n = metrics.n_elements
    neighbour = np.full((n, 3), WALL, dtype=np.int64)
    length = np.zeros((n, 3))
    normal = np.zeros((n, 3, 2))
    slot = np.zeros(n, dtype=np.int64)

    def place(element, other, ell, nx, ny):
        k = slot[element]
        if np.any(k > 2):
            raise ValueError("an element has more than three faces; the mesh is not triangular")
        neighbour[element, k] = other
        length[element, k] = ell
        normal[element, k, 0] = nx
        normal[element, k, 1] = ny
        slot[element] = k + 1

    left, right = metrics.interior_left, metrics.interior_right
    for e, o, ell, nvec in zip(
        left, right, metrics.interior_length, metrics.interior_normal, strict=True
    ):
        place(e, o, ell, nvec[0], nvec[1])
    for e, o, ell, nvec in zip(
        right, left, metrics.interior_length, metrics.interior_normal, strict=True
    ):
        place(e, o, ell, -nvec[0], -nvec[1])  # the same face, seen from the other side

    tags = {int(BoundaryTag.WALL): WALL, int(BoundaryTag.FARFIELD): FARFIELD}
    for e, tag, ell, nvec in zip(
        metrics.boundary_element,
        metrics.boundary_tag,
        metrics.boundary_length,
        metrics.boundary_normal,
        strict=True,
    ):
        place(e, tags[int(tag)], ell, nvec[0], nvec[1])

    if not np.all(slot == 3):
        bad = int(np.flatnonzero(slot != 3)[0])
        raise ValueError(f"element {bad} has {slot[bad]} faces, not 3")
    return ElementFaces(neighbour, length, np.ascontiguousarray(normal))


class Backend(ABC):
    """One execution strategy for a fixed mesh and operating point."""

    name = "abstract"

    def __init__(self, metrics: Metrics, flow: FlowConditions, options: SolverOptions):
        self.metrics = metrics
        self.flow = flow
        self.options = options
        self.component_weight = ts.component_weights(flow)
        # Resolved once, here, so that `cfl=None` means the same thing to all
        # three backends and nothing downstream has to know it can be None.
        self.cfl = ts.recommended_cfl(options.scheme) if options.cfl is None else float(options.cfl)

    # -- primitives a backend must provide --------------------------------
    @abstractmethod
    def residual(self, U):
        """Return ``(R, wave)``, matching :func:`src.spatial.assembly.residual`."""

    @abstractmethod
    def inverse_mass(self, R):
        r"""Apply :math:`M^{-1}`, which for this discretisation is ``R / area``."""

    @abstractmethod
    def time_step(self, wave):
        """Element-local pseudo-time step, shaped to broadcast against ``U``."""

    @abstractmethod
    def norm(self, A) -> float:
        r"""Component-weighted RMS of a rate array.

        See :func:`src.timestepping.step.component_weights` for why the weights
        are there.  Every backend applies the same weights to the same axis,
        which is what keeps their convergence histories comparable to round-off.
        """

    def limit(self, U):
        """Return ``U`` made admissible again.

        Nothing to do yet: the reconstruction is piecewise constant, so a cell
        average is the whole of the solution in that cell and there are no
        sub-cell oscillations to limit.  The steppers below call this at every
        stage regardless, because that is where a limiter has to go for the
        strong-stability argument to hold -- so adding reconstruction means
        implementing this method, not rewriting the schemes.
        """
        return U

    def asarray(self, A):
        return np.asarray(A)

    def to_numpy(self, A) -> np.ndarray:
        return np.asarray(A)

    # -- integrators, shared by every backend ------------------------------
    def rate(self, U):
        r"""``dU/dt = -M^{-1} R``, plus the wave-speed sum the step size needs."""
        R, wave = self.residual(U)
        return -self.inverse_mass(R), wave

    def euler_step(self, U):
        """One forward-Euler step.  Returns ``(U_new, residual_norm)``."""
        F0, wave = self.rate(U)
        dt = self.time_step(wave)
        res = self.norm(F0)
        return self.limit(U + dt * F0), res

    def ssprk3_step(self, U):
        """One three-stage SSP-RK3 step.

        The time step is frozen at the first stage.  That is standard for a
        pseudo-time march -- the intermediate stages are not meant to be
        time-accurate -- and it is also what makes each stage a plain convex
        combination of forward-Euler steps with the *same* ``dt``, which is the
        hypothesis the strong-stability property rests on.
        """
        F0, wave = self.rate(U)
        dt = self.time_step(wave)
        res = self.norm(F0)

        U1 = self.limit(U + dt * F0)
        F1, _ = self.rate(U1)
        U2 = self.limit(0.75 * U + 0.25 * (U1 + dt * F1))
        F2, _ = self.rate(U2)
        U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * F2)
        return self.limit(U3), res

    def make_step(self, scheme: str) -> Callable:
        if scheme == "euler":
            return self.euler_step
        if scheme == "ssprk3":
            return self.ssprk3_step
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")

    def run(self, U, n_steps: int, scheme: str):
        """Advance ``n_steps`` steps.  Returns ``(U, residual_norm_of_the_last)``.

        Backends that can fuse or buffer the loop override this; the default
        simply iterates, which is what the vectorised reference wants anyway.
        """
        step = self.make_step(scheme)
        res = float("nan")
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(n_steps):
                U, res = step(U)
        return U, res


__all__ = ["Backend", "ElementFaces", "FARFIELD", "WALL", "element_faces"]
