r"""What a backend is, and the per-element face data every backend needs.

A backend is one execution strategy for one fixed mesh and operating point.  It
owns no physics -- the equations live in :mod:`src.physics` and the assembly in
:mod:`src.assembly` -- only the question of *how* those get evaluated.

The gather layout
-----------------
:func:`element_faces` turns the edge lists in :class:`~src.metrics.Metrics`
into a per-element view: for each element, its three faces with the neighbour
on the other side, the face length, and the outward normal.  The vectorised
assembly does not need this; the compiled one does, because it is what lets
each thread own an element and write only to that element.  See
:mod:`src.backends._numba_kernels` for why that matters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from ..config import FlowConditions, SolverOptions
from ..mesh import BoundaryTag
from ..metrics import Metrics

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

    @abstractmethod
    def residual(self, U):
        """Return ``(R, wave)``, matching :func:`src.assembly.residual`."""

    @abstractmethod
    def run(self, U, n_steps: int, scale: float):
        """Advance ``n_steps`` and return ``(U, scaled_residual_of_the_last_step)``."""


__all__ = ["Backend", "ElementFaces", "FARFIELD", "WALL", "element_faces"]
