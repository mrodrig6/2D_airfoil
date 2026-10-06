"""Backend registry.

Two interchangeable execution strategies over one set of equations:

=========  ==========================================================
``numpy``  The vectorised reference.  No compile step, easiest to read,
           and what every other backend is checked against.
``numba``  Compiled ``nopython`` kernels with the march fused, so the
           interpreter sees one call per few hundred iterations.
=========  ==========================================================

A third, ``jax``, arrives with the adjoint.
"""

from __future__ import annotations

from ..config import FlowConditions, SolverOptions
from ..metrics import Metrics
from .base import Backend, element_faces
from .numpy_backend import NumpyBackend

__all__ = ["Backend", "NumpyBackend", "available_backends", "element_faces", "get_backend"]


def available_backends() -> tuple[str, ...]:
    """Backends importable in this environment, fastest first."""
    found = []
    try:
        import numba  # noqa: F401

        found.append("numba")
    except ImportError:
        pass
    found.append("numpy")
    return tuple(found)


def get_backend(
    name: str, metrics: Metrics, flow: FlowConditions, options: SolverOptions
) -> Backend:
    """Instantiate a backend by name."""
    if name == "numpy":
        return NumpyBackend(metrics, flow, options)
    if name == "numba":
        try:
            from .numba_backend import NumbaBackend
        except ImportError as exc:  # pragma: no cover - needs numba absent
            raise ImportError(
                "the 'numba' backend needs Numba installed: pip install numba"
            ) from exc
        return NumbaBackend(metrics, flow, options)
    raise ValueError(f"unknown backend {name!r}; choose from {available_backends()}")
