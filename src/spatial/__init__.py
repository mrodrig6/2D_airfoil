"""The right-hand side: geometry turned into numbers, and the residual built from them.

``metrics`` reduces a mesh to the areas, face lengths and normals the scheme
actually reads; ``assembly`` sums the fluxes over them.

Mesh refinement is *not* here, although it is a spatial question:
:mod:`src.adapt` drives whole solves to decide where the mesh should be finer,
so it sits with the other drivers at the top level rather than underneath
them.
"""

from .assembly import Residual, accumulate, accumulate_scalar, residual
from .metrics import Metrics, build_metrics

__all__ = [
    "Metrics",
    "Residual",
    "accumulate",
    "accumulate_scalar",
    "build_metrics",
    "residual",
]
