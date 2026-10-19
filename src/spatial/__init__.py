"""The right-hand side: geometry turned into numbers, and the residual built from them.

``metrics`` reduces a mesh to the areas, face lengths and normals the scheme
actually reads; ``assembly`` sums the fluxes over them; ``adapt`` decides where
the mesh should be finer.

``adapt`` is not re-exported here.  Refinement drives whole solves, so it
reaches back up to :mod:`src.api`; importing it from this package would make
that cycle an import-time one.  Use ``from src.spatial.adapt import ...``.
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
