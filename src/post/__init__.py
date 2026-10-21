"""Reading the answer: integrated loads, derived fields, and figures.

``plotting`` is not re-exported, because matplotlib is an optional dependency
and importing this package should not require it.  ``src.plotting`` is a lazy
attribute on the top-level package for the same reason.
"""

from .postprocess import (
    Forces,
    entropy_error,
    forces,
    legacy_forces,
    report,
    scalar_field,
    shock_location,
    surface_pressure,
    wall_order,
    wall_pressure,
)

__all__ = [
    "Forces",
    "entropy_error",
    "forces",
    "legacy_forces",
    "report",
    "scalar_field",
    "shock_location",
    "surface_pressure",
    "wall_order",
    "wall_pressure",
]
