"""Marching the residual to steady state.

``step`` sizes the time step and measures convergence; ``march`` runs the loop
and owns the stage combinations of each scheme.  The step size is local to each
cell, which is why it belongs here rather than in :mod:`src.spatial`: it is a
property of the march, not of the discretisation.
"""

from .march import SolveResult, initial_state, resolve_backend, solve_steady
from .step import component_weights, local_time_step, residual_norm

__all__ = [
    "SolveResult",
    "component_weights",
    "initial_state",
    "local_time_step",
    "residual_norm",
    "resolve_backend",
    "solve_steady",
]
