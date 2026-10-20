"""Marching the residual to steady state.

``step`` sizes the time step and measures convergence; ``march`` runs the loop,
chunk by chunk, and decides when to stop.  The stage combinations themselves
are *not* here -- they live in :mod:`src.backends.base`, so that all three
backends share one definition of each scheme.  The step size is local to each
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
