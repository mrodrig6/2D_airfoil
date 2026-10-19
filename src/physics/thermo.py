r"""Thermodynamics and the exact Euler flux: the pieces every solver shares.

Pure functions of a state and a normal.  Nothing here knows about the mesh, the
march, or which backend is running, and every function takes an ``xp`` array
module so the same source runs under NumPy and under ``jax.numpy``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: Floor applied to density and pressure inside a flux evaluation.
#:
#: Not a physical model: a transient during the march can hand a flux a state
#: whose pressure has gone slightly negative, and a NaN from the square root
#: destroys the whole field in one step where a floor costs one cell's accuracy
#: for one iteration.  The converged answer never touches it.
FLOOR = 1e-10


class FluxResult(NamedTuple):
    """A numerical flux and the largest signal speed that produced it."""

    flux: object  #: ``(..., 4)``
    max_speed: object  #: ``(...)``


def primitives(U, gamma: float, xp=np):
    r"""Return ``(rho, vx, vy, p, H)``, with density and pressure floored positive."""
    rho = xp.maximum(U[..., 0], FLOOR)
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    p = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * (vx * vx + vy * vy)), FLOOR)
    H = (U[..., 3] + p) / rho
    return rho, vx, vy, p, H


def conservative(rho, vx, vy, p, gamma: float, xp=np):
    """The inverse of :func:`primitives`."""
    return xp.stack(
        [rho, rho * vx, rho * vy, p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)],
        axis=-1,
    )


def euler_flux(U, gamma: float, xp=np):
    r"""The pair :math:`(\mathbf{F}, \mathbf{G})`, for checking :func:`normal_flux`."""
    rho, vx, vy, p, H = primitives(U, gamma, xp=xp)
    F = xp.stack([rho * vx, rho * vx * vx + p, rho * vx * vy, rho * vx * H], axis=-1)
    G = xp.stack([rho * vy, rho * vx * vy, rho * vy * vy + p, rho * vy * H], axis=-1)
    return F, G


def normal_flux(U, nx, ny, gamma: float, xp=np):
    r"""The projected flux :math:`\mathbf{F} n_x + \mathbf{G} n_y`.

    This is the only projection the scheme ever needs, and it depends on the
    normal only through :math:`v_n` -- which is what lets one solver serve every
    face of an unstructured mesh with no per-face rotation.
    """
    rho, vx, vy, p, H = primitives(U, gamma, xp=xp)
    vn = vx * nx + vy * ny
    return xp.stack(
        [rho * vn, rho * vx * vn + p * nx, rho * vy * vn + p * ny, rho * H * vn], axis=-1
    )


def sound_speed(rho, p, gamma: float, xp=np):
    return xp.sqrt(xp.maximum(gamma * p / rho, FLOOR))


__all__ = [
    "FLOOR",
    "FluxResult",
    "conservative",
    "euler_flux",
    "normal_flux",
    "primitives",
    "sound_speed",
]
