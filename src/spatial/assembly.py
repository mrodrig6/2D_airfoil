r"""The finite-volume residual: one numerical flux per face, gathered into cells.

Integrating the Euler equations over a cell and applying the divergence theorem
gives the semi-discrete form the solver marches:

.. math::
    A_e \frac{d\vec{U}_e}{dt}
    + \underbrace{\sum_{f \in \partial e} \hat{\mathbf{F}}_f\,\ell_f}_{R_e}
    = \vec{0}.

The flux on a face is computed **once** and added to one cell and subtracted
from the other, so the scheme is conservative to machine precision whichever
approximate Riemann solver supplies it.

Reconstruction is piecewise constant: the state at a face is the neighbouring
cell average, exactly as ``legacy/residual_calc.m``.  All of the upwinding comes
from the choice of :math:`\hat{\mathbf{F}}`.

Everything here takes an ``xp`` array module, so the same source runs under
NumPy and ``jax.numpy``.  The one operation that genuinely differs between them
is the scatter, which :func:`accumulate` isolates.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .. import physics as ph
from ..config import FlowConditions
from ..geometry.mesh import BoundaryTag
from .metrics import Metrics


def accumulate(n_elements: int, index, values, xp=np):
    """Sum ``values`` into ``n_elements`` bins by ``index``.

    The only place the array libraries genuinely differ.  NumPy's ``add.at`` is
    correct but roughly a hundred times slower than ``bincount``, which matters
    because this runs twice per face per iteration; JAX has a scatter of its
    own and no ``bincount`` that takes a weight per component.
    """
    values = xp.asarray(values)
    if xp is np:
        return np.stack(
            [np.bincount(index, values[..., k], minlength=n_elements) for k in range(4)],
            axis=-1,
        )
    return xp.zeros((n_elements, 4), values.dtype).at[index].add(values)


def accumulate_scalar(n_elements: int, index, values, xp=np):
    """:func:`accumulate` for a scalar per face, used for the wave speeds."""
    if xp is np:
        return np.bincount(index, values, minlength=n_elements)
    return xp.zeros(n_elements, values.dtype).at[index].add(values)


class Residual(NamedTuple):
    r"""The flux residual and the wave-speed sum the time step needs."""

    R: object  #: ``(n_elem, 4)``, the net flux out of each cell
    wave: object  #: ``(n_elem,)``, :math:`\sum_f (|v_n| + a)_f \ell_f`


def residual(U, metrics: Metrics, flow: FlowConditions, *, xp=np, U_inf=None) -> Residual:
    r"""Assemble :math:`R_e = \sum_f \hat{\mathbf{F}}_f \ell_f` over every cell.

    Three kinds of face, one pass each:

    * **interior** -- the chosen approximate Riemann solver between the two
      cell averages, added to the left cell and subtracted from the right,
      which is what the normal convention in :mod:`src.spatial.metrics` is for;
    * **wall** -- :func:`~src.physics.wall_flux`, pressure only;
    * **far field** -- :func:`~src.physics.farfield_flux`, the characteristic
      state and then the same Riemann solver across it.

    ``U_inf`` overrides the free-stream state ``flow`` would supply.  That is
    what lets :mod:`src.sensitivity` differentiate with respect to the angle of
    attack and the Mach number: both enter the problem **only** through this
    vector, so a traced value here is the whole dependence.
    """
    n = metrics.n_elements
    gamma = flow.gamma
    options = flow.flux_options()

    left, right = metrics.interior_left, metrics.interior_right
    nx = metrics.interior_normal[:, 0]
    ny = metrics.interior_normal[:, 1]

    # Piecewise constant hands the solver the two cell averages; MUSCL hands
    # it limited linear extrapolations to the face midpoint.  That is the only
    # difference between first and second order here -- the sum below, and so
    # conservation, is identical either way.
    if flow.reconstruction == "constant":
        UL, UR, UB = U[left], U[right], None
    else:
        from .reconstruct import face_states

        UL, UR, UB = face_states(U, metrics, flow, xp=xp)

    inner = ph.interface_flux(flow.flux, UL, UR, nx, ny, gamma, xp=xp, **options)
    carried = inner.flux * metrics.interior_length[:, None]
    speed = inner.max_speed * metrics.interior_length

    R = accumulate(n, left, carried, xp) - accumulate(n, right, carried, xp)
    wave = accumulate_scalar(n, left, speed, xp) + accumulate_scalar(n, right, speed, xp)

    element = metrics.boundary_element
    bnx = metrics.boundary_normal[:, 0]
    bny = metrics.boundary_normal[:, 1]
    U_inf = xp.asarray(flow.state) if U_inf is None else U_inf

    for tag in (BoundaryTag.WALL, BoundaryTag.FARFIELD):
        s = metrics.tag_slice(tag)
        if s.stop == s.start:
            continue
        face = U[element[s]] if UB is None else UB[s]
        if tag is BoundaryTag.WALL:
            out = ph.wall_flux(face, bnx[s], bny[s], gamma, xp=xp)
        else:
            out = ph.farfield_flux(
                face, U_inf, bnx[s], bny[s], gamma, flux=flow.flux, xp=xp, **options
            )
        length = metrics.boundary_length[s]
        R = R + accumulate(n, element[s], out.flux * length[:, None], xp)
        wave = wave + accumulate_scalar(n, element[s], out.max_speed * length, xp)

    return Residual(R, wave)


__all__ = ["Residual", "accumulate", "accumulate_scalar", "residual"]
