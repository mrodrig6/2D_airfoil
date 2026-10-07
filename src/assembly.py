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

from . import physics as ph
from .config import FlowConditions
from .mesh import BoundaryTag
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
      which is what the normal convention in :mod:`src.metrics` is for;
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
    inner = ph.interface_flux(flow.flux, U[left], U[right], nx, ny, gamma, xp=xp, **options)
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
        face = U[element[s]]
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


def local_time_step(wave, metrics: Metrics, cfl: float, xp=np):
    r"""The element-local pseudo-time step.

    .. math::
        \Delta t_e = \mathrm{CFL}\,\frac{2 A_e}{\sum_f (|v_n| + a)_f\,\ell_f}

    which is ``legacy/main.m``'s :math:`\mathrm{CFL}\,d_h / s` written out:
    with :math:`d_h = 2A/P` and :math:`s = \frac{1}{P}\sum_f (|v_n|+a)_f
    \ell_f`, the perimeter cancels.  Writing it this way removes a quantity
    that has to be kept in step with the faces for no benefit.

    Each cell takes its own step, which is legitimate because only the fixed
    point is wanted -- and necessary, because the cells on this mesh span six
    orders of magnitude in area and a global step would be set by the smallest.
    """
    return cfl * 2.0 * metrics.area / xp.maximum(wave, ph.FLOOR)


def component_weights(flow: FlowConditions, xp=np):
    r"""Scale the four residual components onto a common footing.

    Mass, momentum and energy have different units even non-dimensionally, so
    an unweighted norm is dominated by whichever happens to be largest and is
    not comparable between operating points.  Dividing momentum by
    :math:`a_\infty` and energy by :math:`a_\infty^2` makes every component a
    density rate, which is what lets one tolerance mean the same thing at
    :math:`M = 0.5` and at :math:`M = 0.8`.
    """
    a = flow.sound_speed
    return xp.asarray([1.0, 1.0 / a, 1.0 / a, 1.0 / a**2])


def residual_norm(R, metrics: Metrics, flow: FlowConditions, xp=np) -> float:
    r"""Component-weighted RMS of :math:`d\vec{U}/dt = -R/A`."""
    rate = R / metrics.area[:, None] * component_weights(flow, xp)
    return float(xp.sqrt(xp.mean(rate * rate)))


__all__ = [
    "Residual",
    "accumulate",
    "accumulate_scalar",
    "component_weights",
    "local_time_step",
    "residual",
    "residual_norm",
]
