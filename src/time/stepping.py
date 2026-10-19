r"""How big a step to take, and how converged the march is.

Separated from the residual because these answer a different question.
:mod:`src.spatial.assembly` says what :math:`dU/dt` is; this module says how far
to move along it and when to stop.  Both the march and every backend need them.
"""

from __future__ import annotations

import numpy as np

from ..config import FlowConditions
from ..physics import FLOOR
from ..spatial.metrics import Metrics


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
    return cfl * 2.0 * metrics.area / xp.maximum(wave, FLOOR)


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


__all__ = ["component_weights", "local_time_step", "residual_norm"]
