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

#: Largest ``cfl`` at which the march still converged, by scheme.  Measured by
#: bisection to about 3% over four cases -- the transonic design point on a
#: 96x40 and a 144x56 mesh, a subcritical point and a supercritical one -- and
#: the tightest of the four is recorded.  The spread was narrow: Euler came out
#: between 1.152 and 1.174, SSP-RK3 between 1.514 and 1.605.
#:
#: These are *limits*, not recommendations.  What they say about the two
#: schemes is the uncomfortable part: SSP-RK3 buys only 1.31 times the step for
#: three times the residual evaluations, so forward Euler reaches steady state
#: roughly 2.3 times sooner.  ``docs/performance.md`` has the wall-clock
#: measurement, and the reason the default is SSP-RK3 regardless.
STABILITY_LIMIT = {"euler": 1.15, "ssprk3": 1.51}

#: Fraction of the measured limit that :func:`recommended_cfl` takes.  A scan
#: over four cases is a measurement, not a stability proof, so the shipped
#: default keeps a fifth in hand.  The value is not chosen freshly: 0.78 is the
#: margin the long-standing ``cfl = 0.9`` Euler default already sat at, and
#: holding the *margin* constant rather than the number is what makes one
#: default mean the same thing for both schemes.
CFL_MARGIN = 0.78


def stability_limit(scheme: str) -> float:
    """The measured limit for ``scheme``; see :data:`STABILITY_LIMIT`."""
    try:
        return STABILITY_LIMIT[scheme]
    except KeyError:
        raise ValueError(
            f"no measured stability limit for scheme {scheme!r}; known: {sorted(STABILITY_LIMIT)}"
        ) from None


def recommended_cfl(scheme: str) -> float:
    """The ``cfl`` the solver uses when none is given.

    Scheme-dependent *because* ``cfl`` keeps its traditional meaning: one
    number cannot carry the same safety margin for a one-stage and a
    three-stage method, and the margin is the thing worth holding constant.
    Rounded to two decimals, which is what makes Euler come out at exactly the
    0.9 it has always used.
    """
    return round(CFL_MARGIN * stability_limit(scheme), 2)


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


__all__ = [
    "CFL_MARGIN",
    "STABILITY_LIMIT",
    "component_weights",
    "local_time_step",
    "recommended_cfl",
    "residual_norm",
    "stability_limit",
]
