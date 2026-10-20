r"""Second-order reconstruction: cell gradients, and the limiters that tame them.

A cell-centred finite-volume scheme stores one average per cell.  Handing that
average straight to the Riemann solver -- which is what this code did, and
still does by default -- makes the scheme first order: the solution is
piecewise constant, so a smooth field is resolved no better than a staircase.

MUSCL recovers second order by giving each cell a *slope* as well as a mean,
and evaluating the state at the face rather than at the centroid:

.. math::
    \mathbf{U}_L = \bar{\mathbf{U}}_e
        + \Phi_e\, \nabla \bar{\mathbf{U}}_e \cdot (\vec{x}_f - \vec{x}_e)

with :math:`\Phi_e \in [0, 1]` a limiter.  The cell average is untouched, so
the scheme stays conservative whatever the slope is: the reconstruction only
changes *where* the flux is sampled, never how it is summed.

Why a limiter is not optional
-----------------------------
Second-order reconstruction overshoots at a discontinuity -- that is Godunov's
theorem, not a defect of this implementation -- and an overshoot at a strong
shock produces a negative pressure, which ends the run.  :func:`limiter_factor`
is what keeps the reconstruction inside the data it came from.

===============  ============================================================
``none``         No limiting.  Second order everywhere, and unusable with a
                 shock; here to make the limiter's effect visible by removing
                 it, and for smooth verification cases where it is the most
                 accurate choice.
``positivity``   Scale each cell's slope by the largest factor that keeps the
                 reconstructed density and pressure positive at every face of
                 that cell.  Cheap, and the minimum needed to not crash.
``superbee``     Roe's TVD limiter, the most compressive of the standard
                 family.  Enforces monotonicity against the neighbouring cell
                 averages, which also delivers positivity.
===============  ============================================================

The names and the roles match the companion DG nozzle solver, where the same
three act on the element polynomial rather than on a reconstructed slope --
same question, asked of a different representation.
"""

from __future__ import annotations

import numpy as np

from .. import physics as ph
from ..config import FlowConditions
from .metrics import Metrics

#: Reconstructions the solver implements.
RECONSTRUCTIONS = ("constant", "muscl")

#: Slope limiters.  See the table above.
LIMITERS = ("none", "positivity", "superbee")


def cell_gradients(U, metrics: Metrics, *, xp=np):
    r"""Cell gradients by weighted least squares over the face neighbours.

    For cell :math:`e` with neighbours :math:`j`, the gradient that best fits
    the neighbouring means:

    .. math::
        \min_{\nabla \bar{U}_e} \sum_j w_j \left(
            \nabla \bar{U}_e \cdot (\vec{x}_j - \vec{x}_e)
            - (\bar{U}_j - \bar{U}_e) \right)^2 ,
        \qquad w_j = \lvert \vec{x}_j - \vec{x}_e \rvert^{-2}

    solved through the 2x2 normal equations, whose geometry half is the same
    for every time step and is built once per mesh.

    **Least squares and not Green-Gauss**, which is the cheaper and more
    obvious choice on a code that already has face normals and areas to hand.
    Green-Gauss needs the state at the face *midpoint*; averaging the two cell
    means gives the state midway between the two centroids instead, and on a
    stretched mesh those are not the same point.  This mesh stretches by six
    orders of magnitude in cell area between the wall and the far field, and
    the error is not subtle there: on a field that is exactly linear,
    Green-Gauss returned gradients wrong by a factor of fifteen, while least
    squares is exact to round-off.  Being exact on linear data is the whole
    requirement -- it is what makes the reconstruction second order.

    A cell with fewer than two independent neighbour directions has no
    determined gradient; it gets zero, which makes it first order locally and
    is the safe answer rather than an arbitrary one.

    Returns ``(n_elements, 2, 4)``: the x and y derivative of each conserved
    variable.
    """
    stencil = _cached_stencil(metrics)
    nbr = xp.asarray(stencil.neighbour)
    valid = xp.asarray(stencil.valid)[..., None]
    wd = xp.asarray(stencil.weighted_offset)  # (n, 3, 2), already w * d
    inv = xp.asarray(stencil.inverse_moment)  # (n, 2, 2)

    dU = xp.where(valid, U[nbr] - U[:, None, :], 0.0)  # (n, 3, 4)
    rhs = xp.einsum("ekd,ekc->edc", wd, dU)
    return xp.einsum("eij,ejc->eic", inv, rhs)


class _Stencil:
    """The geometry half of the least-squares fit: fixed for a given mesh."""

    __slots__ = ("neighbour", "valid", "weighted_offset", "inverse_moment", "offset")

    def __init__(self, neighbour, valid, weighted_offset, inverse_moment, offset):
        self.neighbour = neighbour
        self.valid = valid
        self.weighted_offset = weighted_offset
        self.inverse_moment = inverse_moment
        self.offset = offset


_STENCIL_CACHE: dict[int, _Stencil] = {}


def _cached_stencil(metrics: Metrics) -> _Stencil:
    key = id(metrics)
    hit = _STENCIL_CACHE.get(key)
    if hit is None:
        hit = _build_stencil(metrics)
        _STENCIL_CACHE[key] = hit
    return hit


def _build_stencil(metrics: Metrics) -> _Stencil:
    """Neighbour indices, weighted offsets, the inverse moment matrix, face offsets."""
    n = metrics.n_elements
    centroid = metrics.centroid
    neighbour = np.zeros((n, 3), dtype=np.int64)
    valid = np.zeros((n, 3), dtype=bool)
    offset = np.zeros((n, 3, 2))  # x_face - x_cell, for the reconstruction
    slot = np.zeros(n, dtype=np.int64)

    def place(elements, others, midpoints, is_interior):
        for i, e in enumerate(elements):
            k = slot[e]
            offset[e, k] = midpoints[i] - centroid[e]
            if is_interior:
                neighbour[e, k] = others[i]
                valid[e, k] = True
            slot[e] = k + 1

    mid = metrics.interior_midpoint
    place(metrics.interior_left, metrics.interior_right, mid, True)
    place(metrics.interior_right, metrics.interior_left, mid, True)
    place(metrics.boundary_element, None, metrics.boundary_midpoint, False)

    d = np.where(valid[..., None], centroid[neighbour] - centroid[:, None, :], 0.0)
    dist2 = np.sum(d * d, axis=-1)
    w = np.where(valid & (dist2 > 0.0), 1.0 / np.maximum(dist2, _TINY), 0.0)
    weighted = w[..., None] * d

    moment = np.einsum("nka,nkb->nab", weighted, d)  # (n, 2, 2), symmetric
    det = moment[:, 0, 0] * moment[:, 1, 1] - moment[:, 0, 1] * moment[:, 1, 0]
    scale = np.maximum(np.abs(moment).max(axis=(1, 2)), _TINY) ** 2
    # An *over-determined* fit, deliberately: three neighbours, not two.
    #
    # A triangle touching a boundary has two interior neighbours and two
    # unknowns, so the fit is exact -- it reproduces both neighbour
    # differences with nothing left over to damp it.  On a wall cell of
    # aspect ratio a thousand both neighbours lie nearly along the wall, so
    # the wall-normal component of that exact fit is amplified by the
    # reciprocal of a very small number.  Measured: the march was stable in
    # the interior and grew the density of the wall cells from 1.07 to 3.35
    # in forty steps, at every cfl tried.
    #
    # Those cells get no gradient and stay first order, which is the same
    # answer the boundary *faces* already get and for the same reason.
    ok = (valid.sum(axis=1) >= 3) & (np.abs(det) > 1e-12 * scale)
    safe = np.where(ok, det, 1.0)
    inv = np.empty_like(moment)
    inv[:, 0, 0] = moment[:, 1, 1] / safe
    inv[:, 1, 1] = moment[:, 0, 0] / safe
    inv[:, 0, 1] = -moment[:, 0, 1] / safe
    inv[:, 1, 0] = -moment[:, 1, 0] / safe
    inv[~ok] = 0.0  # no determined gradient: first order at this cell
    return _Stencil(neighbour, valid, weighted, inv, offset)


def _face_offsets(metrics: Metrics, xp=np):
    """``x_face - x_cell`` for each face, from both sides; and for boundaries."""
    left, right = metrics.interior_left, metrics.interior_right
    mid = xp.asarray(metrics.interior_midpoint)
    centroid = xp.asarray(metrics.centroid)
    return mid - centroid[left], mid - centroid[right]


def neighbour_bounds(U, metrics: Metrics, xp=np):
    """Per-cell ``(min, max)`` over the cell's own mean and its face neighbours.

    The window a monotone reconstruction may not leave.  One extra scatter
    pass over the interior faces; boundary faces contribute nothing, which
    leaves a boundary cell bounded by its interior neighbours alone.
    """
    left, right = metrics.interior_left, metrics.interior_right
    lo = _scatter(U, left, U[right], xp, "min")
    lo = _scatter(lo, right, U[left], xp, "min")
    hi = _scatter(U, left, U[right], xp, "max")
    hi = _scatter(hi, right, U[left], xp, "max")
    return lo, hi


def _scatter(base, index, values, xp, how: str):
    """``min``/``max`` scatter -- the one reduction NumPy and JAX spell apart."""
    if xp is np:
        out = np.array(base, copy=True)
        (np.minimum if how == "min" else np.maximum).at(out, index, values)
        return out
    return base.at[index].min(values) if how == "min" else base.at[index].max(values)


def _superbee(r, xp=np):
    r"""Roe's limiter, :math:`\phi(r) = \max(0, \min(2r, 1), \min(r, 2))`.

    The most compressive of the standard TVD family, which is why it is the
    one the companion nozzle code also offers: it keeps a contact sharp where
    minmod would smear it over several cells across a long run.
    """
    return xp.maximum(0.0, xp.maximum(xp.minimum(2.0 * r, 1.0), xp.minimum(r, 2.0)))


def limiter_factor(U, grad, metrics: Metrics, kind: str, gamma: float, *, xp=np):
    r"""The per-cell, per-component factor :math:`\Phi \in [0, 1]` on the slope.

    Shaped ``(n_elements, 1, 4)`` so it broadcasts against the ``(n, 2, 4)``
    gradient.  ``positivity`` returns the same scalar for all four components,
    because scaling only one of them would not keep the *state* admissible.
    """
    if kind == "none":
        return xp.ones((metrics.n_elements, 1, 4))
    if kind == "superbee":
        raise ValueError("superbee is applied per face; see _superbee_faces")
    if kind == "positivity":
        return _positivity_factor(U, grad, metrics, gamma, xp=xp)
    raise ValueError(f"unknown limiter {kind!r}; choose from {LIMITERS}")


def _cell_face_increments(grad, metrics: Metrics, xp=np):
    """``grad . (x_f - x_e)`` for all three faces of every cell, shape ``(n, 3, 4)``."""

    offset = xp.asarray(_cached_stencil(metrics).offset)  # (n, 3, 2)
    return xp.einsum("ekd,edc->ekc", offset, grad)


def _superbee_faces(U, grad, metrics: Metrics, *, xp=np):
    r"""Face-based MUSCL with Roe's limiter, along the line joining centroids.

    For an interior face between :math:`L` and :math:`R`, with
    :math:`\Delta = \bar{U}_R - \bar{U}_L` the centred difference and the
    cell gradient supplying the upwind-biased one,

    .. math::
        \tilde{\Delta}_L = 2\,\nabla\bar{U}_L \cdot (\vec{x}_R - \vec{x}_L)
                             - \Delta, \qquad
        U_L^{f} = \bar{U}_L + \tfrac12 \phi(r_L)\, \Delta, \quad
        r_L = \tilde{\Delta}_L / \Delta .

    **Face-based and not cell-based**, which matters and is easy to get
    wrong: Roe's :math:`\phi` reaches 2, and scaling a cell's whole slope by
    up to 2 against the room to its neighbours' extremes overshoots them by
    construction.  Here the same :math:`\phi = 2` gives
    :math:`U_L^{f} = \bar{U}_L + \Delta = \bar{U}_R` exactly -- the
    neighbour's own mean, the most compressive value that is still inside the
    data.  Written the first way this limiter produced negative densities at
    a face; written this way it cannot.

    Boundary faces have no second cell to difference against, so they stay
    first order.  That costs accuracy only in the one layer of cells touching
    the wall and the far field.
    """
    left, right = metrics.interior_left, metrics.interior_right
    centroid = xp.asarray(metrics.centroid)
    dx = centroid[right] - centroid[left]

    delta = U[right] - U[left]
    safe = xp.where(xp.abs(delta) > _TINY, delta, _TINY)
    flat = xp.abs(delta) <= _TINY

    up_l = 2.0 * xp.einsum("fd,fdc->fc", dx, grad[left]) - delta
    up_r = 2.0 * xp.einsum("fd,fdc->fc", dx, grad[right]) - delta
    phi_l = xp.where(flat, 0.0, _superbee(up_l / safe, xp))
    phi_r = xp.where(flat, 0.0, _superbee(up_r / safe, xp))

    UL = U[left] + 0.5 * phi_l * delta
    UR = U[right] - 0.5 * phi_r * delta
    return UL, UR


def _positivity_factor(U, grad, metrics: Metrics, gamma: float, *, xp=np):
    r"""The largest common slope scaling that keeps every face state physical.

    Zhang--Shu, in the form a cell average admits: find the smallest
    :math:`\theta \in [0, 1]` for which

    .. math::
        \bar{U}_e + \theta\, \nabla \bar{U}_e \cdot (\vec{x}_f - \vec{x}_e)

    has positive density and pressure at every face of the cell.  Density is
    linear in :math:`\theta` and solved directly; pressure is not, so it is
    bisected -- eight iterations, which brackets :math:`\theta` to 1/256 and
    always from the safe side, because the interval starts at a
    :math:`\theta = 0` that is admissible by construction.

    One factor for all four components: scaling momentum without density
    would change the velocity the reconstruction implies, and the point is to
    keep the *state* admissible, not each number separately.
    """
    delta = _cell_face_increments(grad, metrics, xp=xp)  # (n, 3, 4)
    mean = U[:, None, :]

    rho_mean = xp.maximum(mean[..., 0], ph.FLOOR)
    floor = _POSITIVITY_FRACTION * rho_mean
    drho = delta[..., 0]
    # rho(theta) = rho_mean + theta * drho >= floor
    need = rho_mean - floor
    theta = xp.where(drho < -_TINY, xp.minimum(1.0, need / xp.maximum(-drho, _TINY)), 1.0)

    def pressure_at(th):
        s = mean + th[..., None] * delta
        rho = xp.maximum(s[..., 0], ph.FLOOR)
        ke = 0.5 * (s[..., 1] ** 2 + s[..., 2] ** 2) / rho
        return (gamma - 1.0) * (s[..., 3] - ke)

    p_floor = _POSITIVITY_FRACTION * xp.maximum(pressure_at(xp.zeros_like(theta)), ph.FLOOR)
    lo = xp.zeros_like(theta)
    hi = theta
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        ok = pressure_at(mid) >= p_floor
        lo = xp.where(ok, mid, lo)
        hi = xp.where(ok, hi, mid)
    return xp.min(lo, axis=1)[:, None, None] * xp.ones((1, 1, 4))


def _pull_back_to_admissible(mean, face, gamma, *, xp=np):
    r"""Blend a face state back toward its cell mean until it is physical.

    Returns :math:`\bar{U} + \theta (U^f - \bar{U})` for the largest
    :math:`\theta \in [0, 1]` that keeps density and pressure above a tenth
    of the cell mean's.  Density is linear in :math:`\theta` and solved;
    pressure is not and is bisected eight times, always from the admissible
    side, since :math:`\theta = 0` returns the mean itself.
    """
    delta = face - mean
    rho_mean = xp.maximum(mean[..., 0], ph.FLOOR)
    floor = _POSITIVITY_FRACTION * rho_mean
    drho = delta[..., 0]
    theta = xp.where(
        drho < -_TINY,
        xp.minimum(1.0, (rho_mean - floor) / xp.maximum(-drho, _TINY)),
        1.0,
    )

    def pressure(th):
        s = mean + th[..., None] * delta
        rho = xp.maximum(s[..., 0], ph.FLOOR)
        ke = 0.5 * (s[..., 1] ** 2 + s[..., 2] ** 2) / rho
        return (gamma - 1.0) * (s[..., 3] - ke)

    p_floor = _POSITIVITY_FRACTION * xp.maximum(pressure(xp.zeros_like(theta)), ph.FLOOR)
    lo = xp.zeros_like(theta)
    hi = theta
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        ok = pressure(mid) >= p_floor
        lo = xp.where(ok, mid, lo)
        hi = xp.where(ok, hi, mid)
    return mean + lo[..., None] * delta


def face_states(U, metrics: Metrics, flow: FlowConditions, *, xp=np):
    r"""Left and right states at every interior face, and at every boundary face.

    With ``reconstruction='constant'`` these are the neighbouring cell
    averages and this function is never called.  With ``'muscl'`` they are the
    limited linear extrapolations to the face midpoint.
    """
    grad = cell_gradients(U, metrics, xp=xp)
    left, right = metrics.interior_left, metrics.interior_right
    element = metrics.boundary_element

    if flow.limiter == "superbee":
        # Face-based, and first order on the boundary layer of cells.
        UL, UR = _superbee_faces(U, grad, metrics, xp=xp)
        # Positivity is still enforced afterwards, exactly as the companion
        # nozzle code does and for the same reason: a slope limiter bounds the
        # *range* of the conserved variables by the neighbouring means, which
        # does not bound a nonlinear function of them.  Measured on a rough
        # state, superbee alone kept density positive and let pressure reach
        # -37; the Riemann solvers need both.
        UL = _pull_back_to_admissible(U[left], UL, flow.gamma, xp=xp)
        UR = _pull_back_to_admissible(U[right], UR, flow.gamma, xp=xp)
        return UL, UR, U[element]

    # `none` and `positivity` scale the cell's own slope, so they share the
    # extrapolation and differ only in the factor.
    phi = limiter_factor(U, grad, metrics, flow.limiter, flow.gamma, xp=xp)
    slope = grad * phi

    off_l, off_r = _face_offsets(metrics, xp)
    UL = U[left] + xp.einsum("fd,fdc->fc", off_l, slope[left])
    UR = U[right] + xp.einsum("fd,fdc->fc", off_r, slope[right])
    # Boundary faces keep the cell mean -- first order in the one layer of
    # cells touching the wall and the far field.  Extrapolating there feeds
    # the wall and characteristic conditions a state the reconstruction
    # invented rather than one the solution supports, and it destabilised the
    # march outright: the residual grew by 1e35 in fifty steps.
    return UL, UR, U[element]


#: Below this a slope increment is treated as exactly zero -- it constrains no
#: limiter and must not reach a division.
_TINY = 1e-300

#: How much of the cell mean the limiter guarantees to leave at a face.  Not
#: zero: a face state of exactly zero density is admissible to the limiter and
#: catastrophic to the flux, which divides by it.
_POSITIVITY_FRACTION = 0.1

__all__ = [
    "LIMITERS",
    "RECONSTRUCTIONS",
    "face_states",
    "cell_gradients",
    "limiter_factor",
    "neighbour_bounds",
]
