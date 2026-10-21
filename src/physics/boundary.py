r"""The two boundary conditions: an inviscid wall, and a characteristic far field.

Each is written as a *flux*, not as a ghost state to be filled in, which is what
keeps the boundary treatment inside the same conservative statement as every
interior face.
"""

from __future__ import annotations

import numpy as np

from .fluxes import interface_flux
from .thermo import FLOOR, FluxResult, conservative, primitives, sound_speed


# --------------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------------
def wall_flux(U, nx, ny, gamma: float, *, xp=np) -> FluxResult:
    r"""Inviscid wall: no flow through the surface, so the flux carries pressure only.

    .. math::
        \hat{\mathbf{F}}_{\text{wall}} = \left(0,\ p_b n_x,\ p_b n_y,\ 0\right)^T,
        \qquad
        p_b = (\gamma - 1)\left(\rho E
              - \tfrac12 \rho\,\lvert \vec{v} - (\vec{v}\cdot\hat{n})\hat{n}\rvert^2\right),

    the pressure of the interior state **with its normal velocity removed**
    rather than the interior pressure itself.  The two differ by
    :math:`\tfrac12(\gamma-1)\rho v_n^2`, which is zero only once the solution
    has converged -- so using the interior pressure directly would add a
    spurious surface force that decays with the residual and contaminates
    :math:`c_\ell` on the way.  This is what ``legacy/residual_calc.m`` did, and
    it is kept unchanged.
    """
    rho = xp.maximum(U[..., 0], FLOOR)
    vx = U[..., 1] / rho
    vy = U[..., 2] / rho
    vn = vx * nx + vy * ny
    qt2 = (vx - vn * nx) ** 2 + (vy - vn * ny) ** 2
    p = xp.maximum((gamma - 1.0) * (U[..., 3] - 0.5 * rho * qt2), FLOOR)
    zero = xp.zeros_like(p)
    flux = xp.stack([zero, p * nx + zero, p * ny + zero, zero], axis=-1)
    return FluxResult(flux, sound_speed(rho, p, gamma, xp=xp) + xp.abs(vn))


def farfield_state(U, U_inf, nx, ny, gamma: float, *, xp=np):
    r"""The far-field state from the one-dimensional Riemann invariants.

    The legacy code imposed the **whole** free-stream state here and let the
    Riemann solver sort it out.  That over-specifies a subsonic boundary: four
    conditions where the characteristics ask for one, which reflects outgoing
    waves back into the domain and is why the legacy domain had to be 50 chords
    across.

    A characteristic condition imposes only what the incoming characteristics
    carry.  With :math:`\hat{n}` pointing out of the domain, the invariant
    riding the outgoing acoustic wave comes from inside and the one riding the
    incoming wave comes from the free stream:

    .. math::
        R^{+} = v_n^{+} + \frac{2a^{+}}{\gamma - 1}, \qquad
        R^{-} = v_{n\infty} - \frac{2a_\infty}{\gamma - 1},

    .. math::
        v_{nb} = \tfrac12\left(R^{+} + R^{-}\right), \qquad
        a_b = \tfrac{\gamma-1}{4}\left(R^{+} - R^{-}\right).

    Entropy and the tangential velocity are convected, so they come from
    whichever side the flow is going: the free stream on an inflow face, the
    interior on an outflow one.  Supersonic faces take everything from one side,
    and all four cases are selected with :func:`numpy.where` rather than a
    branch, so the whole function stays differentiable and vectorised.
    """
    rho, vx, vy, p, _ = primitives(U, gamma, xp=xp)
    a = sound_speed(rho, p, gamma, xp=xp)
    vn = vx * nx + vy * ny

    rho_i, vx_i, vy_i, p_i, _ = primitives(U_inf, gamma, xp=xp)
    a_i = sound_speed(rho_i, p_i, gamma, xp=xp)
    vn_i = vx_i * nx + vy_i * ny

    two = 2.0 / (gamma - 1.0)
    r_plus = vn + two * a
    r_minus = vn_i - two * a_i
    vnb = 0.5 * (r_plus + r_minus)
    ab = 0.25 * (gamma - 1.0) * (r_plus - r_minus)
    ab = xp.maximum(ab, FLOOR)

    outgoing = vnb > 0.0  # the flow leaves the domain through this face
    rho_s = xp.where(outgoing, rho, rho_i)
    p_s = xp.where(outgoing, p, p_i)
    vx_s = xp.where(outgoing, vx, vx_i)
    vy_s = xp.where(outgoing, vy, vy_i)
    vn_s = xp.where(outgoing, vn, vn_i)

    entropy = p_s / rho_s**gamma
    rho_b = (ab * ab / (gamma * entropy)) ** (1.0 / (gamma - 1.0))
    p_b = rho_b * ab * ab / gamma
    vx_b = vx_s + (vnb - vn_s) * nx
    vy_b = vy_s + (vnb - vn_s) * ny
    subsonic = conservative(rho_b, vx_b, vy_b, p_b, gamma, xp=xp)

    # supersonic: every characteristic points the same way, so one side decides
    supersonic_out = (vn >= a)[..., None]
    supersonic_in = (vn <= -a)[..., None]
    return xp.where(supersonic_in, U_inf + xp.zeros_like(U), xp.where(supersonic_out, U, subsonic))


def farfield_flux(
    U, U_inf, nx, ny, gamma: float, *, flux: str = "roe", xp=np, **kwargs
) -> FluxResult:
    """The far-field flux: build the characteristic state, then solve across it."""
    return interface_flux(
        flux, U, farfield_state(U, U_inf, nx, ny, gamma, xp=xp), nx, ny, gamma, xp=xp, **kwargs
    )


__all__ = [
    "farfield_flux",
    "farfield_state",
    "wall_flux",
]
