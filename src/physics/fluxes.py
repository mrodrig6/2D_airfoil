r"""The approximate Riemann solvers, and the dispatch between them.

**The approximate Riemann solver is the upwinding.**  There is no separate
upwind operator anywhere in this code: the reconstruction hands each solver two
cell averages, and the solver's own wave structure decides which side each
characteristic comes from.  Each takes a left state, a right state and the face
normal, and returns the flux together with the largest signal speed it saw --
which is what the local time step is built from.

=============  ==============================================================
``roe``        Roe's linearised solver with the Harten-Hyman entropy fix.  The
               default, and what ``legacy/residual_calc.m`` implements.
``hllc``       HLLC with Batten's wave speeds.  Positivity-preserving, with no
               tunable constant.
``ausmup``     AUSM\ :sup:`+`-up.  A flux split rather than a wave
               decomposition, so it agrees with the other two for completely
               independent reasons.  Converges on every transonic case tried;
               limit-cycles at low free-stream Mach number on a fine mesh.
=============  ==============================================================

A fourth, SLAU2, was implemented, measured over sixty solves and removed: it
stops converging above a peak Mach number of about 0.91, which rules out every
transonic case on this geometry.  ``docs/riemann_fluxes.md`` is the record --
including the retraction of an earlier verdict on AUSM\ :sup:`+`-up that turned
out to be about a bug in its implementation rather than about the scheme.

References
----------
1. P. L. Roe, *J. Comput. Phys.* **43**, 357-372, 1981.
2. A. Harten and J. M. Hyman, *J. Comput. Phys.* **50**, 235-269, 1983.
3. P. Batten et al., *SIAM J. Sci. Comput.* **18**(6), 1553-1570, 1997.
4. M.-S. Liou, *J. Comput. Phys.* **214**, 137-170, 2006.
5. K. Kitamura and E. Shima, *J. Comput. Phys.* **245**, 62-83, 2013.  (SLAU2,
   removed)
"""

from __future__ import annotations

import numpy as np

from .thermo import FLOOR, FluxResult, normal_flux, primitives, sound_speed


# --------------------------------------------------------------------------
# Roe
# --------------------------------------------------------------------------
def roe_flux(UL, UR, nx, ny, gamma: float, *, entropy_fix: float = 0.05, xp=np) -> FluxResult:
    r"""Roe's approximate Riemann solver with the Harten-Hyman entropy fix.

    .. math::
        \hat{\mathbf{F}} = \tfrac12\left(\mathbf{F}_L + \mathbf{F}_R\right)
          - \tfrac12 \lvert \hat{\mathbf{A}} \rvert \,\Delta \mathbf{U},

    where :math:`\hat{\mathbf{A}}` is the flux Jacobian at the Roe-averaged
    state -- the density-weighted average

    .. math::
        \hat{u} = \frac{\sqrt{\rho_L}\,u_L + \sqrt{\rho_R}\,u_R}
                       {\sqrt{\rho_L} + \sqrt{\rho_R}},

    and likewise for :math:`v` and :math:`H`.  That particular weighting is the
    one that makes the linearisation **exact across a single discontinuity**,
    which is why a Roe flux captures a steady shock in one or two cells.

    **The entropy fix.**  :math:`\lvert\hat{\mathbf{A}}\rvert` vanishes where an
    eigenvalue does, so at a sonic point the scheme adds no dissipation and
    admits an expansion shock -- an entropy-violating solution that the equations
    allow and physics does not.  Harten and Hyman replace the kink in
    :math:`\lvert\lambda\rvert` with a parabola over
    :math:`\lvert\lambda\rvert < \epsilon`,

    .. math::
        \lvert\lambda\rvert \;\rightarrow\;
        \frac{\lambda^2 + \epsilon^2}{2\epsilon}, \qquad \epsilon = \delta\,\hat{a},

    which is smooth, never decreases the dissipation, and leaves everything away
    from the sonic point untouched.  ``legacy/residual_calc.m`` used
    :math:`\delta = 0.01`; the default here is 0.05, which is the usual choice
    for a transonic march and matters because this flow *has* a sonic line.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny

    FL = normal_flux(UL, nx, ny, gamma, xp=xp)
    FR = normal_flux(UR, nx, ny, gamma, xp=xp)

    wL = xp.sqrt(rhoL)
    wR = xp.sqrt(rhoR)
    inv = 1.0 / (wL + wR)
    u = (wL * vxL + wR * vxR) * inv
    v = (wL * vyL + wR * vyR) * inv
    H = (wL * HL + wR * HR) * inv
    q2 = u * u + v * v
    a = xp.sqrt(xp.maximum((gamma - 1.0) * (H - 0.5 * q2), FLOOR))
    vn = u * nx + v * ny

    eps = entropy_fix * a
    lam = xp.abs(xp.stack([vn + a, vn - a, vn], axis=-1))
    # xp.where evaluates both branches, so the division needs a denominator that
    # is finite even when the fix is switched off entirely (eps = 0); the branch
    # it feeds is never selected then, since no |lambda| is below zero
    safe = xp.where(eps > 0.0, eps, 1.0)[..., None]
    lam = xp.where(lam < eps[..., None], 0.5 * (lam * lam / safe + eps[..., None]), lam)
    l1, l2, l3 = lam[..., 0], lam[..., 1], lam[..., 2]

    dU = UR - UL
    drho, dru, drv, drE = dU[..., 0], dU[..., 1], dU[..., 2], dU[..., 3]
    g1 = (gamma - 1.0) * (0.5 * q2 * drho - u * dru - v * drv + drE)
    g2 = -vn * drho + dru * nx + drv * ny

    s1 = 0.5 * (l1 + l2)
    s2 = 0.5 * (l1 - l2)
    c1 = (g1 / (a * a)) * (s1 - l3) + (g2 / a) * s2
    c2 = (g1 / a) * s2 + (s1 - l3) * g2

    diss = xp.stack(
        [
            l3 * drho + c1,
            l3 * dru + c1 * u + c2 * nx,
            l3 * drv + c1 * v + c2 * ny,
            l3 * drE + c1 * H + c2 * vn,
        ],
        axis=-1,
    )
    aL = sound_speed(rhoL, pL, gamma, xp=xp)
    aR = sound_speed(rhoR, pR, gamma, xp=xp)
    speed = xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR)
    return FluxResult(0.5 * (FL + FR) - 0.5 * diss, speed)


# --------------------------------------------------------------------------
# HLLC
# --------------------------------------------------------------------------
def hllc_flux(UL, UR, nx, ny, gamma: float, *, xp=np) -> FluxResult:
    r"""HLLC with Batten's wave speeds.

    Where Roe linearises, HLLC *assumes a wave pattern*: two outer waves at
    speeds :math:`S_L` and :math:`S_R` with a contact at :math:`S_*` between
    them, and a constant state in each of the four regions.  Restoring that
    middle wave is what separates HLLC from HLLE, and it is what lets it carry a
    contact discontinuity without smearing it.

    .. math::
        S_* = \frac{p_R - p_L + \rho_L v_{nL}(S_L - v_{nL})
                    - \rho_R v_{nR}(S_R - v_{nR})}
                   {\rho_L (S_L - v_{nL}) - \rho_R (S_R - v_{nR})}

    Batten's choice of :math:`S_L` and :math:`S_R` -- from the Roe-averaged
    eigenvalues, bracketed with the two one-sided ones -- is what makes the
    scheme **provably positivity-preserving**: density and pressure in the star
    states cannot go negative.  It has no tunable constant and needs no entropy
    fix, which is why it is the natural second opinion when a Roe result looks
    suspicious.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = sound_speed(rhoL, pL, gamma, xp=xp)
    aR = sound_speed(rhoR, pR, gamma, xp=xp)

    wL = xp.sqrt(rhoL)
    wR = xp.sqrt(rhoR)
    inv = 1.0 / (wL + wR)
    u = (wL * vxL + wR * vxR) * inv
    v = (wL * vyL + wR * vyR) * inv
    H = (wL * HL + wR * HR) * inv
    a = xp.sqrt(xp.maximum((gamma - 1.0) * (H - 0.5 * (u * u + v * v)), FLOOR))
    vn = u * nx + v * ny

    SL = xp.minimum(vnL - aL, vn - a)
    SR = xp.maximum(vnR + aR, vn + a)

    mL = rhoL * (SL - vnL)
    mR = rhoR * (SR - vnR)
    denom = mL - mR
    denom = xp.where(xp.abs(denom) < FLOOR, xp.where(denom >= 0, FLOOR, -FLOOR), denom)
    SM = (pR - pL + mL * vnL - mR * vnR) / denom

    FL = normal_flux(UL, nx, ny, gamma, xp=xp)
    FR = normal_flux(UR, nx, ny, gamma, xp=xp)

    def star(U, rho, vx, vy, vn_side, p, S):
        """The HLLC star state on one side."""
        factor = (S - vn_side) / (S - SM)
        rho_s = rho * factor
        p_s = p + rho * (vn_side - SM) * (vn_side - S)
        vx_s = vx + (SM - vn_side) * nx
        vy_s = vy + (SM - vn_side) * ny
        rE_s = factor * U[..., 3] + (p_s * SM - p * vn_side) / (S - SM)
        return xp.stack([rho_s, rho_s * vx_s, rho_s * vy_s, rE_s], axis=-1)

    UsL = star(UL, rhoL, vxL, vyL, vnL, pL, SL)
    UsR = star(UR, rhoR, vxR, vyR, vnR, pR, SR)
    FsL = FL + SL[..., None] * (UsL - UL)
    FsR = FR + SR[..., None] * (UsR - UR)

    flux = xp.where(
        (SL >= 0.0)[..., None],
        FL,
        xp.where((SM >= 0.0)[..., None], FsL, xp.where((SR >= 0.0)[..., None], FsR, FR)),
    )
    return FluxResult(flux, xp.maximum(xp.abs(SL), xp.abs(SR)))


# --------------------------------------------------------------------------
# AUSM+-up
# --------------------------------------------------------------------------
# beta = 1/8 and alpha = 3/16 are **not** tunable: they are the unique values
# that make the split polynomials of the stated degrees smooth at |M| = 1
# (Liou, J. Comput. Phys. 129:364, 1996).
_AUSM_BETA = 0.125
_AUSM_ALPHA = 0.1875

#: The two "-up" diffusion coefficients (Liou, J. Comput. Phys. 214:137, 2006).
#: These *are* tunable, and these values are not guesses: they are the ones
#: benchmarked in EMS, the group's Eulerian Multicomponent Solver, carried over
#: through ``mrodrig6/approximate_riemann_solver``.  That repository records
#: what happens when they are reconstructed from memory instead -- a ``K_u``
#: 7.5 times too large, anti-dissipative on a moving contact -- which is why
#: they are quoted here rather than re-derived.
_AUSM_KP = 0.4
_AUSM_KU = 0.2


def _m4_plus(m, beta, xp=np):
    r""":math:`M^{+}_{(4)}`: :math:`M^{+}_{(1)}` supersonic, else the quartic."""
    m1 = 0.5 * (m + xp.abs(m))
    m2p = 0.25 * (m + 1.0) ** 2
    m2m = -0.25 * (m - 1.0) ** 2
    return xp.where(xp.abs(m) >= 1.0, m1, m2p * (1.0 - 16.0 * beta * m2m))


def _m4_minus(m, beta, xp=np):
    r""":math:`M^{-}_{(4)}`."""
    m1 = 0.5 * (m - xp.abs(m))
    m2p = 0.25 * (m + 1.0) ** 2
    m2m = -0.25 * (m - 1.0) ** 2
    return xp.where(xp.abs(m) >= 1.0, m1, m2m * (1.0 + 16.0 * beta * m2p))


def _p5_plus(m, alpha, xp=np):
    r""":math:`P^{+}_{(5)}`: 1 upstream, 0 downstream, the quintic between.

    The supersonic branch is :math:`M^{+}_{(1)}/M`, which divides by zero at
    :math:`M = 0`.  Both branches of an ``xp.where`` are evaluated, so the
    denominator is guarded even though that branch is never selected there --
    otherwise a NaN appears in the unused half and propagates straight through
    into a JAX gradient.
    """
    supersonic = xp.abs(m) >= 1.0
    safe = xp.where(supersonic, m, 1.0)
    m2p = 0.25 * (m + 1.0) ** 2
    m2m = -0.25 * (m - 1.0) ** 2
    return xp.where(
        supersonic,
        0.5 * (m + xp.abs(m)) / safe,
        m2p * ((2.0 - m) - 16.0 * alpha * m * m2m),
    )


def _p5_minus(m, alpha, xp=np):
    r""":math:`P^{-}_{(5)}`; see :func:`_p5_plus` for the guarded denominator."""
    supersonic = xp.abs(m) >= 1.0
    safe = xp.where(supersonic, m, 1.0)
    m2p = 0.25 * (m + 1.0) ** 2
    m2m = -0.25 * (m - 1.0) ** 2
    return xp.where(
        supersonic,
        0.5 * (m - xp.abs(m)) / safe,
        m2m * ((-2.0 - m) + 16.0 * alpha * m * m2p),
    )


def ausmup_flux(UL, UR, nx, ny, gamma: float, *, xp=np) -> FluxResult:
    r"""AUSM\ :sup:`+`-up (Liou, J. Comput. Phys. **214**:137, 2006).

    Neither Roe nor HLLC's idea.  Where those decompose the jump into waves,
    AUSM splits the *flux* into a convective part carried bodily by one
    interface velocity and a pressure part handled separately:

    .. math::
        \mathbf{F} = u_{1/2}\,\mathbf{\Psi}_{\text{upwind}}
                   + p_{1/2}\,[0,\ n_x,\ n_y,\ 0]^{T},
        \qquad
        \mathbf{\Psi} = [\rho,\ \rho v_x,\ \rho v_y,\ \rho E + p]^{T}

    with :math:`\mathbf{\Psi}` taken from whichever side :math:`u_{1/2}` blows
    from.  Because all the pressure lives in :math:`p_{1/2}`, the convective
    part carries no pressure dissipation and a contact -- stationary or moving
    -- comes through exactly.

    The interface velocity and pressure are polynomial splittings of each
    side's normal Mach number :math:`M_K = v_{nK}/a_{1/2}`, plus the two terms
    that put the "up" in the name:

    .. math::
        u_{1/2} = a_{1/2}\left[M^{+}_{(4)}(M_L) + M^{-}_{(4)}(M_R)\right]
                + K_p \max(1 - \bar{M}^2, 0)
                  \frac{p_L - p_R}{\rho_{1/2} a_{1/2}}

    .. math::
        p_{1/2} = P^{+}_{(5)} p_L + P^{-}_{(5)} p_R
                + K_u P^{+}_{(5)} P^{-}_{(5)} \rho_{1/2} a_{1/2}
                  (v_{nL} - v_{nR})

    with :math:`a_{1/2}` and :math:`\rho_{1/2}` plain arithmetic means and
    :math:`\bar{M}^2 = (M_L^2 + M_R^2)/2`.

    **The** :math:`K_p` **term is not optional.**  Without it -- plain AUSM+ --
    the mass flux is identically zero whenever :math:`v_{nL} = v_{nR} = 0`,
    because :math:`M^{+}_{(4)}(0) + M^{-}_{(4)}(0) = 0` exactly, however large
    the pressure jump across the face.  Momentum still feels the jump and
    accelerates the gas while the energy flux stays zero, so the growing
    kinetic energy is taken out of a fixed total and the internal energy goes
    negative on the light side.  It supplies the missing pressure--velocity
    coupling.

    **On this geometry it is the dissipative choice**, not a free upgrade: on
    the transonic NACA 0012 it gives about 15% more drag than Roe at the same
    resolution.  ``docs/riemann_fluxes.md`` measures that.  What it buys is
    robustness where Roe's linearisation is least comfortable, and a second
    opinion from a completely different construction.
    """
    rhoL, vxL, vyL, pL, _ = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, _ = primitives(UR, gamma, xp=xp)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = sound_speed(rhoL, pL, gamma, xp=xp)
    aR = sound_speed(rhoR, pR, gamma, xp=xp)

    # Arithmetic means, as EMS uses -- *not* Liou's critical-speed
    # construction, and with no f_a all-speed scaling.  That matters: this
    # formulation has no tunable cutoff Mach number, which is what an earlier
    # attempt in this repository foundered on.
    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)

    machL = vnL / a_half
    machR = vnR / a_half
    mbar_sq = 0.5 * (machL * machL + machR * machR)

    # max(1 - Mbar^2, 0) switches the pressure diffusion off once the interface
    # is supersonic, where the upwinding is dissipative enough on its own.
    u_half = a_half * (
        _m4_plus(machL, _AUSM_BETA, xp=xp) + _m4_minus(machR, _AUSM_BETA, xp=xp)
    ) + _AUSM_KP * xp.maximum(1.0 - mbar_sq, 0.0) * (pL - pR) / (rho_half * a_half)

    p5p = _p5_plus(machL, _AUSM_ALPHA, xp=xp)
    p5m = _p5_minus(machR, _AUSM_ALPHA, xp=xp)
    p_half = p5p * pL + p5m * pR + _AUSM_KU * p5p * p5m * rho_half * a_half * (vnL - vnR)

    # Psi = [rho, rho vx, rho vy, rho E + p], upwinded on the sign of u_1/2.
    psiL = xp.stack([rhoL, UL[..., 1], UL[..., 2], UL[..., 3] + pL], axis=-1)
    psiR = xp.stack([rhoR, UR[..., 1], UR[..., 2], UR[..., 3] + pR], axis=-1)
    psi = xp.where((u_half > 0.0)[..., None], psiL, psiR)

    zero = xp.zeros_like(p_half)
    pressure_term = xp.stack([zero, p_half * nx, p_half * ny, zero], axis=-1)

    flux = u_half[..., None] * psi + pressure_term
    speed = xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR)
    return FluxResult(flux, speed)


#: The approximate Riemann solvers, by the name :class:`~src.config.FlowConditions`
#: uses.
SOLVERS = {
    "roe": roe_flux,
    "hllc": hllc_flux,
    "ausmup": ausmup_flux,
}


def interface_flux(name: str, UL, UR, nx, ny, gamma: float, *, xp=np, **kwargs):
    """Dispatch to one of :data:`SOLVERS` by name."""
    try:
        solver = SOLVERS[name]
    except KeyError:
        raise ValueError(f"unknown flux {name!r}; choose from {tuple(SOLVERS)}") from None
    return solver(UL, UR, nx, ny, gamma, xp=xp, **kwargs)


__all__ = [
    "SOLVERS",
    "ausmup_flux",
    "hllc_flux",
    "interface_flux",
    "roe_flux",
]
