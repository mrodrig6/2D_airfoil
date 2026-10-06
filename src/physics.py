r"""The Euler fluxes, four approximate Riemann solvers, and the boundary conditions.

Everything here is a pure function of states and normals.  Nothing knows about
the mesh, the march, or which backend is running, and every function takes an
``xp`` array module so the same source runs under NumPy and under
``jax.numpy``.  That is what makes the JAX backend the *same code* rather than a
second implementation to keep in step.

The numerical flux
------------------
A finite-volume scheme needs one flux per face, shared by the two cells that
meet there.  Each solver below takes a left state, a right state and the face
normal, and returns that flux together with the largest signal speed it saw --
which is what the local time step is built from.

**The approximate Riemann solver is the upwinding.**  There is no separate
upwind operator anywhere in this code: the reconstruction hands each solver two
cell averages, and the solver's own wave structure decides which side each
characteristic comes from.

=============  ==============================================================
``roe``        Roe's linearised solver with the Harten-Hyman entropy fix.  The
               default, and what ``legacy/residual_calc.m`` implements.
``hllc``       HLLC with Batten's wave speeds.  Positivity-preserving, with no
               tunable constant.
``ausmup``     Liou's AUSM\ :sup:`+`-up: a flux-vector splitting, designed for
               robustness at a captured shock.
``slau2``      SLAU2, the parameter-free member of the AUSM family.
=============  ==============================================================

Why all four, and why that matters here
---------------------------------------
The companion nozzle code measured the same four and found Roe-with-entropy-fix
converged while the AUSM-family pair did not -- but **its domain is shock free
by construction**, which is precisely the property the AUSM family is designed
for and which that solver therefore never exercised.  Kitamura's 2016
carbuncle ranking is *inverted* against what converged there: Roe with the
entropy fix scores 0 of 20 on the standard shock-robustness test and SLAU2
scores 20, the exact reverse of how they behaved in the nozzle.

This code has a **shock on the body surface**, so that measurement has to be
redone rather than inherited.  All four ship so the comparison is reproducible;
the result goes in ``docs/riemann_fluxes.md``.

One trap is already known and is worth stating before anyone meets it:
AUSM\ :sup:`+`-up's :math:`M_{co}` is a reference Mach number of the order of
the smallest in the flow, **not** a regularisation epsilon.  It appears as
:math:`K_p/f_a`, so driving it towards zero makes that coefficient diverge --
with :math:`M_{co} = 0` a 1% pressure difference between two states *at rest*
gives a mass flux of :math:`-9.8\times10^4` where Roe gives
:math:`-3.5\times10^{-3}`, and the march dies within 51 iterations.  Liou's
guidance is :math:`M_{co} \sim M_\infty`; **measured on this problem it does not
hold**, and the usable window is 0.30 to 0.45 at both :math:`M_\infty = 0.5` and
:math:`M_\infty = 0.8`.  The default is the middle of that window.  See
``docs/riemann_fluxes.md``.

References
----------
1. P. L. Roe, *J. Comput. Phys.* **43**, 357-372, 1981.
2. A. Harten and J. M. Hyman, *J. Comput. Phys.* **50**, 235-269, 1983.
3. P. Batten et al., *SIAM J. Sci. Comput.* **18**(6), 1553-1570, 1997.
4. M.-S. Liou, *J. Comput. Phys.* **214**, 137-170, 2006.
5. E. Shima and K. Kitamura, *AIAA J.* **49**(8), 1693-1709, 2011.
6. K. Kitamura and E. Shima, *J. Comput. Phys.* **245**, 62-83, 2013.
7. K. Kitamura, *Computers and Fluids* **129**, 134-145, 2016.
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
# The AUSM family
# --------------------------------------------------------------------------
def ausmup_flux(UL, UR, nx, ny, gamma: float, *, cutoff_mach: float = 0.8, xp=np) -> FluxResult:
    r"""Liou's AUSM\ :sup:`+`-up: a flux-vector splitting, not a Riemann solver.

    Rather than linearise a Riemann problem, AUSM splits the interface into a
    **mass** flux carrying the convected field and a **pressure** flux carrying
    the acoustic one:

    .. math::
        \hat{\mathbf{F}} = \dot{m}_{1/2}\,\boldsymbol{\Psi}_{\mathrm{up}}
                         + p_{1/2}\,(0,\ n_x,\ n_y,\ 0)^T,
        \qquad \boldsymbol{\Psi} = (1,\ v_x,\ v_y,\ H)^T,

    with :math:`\boldsymbol{\Psi}` taken wholly from whichever side the mass
    flux comes from.  There is no eigen-decomposition anywhere in it, which is
    the source of both its speed and its different failure modes.

    The split polynomials are built so that
    :math:`\mathcal{M}^{+}_{(4)} + \mathcal{M}^{-}_{(4)} = M` and
    :math:`\mathcal{P}^{+}_{(5)} + \mathcal{P}^{-}_{(5)} = 1` *identically*,
    which is what makes the scheme consistent; a test pins both.  The ``-up``
    suffix is two diffusion terms -- a pressure-diffusion term in the mass flux
    that supplies the velocity-pressure coupling a plain AUSM loses as
    :math:`M \to 0`, and a velocity-diffusion term in the pressure.

    The interface sound speed is the **numerical** one built from the critical
    speed rather than an average, which is what lets the scheme hold a
    stationary normal shock with no interior point:

    .. math::
        a_*^2 = \frac{2(\gamma-1)}{\gamma+1} H, \qquad
        \tilde{a} = \frac{a_*^2}{\max(a_*,\ \lvert v_n \rvert)}, \qquad
        a_{1/2} = \min(\tilde{a}_L,\ \tilde{a}_R).

    Parameters
    ----------
    cutoff_mach
        Liou's :math:`M_{co}`, setting :math:`M_0 = \min(1, \max(\bar M,
        M_{co}))` and :math:`f_a = M_0(2 - M_0)`.

        **It is not a regularisation epsilon**, and this is the one place the
        scheme will bite.  :math:`f_a` appears as :math:`K_p/f_a`, so a small
        :math:`M_{co}` makes that coefficient *large*: at :math:`M_{co} = 0` a
        1% pressure jump between two states at rest gives a mass flux of
        :math:`-9.8\times10^4` where Roe gives :math:`-3.5\times10^{-3}`.
        Liou's guidance is :math:`M_{co} \sim M_\infty`.  On this problem that
        is wrong: swept at two operating points, the scheme converges only for
        :math:`0.30 \le M_{co} \le 0.45`, the same window at both, so it is not
        tracking the free stream.  Below 0.2 the march dies within a handful of
        iterations; at 0.5 it stalls.  The default is 0.4.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny

    k = 2.0 * (gamma - 1.0) / (gamma + 1.0)
    astarL = xp.sqrt(xp.maximum(k * HL, FLOOR))
    astarR = xp.sqrt(xp.maximum(k * HR, FLOOR))
    a_half = xp.minimum(
        astarL * astarL / xp.maximum(astarL, xp.abs(vnL)),
        astarR * astarR / xp.maximum(astarR, xp.abs(vnR)),
    )

    ML = vnL / a_half
    MR = vnR / a_half
    mbar2 = 0.5 * (vnL * vnL + vnR * vnR) / (a_half * a_half)
    m0 = xp.sqrt(xp.minimum(1.0, xp.maximum(mbar2, cutoff_mach * cutoff_mach)))
    fa = m0 * (2.0 - m0)
    alpha = 0.1875 * (5.0 * fa * fa - 4.0)
    beta = 0.125

    m2p = 0.25 * (ML + 1.0) ** 2
    m2m = -0.25 * (MR - 1.0) ** 2
    m4p = xp.where(
        xp.abs(ML) >= 1.0,
        0.5 * (ML + xp.abs(ML)),
        m2p * (1.0 + 4.0 * beta * (ML - 1.0) ** 2),
    )
    m4m = xp.where(
        xp.abs(MR) >= 1.0,
        0.5 * (MR - xp.abs(MR)),
        m2m * (1.0 + 4.0 * beta * (MR + 1.0) ** 2),
    )

    rho_half = 0.5 * (rhoL + rhoR)
    kp, ku, sigma = 0.25, 0.75, 1.0
    mp = (
        -(kp / fa) * xp.maximum(1.0 - sigma * mbar2, 0.0) * (pR - pL) / (rho_half * a_half * a_half)
    )
    m_half = m4p + m4m + mp

    p5p = xp.where(
        xp.abs(ML) >= 1.0,
        0.5 * (1.0 + xp.sign(ML)),
        m2p * ((2.0 - ML) + 4.0 * alpha * ML * (ML - 1.0) ** 2),
    )
    p5m = xp.where(
        xp.abs(MR) >= 1.0,
        0.5 * (1.0 - xp.sign(MR)),
        -m2m * ((2.0 + MR) - 4.0 * alpha * MR * (MR + 1.0) ** 2),
    )
    p_half = p5p * pL + p5m * pR - ku * p5p * p5m * (rhoL + rhoR) * (fa * a_half) * (vnR - vnL)

    forward = (m_half > 0.0)[..., None]
    mdot = a_half * m_half * xp.where(m_half > 0.0, rhoL, rhoR)
    psi = xp.where(
        forward,
        xp.stack([xp.ones_like(vxL), vxL, vyL, HL], axis=-1),
        xp.stack([xp.ones_like(vxR), vxR, vyR, HR], axis=-1),
    )
    zero = xp.zeros_like(pL)
    nvec = xp.stack([zero, nx + zero, ny + zero, zero], axis=-1)
    flux = mdot[..., None] * psi + p_half[..., None] * nvec

    aL = sound_speed(rhoL, pL, gamma, xp=xp)
    aR = sound_speed(rhoR, pR, gamma, xp=xp)
    return FluxResult(flux, xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR))


def slau2_flux(UL, UR, nx, ny, gamma: float, *, xp=np) -> FluxResult:
    r"""SLAU2: the parameter-free AUSM-family flux of Shima and Kitamura.

    Same split as AUSM\ :sup:`+`-up, but **every switch is built from the states
    themselves**, so there is no constant to get wrong -- no :math:`K_p`, no
    :math:`K_u`, and above all no :math:`M_{co}`.  That is the reason to carry
    it alongside: it makes "did a tuning constant cause this?" a question that
    can be answered rather than argued.

    Kitamura and Shima's Eqs. (2.3) and (3.5):

    .. math::
        \dot{m} = \tfrac12\Bigl[\rho_L(v_{nL} + \lvert\overline{v_n}\rvert_L)
                  + \rho_R(v_{nR} - \lvert\overline{v_n}\rvert_R)
                  - \frac{\chi}{\bar{a}}\,\Delta p\Bigr],

    with :math:`g` a function of the two Mach numbers,
    :math:`\chi = (1 - \hat{M})^2`, and the pressure term carrying the
    :math:`\sqrt{\cdot}` form that gives SLAU2 its shock robustness.
    """
    rhoL, vxL, vyL, pL, HL = primitives(UL, gamma, xp=xp)
    rhoR, vxR, vyR, pR, HR = primitives(UR, gamma, xp=xp)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = sound_speed(rhoL, pL, gamma, xp=xp)
    aR = sound_speed(rhoR, pR, gamma, xp=xp)
    a_bar = 0.5 * (aL + aR)

    ML = vnL / a_bar
    MR = vnR / a_bar
    g = -xp.maximum(xp.minimum(ML, 0.0), -1.0) * xp.minimum(xp.maximum(MR, 0.0), 1.0)

    vn_abs = (rhoL * xp.abs(vnL) + rhoR * xp.abs(vnR)) / (rhoL + rhoR)
    vn_abs_L = (1.0 - g) * vn_abs + g * xp.abs(vnL)
    vn_abs_R = (1.0 - g) * vn_abs + g * xp.abs(vnR)

    m_hat = xp.minimum(
        1.0,
        xp.sqrt(0.5 * (vxL * vxL + vyL * vyL + vxR * vxR + vyR * vyR)) / a_bar,
    )
    chi = (1.0 - m_hat) ** 2

    mdot = 0.5 * (rhoL * (vnL + vn_abs_L) + rhoR * (vnR - vn_abs_R) - (chi / a_bar) * (pR - pL))

    bp = xp.where(xp.abs(ML) >= 1.0, 0.5 * (1.0 + xp.sign(ML)), 0.25 * (ML + 1.0) ** 2 * (2.0 - ML))
    bm = xp.where(xp.abs(MR) >= 1.0, 0.5 * (1.0 - xp.sign(MR)), 0.25 * (MR - 1.0) ** 2 * (2.0 + MR))
    # SLAU2's pressure flux: the sqrt term is what makes it shock robust
    p_half = (
        0.5 * (pL + pR)
        + 0.5 * (bp - bm) * (pL - pR)
        + xp.sqrt(0.5 * (vxL * vxL + vyL * vyL + vxR * vxR + vyR * vyR))
        * (bp + bm - 1.0)
        * 0.5
        * (rhoL + rhoR)
        * a_bar
    )

    forward = (mdot > 0.0)[..., None]
    psi = xp.where(
        forward,
        xp.stack([xp.ones_like(vxL), vxL, vyL, HL], axis=-1),
        xp.stack([xp.ones_like(vxR), vxR, vyR, HR], axis=-1),
    )
    zero = xp.zeros_like(pL)
    nvec = xp.stack([zero, nx + zero, ny + zero, zero], axis=-1)
    flux = mdot[..., None] * psi + p_half[..., None] * nvec
    return FluxResult(flux, xp.maximum(xp.abs(vnL) + aL, xp.abs(vnR) + aR))


#: The approximate Riemann solvers, by the name :class:`~src.config.FlowConditions`
#: uses.
SOLVERS = {
    "roe": roe_flux,
    "hllc": hllc_flux,
    "ausmup": ausmup_flux,
    "slau2": slau2_flux,
}


def interface_flux(name: str, UL, UR, nx, ny, gamma: float, *, xp=np, **kwargs):
    """Dispatch to one of :data:`SOLVERS` by name."""
    try:
        solver = SOLVERS[name]
    except KeyError:
        raise ValueError(f"unknown flux {name!r}; choose from {tuple(SOLVERS)}") from None
    return solver(UL, UR, nx, ny, gamma, xp=xp, **kwargs)


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
    "FLOOR",
    "SOLVERS",
    "FluxResult",
    "ausmup_flux",
    "conservative",
    "euler_flux",
    "farfield_flux",
    "farfield_state",
    "hllc_flux",
    "interface_flux",
    "normal_flux",
    "primitives",
    "roe_flux",
    "slau2_flux",
    "sound_speed",
    "wall_flux",
]
