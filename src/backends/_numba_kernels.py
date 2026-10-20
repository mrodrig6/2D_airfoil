r"""Compiled kernels: the whole march in ``nopython`` mode.

Why this file looks nothing like :mod:`src.physics`
---------------------------------------------------
Two differences, both deliberate, and both are what the speed comes from.

**Scalars, not arrays.**  Every function here takes and returns loose floats
rather than ``(..., 4)`` arrays.  Allocating a four-element array per face per
iteration is most of the cost of a compiled kernel, and returning a tuple of
floats costs nothing -- Numba unboxes it into registers.  The arithmetic is
line-for-line the same as :mod:`src.physics`, and ``tests/test_backends.py``
pins the two together to :math:`10^{-12}`.

**Gather, not scatter.**  The vectorised residual computes each interior flux
once and adds it to one cell while subtracting it from the other.  That is a
scatter, and under ``prange`` two threads can hit the same cell at once.  The
usual fixes are atomics (slow) or colouring the mesh (complicated).  This file
does neither: every element loops over its **own** three faces and computes
each flux itself, so a face's flux is evaluated **twice**, once from each side.
That is twice the flux work and it is still much the faster option -- the loop
becomes embarrassingly parallel, every write is to the element the thread owns,
and nothing needs synchronising.

Consistency is what makes it legal: :math:`\hat{F}(L, R, \hat{n}) = -\hat{F}(R,
L, -\hat{n})` exactly, so the two evaluations cannot disagree and conservation
survives.  ``tests/test_physics.py`` checks that identity for all four fluxes,
which makes it a load-bearing test rather than a nicety.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

FLOOR = 1e-10

#: Flux selector, matching :data:`src.config.FLUXES` by index.
# Flux codes, which are indices into `src.config.FLUXES` -- the backend passes
# `FLUXES.index(name)`, so these must stay in that order.
ROE, HLLC, AUSMUP = 0, 1, 2

# AUSM+-up constants; see `src.physics.ausmup_flux` for provenance.
_AUSM_BETA = 0.125
_AUSM_ALPHA = 0.1875
_AUSM_KP = 0.4
_AUSM_KU = 0.2

#: Neighbour codes for a face that has no neighbouring element.
WALL, FARFIELD = -1, -2

_FAST = {"fastmath": True, "cache": True, "inline": "always"}


@njit(**_FAST)
def _primitives(r, ru, rv, rE, gamma):
    rho = max(r, FLOOR)
    vx = ru / rho
    vy = rv / rho
    p = max((gamma - 1.0) * (rE - 0.5 * rho * (vx * vx + vy * vy)), FLOOR)
    return rho, vx, vy, p, (rE + p) / rho


@njit(**_FAST)
def _normal_flux(r, ru, rv, rE, nx, ny, gamma):
    rho, vx, vy, p, H = _primitives(r, ru, rv, rE, gamma)
    vn = vx * nx + vy * ny
    return rho * vn, rho * vx * vn + p * nx, rho * vy * vn + p * ny, rho * H * vn


@njit(**_FAST)
def _roe(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma, efix):
    rhoL, vxL, vyL, pL, HL = _primitives(lr, lru, lrv, lrE, gamma)
    rhoR, vxR, vyR, pR, HR = _primitives(rr, rru, rrv, rrE, gamma)
    fL0, fL1, fL2, fL3 = _normal_flux(lr, lru, lrv, lrE, nx, ny, gamma)
    fR0, fR1, fR2, fR3 = _normal_flux(rr, rru, rrv, rrE, nx, ny, gamma)

    wL = np.sqrt(rhoL)
    wR = np.sqrt(rhoR)
    inv = 1.0 / (wL + wR)
    u = (wL * vxL + wR * vxR) * inv
    v = (wL * vyL + wR * vyR) * inv
    H = (wL * HL + wR * HR) * inv
    q2 = u * u + v * v
    a = np.sqrt(max((gamma - 1.0) * (H - 0.5 * q2), FLOOR))
    vn = u * nx + v * ny

    eps = efix * a
    l1 = abs(vn + a)
    l2 = abs(vn - a)
    l3 = abs(vn)
    if eps > 0.0:
        if l1 < eps:
            l1 = 0.5 * (l1 * l1 / eps + eps)
        if l2 < eps:
            l2 = 0.5 * (l2 * l2 / eps + eps)
        if l3 < eps:
            l3 = 0.5 * (l3 * l3 / eps + eps)

    dr = rr - lr
    dru = rru - lru
    drv = rrv - lrv
    drE = rrE - lrE
    g1 = (gamma - 1.0) * (0.5 * q2 * dr - u * dru - v * drv + drE)
    g2 = -vn * dr + dru * nx + drv * ny
    s1 = 0.5 * (l1 + l2)
    s2 = 0.5 * (l1 - l2)
    c1 = (g1 / (a * a)) * (s1 - l3) + (g2 / a) * s2
    c2 = (g1 / a) * s2 + (s1 - l3) * g2

    aL = np.sqrt(max(gamma * pL / rhoL, FLOOR))
    aR = np.sqrt(max(gamma * pR / rhoR, FLOOR))
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    return (
        0.5 * (fL0 + fR0) - 0.5 * (l3 * dr + c1),
        0.5 * (fL1 + fR1) - 0.5 * (l3 * dru + c1 * u + c2 * nx),
        0.5 * (fL2 + fR2) - 0.5 * (l3 * drv + c1 * v + c2 * ny),
        0.5 * (fL3 + fR3) - 0.5 * (l3 * drE + c1 * H + c2 * vn),
        max(abs(vnL) + aL, abs(vnR) + aR),
    )


@njit(**_FAST)
def _hllc(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma):
    rhoL, vxL, vyL, pL, HL = _primitives(lr, lru, lrv, lrE, gamma)
    rhoR, vxR, vyR, pR, HR = _primitives(rr, rru, rrv, rrE, gamma)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = np.sqrt(max(gamma * pL / rhoL, FLOOR))
    aR = np.sqrt(max(gamma * pR / rhoR, FLOOR))

    wL = np.sqrt(rhoL)
    wR = np.sqrt(rhoR)
    inv = 1.0 / (wL + wR)
    u = (wL * vxL + wR * vxR) * inv
    v = (wL * vyL + wR * vyR) * inv
    H = (wL * HL + wR * HR) * inv
    a = np.sqrt(max((gamma - 1.0) * (H - 0.5 * (u * u + v * v)), FLOOR))
    vn = u * nx + v * ny

    SL = min(vnL - aL, vn - a)
    SR = max(vnR + aR, vn + a)
    speed = max(abs(SL), abs(SR))

    if SL >= 0.0:
        f0, f1, f2, f3 = _normal_flux(lr, lru, lrv, lrE, nx, ny, gamma)
        return f0, f1, f2, f3, speed
    if SR <= 0.0:
        f0, f1, f2, f3 = _normal_flux(rr, rru, rrv, rrE, nx, ny, gamma)
        return f0, f1, f2, f3, speed

    mL = rhoL * (SL - vnL)
    mR = rhoR * (SR - vnR)
    den = mL - mR
    if abs(den) < FLOOR:
        den = FLOOR if den >= 0.0 else -FLOOR
    SM = (pR - pL + mL * vnL - mR * vnR) / den

    if SM >= 0.0:
        f0, f1, f2, f3 = _normal_flux(lr, lru, lrv, lrE, nx, ny, gamma)
        fac = (SL - vnL) / (SL - SM)
        rs = rhoL * fac
        ps = pL + rhoL * (vnL - SM) * (vnL - SL)
        vxs = vxL + (SM - vnL) * nx
        vys = vyL + (SM - vnL) * ny
        rEs = fac * lrE + (ps * SM - pL * vnL) / (SL - SM)
        return (
            f0 + SL * (rs - lr),
            f1 + SL * (rs * vxs - lru),
            f2 + SL * (rs * vys - lrv),
            f3 + SL * (rEs - lrE),
            speed,
        )
    f0, f1, f2, f3 = _normal_flux(rr, rru, rrv, rrE, nx, ny, gamma)
    fac = (SR - vnR) / (SR - SM)
    rs = rhoR * fac
    ps = pR + rhoR * (vnR - SM) * (vnR - SR)
    vxs = vxR + (SM - vnR) * nx
    vys = vyR + (SM - vnR) * ny
    rEs = fac * rrE + (ps * SM - pR * vnR) / (SR - SM)
    return (
        f0 + SR * (rs - rr),
        f1 + SR * (rs * vxs - rru),
        f2 + SR * (rs * vys - rrv),
        f3 + SR * (rEs - rrE),
        speed,
    )


@njit(**_FAST)
def _m4_plus(m, beta):
    if abs(m) >= 1.0:
        return 0.5 * (m + abs(m))
    m2p = 0.25 * (m + 1.0) * (m + 1.0)
    m2m = -0.25 * (m - 1.0) * (m - 1.0)
    return m2p * (1.0 - 16.0 * beta * m2m)


@njit(**_FAST)
def _m4_minus(m, beta):
    if abs(m) >= 1.0:
        return 0.5 * (m - abs(m))
    m2p = 0.25 * (m + 1.0) * (m + 1.0)
    m2m = -0.25 * (m - 1.0) * (m - 1.0)
    return m2m * (1.0 + 16.0 * beta * m2p)


@njit(**_FAST)
def _p5_plus(m, alpha):
    if abs(m) >= 1.0:
        return 0.5 * (m + abs(m)) / m
    m2p = 0.25 * (m + 1.0) * (m + 1.0)
    m2m = -0.25 * (m - 1.0) * (m - 1.0)
    return m2p * ((2.0 - m) - 16.0 * alpha * m * m2m)


@njit(**_FAST)
def _p5_minus(m, alpha):
    if abs(m) >= 1.0:
        return 0.5 * (m - abs(m)) / m
    m2p = 0.25 * (m + 1.0) * (m + 1.0)
    m2m = -0.25 * (m - 1.0) * (m - 1.0)
    return m2m * ((-2.0 - m) + 16.0 * alpha * m * m2p)


@njit(**_FAST)
def _ausmup(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma):
    """AUSM+-up; see :func:`src.physics.ausmup_flux`.

    A scalar branch where the vectorised one needs a guarded `where`: the
    supersonic branch of P+-_(5) divides by M, and here the division simply is
    not reached when |M| < 1.
    """
    rhoL, vxL, vyL, pL, _ = _primitives(lr, lru, lrv, lrE, gamma)
    rhoR, vxR, vyR, pR, _ = _primitives(rr, rru, rrv, rrE, gamma)
    vnL = vxL * nx + vyL * ny
    vnR = vxR * nx + vyR * ny
    aL = np.sqrt(max(gamma * pL / rhoL, FLOOR))
    aR = np.sqrt(max(gamma * pR / rhoR, FLOOR))

    a_half = 0.5 * (aL + aR)
    rho_half = 0.5 * (rhoL + rhoR)
    machL = vnL / a_half
    machR = vnR / a_half
    mbar_sq = 0.5 * (machL * machL + machR * machR)

    u_half = a_half * (_m4_plus(machL, _AUSM_BETA) + _m4_minus(machR, _AUSM_BETA)) + _AUSM_KP * max(
        1.0 - mbar_sq, 0.0
    ) * (pL - pR) / (rho_half * a_half)

    p5p = _p5_plus(machL, _AUSM_ALPHA)
    p5m = _p5_minus(machR, _AUSM_ALPHA)
    p_half = p5p * pL + p5m * pR + _AUSM_KU * p5p * p5m * rho_half * a_half * (vnL - vnR)

    speed = max(abs(vnL) + aL, abs(vnR) + aR)
    if u_half > 0.0:
        return (
            u_half * rhoL,
            u_half * lru + p_half * nx,
            u_half * lrv + p_half * ny,
            u_half * (lrE + pL),
            speed,
        )
    return (
        u_half * rhoR,
        u_half * rru + p_half * nx,
        u_half * rrv + p_half * ny,
        u_half * (rrE + pR),
        speed,
    )


@njit(**_FAST)
def _wall(r, ru, rv, rE, nx, ny, gamma):
    rho = max(r, FLOOR)
    vx = ru / rho
    vy = rv / rho
    vn = vx * nx + vy * ny
    tx = vx - vn * nx
    ty = vy - vn * ny
    p = max((gamma - 1.0) * (rE - 0.5 * rho * (tx * tx + ty * ty)), FLOOR)
    return 0.0, p * nx, p * ny, 0.0, np.sqrt(max(gamma * p / rho, FLOOR)) + abs(vn)


@njit(**_FAST)
def _farfield_state(r, ru, rv, rE, ir, iru, irv, irE, nx, ny, gamma):
    """The characteristic exterior state; see :func:`src.physics.farfield_state`."""
    rho, vx, vy, p, _ = _primitives(r, ru, rv, rE, gamma)
    a = np.sqrt(max(gamma * p / rho, FLOOR))
    vn = vx * nx + vy * ny
    rhoi, vxi, vyi, pi, _ = _primitives(ir, iru, irv, irE, gamma)
    ai = np.sqrt(max(gamma * pi / rhoi, FLOOR))
    vni = vxi * nx + vyi * ny

    if vn >= a:  # supersonic outflow: everything leaves
        return r, ru, rv, rE
    if vn <= -a:  # supersonic inflow: everything is imposed
        return ir, iru, irv, irE

    two = 2.0 / (gamma - 1.0)
    vnb = 0.5 * ((vn + two * a) + (vni - two * ai))
    ab = max(0.25 * (gamma - 1.0) * ((vn + two * a) - (vni - two * ai)), FLOOR)
    if vnb > 0.0:
        rs, ps, vxs, vys, vns = rho, p, vx, vy, vn
    else:
        rs, ps, vxs, vys, vns = rhoi, pi, vxi, vyi, vni
    entropy = ps / rs**gamma
    rb = (ab * ab / (gamma * entropy)) ** (1.0 / (gamma - 1.0))
    pb = rb * ab * ab / gamma
    vxb = vxs + (vnb - vns) * nx
    vyb = vys + (vnb - vns) * ny
    return rb, rb * vxb, rb * vyb, pb / (gamma - 1.0) + 0.5 * rb * (vxb * vxb + vyb * vyb)


@njit(**_FAST)
def _interface(kind, lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma, efix):
    if kind == ROE:
        return _roe(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma, efix)
    if kind == HLLC:
        return _hllc(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma)
    return _ausmup(lr, lru, lrv, lrE, rr, rru, rrv, rrE, nx, ny, gamma)


@njit(fastmath=True, cache=True, parallel=True)
def residual_gather(U, neighbour, length, normal, U_inf, gamma, kind, efix, R, wave):
    """One residual, every element gathering its own three faces.

    Each thread writes only to the element it owns, so ``prange`` needs no
    atomics and no mesh colouring.  The price is that every interior flux is
    evaluated twice; consistency makes the two agree exactly.
    """
    n = U.shape[0]
    for e in prange(n):
        r0 = 0.0
        r1 = 0.0
        r2 = 0.0
        r3 = 0.0
        s = 0.0
        for k in range(3):
            j = neighbour[e, k]
            nx = normal[e, k, 0]
            ny = normal[e, k, 1]
            ell = length[e, k]
            if j >= 0:
                f0, f1, f2, f3, sp = _interface(
                    kind,
                    U[e, 0],
                    U[e, 1],
                    U[e, 2],
                    U[e, 3],
                    U[j, 0],
                    U[j, 1],
                    U[j, 2],
                    U[j, 3],
                    nx,
                    ny,
                    gamma,
                    efix,
                )
            elif j == WALL:
                f0, f1, f2, f3, sp = _wall(U[e, 0], U[e, 1], U[e, 2], U[e, 3], nx, ny, gamma)
            else:
                b0, b1, b2, b3 = _farfield_state(
                    U[e, 0],
                    U[e, 1],
                    U[e, 2],
                    U[e, 3],
                    U_inf[0],
                    U_inf[1],
                    U_inf[2],
                    U_inf[3],
                    nx,
                    ny,
                    gamma,
                )
                f0, f1, f2, f3, sp = _interface(
                    kind,
                    U[e, 0],
                    U[e, 1],
                    U[e, 2],
                    U[e, 3],
                    b0,
                    b1,
                    b2,
                    b3,
                    nx,
                    ny,
                    gamma,
                    efix,
                )
            r0 += f0 * ell
            r1 += f1 * ell
            r2 += f2 * ell
            r3 += f3 * ell
            s += sp * ell
        R[e, 0] = r0
        R[e, 1] = r1
        R[e, 2] = r2
        R[e, 3] = r3
        wave[e] = s


# ==========================================================================
#  The time steppers' building blocks
#
#  One kernel per arithmetic shape, orchestrated from Python by
#  :class:`~src.backends.numba_backend.NumbaBackend`.  Writing the stage
#  algebra as separate kernels rather than one fused march per scheme is a
#  measured choice, not a concession: on this problem it came out within the
#  run-to-run noise of a fully fused loop (see ``docs/performance.md``), and it
#  keeps each arithmetic shape in one place rather than one fused kernel per
#  scheme.  It does *not* keep the stage coefficients in one place: the fused
#  paths in :class:`~src.backends.numba_backend.NumbaBackend` restate the
#  0.75/0.25 and 1/3, 2/3 of SSP-RK3 that
#  :meth:`src.backends.base.Backend.ssprk3_step` also carries.  That is the
#  same bargain the rest of this module makes -- a compiled copy of something
#  defined once in readable form -- and ``tests/test_backends.py`` is what
#  holds the two in step, by running every scheme on every backend.
# ==========================================================================


@njit(fastmath=True, cache=True, parallel=True)
def rate_into(R, area, F):
    r"""``F = -R / area``, the semi-discrete :math:`dU/dt`."""
    n = R.shape[0]
    for e in prange(n):
        inv = 1.0 / area[e]
        for c in range(4):
            F[e, c] = -R[e, c] * inv


@njit(fastmath=True, cache=True, parallel=True)
def dt_into(wave, area, cfl, dt):
    r"""The element-local step :math:`\mathrm{CFL}\,2A_e/\sum_f(|v_n|+a)\ell_f`."""
    n = wave.shape[0]
    for e in prange(n):
        dt[e] = cfl * 2.0 * area[e] / max(wave[e], FLOOR)


@njit(fastmath=True, cache=True, parallel=True)
def rms_weighted(F, w):
    """RMS of ``F`` with component ``c`` scaled by ``w[c]`` first.

    The weights are what make the norm independent of the operating point --
    see :func:`src.timestepping.step.component_weights`.  Folded into the
    reduction rather than applied as ``F * w`` first, because this runs once
    per step and materialising a second state array for it would undo the
    allocation work the rest of this module does.

    There is deliberately **no unweighted companion** to this, so that a future
    stepper cannot reach for one by mistake and get a norm that is correct at
    the default operating point and silently wrong away from it.
    """
    n = F.shape[0]
    total = 0.0
    for e in prange(n):
        acc = 0.0
        for c in range(4):
            v = F[e, c] * w[c]
            acc += v * v
        total += acc
    return np.sqrt(total / (4.0 * n))


@njit(fastmath=True, cache=True, parallel=True)
def stage(U, F, coef, dt, out):
    """``out = U + coef * dt * F``, with ``dt`` per element.

    Safe when ``out`` aliases ``U``: element ``(e, c)`` of the output depends
    only on element ``(e, c)`` of the inputs.
    """
    n = U.shape[0]
    for e in prange(n):
        c0 = coef * dt[e]
        for c in range(4):
            out[e, c] = U[e, c] + c0 * F[e, c]


@njit(fastmath=True, cache=True, parallel=True)
def combine_ssp(U, V, F, dt, a, b, out):
    """``out = a * U + b * (V + dt * F)`` -- both non-trivial SSP-RK3 stages."""
    n = U.shape[0]
    for e in prange(n):
        d = dt[e]
        for c in range(4):
            out[e, c] = a * U[e, c] + b * (V[e, c] + d * F[e, c])
