"""The fluxes and the boundary conditions.

Consistency and conservation are the two properties the whole scheme rests on,
and every solver must have both or it is not a numerical flux at all.  Each
boundary condition is checked against a state it should reproduce exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import FlowConditions
from src import physics as ph

GAMMA = 1.4
FLUXES = tuple(ph.SOLVERS)


@pytest.fixture
def states():
    return np.array([[1.2, 0.30, -0.10, 2.6], [0.8, 0.50, 0.20, 2.0], [2.0, 1.40, 0.05, 6.0]])


@pytest.fixture
def normals():
    n = np.array([[0.6, -0.8], [1.0, 0.0], [-0.28, 0.96]])
    return n[:, 0], n[:, 1]


def test_primitives_round_trip(states):
    rho, vx, vy, p, H = ph.primitives(states, GAMMA)
    assert np.allclose(ph.conservative(rho, vx, vy, p, GAMMA), states)
    assert np.allclose(H, (states[:, 3] + p) / rho)


def test_normal_flux_matches_the_projected_pair(states, normals):
    nx, ny = normals
    F, G = ph.euler_flux(states, GAMMA)
    assert np.allclose(ph.normal_flux(states, nx, ny, GAMMA), F * nx[:, None] + G * ny[:, None])


@pytest.mark.parametrize("flux", FLUXES)
def test_every_flux_is_consistent(flux, states, normals):
    """F_hat(U, U, n) must be the exact flux, or the scheme solves a different PDE."""
    nx, ny = normals
    out = ph.interface_flux(flux, states, states, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(states, nx, ny, GAMMA), atol=1e-12)


@pytest.mark.parametrize("flux", FLUXES)
def test_every_flux_is_conservative(flux, states, normals):
    """F_hat(L, R, n) = -F_hat(R, L, -n): what leaves one cell enters the other."""
    nx, ny = normals
    L, R = states[:2], states[1:]
    a = ph.interface_flux(flux, L, R, nx[:2], ny[:2], GAMMA).flux
    b = ph.interface_flux(flux, R, L, -nx[:2], -ny[:2], GAMMA).flux
    assert np.allclose(a, -b, atol=1e-12)


@pytest.mark.parametrize("flux", FLUXES)
def test_every_flux_is_fully_upwind_when_supersonic(flux):
    """At Mach 3 no information travels upstream, so the flux is the left one."""
    U = np.array([[1.0, 3.0, 0.0, 1.0 / (GAMMA * (GAMMA - 1.0)) + 4.5]])
    nx, ny = np.array([1.0]), np.array([0.0])
    out = ph.interface_flux(flux, U, 1.05 * U, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(U, nx, ny, GAMMA), atol=1e-10)


@pytest.mark.parametrize("flux", FLUXES)
def test_every_flux_survives_a_near_vacuum(flux, normals):
    """A transient can hand a flux almost anything; a NaN kills the whole field."""
    nx, ny = normals[0][:1], normals[1][:1]
    weak = np.array([[1e-9, 0.0, 0.0, 1e-9]])
    strong = np.array([[1.0, 0.5, 0.0, 3.0]])
    out = ph.interface_flux(flux, weak, strong, nx, ny, GAMMA)
    assert np.all(np.isfinite(out.flux)) and np.all(np.isfinite(out.max_speed))


def test_the_entropy_fix_bounds_the_eigenvalues_away_from_zero():
    """Without it the Roe flux adds no dissipation at a sonic point and admits
    an expansion shock -- which this flow, having a sonic line, would find."""
    U = np.array([[1.0, 0.0, 0.0, 2.5]])
    nx, ny = np.array([1.0]), np.array([0.0])
    sonic = np.array([[1.0, 1e-9, 0.0, 2.5]])
    fixed = ph.roe_flux(U, sonic, nx, ny, GAMMA, entropy_fix=0.05).flux
    bare = ph.roe_flux(U, sonic, nx, ny, GAMMA, entropy_fix=0.0).flux
    assert np.all(np.isfinite(fixed)) and np.all(np.isfinite(bare))
    # the fix never *reduces* the dissipation
    assert np.abs(fixed).sum() >= np.abs(bare).sum() - 1e-12


def test_the_wall_flux_carries_pressure_and_nothing_else(states, normals):
    nx, ny = normals
    out = ph.wall_flux(states, nx, ny, GAMMA)
    assert np.allclose(out.flux[:, 0], 0.0)  # no mass through a wall
    assert np.allclose(out.flux[:, 3], 0.0)  # and so no energy
    assert np.allclose(out.flux[:, 1] * ny, out.flux[:, 2] * nx)  # force along n


def test_only_the_fluxes_that_work_are_shipped():
    """SLAU2 was implemented, measured and removed, and stays removed.

    AUSM+-up came back: the first attempt carried a tunable cutoff Mach number
    and only converged inside a narrow window of it, which is what got it
    removed.  The formulation here has no such constant.  See
    docs/riemann_fluxes.md.
    """
    assert set(ph.SOLVERS) == {"roe", "hllc", "ausmup"}
    assert not hasattr(ph, "slau2_flux")
    with pytest.raises(ValueError, match="unknown flux"):
        ph.interface_flux("slau2", np.zeros((1, 4)), np.zeros((1, 4)), 1.0, 0.0, GAMMA)


# -- AUSM+-up ---------------------------------------------------------------
@pytest.mark.parametrize("beta", [0.0, ph._AUSM_BETA, 0.3])
def test_the_split_mach_polynomials_sum_to_the_mach_number(beta):
    r"""``M+_(4)(M) + M-_(4)(M) = M``, for every M and *every* beta.

    One of the two identities that pin the AUSM splittings.  It holds
    independently of beta because the cross terms cancel, so a wrong beta
    would not show up here -- which is the point of also checking the closed
    forms below.
    """
    m = np.linspace(-3.0, 3.0, 1001)
    total = ph._m4_plus(m, beta) + ph._m4_minus(m, beta)
    assert np.abs(total - m).max() < 1e-14


@pytest.mark.parametrize("alpha", [0.0, ph._AUSM_ALPHA, 0.5])
def test_the_pressure_splittings_sum_to_one(alpha):
    r"""``P+_(5)(M) + P-_(5)(M) = 1``, for every M and every alpha.

    This is what makes the pressure splitting consistent: at a uniform state
    the two halves return exactly the one pressure.
    """
    m = np.linspace(-3.0, 3.0, 1001)
    total = ph._p5_plus(m, alpha) + ph._p5_minus(m, alpha)
    assert np.abs(total - 1.0).max() < 1e-14


def test_the_split_polynomials_match_their_closed_forms():
    """The identities above are blind to alpha and beta, so pin the values too.

    These are the expanded forms quoted in the reference implementation, and
    they are what fixes beta = 1/8 and alpha = 3/16.
    """
    m = np.linspace(-0.999, 0.999, 999)
    assert np.abs(ph._m4_plus(m, ph._AUSM_BETA) - (m**4 / 8 + m / 2 + 3 / 8)).max() < 1e-14
    closed = 3 * m**5 / 16 - 5 * m**3 / 8 + 15 * m / 16 + 0.5
    assert np.abs(ph._p5_plus(m, ph._AUSM_ALPHA) - closed).max() < 1e-14


def test_the_pressure_splitting_is_finite_at_zero_mach():
    """Its supersonic branch divides by M, and a vectorised `where` evaluates
    both branches -- so an unguarded denominator puts a NaN in the half that is
    thrown away, and JAX would carry it into the gradient anyway."""
    m = np.array([-2.0, -1.0, -1e-300, 0.0, 1e-300, 1.0, 2.0])
    for split in (ph._p5_plus, ph._p5_minus):
        assert np.all(np.isfinite(split(m, ph._AUSM_ALPHA)))


def test_ausm_moves_mass_across_a_pressure_jump_from_rest():
    r"""The K_p term, which is the whole reason this is AUSM+-**up**.

    With both normal velocities zero, ``M+_(4)(0) + M-_(4)(0) = 0`` exactly, so
    plain AUSM+ gives *no* mass flux however large the pressure jump.  Momentum
    still feels the jump, the energy flux stays zero, and the internal energy on
    the light side goes negative within one step.  The pressure-diffusion term
    supplies the missing coupling, and it must point from high pressure to low.
    """
    left = ph.conservative(
        np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]), GAMMA
    )
    right = ph.conservative(
        np.array([0.125]), np.array([0.0]), np.array([0.0]), np.array([0.1]), GAMMA
    )
    nx, ny = np.array([1.0]), np.array([0.0])
    flux = ph.ausmup_flux(left, right, nx, ny, GAMMA).flux
    assert flux[0, 0] > 0.0  # mass moves, and towards the low-pressure side
    assert flux[0, 3] > 0.0  # so does energy

    # and with no pressure jump it is exactly zero again, as the splitting says
    still = ph.ausmup_flux(left, left, nx, ny, GAMMA).flux
    assert still[0, 0] == pytest.approx(0.0, abs=1e-15)


def test_ausm_carries_a_moving_contact_without_touching_the_pressure():
    """The guard on the two diffusion coefficients.

    A contact has uniform pressure and velocity with a density jump, and the
    exact answer is that the pressure stays put.  The consistency identities
    above constrain the split polynomials but say nothing about K_p and K_u, so
    this is what catches those drifting -- a K_u too large is anti-dissipative
    here and amplifies round-off until the pressure is visibly wrong.
    """
    p0, v0 = 1.0, 0.7
    left = ph.conservative(np.array([1.0]), np.array([v0]), np.array([0.0]), np.array([p0]), GAMMA)
    right = ph.conservative(np.array([0.2]), np.array([v0]), np.array([0.0]), np.array([p0]), GAMMA)
    nx, ny = np.array([1.0]), np.array([0.0])
    flux = ph.ausmup_flux(left, right, nx, ny, GAMMA).flux

    # uniform pressure and velocity: the momentum flux is rho v^2 + p with the
    # upwind density, and the pressure part must be exactly p0
    assert flux[0, 1] - flux[0, 0] * v0 == pytest.approx(p0, rel=1e-12)
    # the contact moves right, so everything is drawn from the left state
    exact = ph.normal_flux(left, nx, ny, GAMMA)
    assert np.abs(flux - exact).max() < 1e-12


def test_the_wall_pressure_removes_only_the_normal_velocity():
    """p_b differs from the interior pressure by rho v_n^2 / 2 * (gamma-1), which
    is zero only at convergence -- using the interior pressure would add a
    surface force that decays with the residual."""
    U = np.array([[1.0, 0.3, 0.4, 3.0]])
    nx, ny = np.array([0.0]), np.array([1.0])
    _, _, _, p_interior, _ = ph.primitives(U, GAMMA)
    p_wall = ph.wall_flux(U, nx, ny, GAMMA).flux[:, 2]
    assert p_wall > p_interior
    assert p_wall - p_interior == pytest.approx(0.5 * (GAMMA - 1.0) * 1.0 * 0.4**2)


@pytest.mark.parametrize("alpha", [-3.0, 0.0, 1.25, 5.0])
def test_the_far_field_reproduces_the_free_stream_exactly(alpha, normals):
    """If the interior already is the free stream, the boundary must not disturb it."""
    nx, ny = normals
    flow = FlowConditions(mach=0.8, alpha_deg=alpha)
    U_inf = np.broadcast_to(flow.state, (3, 4))
    assert np.allclose(ph.farfield_state(U_inf, U_inf, nx, ny, GAMMA), U_inf)
    out = ph.farfield_flux(U_inf, U_inf, nx, ny, GAMMA)
    assert np.allclose(out.flux, ph.normal_flux(U_inf, nx, ny, GAMMA))


def test_the_far_field_takes_everything_from_one_side_when_supersonic():
    """Supersonic inflow imposes the free stream; supersonic outflow extrapolates."""
    flow = FlowConditions(mach=2.0)
    U_inf = np.broadcast_to(flow.state, (1, 4))
    interior = np.array([[1.3, 2.6, 0.1, 5.0]])
    inflow = ph.farfield_state(interior, U_inf, np.array([-1.0]), np.array([0.0]), GAMMA)
    assert np.allclose(inflow, U_inf)
    outflow = ph.farfield_state(interior, U_inf, np.array([1.0]), np.array([0.0]), GAMMA)
    assert np.allclose(outflow, interior)


def test_an_unknown_flux_is_refused():
    with pytest.raises(ValueError, match="unknown flux"):
        ph.interface_flux("godunov", np.zeros((1, 4)), np.zeros((1, 4)), 1.0, 0.0, GAMMA)
