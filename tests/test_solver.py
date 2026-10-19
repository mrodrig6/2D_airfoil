"""Assembly, the march, and the loads that come out of it.

The fast tests here are worth more than they look.  Free-stream preservation in
particular exercises the fluxes, the metrics, the assembly and the far-field
condition **together**, and nothing else catches a sign error in a normal or a
left/right pair swapped: each of those leaves a uniform flow visibly decaying.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import Discretization, FlowConditions, SolverOptions, build_mesh, build_metrics, naca4
from src import physics as ph
from src.geometry.mesh import BoundaryTag
from src.post.postprocess import entropy_error, forces, legacy_forces, surface_pressure
from src.spatial import assembly as asm
from src.timestepping import step as ts
from src.timestepping.march import initial_state, solve_steady

SMALL = Discretization(
    n_surface=64, n_radial=32, far_field_radius=12.0, first_cell_height=0.01, growth_limit=1.3
)


@pytest.fixture(scope="module")
def small():
    mesh, _ = build_mesh(naca4("0012", SMALL.n_surface), SMALL)
    return build_metrics(mesh)


def cells_off_the_wall(metrics):
    """Mask of cells with no wall face: a uniform flow is exact only there."""
    s = metrics.tag_slice(BoundaryTag.WALL)
    mask = np.ones(metrics.n_elements, dtype=bool)
    mask[metrics.boundary_element[s]] = False
    return mask


# -- assembly ---------------------------------------------------------------
@pytest.mark.parametrize("flux", tuple(ph.SOLVERS))
@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_the_free_stream_is_preserved_exactly(small, flux, alpha):
    r"""A uniform free stream must give zero residual away from the wall.

    Every interior face returns the exact flux by consistency, so the cell sum
    collapses to :math:`\mathbf{F}_\infty \cdot \sum_f \hat{n}_f \ell_f`, which
    the discrete Gauss identity makes zero.  The far-field condition reproduces
    the free stream, so those faces behave the same.  Only wall cells differ --
    a uniform flow does not satisfy a wall condition, and that is the whole
    reason the solution is not uniform.
    """
    flow = FlowConditions(mach=0.8, alpha_deg=alpha, flux=flux)
    U = initial_state(small, flow)
    R = asm.residual(U, small, flow).R
    off = cells_off_the_wall(small)
    scale = np.abs(ph.normal_flux(flow.state, 1.0, 0.0, flow.gamma)).max()
    assert np.abs(R[off]).max() < 1e-11 * scale * small.perimeter.max()
    assert np.abs(R[~off]).max() > 0.0  # the wall is what breaks it


def test_the_local_time_step_is_the_legacy_formula(small):
    r"""CFL * 2A / sum(s l) is CFL * d_h / s written out; the perimeter cancels."""
    flow = FlowConditions()
    U = initial_state(small, flow)
    out = asm.residual(U, small, flow)
    dt = ts.local_time_step(out.wave, small, 0.9)
    legacy = 0.9 * small.hydraulic_diameter / (out.wave / small.perimeter)
    assert np.allclose(dt, legacy)
    assert np.all(dt > 0.0)


def test_the_component_weights_put_every_residual_on_one_footing():
    flow = FlowConditions(mach=0.8)
    w = ts.component_weights(flow)
    assert w[0] == 1.0
    assert w[1] == w[2] == pytest.approx(flow.mach)  # 1 / a_inf = M
    assert w[3] == pytest.approx(flow.mach**2)


# -- the march --------------------------------------------------------------
@pytest.mark.slow
def test_a_symmetric_section_at_zero_incidence_makes_no_lift(small):
    """Exactly zero, not approximately: the mesh is symmetric and so is the flow."""
    result = solve_steady(
        small,
        FlowConditions(mach=0.5, alpha_deg=0.0),
        SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000),
    )
    assert result.converged, result.message
    load = forces(result)
    assert abs(load.cl) < 1e-10
    assert abs(load.cm) < 1e-10


@pytest.mark.slow
def test_lift_follows_thin_airfoil_theory_at_small_incidence(small):
    r""":math:`c_\ell = 2\pi\alpha/\sqrt{1 - M^2}`, the one closed form available."""
    alpha = 2.0
    result = solve_steady(
        small,
        FlowConditions(mach=0.5, alpha_deg=alpha),
        SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000),
    )
    assert result.converged, result.message
    predicted = 2 * math.pi * math.radians(alpha) / math.sqrt(1 - 0.5**2)
    assert forces(result).cl == pytest.approx(predicted, rel=0.15)


@pytest.mark.slow
def test_lift_is_antisymmetric_in_the_angle_of_attack(small):
    """A symmetric section cannot tell +alpha from -alpha except by sign, and
    alpha enters only through the free-stream direction -- so this also checks
    that the mesh really is independent of it."""
    options = SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000)
    up = solve_steady(small, FlowConditions(mach=0.5, alpha_deg=2.0), options)
    down = solve_steady(small, FlowConditions(mach=0.5, alpha_deg=-2.0), options)
    assert up.converged and down.converged
    assert forces(up).cl == pytest.approx(-forces(down).cl, rel=1e-6)
    assert forces(up).cd == pytest.approx(forces(down).cd, rel=1e-6)


@pytest.mark.slow
def test_the_legacy_force_formula_really_is_wrong(small):
    """Not asserted -- measured.  The legacy drag is several times the correct
    one, because every panel contributes its full length instead of its
    projected area."""
    result = solve_steady(
        small,
        FlowConditions(mach=0.5, alpha_deg=2.0),
        SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000),
    )
    correct = forces(result)
    cl_legacy, cd_legacy = legacy_forces(result)
    assert cd_legacy > 3.0 * correct.cd  # the formula is not close
    assert cl_legacy == pytest.approx(correct.cl, rel=0.1)  # lift happens to be nearer


@pytest.mark.slow
def test_a_warm_start_reaches_the_same_answer_sooner(small):
    """Which is what makes a sweep cost less than one solve per point."""
    options = SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000)
    first = solve_steady(small, FlowConditions(mach=0.5, alpha_deg=2.0), options)
    warm = solve_steady(small, FlowConditions(mach=0.5, alpha_deg=2.2), options, state=first.state)
    cold = solve_steady(small, FlowConditions(mach=0.5, alpha_deg=2.2), options)
    assert warm.converged and cold.converged
    assert warm.iterations < cold.iterations
    assert forces(warm).cl == pytest.approx(forces(cold).cl, rel=1e-3)


@pytest.mark.slow
def test_the_transonic_case_puts_a_shock_on_the_upper_surface(small):
    """The legacy operating point.  Loose bounds: this is a coarse mesh and a
    first-order scheme, so the test is that the physics is there, not that the
    numbers are converged."""
    from src.post.postprocess import shock_location

    result = solve_steady(
        small,
        FlowConditions(mach=0.8, alpha_deg=1.25),
        SolverOptions(tolerance=1e-5, print_interval=0, max_iterations=30_000),
    )
    assert result.converged, result.message
    mach = np.hypot(result.state[:, 1], result.state[:, 2]) / result.state[:, 0]
    rho, _, _, p, _ = ph.primitives(result.state, result.flow.gamma)
    assert (mach / ph.sound_speed(rho, p, result.flow.gamma)).max() > 1.1  # supersonic pocket
    where = shock_location(result)
    assert where is not None and 0.2 < where < 0.95
    assert forces(result).cl > 0.15
    trace = surface_pressure(result)
    assert trace["upper"].sum() > 10 and (~trace["upper"]).sum() > 10


@pytest.mark.slow
def test_a_diverging_run_says_so_rather_than_returning_nonsense(small):
    """A CFL far past the stability limit must stop and explain, not return."""
    result = solve_steady(
        small,
        FlowConditions(mach=0.8, alpha_deg=1.25),
        SolverOptions(cfl=50.0, print_interval=0, max_iterations=2000, tolerance=1e-6),
    )
    assert not result.converged
    assert "diverg" in result.message or "residual became" in result.message


# -- post-processing --------------------------------------------------------
@pytest.mark.slow
def test_entropy_and_the_surface_trace_are_sane(small):
    result = solve_steady(
        small,
        FlowConditions(mach=0.5, alpha_deg=0.0),
        SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=20_000),
    )
    assert 0.0 < entropy_error(result) < 1e-2  # small, but not zero: this is first order
    trace = surface_pressure(result)
    assert trace["x"].min() < 0.01 and trace["x"].max() > 0.99
    assert np.all(np.isfinite(trace["cp"]))
    # A symmetric section at zero incidence: every face has a mirror face at
    # the same x and opposite y, carrying the same cp.  Compared face by face
    # rather than by the upper/lower split, which is ambiguous by one face --
    # no face sits exactly at the leading edge, two straddle it.
    x, y, cp = trace["x"], trace["y"], trace["cp"]
    order = np.lexsort((np.abs(y), np.round(x, 9)))
    pairs = cp[order].reshape(-1, 2)
    assert np.allclose(pairs[:, 0], pairs[:, 1], atol=1e-8)
