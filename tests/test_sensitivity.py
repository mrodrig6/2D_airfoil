"""The adjoint gradient, and the three places it could silently be wrong.

The module under test duplicates two formulas it has to duplicate -- the
free-stream state and the surface-load integral -- because a tracer cannot be
pushed through a frozen dataclass or through a function that returns floats.
Duplicated physics drifts, so the first two tests pin each copy to the original
and would catch the drift immediately.

The gradient itself is checked against a central difference of this solver's own
output.  That is the only honest reference: the adjoint differentiates the
*discrete* problem, so agreeing with the exact Euler equations is not the claim
being made, and would not be what an optimiser needs.  Those checks cost flow
solves, so they are marked ``slow`` and run on the smallest mesh that is still a
real O-grid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import FlowConditions, SolverOptions, build_case, naca4
from src.geometry import naca_section
from src.post.postprocess import forces
from src.sensitivity import (
    DESIGN_VARIABLES,
    FLOW_VARIABLES,
    check_gradient,
    differentiable_case,
    four_digit_parameters,
    freestream_state,
    wall_loads,
)
from src.timestepping.march import SolveResult, initial_state

jax = pytest.importorskip("jax", reason="the adjoint is JAX only")


@pytest.fixture(scope="module")
def tiny_case(tiny, section):
    """The ``tiny`` mesh, built once.  Reused by every test that only needs a
    case to refuse something -- none of those solve anything."""
    return build_case(section, tiny)


# -- the two duplicated formulas --------------------------------------------
@pytest.mark.jax
@pytest.mark.parametrize("mach", [0.3, 0.8, 1.4])
@pytest.mark.parametrize("alpha", [-3.0, 0.0, 5.0])
def test_the_traceable_free_stream_is_the_frozen_one(mach, alpha):
    """``freestream_state`` exists only so JAX can follow mach and alpha into
    it.  If it ever disagrees with ``FlowConditions.state`` the gradient is of a
    flow nobody solved."""
    flow = FlowConditions(mach=mach, alpha_deg=alpha)
    traced = freestream_state(mach, math.radians(alpha), flow.gamma, np)
    assert np.abs(np.asarray(traced) - flow.state).max() < 1e-15


@pytest.mark.jax
@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_the_traceable_loads_are_the_reported_ones(tiny_metrics, alpha):
    """Same argument for the load integral, on a state that is deliberately not
    the free stream -- a uniform state makes too many terms vanish to catch a
    sign error."""
    flow = FlowConditions(mach=0.6, alpha_deg=alpha)
    rng = np.random.default_rng(7)
    U = initial_state(tiny_metrics, flow) * (
        1.0 + 0.08 * rng.standard_normal((tiny_metrics.n_elements, 4))
    )
    result = SolveResult(
        state=U,
        metrics=tiny_metrics,
        flow=flow,
        options=SolverOptions(print_interval=0),
        converged=True,
        iterations=0,
        residual=0.0,
        history=np.zeros(0),
    )
    reported = forces(result)
    cl, cd, cm = wall_loads(U, flow.alpha, tiny_metrics, flow.gamma, np)
    assert float(cl) == pytest.approx(reported.cl, rel=1e-12)
    assert float(cd) == pytest.approx(reported.cd, rel=1e-12)
    assert float(cm) == pytest.approx(reported.cm, rel=1e-12)


# -- the shape parameters ---------------------------------------------------
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("0012", (0.0, 0.4, 0.12)),
        ("2412", (0.02, 0.4, 0.12)),
        ("naca 4415", (0.04, 0.4, 0.15)),
        ("NACA-6409", (0.06, 0.4, 0.09)),
    ],
)
def test_the_four_digit_parameters_are_parsed_not_fitted(code, expected):
    """Exactly, from the digits.  Recovering them by sampling the surface gives
    0.441 for a 2412 whose camber position is 0.4 -- the camber line is flat
    near its crest -- and the perturbed meshes would then surround a different
    section from the one the flow was solved on."""
    got = four_digit_parameters(code)
    assert (got["camber"], got["camber_position"], got["thickness"]) == expected


@pytest.mark.parametrize("spec", ["23012", "my_design.dat", "0012x", ""])
def test_a_section_with_no_four_digit_code_has_no_parameters(spec):
    assert four_digit_parameters(spec) is None


@pytest.mark.parametrize("code", ["0012", "2412", "4415"])
def test_the_continuous_section_reproduces_the_coded_one(code):
    """``naca_section`` takes the three numbers as floats so a shape derivative
    can perturb them; it must still be the *same* section, to the last bit, or
    the gradient is of a neighbouring airfoil."""
    p = four_digit_parameters(code)
    continuous = naca_section(p["camber"], p["camber_position"], p["thickness"], 64)
    assert np.array_equal(continuous.points, naca4(code, 64).points)


# -- what it refuses --------------------------------------------------------
@pytest.mark.jax
def test_a_section_read_from_a_file_offers_only_the_flow_variables(tiny_case):
    """And says so rather than inventing parameters.  No solve happens: the
    refusal has to come before the expensive part."""
    dc = differentiable_case(case=tiny_case, mach=0.5, print_interval=0)
    dc.shape = None  # as if the section had come from a .dat file
    assert dc.variables == FLOW_VARIABLES
    with pytest.raises(ValueError, match="four-digit section"):
        dc.value_and_gradient("cl", ("thickness",))
    assert dc.result is None


@pytest.mark.jax
def test_an_unknown_design_variable_or_objective_is_refused(tiny_case):
    dc = differentiable_case(case=tiny_case, print_interval=0)
    with pytest.raises(ValueError, match="unknown design variable"):
        dc.value_and_gradient("cl", ("sweep",))
    with pytest.raises(ValueError, match="unknown design variable"):
        dc.params(sweep=30.0)
    with pytest.raises(ValueError, match="unknown objective"):
        dc._objective(np.zeros((tiny_case.metrics.n_elements, 4)), dc.params(), "range")


@pytest.mark.jax
def test_a_four_digit_case_offers_every_variable(tiny_case):
    dc = differentiable_case("2412", case=tiny_case, mach=0.5, alpha=2.0, print_interval=0)
    assert dc.variables == DESIGN_VARIABLES
    assert dc.params()["thickness"] == 0.12
    assert dc.params()["alpha"] == 2.0


# -- the gradient itself ----------------------------------------------------
@pytest.mark.jax
@pytest.mark.slow
def test_the_adjoint_gradient_matches_a_finite_difference(tiny):
    """The one that matters, and the reason this file exists.

    One flow variable and one shape variable, because they travel by completely
    different routes: ``alpha`` is exact through ``jax.vjp`` on the residual,
    while ``thickness`` moves the mesh and takes the hybrid route -- exact
    adjoint, finite-differenced mesh term at a frozen state.  A bug in either
    route shows up as a disagreement here.

    Subcritical on purpose: with a shock the discrete objective is only
    piecewise differentiable in the design variables -- the shock jumps from one
    cell to the next rather than sliding -- so a finite difference at a
    transonic point measures that jump and is the wrong reference.
    """
    dc = differentiable_case(
        "2412",
        **tiny.__dict__,
        mach=0.5,
        alpha=2.0,
        tolerance=1e-10,
        max_iterations=60_000,
        print_interval=0,
    )
    report = check_gradient(dc, "cl", ("alpha", "thickness"), step=1e-4)
    assert report["alpha"]["relative_error"] < 1e-6
    assert report["thickness"]["relative_error"] < 1e-4
    # and the signs are the physics: a 2412 at 2 degrees gains lift with
    # incidence, and the cambered section is nowhere near stall
    assert report["alpha"]["adjoint"] > 0.0


@pytest.mark.jax
@pytest.mark.slow
def test_one_adjoint_serves_every_variable(tiny):
    """The claim that makes the adjoint worth having: the cost of the gradient
    does not grow with the number of design variables.  Here, the second call
    asks for three more variables and solves no second adjoint."""
    dc = differentiable_case(
        "2412",
        **tiny.__dict__,
        mach=0.5,
        alpha=2.0,
        tolerance=1e-10,
        max_iterations=60_000,
        print_interval=0,
    )
    value, one = dc.value_and_gradient("cl", ("alpha",))
    assert set(dc._cache) == {"cl"}
    costate = dc._cache["cl"].copy()
    again, many = dc.value_and_gradient("cl", DESIGN_VARIABLES)
    assert np.array_equal(dc._cache["cl"], costate)  # not re-solved
    assert again == pytest.approx(value, rel=1e-14)
    assert many["alpha"] == pytest.approx(one["alpha"], rel=1e-12)
    assert tuple(many) == DESIGN_VARIABLES  # in the order asked for
    assert value == pytest.approx(forces(dc.result).cl, rel=1e-10)


@pytest.mark.jax
@pytest.mark.slow
def test_the_adjoint_converges_on_a_realistic_mesh_with_stock_options():
    """Stock options, a mesh big enough to be unforgiving.

    The adjoint sweep is forward Euler whatever the forward solve used, so it
    must be sized by Euler's stability limit.  It was briefly sized by the
    *forward* scheme's instead, which meant the default ``ssprk3`` handed it
    1.18 against Euler's measured limit of 1.15 and every default-options
    gradient diverged.  The rest of this module runs on the 48x24 fixture,
    where 1.18 is still stable, so nothing here caught it.  96x40 does.
    """
    dc = differentiable_case(
        "2412",
        mach=0.5,
        alpha=2.0,
        n_surface=96,
        n_radial=40,
        tolerance=1e-6,
        print_interval=0,
    )
    value, gradient = dc.value_and_gradient("cl")
    assert np.isfinite(value)
    assert all(np.isfinite(v) for v in gradient.values())
