"""The non-dimensionalisation.

Every claim ``docs/theory.md`` makes about the free stream is checked here
rather than asserted there, because a documented convention that the code does
not keep is worse than none.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import Discretization, FlowConditions


@pytest.mark.parametrize("mach", [0.1, 0.5, 0.8, 1.2, 2.5])
def test_the_free_stream_has_the_mach_number_it_was_given(mach):
    flow = FlowConditions(mach=mach)
    assert flow.speed / flow.sound_speed == pytest.approx(mach)
    assert flow.sound_speed == pytest.approx(math.sqrt(flow.gamma * flow.pressure / flow.density))


def test_pressure_and_dynamic_pressure_are_what_the_documentation_says():
    flow = FlowConditions(mach=0.8)
    assert flow.pressure == pytest.approx(1.0 / (flow.gamma * 0.8**2))
    assert flow.pressure == pytest.approx(1.1160714285714286)
    assert flow.dynamic_pressure == 0.5  # exactly, at every operating point
    assert FlowConditions(mach=2.0).dynamic_pressure == 0.5


def test_the_gas_constant_choice_makes_cv_one():
    flow = FlowConditions()
    assert flow.gas_constant == pytest.approx(flow.gamma - 1.0)
    internal = flow.pressure / ((flow.gamma - 1.0) * flow.density)
    assert internal == pytest.approx(flow.temperature)  # c_v = 1


@pytest.mark.parametrize("alpha", [-5.0, 0.0, 1.25, 10.0])
def test_angle_of_attack_only_turns_the_free_stream(alpha):
    flow = FlowConditions(mach=0.8, alpha_deg=alpha)
    u, v = flow.velocity
    assert math.hypot(u, v) == pytest.approx(1.0)
    assert math.degrees(math.atan2(v, u)) == pytest.approx(alpha)
    # turning the free stream cannot change a scalar
    assert flow.pressure == pytest.approx(FlowConditions(mach=0.8).pressure)
    assert flow.state[0] == 1.0


def test_the_conservative_state_inverts_to_the_primitives():
    flow = FlowConditions(mach=0.8, alpha_deg=1.25)
    rho, rhou, rhov, rhoE = flow.state
    assert rho == pytest.approx(flow.density)
    assert (rhou / rho, rhov / rho) == pytest.approx(flow.velocity)
    p = (flow.gamma - 1.0) * (rhoE - 0.5 * (rhou**2 + rhov**2) / rho)
    assert p == pytest.approx(flow.pressure)


def test_legacy_initialstate_agrees():
    """``legacy/initialstate.m`` at its own operating point, transcribed."""
    gamma, mach, alpha = 1.4, 0.8, math.radians(1.25)
    p_inf = (1.0 / gamma) * (1.0 / mach) ** 2
    expected = np.array([1.0, math.cos(alpha), math.sin(alpha), p_inf / (gamma - 1.0) + 0.5])
    assert np.allclose(FlowConditions(mach=0.8, alpha_deg=1.25).state, expected)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"mach": 0.0}, "mach must be positive"),
        ({"gamma": 1.0}, "gamma must exceed 1"),
    ],
)
def test_an_impossible_operating_point_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FlowConditions(**kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"n_surface": 17}, "even"),
        ({"n_radial": 2}, "n_radial"),
        ({"far_field_radius": 1.0}, "far_field_radius"),
        ({"first_cell_height": 0.5}, "first_cell_height"),
        ({"clustering": "spline"}, "unknown clustering"),
    ],
)
def test_an_impossible_mesh_request_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Discretization(**kwargs)
