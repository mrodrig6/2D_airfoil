"""Prepared sensitivity cases and repeated solves must describe the same flow."""

from dataclasses import replace

import numpy as np
import pytest

from src import build_case
from src.sensitivity import FLOW_VARIABLES, differentiable_case
from src.timestepping.march import SolveResult, initial_state


def test_prepared_cambered_case_uses_its_own_shape_parameters(tiny):
    case = build_case("2412", tiny)
    dc = differentiable_case(case=case)
    assert dc.shape == {"camber": 0.02, "camber_position": 0.4, "thickness": 0.12}
    moved = dc._case_at(dc.params())
    np.testing.assert_allclose(moved.airfoil.points, case.airfoil.points, atol=1e-14)


def test_prepared_custom_section_does_not_invent_shape_parameters(tiny, section):
    custom = replace(section, name="custom section")
    dc = differentiable_case(case=build_case(custom, tiny))
    assert dc.shape is None
    assert dc.variables == FLOW_VARIABLES


def test_a_modified_section_cannot_inherit_its_old_naca_parameters(tiny, section):
    custom = replace(section, points=section.points * [1.0, 1.1])
    dc = differentiable_case(case=build_case(custom, tiny))
    assert dc.shape is None


@pytest.mark.parametrize("name", ["NACA 0000", "NACA 1012"])
def test_a_custom_section_with_an_invalid_naca_name_still_allows_flow_gradients(
    tiny, section, name
):
    custom = replace(section, name=name)
    dc = differentiable_case(case=build_case(custom, tiny))
    assert dc.variables == FLOW_VARIABLES


def test_prepared_case_geometry_overrides_are_not_ignored(tiny):
    case = build_case("2412", tiny)
    with pytest.raises(TypeError, match="invalidate a prepared case"):
        differentiable_case(case=case, n_radial=32)


@pytest.fixture
def dc_and_result(tiny, monkeypatch):
    case = build_case("0012", tiny)
    dc = differentiable_case(case=case)
    result = SolveResult(
        state=initial_state(case.metrics, dc.flow),
        metrics=case.metrics,
        flow=dc.flow,
        options=dc.options,
        converged=True,
        iterations=1,
        residual=0.0,
        history=np.zeros(1),
    )
    monkeypatch.setattr(case, "solve", lambda *args: result)
    return dc, result


def test_force_solve_invalidates_the_old_adjoint(dc_and_result):
    dc, _ = dc_and_result
    dc.solve()
    dc._cache["cl"] = np.zeros_like(dc.result.state)
    dc.solve(force=True)
    assert not dc._cache


def test_failed_forward_solve_is_never_reused(dc_and_result):
    dc, result = dc_and_result
    result.converged = False
    result.message = "iteration limit"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="flow solve did not converge"):
            dc.solve()


@pytest.mark.jax
@pytest.mark.parametrize("budget", [3, 7])
def test_adjoint_uses_the_exact_iteration_budget(dc_and_result, budget):
    dc, _ = dc_and_result
    dc.adjoint_iterations = budget
    dc.adjoint_batch = 5
    dc.adjoint_tolerance = 1e-30
    with pytest.raises(RuntimeError, match=rf"did not converge in {budget} sweeps"):
        dc.adjoint()
