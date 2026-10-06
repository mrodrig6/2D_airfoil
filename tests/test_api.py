"""The one call students make, and the option plumbing behind it."""

from __future__ import annotations

import numpy as np
import pytest

from src import Discretization, build_case, solve_airfoil
from src.api import Case

SMALL = dict(
    n_surface=48,
    n_radial=24,
    far_field_radius=10.0,
    first_cell_height=0.02,
)


def test_build_case_resolves_a_section_and_meshes_it():
    case = build_case("2412", **SMALL)
    assert isinstance(case, Case)
    assert case.airfoil.name == "NACA 2412"
    assert case.mesh.n_elements == Discretization(**SMALL).n_elements
    assert case.metrics.closure_error() < 1e-12
    assert "NACA 2412" in case.summary()


def test_options_are_routed_to_the_object_that_owns_them():
    """One vocabulary, not four: every option goes by its own name."""
    case = build_case("0012", **SMALL, topology="cgrid", n_wake=8, wake_length=8.0)
    assert case.discretization.topology == "cgrid"
    assert case.mesh.source == "C-grid"


def test_an_unknown_option_names_the_ones_that_exist():
    with pytest.raises(TypeError, match="unknown option 'mahc'"):
        build_case("0012", mahc=0.8)


def test_build_case_refuses_options_that_belong_to_the_solve():
    with pytest.raises(TypeError, match="belong to the solve"):
        build_case("0012", mach=0.8)


def test_reusing_a_case_refuses_options_that_would_change_the_mesh():
    """Silently rebuilding would make a sweep quietly inconsistent."""
    case = build_case("0012", **SMALL)
    with pytest.raises(TypeError, match="prepared case"):
        solve_airfoil(case=case, n_radial=48)


def test_a_mesh_from_a_file_can_replace_the_generated_one(tmp_path):
    from src import write_su2

    generated = build_case("0012", **SMALL)
    path = write_su2(generated.mesh, tmp_path / "m.su2")
    read = build_case("0012", mesh_file=path)
    assert read.mesh.n_elements == generated.mesh.n_elements
    assert np.allclose(read.metrics.reference_area(), generated.metrics.reference_area())


@pytest.mark.slow
def test_solve_airfoil_takes_a_section_to_loads_in_one_call():
    result = solve_airfoil(
        "0012",
        **SMALL,
        mach=0.5,
        alpha=2.0,
        tolerance=1e-5,
        print_interval=0,
        max_iterations=20_000,
    )
    assert result.converged, result.message
    load = result.forces()
    assert load.cl > 0.15
    assert "cl =" in result.report()


@pytest.mark.slow
@pytest.mark.parametrize("flux", ["roe", "hllc", "ausmup", "slau2"])
def test_every_flux_reaches_the_same_subcritical_answer(flux):
    """A shock-free case is where all four should agree; if they do not, the
    disagreement is in the scheme rather than in the shock capturing."""
    result = solve_airfoil(
        "0012",
        **SMALL,
        mach=0.5,
        alpha=2.0,
        flux=flux,
        tolerance=1e-5,
        print_interval=0,
        max_iterations=30_000,
    )
    assert result.converged, f"{flux}: {result.message}"
    assert result.forces().cl == pytest.approx(0.247, rel=0.12)
