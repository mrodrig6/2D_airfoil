"""The compiled backend must agree with the vectorised one, exactly enough.

"Exactly enough" is the whole question.  The two evaluate the same equations by
different routes -- the compiled one computes every interior flux twice, once
from each side, to make the loop lock-free -- so they cannot be bit-identical.
They can be, and are, identical to round-off, and the *answer* must not depend
on which one ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import Discretization, FlowConditions, SolverOptions, build_mesh, build_metrics, naca4
from src.backends import available_backends, element_faces, get_backend
from src.backends.base import FARFIELD, WALL
from src.mesh import BoundaryTag
from src.postprocess import forces
from src.solver import initial_state, resolve_backend, solve_steady

TINY = Discretization(
    n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02, growth_limit=1.3
)


@pytest.fixture(scope="module")
def metrics():
    mesh, _ = build_mesh(naca4("0012", TINY.n_surface), TINY)
    return build_metrics(mesh)


# -- the gather layout ------------------------------------------------------
def test_every_element_gets_exactly_three_faces(metrics):
    faces = element_faces(metrics)
    assert faces.neighbour.shape == (metrics.n_elements, 3)
    assert np.all(faces.length > 0.0)
    assert np.allclose(np.hypot(faces.normal[..., 0], faces.normal[..., 1]), 1.0)


def test_the_gathered_normals_close_over_every_element(metrics):
    """Discrete Gauss again, now in the layout the compiled kernel reads."""
    faces = element_faces(metrics)
    total = (faces.normal * faces.length[..., None]).sum(axis=1)
    assert np.abs(total).max() < 1e-12


def test_an_interior_face_appears_once_from_each_side(metrics):
    """The duplication the gather trades for being lock-free.  Consistency is
    what makes it legal, so the two evaluations cannot disagree."""
    faces = element_faces(metrics)
    for e in range(0, metrics.n_elements, 37):
        for k in range(3):
            j = faces.neighbour[e, k]
            if j < 0:
                continue
            back = np.flatnonzero(faces.neighbour[j] == e)
            assert back.size == 1
            assert faces.length[j, back[0]] == pytest.approx(faces.length[e, k])
            assert faces.normal[j, back[0]] == pytest.approx(-faces.normal[e, k])


def test_boundary_faces_are_tagged_not_numbered(metrics):
    faces = element_faces(metrics)
    assert (faces.neighbour == WALL).sum() == metrics.mesh.n_faces_on(BoundaryTag.WALL)
    assert (faces.neighbour == FARFIELD).sum() == metrics.mesh.n_faces_on(BoundaryTag.FARFIELD)


# -- backend agreement ------------------------------------------------------
def test_resolve_backend_prefers_the_compiled_one():
    assert resolve_backend(SolverOptions(backend="numpy")) == "numpy"
    assert resolve_backend(SolverOptions()) == available_backends()[0]
    # the multi-stage schemes are vectorised only, so auto must not pick numba
    assert resolve_backend(SolverOptions(scheme="rk4")) == "numpy"


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown backend"):
        SolverOptions(backend="cuda")


@pytest.mark.numba
@pytest.mark.parametrize("flux", ["roe", "hllc"])
@pytest.mark.parametrize("alpha", [0.0, 2.0])
def test_the_compiled_residual_matches_the_vectorised_one(metrics, flux, alpha):
    flow = FlowConditions(mach=0.8, alpha_deg=alpha, flux=flux)
    options = SolverOptions()
    rng = np.random.default_rng(0)
    U = initial_state(metrics, flow) * (1.0 + 0.05 * rng.standard_normal((metrics.n_elements, 4)))

    reference, wave_ref = get_backend("numpy", metrics, flow, options).residual(U)
    R, wave = get_backend("numba", metrics, flow, options).residual(U)
    assert np.abs(R - reference).max() / np.abs(reference).max() < 1e-11
    assert np.abs(wave - wave_ref).max() / np.abs(wave_ref).max() < 1e-12


@pytest.mark.numba
def test_the_compiled_residual_also_preserves_the_free_stream(metrics):
    """The gather duplicates every interior flux, so this is worth repeating
    for it rather than inheriting from the vectorised path."""
    flow = FlowConditions(mach=0.8, alpha_deg=1.25)
    R, _ = get_backend("numba", metrics, flow, SolverOptions()).residual(
        initial_state(metrics, flow)
    )
    off = np.ones(metrics.n_elements, dtype=bool)
    off[metrics.boundary_element[metrics.tag_slice(BoundaryTag.WALL)]] = False
    assert np.abs(R[off]).max() < 1e-10


@pytest.mark.numba
def test_the_numba_backend_refuses_a_scheme_it_does_not_implement(metrics):
    with pytest.raises(ValueError, match="forward Euler only"):
        get_backend("numba", metrics, FlowConditions(), SolverOptions(scheme="rk4"))


@pytest.mark.numba
@pytest.mark.slow
def test_the_answer_does_not_depend_on_the_backend(metrics):
    """The one that matters.  Six significant figures on lift, from two
    different evaluation strategies."""
    flow = FlowConditions(mach=0.8, alpha_deg=1.25)
    options = SolverOptions(tolerance=1e-6, print_interval=0, max_iterations=30_000)
    slow = solve_steady(
        metrics, flow, options.__class__(**{**options.__dict__, "backend": "numpy"})
    )
    fast = solve_steady(
        metrics, flow, options.__class__(**{**options.__dict__, "backend": "numba"})
    )
    assert slow.converged and fast.converged
    assert slow.backend == "numpy" and fast.backend == "numba"
    assert forces(fast).cl == pytest.approx(forces(slow).cl, rel=1e-5)
    assert forces(fast).cd == pytest.approx(forces(slow).cd, rel=1e-5)
    # and it is actually faster, which is the entire point
    assert fast.seconds < slow.seconds


@pytest.mark.numba
@pytest.mark.slow
def test_batching_only_changes_where_the_march_stops(metrics):
    """Convergence is tested every `batch` steps, so a run can overshoot by up
    to that many -- and must reach the same answer."""
    flow = FlowConditions(mach=0.5, alpha_deg=2.0)
    base = dict(tolerance=1e-6, print_interval=0, max_iterations=30_000, backend="numba")
    few = solve_steady(metrics, flow, SolverOptions(**base, batch=5))
    many = solve_steady(metrics, flow, SolverOptions(**base, batch=200))
    assert few.converged and many.converged
    assert abs(few.iterations - many.iterations) <= 200
    assert forces(few).cl == pytest.approx(forces(many).cl, rel=1e-4)


# -- the traced backend -----------------------------------------------------
@pytest.mark.jax
@pytest.mark.parametrize("flux", ["roe", "hllc"])
def test_the_jax_residual_matches_the_vectorised_one(metrics, flux):
    """JAX runs the *same* ``src/assembly.py``, with ``jax.numpy`` injected, so
    this is checking that the injection is complete -- that nothing fell back to
    NumPy and silently broke the trace -- and that float64 is actually on.  At
    single precision this test fails by about 1e-7, which is the point of
    ``enable_double_precision``."""
    import jax.numpy as jnp

    flow = FlowConditions(mach=0.8, alpha_deg=1.25, flux=flux)
    options = SolverOptions()
    rng = np.random.default_rng(1)
    U = initial_state(metrics, flow) * (1.0 + 0.05 * rng.standard_normal((metrics.n_elements, 4)))

    reference, wave_ref = get_backend("numpy", metrics, flow, options).residual(U)
    R, wave = get_backend("jax", metrics, flow, options).residual(U)
    assert jnp.asarray(R).dtype == jnp.float64
    assert np.abs(np.asarray(R) - reference).max() / np.abs(reference).max() < 1e-13
    assert np.abs(np.asarray(wave) - wave_ref).max() / np.abs(wave_ref).max() < 1e-13


@pytest.mark.jax
def test_the_jax_residual_is_a_pure_function_of_the_state(metrics):
    """What the adjoint needs from it: ``residual_fn`` must be differentiable,
    which means no side effects and no NumPy array in the middle of it."""
    import jax

    flow = FlowConditions(mach=0.5, alpha_deg=2.0)
    backend = get_backend("jax", metrics, flow, SolverOptions())
    fn = backend.residual_fn()
    U = initial_state(metrics, flow)
    # one reverse-mode pass; it either traces end to end or it raises
    _, vjp = jax.vjp(fn, U)
    (seed,) = vjp(np.ones_like(U))
    assert seed.shape == U.shape
    assert np.all(np.isfinite(np.asarray(seed)))
