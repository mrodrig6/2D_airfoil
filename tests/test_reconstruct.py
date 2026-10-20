"""The second-order reconstruction, checked by the properties that define it.

None of these run a solve.  They pin the pieces the march will rest on once
it is stable: a gradient that is exact on linear data, a reconstruction that
reproduces a linear field at the face, and limiters that leave a physical
state behind.  Each of the first four caught a real defect while being
written, which is recorded against the one it caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import Discretization, FlowConditions, build_mesh, build_metrics, naca4
from src.physics import conservative, primitives
from src.spatial.reconstruct import LIMITERS, cell_gradients, face_states
from src.timestepping.march import initial_state

GAMMA = 1.4


@pytest.fixture(scope="module")
def metrics():
    disc = Discretization(n_surface=64, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    mesh, _ = build_mesh(naca4("0012", 64), disc)
    return build_metrics(mesh)


@pytest.fixture(scope="module")
def interior(metrics):
    """Cells with no boundary face -- the ones that carry a gradient."""
    mask = np.ones(metrics.n_elements, dtype=bool)
    mask[metrics.boundary_element] = False
    return mask


def _linear_field(metrics):
    slope = np.array([[0.3, -0.7], [1.1, 0.2], [-0.4, 0.9], [0.6, 0.5]])
    return 2.0 + metrics.centroid @ slope.T, slope


def test_the_gradient_is_exact_on_a_linear_field(metrics, interior):
    """What makes the reconstruction second order, and nothing less will do.

    Green-Gauss with an averaged face value -- the obvious choice on a code
    that already has normals and areas -- fails this by a factor of fifteen
    on this mesh, because the mean of two cell values sits midway between the
    centroids rather than at the face midpoint, and the two are far apart
    when the cells differ in area by six orders of magnitude.
    """
    U, slope = _linear_field(metrics)
    grad = cell_gradients(U, metrics)
    assert np.abs(grad[interior] - slope.T[None, :, :]).max() < 1e-11


def test_unlimited_reconstruction_reproduces_a_linear_field_at_the_face(metrics, interior):
    U, slope = _linear_field(metrics)
    flow = FlowConditions(reconstruction="muscl", limiter="none")
    UL, UR, _ = face_states(U, metrics, flow)

    exact = 2.0 + metrics.interior_midpoint @ slope.T
    both = interior[metrics.interior_left] & interior[metrics.interior_right]
    assert np.abs(UL[both] - exact[both]).max() < 1e-11
    assert np.abs(UR[both] - exact[both]).max() < 1e-11
    # and first order is wrong by a cell width, which is the point of all this
    assert np.abs(U[metrics.interior_left][both] - exact[both]).max() > 1e-3


@pytest.mark.parametrize("limiter", LIMITERS)
@pytest.mark.parametrize("reconstruction", ["constant", "muscl"])
def test_a_uniform_state_stays_uniform(metrics, reconstruction, limiter):
    """Free-stream preservation, which any reconstruction can break and must not."""
    from src.spatial import assembly as asm

    flow = FlowConditions(mach=0.8, alpha_deg=1.25, reconstruction=reconstruction, limiter=limiter)
    U = initial_state(metrics, flow)
    R = asm.residual(U, metrics, flow).R
    off_wall = np.ones(metrics.n_elements, dtype=bool)
    off_wall[metrics.boundary_element] = False
    assert np.abs(R[off_wall]).max() < 1e-12


def _rough_but_admissible(metrics, seed=3):
    """A violently varying state whose every *cell mean* is still physical.

    Built in primitive variables on purpose.  Perturbing the conserved ones
    independently inflates the kinetic energy and drives the mean pressure
    negative, and then no slope limiter can promise anything -- the limiter
    scales a slope, it cannot repair a mean.  Getting this wrong made the
    limiters look broken when they were not.
    """
    rng = np.random.default_rng(seed)
    n = metrics.n_elements
    base = initial_state(metrics, FlowConditions(mach=0.8, alpha_deg=1.25))
    rho, vx, vy, p, _ = primitives(base, GAMMA)
    return conservative(
        rho * np.exp(rng.standard_normal(n)),
        vx * (1.0 + 0.8 * rng.standard_normal(n)),
        vy * (1.0 + 0.8 * rng.standard_normal(n)),
        p * np.exp(rng.standard_normal(n)),
        GAMMA,
    )


def _face_minima(states):
    worst_rho, worst_p = np.inf, np.inf
    for S in states:
        rho = S[..., 0]
        safe = np.where(rho > 0.0, rho, np.nan)
        ke = 0.5 * (S[..., 1] ** 2 + S[..., 2] ** 2) / safe
        worst_rho = min(worst_rho, float(np.nanmin(rho)))
        worst_p = min(worst_p, float(np.nanmin((GAMMA - 1.0) * (S[..., 3] - ke))))
    return worst_rho, worst_p


@pytest.mark.parametrize("limiter", ["positivity", "superbee"])
def test_the_limiters_leave_every_face_state_physical(metrics, limiter):
    """The contract: admissible means in, admissible face states out.

    ``superbee`` needs ``positivity`` applied after it to satisfy this, and
    that is not a belt-and-braces precaution -- a slope limiter bounds the
    *range* of the conserved variables by the neighbouring means, which says
    nothing about a nonlinear function of them.  Measured on this state,
    superbee alone held density positive and let pressure reach -37.
    """
    U = _rough_but_admissible(metrics)
    mean_rho, mean_p = _face_minima([U])
    assert mean_rho > 0.0 and mean_p > 0.0, "the fixture must hand the limiter a valid mean"

    flow = FlowConditions(mach=0.8, alpha_deg=1.25, reconstruction="muscl", limiter=limiter)
    rho, p = _face_minima(face_states(U, metrics, flow))
    assert rho > 0.0, f"{limiter} left a non-positive face density"
    assert p > 0.0, f"{limiter} left a non-positive face pressure"


def test_no_boundary_cell_carries_a_gradient(metrics):
    """Boundary cells stay first order, and the reason is worth keeping.

    A triangle on the wall has two interior neighbours and the fit has two
    unknowns, so it reproduces both differences exactly with nothing left to
    damp it.  On a wall cell of aspect ratio a thousand the wall-normal
    component of that exact fit is enormous: the march grew wall density from
    1.07 to 3.35 in forty steps before this was required to be
    over-determined.
    """
    from src.spatial.reconstruct import _cached_stencil

    carries = np.abs(_cached_stencil(metrics).inverse_moment).sum(axis=(1, 2)) > 0.0
    on_boundary = np.zeros(metrics.n_elements, dtype=bool)
    on_boundary[metrics.boundary_element] = True
    assert not (carries & on_boundary).any()
    assert carries[~on_boundary].all(), "every interior cell should carry one"


def test_marching_with_muscl_refuses_rather_than_diverging():
    """It is not ready, and says so instead of returning a blown-up field."""
    from src.timestepping.march import solve_steady

    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    mesh, _ = build_mesh(naca4("0012", 48), disc)
    m = build_metrics(mesh)
    with pytest.raises(NotImplementedError, match="not ready to march"):
        solve_steady(m, FlowConditions(reconstruction="muscl"))
