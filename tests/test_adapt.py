"""Refinement has to leave a mesh the solver can still use.

Three things can go wrong and only one of them is loud.  A non-conforming mesh
raises on the next connectivity build, so it finds itself.  The other two are
silent: a new wall node left on the chord between two surface points means the
section never converges to its own shape, and green closure applied to a green
triangle drives the smallest angle to zero, which shows up only as a march that
has mysteriously stopped making progress.  Both are tested here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import BoundaryTag, Discretization, FlowConditions, SolverOptions, build_metrics, naca4
from src.adapt import (
    BoundaryGeometry,
    RefinementTree,
    _project_to_polyline,
    adaptive_solve,
    flag_fraction,
    mach_jump_indicator,
    transfer_state,
)
from src.postprocess import forces
from src.solver import SolveResult, initial_state


def smallest_angle(mesh) -> float:
    """The worst angle anywhere in the mesh, in degrees."""
    p = mesh.node_coords[mesh.elements]
    angles = []
    for k in range(3):
        u = p[:, (k + 1) % 3] - p[:, k]
        v = p[:, (k + 2) % 3] - p[:, k]
        cosine = np.einsum("ij,ij->i", u, v) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return float(np.min(angles))


@pytest.fixture(scope="module")
def tree(tiny_mesh, section):
    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    return RefinementTree(tiny_mesh, BoundaryGeometry.from_case(section, disc))


# -- conformity -------------------------------------------------------------
def test_refining_a_few_cells_leaves_a_usable_mesh(tree, tiny_mesh):
    """The discrete Gauss identity is the real test: it closes to round-off only
    if every face has exactly one element on each side and the normals agree."""
    tree.refine([0, 1, 500, 501])
    mesh, leaf_of = tree.build()

    assert mesh.n_elements > tiny_mesh.n_elements
    assert leaf_of.size == mesh.n_elements
    metrics = build_metrics(mesh)
    assert np.abs(metrics.closure()).max() < 1e-12 if hasattr(metrics, "closure") else True
    # every element's face normals must sum to zero, which is the same statement
    total = np.zeros((mesh.n_elements, 2))
    np.add.at(
        total, metrics.interior_left, metrics.interior_normal * metrics.interior_length[:, None]
    )
    np.add.at(
        total, metrics.interior_right, -metrics.interior_normal * metrics.interior_length[:, None]
    )
    np.add.at(
        total, metrics.boundary_element, metrics.boundary_normal * metrics.boundary_length[:, None]
    )
    assert np.abs(total).max() < 1e-12


def test_every_leaf_ends_with_at_most_one_split_edge(tree):
    """One-irregularity.  Two split edges would need two cuts to close, and two
    cuts of a triangle is a red refinement done badly."""
    tree.close()
    for leaf in tree.leaves:
        a, b, c = tree.triangles[leaf]
        split = sum(tree._is_split(*e) for e in ((a, b), (b, c), (c, a)))
        assert split <= 1


def test_repeated_cycles_do_not_produce_slivers(tiny_mesh, section):
    """The failure this design exists to prevent.

    Bisecting a green triangle, and bisecting the result, drives the smallest
    angle to zero.  Here green closure is recomputed from the red tree every
    cycle, so a green triangle is never refined -- it does not survive between
    cycles at all.  Five cycles must therefore not degrade the mesh.
    """
    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    tree = RefinementTree(tiny_mesh, BoundaryGeometry.from_case(section, disc))
    rng = np.random.default_rng(0)

    worst = [smallest_angle(tiny_mesh)]
    leaf_of = np.arange(tiny_mesh.n_elements)
    for _ in range(5):
        tree.refine(leaf_of[rng.choice(leaf_of.size, size=leaf_of.size // 20, replace=False)])
        mesh, leaf_of = tree.build()
        worst.append(smallest_angle(mesh))

    # The honest claim, which is not "the angle never falls": the first cycle's
    # green closure does cost angle, because bisecting a triangle from a vertex
    # splits the angle there.  What must not happen is that it keeps costing --
    # the compounding is the failure, not the single step.
    assert worst[1] < worst[0]
    assert min(worst[2:]) >= worst[1] - 1e-9


# -- the geometry -----------------------------------------------------------
def test_refining_the_wall_makes_the_section_rounder(tiny_mesh, section):
    """The statement that matters, and it is global rather than per-node.

    A midpoint left on the chord is *inside* the body, so a refined wall made of
    chord midpoints encloses no more area than the coarse one did and the section
    never converges to its own shape.  Projected onto the surface, the polygon
    has to grow -- towards the true area, never past it.
    """
    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    tree = RefinementTree(tiny_mesh, BoundaryGeometry.from_case(section, disc))
    wall = tiny_mesh.boundary_edges[tiny_mesh.tag_slice(BoundaryTag.WALL)]

    def area(points):
        x, y = points[:, 0], points[:, 1]
        return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    # The wall faces come out of the connectivity sort in no particular order,
    # so the loop has to be walked before it can be a polygon.
    nxt = {int(a): int(b) for a, b in wall[:, 1:]}
    chain = [next(iter(nxt))]
    while nxt[chain[-1]] != chain[0]:
        chain.append(nxt[chain[-1]])
    assert len(chain) == len(nxt)

    coarse, refined, inserted = [], [], []
    for node in chain:
        new = np.asarray(tree.nodes[tree._split(node, nxt[node])])
        coarse.append(tree.nodes[node])
        refined.extend([tree.nodes[node], tuple(new)])
        inserted.append(new)

    dense = naca4("0012", 2048).points
    # On the section, measured against the *segments* of a finer reference and
    # not against its vertices -- near the trailing edge the vertices are 2e-4
    # apart, so a vertex distance would report that spacing and not the error.
    inserted = np.asarray(inserted)
    loop = np.vstack([dense, dense[:1]])
    gap = np.hypot(*(inserted - _project_to_polyline(inserted, loop)).T)
    assert gap.max() < 1e-5

    truth = area(dense)
    assert area(np.asarray(coarse)) < area(np.asarray(refined)) <= truth * (1.0 + 1e-9)
    # and the gain is real, not round-off
    assert area(np.asarray(refined)) - area(np.asarray(coarse)) > 1e-5 * truth


def test_a_new_far_field_node_lands_on_the_circle(tiny_mesh, section):
    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    tree = RefinementTree(tiny_mesh, BoundaryGeometry.from_case(section, disc))
    far = tiny_mesh.boundary_edges[tiny_mesh.tag_slice(BoundaryTag.FARFIELD)]

    a, b = int(far[0, 1]), int(far[0, 2])
    new = np.asarray(tree.nodes[tree._split(a, b)])
    centre = np.asarray(disc.far_field_centre)
    assert np.hypot(*(new - centre)) == pytest.approx(disc.far_field_radius, rel=1e-12)


def test_a_node_is_on_a_boundary_only_when_both_its_parents_are(tree, tiny_mesh):
    """The one-sided test would drag a midpoint of an interior edge onto the
    section, tangling the mesh at the wall."""
    wall_nodes = set(tiny_mesh.boundary_nodes(BoundaryTag.WALL).tolist())
    interior = next(
        (a, b)
        for a, b in tiny_mesh.interior_edges[:, 2:4]
        if (int(a) in wall_nodes) != (int(b) in wall_nodes)
    )
    assert tree._edge_tag(int(interior[0]), int(interior[1])) == 0


# -- the indicator ----------------------------------------------------------
def test_the_indicator_sees_no_jump_in_a_uniform_field(tiny_metrics):
    """Away from the wall, where the free stream does have a normal velocity --
    that term is the wall condition's own residual, and it is supposed to fire."""
    flow = FlowConditions(mach=0.5, alpha_deg=2.0)
    result = SolveResult(
        state=initial_state(tiny_metrics, flow),
        metrics=tiny_metrics,
        flow=flow,
        options=SolverOptions(print_interval=0),
        converged=True,
        iterations=0,
        residual=0.0,
        history=np.zeros(0),
    )
    error = mach_jump_indicator(result)

    off_wall = np.ones(tiny_metrics.n_elements, dtype=bool)
    off_wall[tiny_metrics.boundary_element[tiny_metrics.tag_slice(BoundaryTag.WALL)]] = False
    assert np.abs(error[off_wall]).max() < 1e-12
    assert error[~off_wall].max() > 0.0


def test_flagging_takes_the_worst_cells_and_the_right_number_of_them():
    error = np.array([0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])
    assert flag_fraction(error, 0.3).tolist() == [1, 3, 5]
    assert flag_fraction(error, 1.0).size == error.size
    assert flag_fraction(error, 0.001).size == 1  # never zero cells
    with pytest.raises(ValueError, match="fraction"):
        flag_fraction(error, 0.0)


# -- the state --------------------------------------------------------------
def test_each_child_inherits_its_parent_exactly(tiny_mesh, tiny_metrics, section):
    disc = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    tree = RefinementTree(tiny_mesh, BoundaryGeometry.from_case(section, disc))
    U = initial_state(tiny_metrics, FlowConditions(mach=0.5)) * (
        1.0 + 0.01 * np.arange(tiny_metrics.n_elements)[:, None]
    )
    old_leaf_of = np.arange(tiny_mesh.n_elements)

    tree.refine([7, 8, 9])
    mesh, new_leaf_of = tree.build()
    moved = transfer_state(U, tree, old_leaf_of, new_leaf_of)

    assert moved.shape == (mesh.n_elements, 4)
    # every new value is one of the old ones, and the unrefined cells keep theirs
    assert np.isin(moved[:, 0], U[:, 0]).all()
    for new, leaf in enumerate(new_leaf_of):
        if leaf < tiny_mesh.n_elements and tree.children[int(leaf)] is None:
            assert moved[new] == pytest.approx(U[int(leaf)])


# -- the loop ---------------------------------------------------------------
@pytest.mark.slow
def test_adaptation_moves_the_loads_towards_the_fine_mesh_answer():
    """The claim worth making: cells spent at the shock buy accuracy.

    Checked against the direction of the uniformly refined answer rather than
    against a fixed number, because the point is the trend and a threshold here
    would be a number copied from one machine.
    """
    disc = Discretization(n_surface=64, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
    history = adaptive_solve(
        "0012",
        disc,
        FlowConditions(mach=0.8, alpha_deg=1.25),
        SolverOptions(tolerance=1e-6, max_iterations=80_000, print_interval=0),
        cycles=4,
        verbose=False,
    )
    first, last = history.cycles[0], history.cycles[-1]
    assert last.n_elements > 1.5 * first.n_elements
    # against the starting mesh's own worst angle, not an absolute number: an
    # O-grid is already slightly skewed at the trailing edge, and the claim is
    # that refinement does not make it meaningfully worse
    assert smallest_angle(history.mesh) > 0.8 * smallest_angle(first.result.metrics.mesh)

    from src import build_case

    fine = build_case(
        "0012",
        Discretization(n_surface=192, n_radial=64, far_field_radius=10.0, first_cell_height=0.02),
    ).solve(
        FlowConditions(mach=0.8, alpha_deg=1.25),
        SolverOptions(tolerance=1e-6, max_iterations=80_000, print_interval=0),
    )
    target = forces(fine).cl
    assert abs(last.loads.cl - target) < abs(first.loads.cl - target)
