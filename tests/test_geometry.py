"""The airfoil surface: the NACA family, coordinate files, and the winding.

The decisive test is :func:`test_naca0012_reproduces_the_legacy_geometry`.  The
legacy MATLAB case is a NACA 0012 stored as a decomposed geometry matrix, and
this port is only a port if the Python generator puts the surface in the same
place.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import geometry as geo

LEGACY_MAT = "legacy/nacageom.mat"


def naca0012_thickness(x, a4=0.1036):
    """The four-digit thickness polynomial, written out so the test is independent."""
    return 5 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - a4 * x**4)


def test_naca0012_reproduces_the_legacy_geometry():
    """Every point of ``legacy/nacageom.mat`` must lie on the generated surface.

    The legacy segments are not stored in loop order, so this compares each
    endpoint against the thickness polynomial rather than against our point
    list.  It also pins the closed-trailing-edge coefficient: the legacy data
    matches ``a4 = 0.1036`` to round-off and ``a4 = 0.1015`` only to 1.3e-3,
    which is why the closed form is the default.
    """
    sio = pytest.importorskip("scipy.io", reason="reading the legacy .mat needs scipy")
    matrix = sio.loadmat(LEGACY_MAT)["nacageom"]
    segments = matrix[:, matrix[0] == 2]
    points = np.unique(
        np.column_stack(
            [
                np.concatenate([segments[1], segments[2]]),
                np.concatenate([segments[3], segments[4]]),
            ]
        ),
        axis=0,
    )
    x, y = points[:, 0], points[:, 1]
    assert np.abs(np.abs(y) - naca0012_thickness(x, 0.1036)).max() < 1e-12
    assert np.abs(np.abs(y) - naca0012_thickness(x, 0.1015)).max() > 1e-3

    generated = geo.naca4("0012", 200)
    xs, thickness, _ = generated.thickness_and_camber(400)
    inside = (xs > 0.02) & (xs < 0.98)  # the leading edge is a sqrt, not a polynomial
    assert np.abs(thickness[inside] - 2 * naca0012_thickness(xs[inside])).max() < 2e-4


def test_the_surface_is_counter_clockwise_with_the_trailing_edge_first(section):
    assert section.area > 0.0
    assert section.points[0, 0] == pytest.approx(1.0)
    assert section.points[section.leading_edge_index] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_normals_point_away_from_the_section(section):
    """The outward normal convention the mesh generator and the wall flux both rely on."""
    n = geo.surface_normals(section.points)
    assert np.allclose(np.hypot(n[:, 0], n[:, 1]), 1.0)
    inside = np.array([0.35, 0.0])  # comfortably within a NACA 0012
    assert np.all(np.einsum("ij,ij->i", n, section.points - inside) > 0.0)


def test_a_sharp_trailing_edge_is_one_point_and_an_open_one_is_two():
    sharp = geo.naca4("0012", 120)
    assert sharp.is_sharp and sharp.n_points == 120
    open_te = geo.naca4("0012", 120, closed_trailing_edge=False)
    assert not open_te.is_sharp and open_te.n_points == 121
    assert open_te.trailing_edge_gap == pytest.approx(2 * naca0012_thickness(1.0, 0.1015))


def test_camber_lands_where_the_digits_say(cambered):
    x, _, camber = cambered.thickness_and_camber(400)
    assert camber.max() == pytest.approx(0.02, abs=1e-4)
    assert x[np.argmax(camber)] == pytest.approx(0.4, abs=0.01)


@pytest.mark.parametrize("code", ["12", "naca00123", "0000", "abcd"])
def test_a_bad_code_is_refused(code):
    with pytest.raises(ValueError):
        geo.naca4(code)


def test_a_selig_file_round_trips_either_way_round(section, tmp_path):
    """Reading must not depend on how the file happened to be wound."""
    path = geo.write_dat(section, tmp_path / "s.dat")
    assert np.allclose(geo.read_dat(path).points, section.points, atol=1e-7)

    reversed_file = geo.write_dat(
        geo.Airfoil("reversed", section.points[::-1].copy()), tmp_path / "r.dat"
    )
    assert np.allclose(geo.read_dat(reversed_file).points, section.points, atol=1e-7)


def test_a_lednicer_file_reads_as_the_same_section(section, tmp_path):
    upper, lower = section.sides()
    body = "\n".join(f"{x:10.6f} {y:10.6f}" for x, y in upper)
    body += "\n\n" + "\n".join(f"{x:10.6f} {y:10.6f}" for x, y in lower)
    path = tmp_path / "l.dat"
    path.write_text(f"NACA 0012\n  {len(upper)}.  {len(lower)}.\n\n{body}\n")
    assert np.allclose(geo.read_dat(path).points, section.points, atol=1e-6)


def test_resampling_preserves_the_section():
    """A coarse section resampled finer must still be the same shape."""
    coarse = geo.naca4("2412", 64)
    fine = geo.resample(coarse, 256)
    assert fine.n_points == 256
    assert fine.max_thickness == pytest.approx(coarse.max_thickness, abs=2e-4)
    assert fine.max_camber == pytest.approx(coarse.max_camber, abs=2e-4)
    assert fine.area == pytest.approx(coarse.area, rel=2e-3)


def test_load_airfoil_resolves_a_code_and_a_path(section, tmp_path):
    assert geo.load_airfoil("naca 0012", 48).n_points == 48
    path = geo.write_dat(section, tmp_path / "x.dat")
    assert geo.load_airfoil(path, 64).n_points == 64
    with pytest.raises(ValueError, match="cannot resolve"):
        geo.load_airfoil("not-an-airfoil")
