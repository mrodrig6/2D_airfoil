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


# -- the five-digit family --------------------------------------------------
def naca5_camber(code, x):
    """The five-digit camber line, written out so the test is independent."""
    from src.geometry import _NACA5_PLAIN, _NACA5_REFLEXED

    lift, position, reflexed = int(code[0]), int(code[1]), int(code[2])
    scale = 0.15 * lift / 0.3
    if reflexed:
        m, k1, ratio = _NACA5_REFLEXED[position]
        k1 *= scale
        tail = ratio * (1 - m) ** 3 + m**3
        return (k1 / 6) * np.where(
            x <= m, (x - m) ** 3 - tail * x + m**3, ratio * (x - m) ** 3 - tail * x + m**3
        )
    m, k1 = _NACA5_PLAIN[position]
    k1 *= scale
    return (k1 / 6) * np.where(x <= m, x**3 - 3 * m * x**2 + m * m * (3 - m) * x, m**3 * (1 - x))


@pytest.mark.parametrize("code", ["12012", "23012", "24012", "25012", "43012", "23112", "25112"])
def test_the_five_digit_crest_lands_where_the_second_digit_says(code):
    """The tabulated m and k1 exist to put the crest at P/20; check that they do."""
    x = np.linspace(0.0, 1.0, 200_001)
    camber = naca5_camber(code, x)
    assert abs(x[np.argmax(np.abs(camber))] - int(code[1]) / 20) < 2e-3


def test_naca23012_matches_the_published_section():
    """0.3 design lift, crest at 15% chord, 12% thick -- the canonical one."""
    x = np.linspace(0.0, 1.0, 200_001)
    assert naca5_camber("23012", x).max() == pytest.approx(0.0183, abs=1e-4)
    section = geo.naca5("23012", 400)
    xs, thickness, _ = section.thickness_and_camber(2000)
    assert thickness.max() == pytest.approx(0.12, abs=5e-4)
    assert xs[np.argmax(thickness)] == pytest.approx(0.30, abs=0.01)
    assert section.is_sharp and section.area > 0.0


def test_the_generated_five_digit_surface_sits_on_its_camber_line():
    section = geo.naca5("23012", 400)
    x, _, camber = section.thickness_and_camber(800)
    inside = (x > 0.05) & (x < 0.95)  # the leading edge is a sqrt, not a polynomial
    assert np.abs(camber[inside] - naca5_camber("23012", x[inside])).max() < 2e-3


def test_the_reflexed_camber_line_turns_up_at_the_tail():
    """Which is the point of the third digit: it is what nulls the pitching moment."""
    x = np.linspace(0.0, 1.0, 2001)
    plain, reflexed = naca5_camber("23012", x), naca5_camber("23112", x)
    assert np.interp(0.9, x, reflexed) < np.interp(0.9, x, plain)
    assert plain.max() > 0.0 and reflexed.max() > 0.0


@pytest.mark.parametrize(
    "code,message",
    [
        ("26012", "camber position 6 is not tabulated"),
        ("21112", "camber position 1 is not tabulated"),
        ("23212", "plain .0. or reflexed .1."),
        ("2301", "not a NACA five-digit code"),
    ],
)
def test_an_untabulated_five_digit_section_is_refused(code, message):
    """Interpolating the table would produce something that is not a NACA section."""
    with pytest.raises(ValueError, match=message):
        geo.naca5(code)


@pytest.mark.parametrize(
    "code,name",
    [
        ("0012", "NACA 0012"),
        ("naca 2412", "NACA 2412"),
        ("23012", "NACA 23012"),
        ("NACA-23112", "NACA 23112"),
    ],
)
def test_naca_dispatches_on_the_digit_count(code, name):
    assert geo.naca(code, 64).name == name


@pytest.mark.parametrize("code", ["63-012", "sc20612", "012"])
def test_naca_refuses_what_it_has_no_closed_form_for(code):
    with pytest.raises(ValueError, match="six-series"):
        geo.naca(code)


def test_load_airfoil_takes_a_five_digit_code_too():
    assert geo.load_airfoil("23012", 96).name == "NACA 23012"


def test_normals_of_an_open_path_do_not_wrap_around():
    """The C-grid's inner path is open, and joining its ends would put a normal
    where the outflow boundary is.

    An L: right along the x axis, then up.  Each end must take its own single
    edge's normal; only the corner is a bisector.
    """
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    n = geo.surface_normals(path, closed=False)
    assert np.allclose(n[0], [0.0, -1.0])  # from the first edge alone
    assert np.allclose(n[-1], [1.0, 0.0])  # from the last edge alone
    assert np.allclose(n[1], [np.sqrt(0.5), -np.sqrt(0.5)])  # the corner bisects
    closed = geo.surface_normals(path, closed=True)
    assert not np.allclose(closed[0], n[0])  # wrapping changes both ends
    assert not np.allclose(closed[-1], n[-1])
