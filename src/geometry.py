r"""Airfoil geometry: the NACA four-digit family, and coordinates from a file.

This module is the only part of the discretisation that knows what an airfoil
is.  Everything downstream -- the mesh generator, the metrics, the solver --
takes a closed list of surface points and does not care where they came from.
That is what makes an externally designed section (from XFOIL, from a wind
tunnel model, from an optimiser) a first-class input rather than a special case.

The ordering convention
-----------------------
A surface is stored as ``(n, 2)`` points forming a **closed counter-clockwise
loop with no repeated point**: index 0 is the trailing edge, the loop runs
forward over the *upper* surface to the leading edge and back along the *lower*
surface, and the segment from the last point to point 0 closes it.

Counter-clockwise is not arbitrary.  For a curve traversed counter-clockwise the
outward normal -- pointing away from the body, into the flow -- is

.. math::
    \hat{n} = (t_y,\ -t_x), \qquad \hat{t} = \text{unit tangent},

with no sign test anywhere, which is what lets the mesh generator march outward
and the solver orient a wall flux without special-casing either.  It happens to
coincide with the order XFOIL writes (:func:`read_dat` flips anything else), and
it is the *reverse* of the legacy ``nacageom.mat``, which ran trailing edge ->
lower -> leading edge -> upper.

The four-digit thickness distribution
-------------------------------------
For a NACA ``MPXX`` section with thickness ratio :math:`t = XX/100`,

.. math::
    y_t(x) = 5t\left(0.2969\sqrt{x} - 0.1260\,x - 0.3516\,x^2
             + 0.2843\,x^3 - a_4\,x^4\right),

with :math:`a_4 = 0.1015` for the original (open) trailing edge and
:math:`a_4 = 0.1036` for the closed one.  **The default here is the closed
form**, because that is what the legacy ``nacageom.mat`` used: its points agree
with :math:`a_4 = 0.1036` to better than :math:`10^{-6}` and with
:math:`a_4 = 0.1015` only to :math:`1.4\times 10^{-3}`.  A closed trailing edge
also removes a blunt-base cell from the O-grid, which matters more here than the
historical coefficient does.

Camber, for a non-zero first digit :math:`m = M/100` with position
:math:`p = P/10`:

.. math::
    y_c(x) = \begin{cases}
      \dfrac{m}{p^2}\left(2px - x^2\right) & x \le p \\[2mm]
      \dfrac{m}{(1-p)^2}\left(1 - 2p + 2px - x^2\right) & x > p
    \end{cases}

and the surface is laid out **normal to the camber line**, which is the correct
construction rather than the common shortcut of adding :math:`y_t` vertically.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Thickness-polynomial quartic coefficient, by trailing-edge treatment.
_A4 = {True: 0.1036, False: 0.1015}

#: Recognised surface point distributions.
CLUSTERINGS = ("cosine", "uniform")


# --------------------------------------------------------------------------
# The surface
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Airfoil:
    """A closed airfoil surface, normalised to unit chord.

    Build one with :func:`naca4`, :func:`read_dat` or :func:`load_airfoil`; the
    constructor does no validation, so prefer those.
    """

    name: str
    points: np.ndarray

    def __post_init__(self) -> None:
        pts = np.ascontiguousarray(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
        if pts.shape[0] < 16:
            raise ValueError(f"an airfoil needs at least 16 points, got {pts.shape[0]}")
        object.__setattr__(self, "points", pts)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def x(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.points[:, 1]

    @property
    def chord(self) -> float:
        return float(self.x.max() - self.x.min())

    @property
    def area(self) -> float:
        """Enclosed area, positive for the counter-clockwise storage order."""
        return float(signed_area(self.points))

    @property
    def trailing_edge_gap(self) -> float:
        """Thickness of the base at the trailing edge; exactly 0 when it is sharp.

        Index 0 sits at maximum ``x`` by construction.  A *sharp* section has one
        point there and the loop closes along a panel that runs back up to it; an
        *open* one has a second point at the same ``x``, and the closing segment
        is the blunt base whose length this returns.
        """
        if abs(self.points[-1, 0] - self.points[0, 0]) > 1e-9 * max(self.chord, 1.0):
            return 0.0
        return float(abs(self.points[-1, 1] - self.points[0, 1]))

    @property
    def is_sharp(self) -> bool:
        return self.trailing_edge_gap < 1e-10

    @property
    def leading_edge_index(self) -> int:
        """Index of the point furthest from the trailing edge along the chord."""
        return int(np.argmin(self.x))

    def sides(self) -> tuple[np.ndarray, np.ndarray]:
        """Split the loop into ``(upper, lower)``, both running leading edge -> trailing edge.

        The leading-edge point belongs to both and the trailing edge closes each,
        so a symmetric section gives two arrays that mirror exactly.
        """
        i = self.leading_edge_index
        upper = self.points[i::-1]  # index 0 is the trailing edge already
        lower = self.points[i:]
        if self.is_sharp:
            # a sharp section stores its single trailing-edge point once, at
            # index 0, so the lower side has to be closed back onto it
            lower = np.concatenate([lower, self.points[:1]])
        return upper, lower

    def thickness_and_camber(self, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Sample ``(x, thickness, camber)`` on a cosine distribution in ``x``.

        Both surfaces are interpolated onto a common :math:`x`, so this works for
        a section read from a file as well as a generated one.  The interpolation
        is cubic rather than linear because the leading edge turns through 180
        degrees over a few points, and a linear read of it understates the
        thickness by around 1% of the thickness itself.
        """
        upper, lower = self.sides()  # both monotone in x, so x parameterises them
        x = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n))) * self.chord + self.x.min()
        yu = _hermite(upper, upper[:, 0], np.clip(x, *upper[[0, -1], 0]))[:, 1]
        yl = _hermite(lower, lower[:, 0], np.clip(x, *lower[[0, -1], 0]))[:, 1]
        return x, yu - yl, 0.5 * (yu + yl)

    @property
    def max_thickness(self) -> float:
        return float(self.thickness_and_camber()[1].max())

    @property
    def max_camber(self) -> float:
        camber = self.thickness_and_camber()[2]
        return float(camber[np.argmax(np.abs(camber))])

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_points} points, chord {self.chord:.6f}, "
            f"max thickness {self.max_thickness:.5f}, max camber {self.max_camber:+.5f}, "
            f"trailing edge {'sharp' if self.is_sharp else f'open by {self.trailing_edge_gap:.2e}'}"
        )


# --------------------------------------------------------------------------
# Orientation and normalisation
# --------------------------------------------------------------------------
def signed_area(points: np.ndarray) -> float:
    """Shoelace area of a closed loop: positive counter-clockwise."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def surface_normals(points: np.ndarray, *, closed: bool = True) -> np.ndarray:
    r"""Outward unit normal at every point of a path traversed counter-clockwise.

    The normal at a point is the normalised sum of the two adjacent edge
    normals, which is the angle bisector -- the right thing at a smooth point and
    also at the trailing-edge corner, where it points aft along the bisector of
    the two surfaces rather than along either of them.

    ``closed=False`` treats the points as an **open** polyline instead of a loop,
    so the two ends take their single adjacent edge's normal rather than wrapping
    around. That is what the C-grid's inner path needs: it runs from downstream,
    up the wake cut, around the section and back down the cut, and joining its
    ends would put a normal where the outflow boundary is.
    """
    edge = np.diff(points, axis=0)  # edge i joins point i to point i + 1
    if closed:
        edge = np.vstack([edge, points[0] - points[-1]])
    length = np.hypot(edge[:, 0], edge[:, 1])
    if np.any(length <= 0.0):
        raise ValueError("the path has a repeated point, so an edge has zero length")
    # outward normal of edge i, for counter-clockwise traversal
    n_edge = np.stack([edge[:, 1] / length, -edge[:, 0] / length], axis=-1)
    if closed:
        n = n_edge + np.roll(n_edge, 1, axis=0)  # bisector at point i: edges i-1 and i
    else:
        n = np.vstack([n_edge[:1], n_edge[1:] + n_edge[:-1], n_edge[-1:]])
    norm = np.hypot(n[:, 0], n[:, 1])
    if np.any(norm < 1e-12):
        raise ValueError("the path doubles back on itself; cannot define a normal")
    return n / norm[:, None]


def normalise(points: np.ndarray, *, name: str = "airfoil") -> Airfoil:
    """Put raw coordinates into the stored convention.

    Scales to unit chord with the leading edge at the origin, drops a repeated
    closing point, orients counter-clockwise, and rotates the list so index 0 is
    the trailing edge.
    """
    pts = np.ascontiguousarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {pts.shape}")
    if np.hypot(*(pts[0] - pts[-1])) < 1e-12:
        pts = pts[:-1]  # the loop is implicit; a repeated point breaks the normals

    chord = pts[:, 0].max() - pts[:, 0].min()
    if chord <= 0.0:
        raise ValueError("the coordinates have zero chord")
    pts = pts.copy()
    pts[:, 0] -= pts[:, 0].min()
    pts /= chord

    if signed_area(pts) < 0.0:
        pts = pts[::-1]
    pts = np.roll(pts, -int(np.argmax(pts[:, 0])), axis=0)
    return Airfoil(name=name, points=np.ascontiguousarray(pts))


# --------------------------------------------------------------------------
# Point distributions
# --------------------------------------------------------------------------
def side_distribution(n_intervals: int, clustering: str = "cosine") -> np.ndarray:
    r"""``n_intervals + 1`` parameters on :math:`[0, 1]`, clustered at both ends.

    ``'cosine'`` is :math:`\tfrac12(1 - \cos \pi k/n)`, which packs points where
    the curvature is -- the leading edge, and the trailing edge where the
    pressure recovery is steepest.
    """
    k = np.arange(n_intervals + 1) / n_intervals
    if clustering == "uniform":
        return k
    if clustering == "cosine":
        return 0.5 * (1.0 - np.cos(math.pi * k))
    raise ValueError(f"unknown clustering {clustering!r}; use one of {CLUSTERINGS}")


def _arclength(points: np.ndarray) -> np.ndarray:
    """Cumulative normalised arc length along an open polyline, from 0 to 1."""
    step = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(step)])
    return s / s[-1]


def _hermite(points: np.ndarray, s_source: np.ndarray, s_target: np.ndarray) -> np.ndarray:
    """Cubic-Hermite interpolation of a polyline, with finite-difference tangents.

    Linear interpolation would be enough almost everywhere and badly wrong at the
    leading edge, where the surface turns through 180 degrees over a few points.
    Tangents come from a centred difference in the arc-length parameter, which
    makes the result a Catmull-Rom spline: it passes through every source point
    and stays local, so one bad point in a file cannot ring across the section.
    """
    n = len(s_source)
    tangent = np.empty_like(points)
    tangent[1:-1] = (points[2:] - points[:-2]) / (s_source[2:] - s_source[:-2])[:, None]
    tangent[0] = (points[1] - points[0]) / (s_source[1] - s_source[0])
    tangent[-1] = (points[-1] - points[-2]) / (s_source[-1] - s_source[-2])

    j = np.clip(np.searchsorted(s_source, s_target, side="right") - 1, 0, n - 2)
    h = s_source[j + 1] - s_source[j]
    u = ((s_target - s_source[j]) / h)[:, None]
    h = h[:, None]
    u2, u3 = u * u, u * u * u
    return (
        (2.0 * u3 - 3.0 * u2 + 1.0) * points[j]
        + (u3 - 2.0 * u2 + u) * h * tangent[j]
        + (-2.0 * u3 + 3.0 * u2) * points[j + 1]
        + (u3 - u2) * h * tangent[j + 1]
    )


def _close_loop(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Join two sides, each running leading edge -> trailing edge, into one loop.

    The result runs trailing edge -> upper -> leading edge -> lower, with the
    shared leading edge stored once.  A sharp section also shares its
    trailing-edge point, so that duplicate goes too; an open one keeps both, and
    the closing segment between them is the blunt base.
    """
    loop = np.concatenate([upper[::-1], lower[1:]])
    if np.hypot(*(loop[0] - loop[-1])) < 1e-12:
        loop = loop[:-1]
    return loop


def resample(airfoil: Airfoil, n_points: int, clustering: str = "cosine") -> Airfoil:
    """Redistribute a surface onto ``n_points``, clustered per ``clustering``.

    Each side is resampled independently in its own arc length, so the leading
    and trailing edge land exactly on the original points and a symmetric section
    stays symmetric.
    """
    if n_points % 2 or n_points < 16:
        raise ValueError(f"n_points must be even and at least 16, got {n_points}")
    per_side = n_points // 2
    target = side_distribution(per_side, clustering)

    upper, lower = airfoil.sides()  # both run leading edge -> trailing edge
    new_upper = _hermite(upper, _arclength(upper), target)
    new_lower = _hermite(lower, _arclength(lower), target)
    return normalise(_close_loop(new_upper, new_lower), name=airfoil.name)


# --------------------------------------------------------------------------
# The NACA four-digit family
# --------------------------------------------------------------------------
def _check_n_points(n_points: int) -> int:
    if n_points % 2 or n_points < 16:
        raise ValueError(f"n_points must be even and at least 16, got {n_points}")
    return n_points


def _thickness(x: np.ndarray, t: float, closed: bool) -> np.ndarray:
    """The four-digit thickness quartic, shared by both NACA families."""
    a4 = _A4[bool(closed)]
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - a4 * x**4)


def _offset_from_camber(
    x: np.ndarray, yt: np.ndarray, yc: np.ndarray, theta: np.ndarray, *, name: str
) -> Airfoil:
    r"""Offset the thickness **normal to the camber line**, not vertically.

    The two agree to :math:`O(\theta^2)` and differ visibly near the leading
    edge of a cambered section -- which is exactly where the suction peak is.
    """
    upper = np.stack([x - yt * np.sin(theta), yc + yt * np.cos(theta)], axis=-1)
    lower = np.stack([x + yt * np.sin(theta), yc - yt * np.cos(theta)], axis=-1)
    return normalise(_close_loop(upper, lower), name=name)


def naca4(
    code: str | int = "0012",
    n_points: int = 192,
    *,
    clustering: str = "cosine",
    closed_trailing_edge: bool = True,
) -> Airfoil:
    r"""Generate a NACA four-digit section.

    Parameters
    ----------
    code
        Four digits, with or without a leading ``NACA``: ``'0012'``, ``'naca2412'``.
    n_points
        Points around the closed surface; must be even, so a symmetric section
        gets a point exactly at the leading edge.  With
        ``closed_trailing_edge=False`` the section carries a *second* point at
        the trailing edge -- the other corner of the blunt base -- and so comes
        back with ``n_points + 1``.
    closed_trailing_edge
        Use :math:`a_4 = 0.1036`, closing the trailing edge exactly.  See the
        module docstring for why this is the default.
    """
    digits = re.fullmatch(r"(?:naca[\s-]*)?(\d{4})", str(code).strip(), re.IGNORECASE)
    if digits is None:
        raise ValueError(f"not a NACA four-digit code: {code!r}")
    text = digits.group(1)
    m, p, t = int(text[0]) / 100.0, int(text[1]) / 10.0, int(text[2:]) / 100.0
    if t <= 0.0:
        raise ValueError(f"NACA {text} has zero thickness")
    if m > 0.0 and not 0.0 < p < 1.0:
        raise ValueError(f"NACA {text} is cambered but its camber position is {p}")

    x = side_distribution(_check_n_points(n_points) // 2, clustering)
    yt = _thickness(x, t, closed_trailing_edge)

    if m == 0.0:
        yc = np.zeros_like(x)
        theta = np.zeros_like(x)
    else:
        fore = x <= p
        yc = np.where(
            fore,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2 * (1.0 - 2.0 * p + 2.0 * p * x - x**2),
        )
        # the surface is offset normal to the camber line, not vertically
        dyc = np.where(
            fore,
            2.0 * m / p**2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )
        theta = np.arctan(dyc)

    return _offset_from_camber(x, yt, yc, theta, name=f"NACA {text}")


#: NACA five-digit camber lines, tabulated by the second digit P.  Both tables
#: are stated for a design lift coefficient of 0.3 (the 230-series); :func:`naca5`
#: rescales ``k1`` for any other first digit.
_NACA5_PLAIN = {  # P: (m, k1)
    1: (0.0580, 361.400),
    2: (0.1260, 51.640),
    3: (0.2025, 15.957),
    4: (0.2900, 6.643),
    5: (0.3910, 3.230),
}
_NACA5_REFLEXED = {  # P: (m, k1, k2/k1)
    2: (0.1300, 51.990, 0.000764),
    3: (0.2170, 15.793, 0.006770),
    4: (0.3180, 6.520, 0.030300),
    5: (0.4410, 3.191, 0.135500),
}


def naca5(
    code: str | int = "23012",
    n_points: int = 192,
    *,
    clustering: str = "cosine",
    closed_trailing_edge: bool = True,
) -> Airfoil:
    r"""Generate a NACA five-digit section.

    The digits are ``LPSTT``: :math:`L` sets the design lift coefficient as
    :math:`c_{\ell,\text{design}} = 0.15 L`, :math:`P` puts the camber crest at
    :math:`x/c = P/20`, :math:`S` selects a plain (0) or **reflexed** (1) camber
    line, and :math:`TT` is the thickness in per cent.  NACA 23012 is the
    familiar one: 0.3 design lift, crest at 15% chord, 12% thick.

    The thickness distribution is the same quartic as the four-digit family --
    :func:`naca4` and this share it.  What differs is the camber line, which is
    a **cubic forward of the crest and straight behind it** rather than two
    parabolas.  That is the whole point of the family: it puts the camber well
    forward, which moves the aerodynamic centre far less with lift.

    .. math::
        y_c = \begin{cases}
          \dfrac{k_1}{6}\left(x^3 - 3mx^2 + m^2(3-m)x\right), & x \le m,\\[3mm]
          \dfrac{k_1 m^3}{6}\left(1 - x\right), & x > m,
        \end{cases}

    and for a reflexed line (``S = 1``), which carries a slight upward turn at
    the tail so the section's pitching moment is near zero,

    .. math::
        y_c = \frac{k_1}{6}\begin{cases}
          (x-m)^3 - \frac{k_2}{k_1}(1-m)^3 x - m^3 x + m^3, & x \le m,\\[2mm]
          \frac{k_2}{k_1}(x-m)^3 - \frac{k_2}{k_1}(1-m)^3 x - m^3 x + m^3, & x > m .
        \end{cases}

    :math:`m` and :math:`k_1` are **not** free parameters: they are tabulated
    against :math:`P` (Abbott & von Doenhoff, Table 6.1), chosen so the crest
    lands at :math:`P/20` and the design lift comes out right.  Only the
    tabulated positions exist, so a code naming any other is refused rather
    than interpolated -- an interpolated five-digit section is not a NACA
    section, and silently inventing one would be worse than saying no.
    """
    digits = re.fullmatch(r"(?:naca[\s-]*)?(\d{5})", str(code).strip(), re.IGNORECASE)
    if digits is None:
        raise ValueError(f"not a NACA five-digit code: {code!r}")
    text = digits.group(1)
    design_lift = 0.15 * int(text[0])
    position, reflexed, t = int(text[1]), int(text[2]), int(text[3:]) / 100.0
    if t <= 0.0:
        raise ValueError(f"NACA {text} has zero thickness")
    if reflexed not in (0, 1):
        raise ValueError(
            f"NACA {text}: the third digit selects a plain (0) or reflexed (1) camber "
            f"line; {reflexed} is neither"
        )
    table = _NACA5_REFLEXED if reflexed else _NACA5_PLAIN
    if position not in table:
        raise ValueError(
            f"NACA {text}: camber position {position} is not tabulated for a "
            f"{'reflexed' if reflexed else 'plain'} five-digit camber line.  "
            f"Available: {sorted(table)}, putting the crest at "
            f"{sorted(q / 20 for q in table)} chord."
        )

    x = side_distribution(_check_n_points(n_points) // 2, clustering)
    yt = _thickness(x, t, closed_trailing_edge)

    scale = design_lift / 0.3  # the tables are stated for the 230-series
    fore = x <= table[position][0]
    if reflexed:
        m, k1, ratio = table[position]
        k1 *= scale
        tail = ratio * (1.0 - m) ** 3 + m**3
        yc = (k1 / 6.0) * np.where(
            fore, (x - m) ** 3 - tail * x + m**3, ratio * (x - m) ** 3 - tail * x + m**3
        )
        dyc = (k1 / 6.0) * np.where(
            fore, 3.0 * (x - m) ** 2 - tail, 3.0 * ratio * (x - m) ** 2 - tail
        )
    else:
        m, k1 = table[position]
        k1 *= scale
        yc = (k1 / 6.0) * np.where(
            fore, x**3 - 3.0 * m * x**2 + m * m * (3.0 - m) * x, m**3 * (1.0 - x)
        )
        dyc = (k1 / 6.0) * np.where(
            fore, 3.0 * x**2 - 6.0 * m * x + m * m * (3.0 - m), -(m**3) * np.ones_like(x)
        )

    return _offset_from_camber(x, yt, yc, np.arctan(dyc), name=f"NACA {text}")


def naca(
    code: str | int,
    n_points: int = 192,
    *,
    clustering: str = "cosine",
    closed_trailing_edge: bool = True,
) -> Airfoil:
    """Generate a NACA section, four-digit or five-digit, chosen by the code."""
    text = re.sub(r"^naca[\s-]*", "", str(code).strip(), count=1, flags=re.IGNORECASE)
    builder = {4: naca4, 5: naca5}.get(len(text)) if text.isdigit() else None
    if builder is None:
        raise ValueError(
            f"not a NACA four- or five-digit code: {code!r}.  The six-series and the "
            f"modified four-digit sections have no closed form, so read those from a "
            f"coordinate file instead."
        )
    return builder(text, n_points, clustering=clustering, closed_trailing_edge=closed_trailing_edge)


# --------------------------------------------------------------------------
# Coordinates from a file
# --------------------------------------------------------------------------
def _float_pairs(lines: list[str]) -> list[tuple[int, np.ndarray]]:
    """Every line that is exactly two numbers, with its index."""
    out = []
    for i, line in enumerate(lines):
        field = line.replace(",", " ").split()
        if len(field) != 2:
            continue
        try:
            out.append((i, np.array([float(field[0]), float(field[1])])))
        except ValueError:
            continue
    return out


def read_dat(path: str | Path, *, name: str | None = None) -> Airfoil:
    """Read an airfoil coordinate file in Selig or Lednicer format.

    Both are plain text, both start with a name line, and they differ in what
    follows -- which is detectable rather than something you have to be told:

    **Selig** (what XFOIL's ``SAVE`` writes) is one loop, trailing edge over the
    upper surface to the leading edge and back along the lower surface.

    **Lednicer** opens with the two surface point counts, then gives the upper
    surface from the leading edge and the lower surface from the leading edge,
    usually separated by a blank line.  The counts are what identifies it: a pair
    of numbers both greater than one cannot be an airfoil coordinate.

    The result is put into this module's convention regardless of which was
    given, so a file in either format, wound either way, round-trips.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    pairs = _float_pairs(lines)
    if len(pairs) < 16:
        raise ValueError(f"{path}: found only {len(pairs)} coordinate lines")

    label = name
    if label is None:
        head = lines[0].strip() if lines and pairs[0][0] > 0 else ""
        label = head or path.stem

    first = pairs[0][1]
    is_lednicer = bool(first[0] > 1.0 and first[1] > 1.0)
    if is_lednicer:
        n_upper, n_lower = (int(round(v)) for v in first)
        body = [p for _, p in pairs[1:]]
        if n_upper + n_lower > len(body):
            raise ValueError(
                f"{path}: header asks for {n_upper}+{n_lower} points but only {len(body)} follow"
            )
        upper = np.array(body[:n_upper])
        lower = np.array(body[n_upper : n_upper + n_lower])
        # both run leading edge -> trailing edge; reverse the upper to get one loop
        loop = np.concatenate([upper[::-1], lower[1:]])
    else:
        loop = np.array([p for _, p in pairs])

    return normalise(loop, name=label)


def write_dat(airfoil: Airfoil, path: str | Path) -> Path:
    """Write a section in Selig format, which is what XFOIL reads back."""
    path = Path(path)
    rows = np.concatenate([airfoil.points, airfoil.points[:1]])
    body = "\n".join(f"{x:12.8f} {y:12.8f}" for x, y in rows)
    path.write_text(f"{airfoil.name}\n{body}\n")
    return path


def load_airfoil(
    spec: str | int | Path | Airfoil,
    n_points: int = 192,
    *,
    clustering: str = "cosine",
) -> Airfoil:
    """Resolve whatever the user named into an :class:`Airfoil` of ``n_points``.

    Accepts an :class:`Airfoil`, a path to a coordinate file, or a NACA
    four-digit code.  A generated section is produced directly at the requested
    distribution; one read from a file is resampled onto it.

    >>> load_airfoil("0012", 64).summary()        # doctest: +ELLIPSIS
    'NACA 0012: 64 points, ...'
    """
    if isinstance(spec, Airfoil):
        return spec if spec.n_points == n_points else resample(spec, n_points, clustering)
    text = str(spec).strip()
    if re.fullmatch(r"(?:naca[\s-]*)?\d{4,5}", text, re.IGNORECASE):
        return naca(text, n_points, clustering=clustering)
    candidate = Path(text)
    if candidate.exists():
        return resample(read_dat(candidate), n_points, clustering)
    raise ValueError(
        f"cannot resolve airfoil {spec!r}: it is neither a NACA four-digit code "
        f"nor an existing coordinate file"
    )
