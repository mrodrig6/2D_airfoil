r"""Forces, surface pressure, and scalar fields.

The force integral, and the defect it corrects
----------------------------------------------
For an inviscid flow the only force on the section is the pressure integral
over its surface.  With :math:`\hat{n}` the outward normal of the *flow domain*
-- which at the wall points **into** the section -- the force per unit span is

.. math::
    \vec{F} = \oint_{\text{wall}} p\,\hat{n}\,ds .

The sign is worth deriving rather than remembering.  The stress in a fluid is
:math:`\sigma = -p\mathbf{I}`, so the traction it exerts on the body is
:math:`-p\,\hat{n}_{\text{body}}` with :math:`\hat{n}_{\text{body}}` pointing
out of the body into the flow.  This code's :math:`\hat{n}` points out of the
*flow domain*, which at the wall is the opposite vector, so the two signs
cancel.  A check that settles it in one line: on the lower surface of a lifting
section :math:`\hat{n}` points up and the pressure there is high, so the
contribution must be upward -- which :math:`+p\,\hat{n}` gives and
:math:`-p\,\hat{n}` does not.

and lift and drag are that vector resolved perpendicular and parallel to the
free stream, which is a **rotation through** :math:`\alpha`:

.. math::
    \begin{pmatrix} D \\ L \end{pmatrix}
    = \begin{pmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{pmatrix}
      \begin{pmatrix} F_x \\ F_y \end{pmatrix},
    \qquad c_\ell = \frac{L}{q_\infty c}, \quad c_d = \frac{D}{q_\infty c},

with :math:`q_\infty = \tfrac12` and :math:`c = 1` in this code's units.

``legacy/post_calc.m`` computed

.. code-block:: matlab

    D = -sum(p_b.*sign(n(:,1)).*s_edge)*sin(alpha);
    L =  sum(p_b.*sign(n(:,2)).*s_edge)*cos(alpha);

which is wrong in two independent ways.  It uses ``sign(n_x)`` and ``sign(n_y)``
where the formula needs :math:`n_x` and :math:`n_y`, so every panel contributes
its **full length** instead of its projected area and the sum is not
:math:`\oint p\,\hat n\,ds` at all; and it scales two separately accumulated
integrals by :math:`\sin\alpha` and :math:`\cos\alpha`, where the correct
operation is a rotation of the force vector -- the two agree only when one
component vanishes.  :func:`legacy_forces` reproduces it so the difference can
be measured rather than asserted.

Pressure coefficient
--------------------
:math:`c_p = (p - p_\infty)/q_\infty = 2(p - p_\infty)`, plotted as
:math:`-c_p` by convention so suction is up.  The legacy plotted
:math:`(p_\infty - p)/q_\infty` against an axis labelled :math:`-c_p`, which is
self-consistent but defines the quantity with the opposite sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import physics as ph
from ..config import FlowConditions
from ..geometry.mesh import BoundaryTag, MeshTopology
from ..timestepping.march import SolveResult

#: Scalars :func:`scalar_field` can compute.
SCALARS = ("mach", "pressure", "density", "velocity", "entropy", "cp")


@dataclass(frozen=True)
class Forces:
    """Integrated loads on the section, per unit span."""

    cl: float
    cd: float
    cm: float
    fx: float
    fy: float

    def summary(self) -> str:
        return (
            f"cl = {self.cl:+.6f}   cd = {self.cd:+.6f}   "
            f"cm(c/4) = {self.cm:+.6f}   L/D = {self.cl / self.cd:+.1f}"
            if abs(self.cd) > 1e-12
            else f"cl = {self.cl:+.6f}   cd = {self.cd:+.6f}   cm(c/4) = {self.cm:+.6f}"
        )


def wall_pressure(result: SolveResult):
    """``(pressure, midpoint, normal, length)`` on every wall face.

    The pressure is the one the wall condition itself uses -- the interior state
    with its normal velocity removed -- so the force integral and the flux the
    solver actually applied agree.  Taking the raw interior pressure instead
    would add a spurious force proportional to :math:`\\rho v_n^2`, which decays
    with the residual and so quietly contaminates a partly converged run.
    """
    metrics = result.metrics
    s = metrics.tag_slice(BoundaryTag.WALL)
    U = result.state[metrics.boundary_element[s]]
    nx = metrics.boundary_normal[s, 0]
    ny = metrics.boundary_normal[s, 1]

    rho = np.maximum(U[:, 0], ph.FLOOR)
    vx, vy = U[:, 1] / rho, U[:, 2] / rho
    vn = vx * nx + vy * ny
    qt2 = (vx - vn * nx) ** 2 + (vy - vn * ny) ** 2
    p = np.maximum((result.flow.gamma - 1.0) * (U[:, 3] - 0.5 * rho * qt2), ph.FLOOR)
    return p, metrics.boundary_midpoint[s], metrics.boundary_normal[s], metrics.boundary_length[s]


def forces(result: SolveResult, *, moment_centre: float = 0.25) -> Forces:
    """Lift, drag and pitching moment from the surface pressure integral."""
    p, mid, normal, length = wall_pressure(result)
    flow = result.flow

    # The stress in a fluid is sigma = -p I, so the traction the fluid exerts
    # on the body is -p times the body's *outward* normal.  Here n points out
    # of the *flow domain*, which at the wall is into the body -- the opposite
    # vector -- so the two minus signs cancel and the integrand is +p n.
    fx = float(np.sum(p * normal[:, 0] * length))
    fy = float(np.sum(p * normal[:, 1] * length))

    alpha = flow.alpha
    ca, sa = math.cos(alpha), math.sin(alpha)
    drag = fx * ca + fy * sa
    lift = -fx * sa + fy * ca

    arm_x = mid[:, 0] - moment_centre
    arm_y = mid[:, 1]
    # Nose-up positive, which is **minus** the z-component of r x F.  The two
    # sign questions here are separate and easy to conflate: the force one is
    # about which way n points, this one is about the axis convention.  With x
    # running aft and y up, a positive z-moment rotates the tail up, which is
    # nose *down*.  The check: a lift acting aft of the reference point must
    # give a negative cm, and only this sign does.
    moment = -float(np.sum(p * (arm_x * normal[:, 1] - arm_y * normal[:, 0]) * length))

    scale = flow.dynamic_pressure  # chord is 1
    return Forces(cl=lift / scale, cd=drag / scale, cm=moment / scale, fx=fx, fy=fy)


def legacy_forces(result: SolveResult) -> tuple[float, float]:
    """``legacy/post_calc.m``'s formula, reproduced so the error can be measured.

    See the module docstring for why it is wrong.  Kept because "the port
    changed the answer" is a claim that deserves a number.
    """
    p, _, normal, length = wall_pressure(result)
    alpha = result.flow.alpha
    drag = -float(np.sum(p * np.sign(normal[:, 0]) * length)) * math.sin(alpha)
    lift = float(np.sum(p * np.sign(normal[:, 1]) * length)) * math.cos(alpha)
    scale = result.flow.dynamic_pressure
    return lift / scale, drag / scale


def wall_order(mesh: MeshTopology) -> np.ndarray:
    """Indices into the wall faces, chained around the section from the trailing edge.

    The faces come out of the connectivity sort in no useful order, and a
    surface-pressure plot needs them joined up.  Each wall face is oriented by
    its element's counter-clockwise traversal, so the loop chains head to tail;
    walking it gives the order, which is then rotated to start at the trailing
    edge and flipped if needed so it runs over the upper surface first.
    """
    s = mesh.tag_slice(BoundaryTag.WALL)
    edges = mesh.boundary_edges[s, 1:]
    nxt = {int(a): i for i, (a, _) in enumerate(edges)}
    order = [int(np.argmax(mesh.node_coords[edges[:, 0], 0]))]
    for _ in range(len(edges) - 1):
        following = nxt.get(int(edges[order[-1], 1]))
        if following is None or following in order[:1]:
            break
        order.append(following)
    order = np.asarray(order)
    midpoint = 0.5 * (mesh.node_coords[edges[order, 0]] + mesh.node_coords[edges[order, 1]])
    if len(order) > 2 and midpoint[1, 1] < midpoint[-1, 1]:
        order = np.concatenate([order[:1], order[:0:-1]])  # keep the trailing edge first
    return order


def surface_pressure(result: SolveResult) -> dict[str, np.ndarray]:
    r"""The surface :math:`c_p` distribution, ordered along the section.

    Returns ``x``, ``y``, ``cp`` and ``upper``, the last a boolean mask so a
    plot can draw the two surfaces separately.
    """
    p, mid, _, _ = wall_pressure(result)
    order = wall_order(result.mesh)
    flow = result.flow
    cp = (p[order] - flow.pressure) / flow.dynamic_pressure
    x, y = mid[order, 0], mid[order, 1]
    # the chained order runs trailing edge -> upper -> leading edge -> lower
    leading = int(np.argmin(x))
    upper = np.zeros(x.size, dtype=bool)
    upper[: leading + 1] = True
    return {"x": x, "y": y, "cp": cp, "upper": upper}


def scalar_field(state: np.ndarray, flow: FlowConditions, name: str) -> np.ndarray:
    """A cell-centred scalar derived from the conservative state."""
    rho, vx, vy, p, _ = ph.primitives(state, flow.gamma)
    speed = np.hypot(vx, vy)
    if name == "density":
        return rho
    if name == "pressure":
        return p
    if name == "velocity":
        return speed
    if name == "mach":
        return speed / ph.sound_speed(rho, p, flow.gamma)
    if name == "cp":
        return (p - flow.pressure) / flow.dynamic_pressure
    if name == "entropy":
        # s - s_inf, with c_v = 1; zero everywhere in a shock-free flow
        return np.log(p / flow.pressure) - flow.gamma * np.log(rho / flow.density)
    raise ValueError(f"unknown scalar {name!r}; choose from {SCALARS}")


def entropy_error(result: SolveResult) -> float:
    r"""Area-weighted RMS of :math:`\Delta s`, which is zero for an exact solution.

    Upstream of the shock the flow is isentropic and a numerical solution is not,
    so this measures the scheme's own error where there is nothing else to
    compare against.  It is **not** zero for a transonic case -- a shock makes
    real entropy -- so read it as a resolution measure, not a correctness one.
    """
    s = scalar_field(result.state, result.flow, "entropy")
    area = result.metrics.area
    return float(np.sqrt(np.sum(area * s * s) / np.sum(area)))


def shock_location(result: SolveResult) -> float | None:
    """Chordwise position of the strongest upper-surface pressure rise, or ``None``.

    The crudest useful shock detector: the largest adverse jump in surface
    :math:`c_p` between neighbouring faces.  Enough to say whether two fluxes
    put the shock in the same place, which is what the flux study needs.
    """
    trace = surface_pressure(result)
    x, cp, upper = trace["x"], trace["cp"], trace["upper"]
    xs, cps = x[upper][::-1], cp[upper][::-1]  # leading edge -> trailing edge
    if xs.size < 3:
        return None
    jump = np.diff(cps)
    i = int(np.argmax(jump))
    return float(0.5 * (xs[i] + xs[i + 1])) if jump[i] > 0.1 else None


def report(result: SolveResult) -> str:
    """One block of text with everything worth printing after a solve."""
    load = forces(result)
    lines = [
        result.summary(),
        f"  {load.summary()}",
        f"  entropy error {entropy_error(result):.4e}",
    ]
    where = shock_location(result)
    lines.append(
        f"  upper-surface shock near x/c = {where:.3f}"
        if where is not None
        else "  no upper-surface shock detected"
    )
    mach = scalar_field(result.state, result.flow, "mach")
    lines.append(f"  Mach number {mach.min():.4f} to {mach.max():.4f}")
    return "\n".join(lines)


__all__ = [
    "SCALARS",
    "Forces",
    "entropy_error",
    "forces",
    "legacy_forces",
    "report",
    "scalar_field",
    "shock_location",
    "surface_pressure",
    "wall_order",
    "wall_pressure",
]
