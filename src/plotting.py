r"""Matplotlib figures: the airfoil, and the mesh.

Matplotlib is an **optional** dependency.  Importing this module without it
raises with the install command rather than failing somewhere deep inside a
draw call, and nothing in the solver imports it.

Style, decided once
-------------------
Every figure in this project follows three rules, so a set of them reads as one
document rather than as a pile of defaults:

* **Recessive axes.** The data is the only thing with full contrast; the frame,
  the ticks and any grid are light grey and thin.  A mesh plot drops the frame
  entirely -- the mesh *is* the picture, and a box around it adds nothing.
* **Thin marks, one series, no legend box.**  A single-series figure is named by
  its title; a legend repeating that name is noise.
* **One light figure, not a theme pair.**  These are written to PNG for a report
  or a slide, where a dark variant has nowhere to go.

Colour maps for a scalar field (which arrive with the solver)
-------------------------------------------------------------
:data:`FIELD_CMAP` is the project's sequential map, and the choice is deliberate:
``legacy/plotsolution.m`` drew the Mach field with ``jet``, and a rainbow map is
the classic way to put structure in a picture that is not in the data.  Its
lightness is non-monotonic, so it invents visual edges at the yellow and cyan
bands -- which on a transonic field is indistinguishable from the shock the
figure exists to show -- and it is unreadable in greyscale and to a colour-blind
reader.  ``magma`` rises monotonically in lightness, so a brighter pixel is a
larger value everywhere, and it survives both.

Any map used here must be monotone in lightness.  That rules out ``jet``,
``rainbow``, ``hsv`` and ``nipy_spectral``, and :func:`field_cmap` refuses them
by name rather than leaving it to reviewer memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Airfoil, surface_normals
from .mesh import BoundaryTag, MeshTopology

#: Sequential map for a scalar field.  See the module docstring.
FIELD_CMAP = "magma"

#: Maps whose lightness is not monotonic, so they manufacture edges.
_BANNED_CMAPS = frozenset(
    {"jet", "rainbow", "gist_rainbow", "hsv", "nipy_spectral", "gist_ncar", "prism", "flag"}
)

_INK = "#1a1a1a"
_MUTED = "#9aa0a6"
_WALL = "#1f2933"
_MESH = "#5b6b7a"
_ACCENT = "#eb6834"

#: Axis and colour-bar labels for the fields :func:`~src.postprocess.scalar_field`
#: can produce.  A colour bar labelled ``mach`` is a variable name leaking into a
#: figure; these are what the quantity is called.
SCALAR_LABELS = {
    "mach": "Mach number",
    "pressure": r"pressure  $p / \rho_\infty v_\infty^2$",
    "density": r"density  $\rho / \rho_\infty$",
    "velocity": r"speed  $|v| / v_\infty$",
    "cp": "$c_p$",
    "entropy": r"entropy  $\Delta s / c_v$",
}

#: Categorical series colours, assigned in this **fixed order** and never
#: cycled: a colour follows the entity, not its rank, so filtering a flux out
#: of a comparison must not repaint the others.
#:
#: These are not hand-picked.  A first attempt was, and it failed two checks --
#: three of the four read as grey, and the green and the orange were 2.2 Delta E
#: apart for a protanope, which is indistinguishable.  These are the validated
#: slots; the worst adjacent pair is 9.2 under deuteranopia and 27.6 for normal
#: vision.  The green is below 3:1 against white, which is why every multi-series
#: figure here carries a legend rather than relying on the colour alone.
#:
#: A fifth series folds into "other" or becomes a small multiple.  It does not
#: get a generated hue.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise ImportError(
            "plotting needs matplotlib, which is optional: pip install matplotlib "
            "(or ./fv2d.sh install)"
        ) from exc
    return plt


def field_cmap(name: str | None = None) -> str:
    """Validate and return a sequential colour map name.

    Rejects the non-monotonic maps by name: a rainbow invents edges that are not
    in the field, which is exactly the failure mode a transonic figure cannot
    afford.
    """
    name = name or FIELD_CMAP
    if name.removesuffix("_r") in _BANNED_CMAPS:
        raise ValueError(
            f"{name!r} is not monotonic in lightness, so it manufactures edges that "
            f"are not in the data -- on a transonic field that is indistinguishable "
            f"from the shock.  Use {FIELD_CMAP!r}, 'viridis' or 'cividis'."
        )
    return name


def _operating_point(flow, lead: str) -> str:
    r"""``lead`` followed by the operating point, which every result figure needs.

    A figure that does not say its own Mach number and incidence is useless the
    moment it leaves the terminal it was made in -- which is the only place
    these are ever looked at.
    """
    return f"{lead}  \u2014  $M_\\infty = {flow.mach:g}$, $\\alpha = {flow.alpha_deg:g}^\\circ$"


def _style(ax, *, frame: bool = True) -> None:
    """Push the frame and ticks back so only the data carries contrast."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(frame)
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=9, width=0.8, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_INK)


def _finish(fig, path: str | Path | None, dpi: int = 160):
    if path is None:
        return fig
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    return path


def plot_airfoil(
    airfoil: Airfoil,
    path: str | Path | None = None,
    *,
    normals: bool = False,
    title: str | None = None,
    **kwargs: Any,
):
    """Draw a section, optionally with its outward normals.

    The normals are the diagnostic worth having: if a coordinate file was read
    in a winding this code did not expect, they point *into* the section, and
    that is visible at a glance where a list of numbers is not.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 2.6), **kwargs)
    loop = np.vstack([airfoil.points, airfoil.points[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color=_WALL, linewidth=1.4, solid_joinstyle="round")
    ax.plot(airfoil.x, airfoil.y, ".", color=_WALL, markersize=2.0, alpha=0.55)
    if normals:
        n = surface_normals(airfoil.points)
        scale = 0.04 * airfoil.chord
        ax.quiver(
            airfoil.x,
            airfoil.y,
            n[:, 0],
            n[:, 1],
            color=_MUTED,
            width=0.0022,
            scale=1.0 / scale,
            scale_units="xy",
            alpha=0.9,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title(title or airfoil.name, color=_INK, fontsize=11, loc="left")
    _style(ax)
    fig.tight_layout()
    return _finish(fig, path)


def plot_mesh(
    mesh: MeshTopology,
    path: str | Path | None = None,
    *,
    window: tuple[float, float, float, float] | None = (-0.6, 1.6, -0.7, 0.7),
    title: str | None = None,
    **kwargs: Any,
):
    """Draw the mesh as a wireframe, with the wall picked out.

    ``window`` is ``(xmin, xmax, ymin, ymax)``; the default frames the airfoil
    rather than the far field, because at the far-field scale the section is two
    pixels wide.  Pass ``window=None`` for the whole domain.
    """
    plt = _pyplot()
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(7.2, 5.0), **kwargs)
    p = mesh.node_coords

    pairs = np.concatenate([mesh.interior_edges[:, 2:4], mesh.boundary_edges[:, 1:3]])
    if window is not None:
        x0, x1, y0, y1 = window
        inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        pairs = pairs[inside[pairs].any(axis=1)]
    ax.add_collection(LineCollection(p[pairs], colors=_MESH, linewidths=0.35, alpha=0.75))

    wall = mesh.boundary_edges[mesh.tag_slice(BoundaryTag.WALL), 1:3]
    ax.add_collection(LineCollection(p[wall], colors=_WALL, linewidths=1.5))

    if window is None:
        ax.autoscale_view()
        span = np.abs(p).max()
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
    else:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title(title or mesh.summary(), color=_INK, fontsize=10, loc="left")
    _style(ax, frame=False)
    fig.tight_layout()
    # a wireframe of tens of thousands of segments is all edges, so it costs far
    # more per pixel than a filled plot; 100 dpi keeps a committed figure small
    # without the lines turning to mush
    return _finish(fig, path, dpi=100)


def plot_surface_pressure(
    result,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    **kwargs: Any,
):
    r"""The surface pressure distribution, drawn as :math:`-c_p` so suction is up.

    Two series, so a legend is required -- identity must never be carried by
    colour alone.  The sonic line :math:`c_p^*` is drawn because on a transonic
    case it is the single most informative thing on the plot: everything above
    it is supersonic, and where the upper-surface curve crosses back through it
    is the shock.
    """
    plt = _pyplot()
    from .postprocess import surface_pressure

    trace = surface_pressure(result)
    flow = result.flow
    fig, ax = plt.subplots(figsize=(7.0, 4.6), **kwargs)

    upper = trace["upper"]
    for mask, label, colour in ((upper, "upper", _WALL), (~upper, "lower", _ACCENT)):
        order = np.argsort(trace["x"][mask])
        ax.plot(
            trace["x"][mask][order],
            -trace["cp"][mask][order],
            color=colour,
            linewidth=2.0,
            label=label,
            solid_joinstyle="round",
        )

    g = flow.gamma
    cp_star = (2.0 / (g * flow.mach**2)) * (
        ((2.0 + (g - 1.0) * flow.mach**2) / (g + 1.0)) ** (g / (g - 1.0)) - 1.0
    )
    ax.axhline(-cp_star, color=_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        "sonic",
        xy=(0.985, -cp_star),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=_MUTED,
        fontsize=9,
    )

    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$-c_p$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        title or _operating_point(result.flow, f"{result.mesh.n_elements} cells"),
        color=_INK,
        fontsize=11,
        loc="left",
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(_INK)
    _style(ax)
    fig.tight_layout()
    return _finish(fig, path)


def plot_field(
    result,
    path: str | Path | None = None,
    *,
    scalar: str = "mach",
    window: tuple[float, float, float, float] = (-0.6, 1.8, -0.8, 0.8),
    levels: int = 24,
    cmap: str | None = None,
    title: str | None = None,
    **kwargs: Any,
):
    """A cell-centred scalar field over the mesh, with the section drawn on top.

    Uses :data:`FIELD_CMAP` through :func:`field_cmap`, which refuses the
    non-monotonic maps -- see the module docstring for why that matters more on
    a transonic field than on most.
    """
    plt = _pyplot()
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation

    from .postprocess import scalar_field

    values = scalar_field(result.state, result.flow, scalar)
    mesh = result.mesh
    tri = Triangulation(mesh.node_coords[:, 0], mesh.node_coords[:, 1], mesh.elements)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), **kwargs)
    art = ax.tripcolor(
        tri, facecolors=values, cmap=field_cmap(cmap), shading="flat", rasterized=True
    )
    if scalar == "mach":
        # the sonic line is the one contour worth naming on a transonic field
        ax.tricontour(
            tri,
            _node_average(mesh, values),
            levels=[1.0],
            colors="white",
            linewidths=1.2,
        )
    wall = mesh.boundary_edges[mesh.tag_slice(BoundaryTag.WALL), 1:3]
    ax.add_collection(
        LineCollection(mesh.node_coords[wall], colors=_WALL, linewidths=1.6, zorder=3)
    )

    bar = fig.colorbar(art, ax=ax, fraction=0.035, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=_MUTED, labelsize=9, width=0.8, length=3)
    bar.set_label(SCALAR_LABELS.get(scalar, scalar), color=_INK, fontsize=10)

    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal")
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title(
        title or _operating_point(result.flow, SCALAR_LABELS.get(scalar, scalar)),
        color=_INK,
        fontsize=11,
        loc="left",
    )
    _style(ax, frame=False)
    fig.tight_layout()
    return _finish(fig, path)


def _node_average(mesh, values):
    """Cell values averaged onto nodes, for a contour line."""
    total = np.bincount(mesh.elements.ravel(), np.repeat(values, 3), minlength=mesh.n_nodes)
    count = np.bincount(mesh.elements.ravel(), minlength=mesh.n_nodes)
    return total / np.maximum(count, 1)


def plot_convergence(
    results,
    path: str | Path | None = None,
    *,
    labels: list[str] | None = None,
    title: str | None = None,
    **kwargs: Any,
):
    """Residual histories, one line each, on a log axis.

    Takes one result or several, which is what the flux comparison needs: the
    whole question there is whether a scheme's residual decays or stalls, and
    that is only legible with the histories on one pair of axes.
    """
    plt = _pyplot()

    if not isinstance(results, (list, tuple)):
        results = [results]
    fig, ax = plt.subplots(figsize=(7.0, 4.4), **kwargs)
    for i, result in enumerate(results):
        name = labels[i] if labels else getattr(result.flow, "flux", f"run {i + 1}")
        ax.semilogy(
            np.arange(1, result.history.size + 1),
            result.history,
            color=_SERIES[i % len(_SERIES)],
            linewidth=1.8,
            label=name,
        )
    ax.axhline(results[0].options.tolerance, color=_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlabel("iteration")
    ax.set_ylabel("scaled residual")
    ax.set_title(title or "convergence", color=_INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color=_MUTED, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    if len(results) > 1:
        legend = ax.legend(frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(_INK)
    _style(ax)
    fig.tight_layout()
    return _finish(fig, path)


def plot_polar(
    angles,
    loads,
    path: str | Path | None = None,
    *,
    title: str | None = None,
    **kwargs: Any,
):
    r"""The three load curves against incidence, as small multiples.

    Three panels rather than three lines on one axis, because $c_\ell$, $c_d$ and
    $c_m$ differ by an order of magnitude and a shared axis flattens two of them
    into the baseline.  A twin axis would fix the scale and break the reading --
    which curve belongs to which axis is then a legend lookup on every glance.

    Parameters
    ----------
    angles
        Incidences in degrees.
    loads
        One :class:`~src.postprocess.Forces` per angle, in the same order.
    """
    plt = _pyplot()

    alpha = np.asarray(angles, dtype=float)
    series = (
        ("$c_\\ell$", np.array([load.cl for load in loads]), _SERIES[0]),
        ("$c_d$", np.array([load.cd for load in loads]), _SERIES[1]),
        ("$c_m$ about $c/4$", np.array([load.cm for load in loads]), _SERIES[2]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), **kwargs)
    for ax, (label, value, colour) in zip(axes, series, strict=True):
        ax.plot(alpha, value, color=colour, linewidth=1.8, marker="o", markersize=3.5)
        ax.axhline(0.0, color=_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_xlabel(r"$\alpha$  [deg]")
        ax.set_title(label, color=_INK, fontsize=11, loc="left")
        ax.grid(True, color=_MUTED, alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    if title:
        fig.suptitle(title, color=_INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, path)


def plot_adaptation(history, path: str | Path | None = None, *, title: str | None = None, **kwargs):
    """Lift and drag against cell count over an adaptive run.

    The cell count is the x axis, on a log scale, because the question an
    adaptive run answers is *what did the cells buy* -- and the honest comparison
    against a uniformly refined mesh is a comparison at equal cell count, not at
    equal cycle number.
    """
    plt = _pyplot()

    cells = np.array([c.n_elements for c in history.cycles], dtype=float)
    panels = (
        ("$c_\\ell$", np.array([c.loads.cl for c in history.cycles]), _SERIES[0]),
        ("$c_d$", np.array([c.loads.cd for c in history.cycles]), _SERIES[1]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), **kwargs)
    for ax, (label, value, colour) in zip(axes, panels, strict=True):
        ax.semilogx(cells, value, color=colour, linewidth=1.8, marker="o", markersize=4)
        ax.set_xlabel("cells")
        ax.set_title(label, color=_INK, fontsize=11, loc="left")
        ax.grid(True, which="both", color=_MUTED, alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    if title:
        fig.suptitle(title, color=_INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return _finish(fig, path)


__all__ = [
    "FIELD_CMAP",
    "SCALAR_LABELS",
    "field_cmap",
    "plot_adaptation",
    "plot_airfoil",
    "plot_convergence",
    "plot_field",
    "plot_mesh",
    "plot_polar",
    "plot_surface_pressure",
]
