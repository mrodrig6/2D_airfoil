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


__all__ = ["FIELD_CMAP", "field_cmap", "plot_airfoil", "plot_mesh"]
