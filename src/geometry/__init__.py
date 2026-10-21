"""What shape, and what mesh around it.

``section`` describes the airfoil; ``mesh`` wraps a grid around it; ``meshio``
reads one somebody else generated.  Nothing here knows about the flow.
"""

from .mesh import (
    BoundaryTag,
    CGrid,
    MeshTopology,
    OGrid,
    build_mesh,
    generate_cgrid,
    generate_ogrid,
    radial_distribution,
    triangulate,
)
from .meshio import read_mesh, write_su2
from .section import (
    CLUSTERINGS,
    Airfoil,
    load_airfoil,
    naca,
    naca4,
    naca5,
    naca_section,
    read_dat,
    resample,
    surface_normals,
    write_dat,
)

__all__ = [
    "Airfoil",
    "BoundaryTag",
    "CGrid",
    "CLUSTERINGS",
    "MeshTopology",
    "OGrid",
    "build_mesh",
    "generate_cgrid",
    "generate_ogrid",
    "load_airfoil",
    "naca",
    "naca4",
    "naca5",
    "naca_section",
    "radial_distribution",
    "read_dat",
    "read_mesh",
    "resample",
    "surface_normals",
    "triangulate",
    "write_dat",
    "write_su2",
]
