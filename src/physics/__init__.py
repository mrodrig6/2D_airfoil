r"""The Euler equations: thermodynamics, the Riemann solvers, the boundaries.

Everything here is a pure function of states and normals.  Nothing knows about
the mesh, the march, or which backend is running, and every function takes an
``xp`` array module so the same source runs under NumPy and under
``jax.numpy``.  That is what makes the JAX backend the *same code* rather than
a second implementation to keep in step.

================================  ===========================================
:mod:`~src.physics.thermo`        the state vector: primitive and conservative
                                  variables, the ideal-gas relations, the
                                  Euler flux itself
:mod:`~src.physics.fluxes`        the ``roe``, ``hllc`` and ``ausmup``
                                  approximate Riemann solvers, and the
                                  dispatch between them
:mod:`~src.physics.boundary`      the wall and far-field fluxes
================================  ===========================================

A finite-volume scheme needs one flux per face, shared by the two cells that
meet there.  Each solver in :mod:`~src.physics.fluxes` takes a left state, a
right state and the face normal, and returns that flux together with the
largest signal speed it saw -- which is what the local time step is built
from.  **The approximate Riemann solver is the upwinding:** there is no
separate upwind operator anywhere in this code.

``docs/riemann_fluxes.md`` records what each solver was measured to do, which
fourth solver was removed and why, and the retraction of an earlier verdict on
AUSM\ :sup:`+`-up that turned out to be about a bug rather than the scheme.

References
----------
1. P. L. Roe, *J. Comput. Phys.* **43**, 357-372, 1981.
2. A. Harten and J. M. Hyman, *J. Comput. Phys.* **50**, 235-269, 1983.
3. P. Batten et al., *SIAM J. Sci. Comput.* **18**(6), 1553-1570, 1997.
4. M.-S. Liou, *J. Comput. Phys.* **214**, 137-170, 2006.
"""

from .boundary import farfield_flux, farfield_state, wall_flux
from .fluxes import SOLVERS, ausmup_flux, hllc_flux, interface_flux, roe_flux
from .thermo import (
    FLOOR,
    FluxResult,
    conservative,
    euler_flux,
    normal_flux,
    primitives,
    sound_speed,
)

__all__ = [
    "FLOOR",
    "SOLVERS",
    "FluxResult",
    "ausmup_flux",
    "conservative",
    "euler_flux",
    "farfield_flux",
    "farfield_state",
    "hllc_flux",
    "interface_flux",
    "normal_flux",
    "primitives",
    "roe_flux",
    "sound_speed",
    "wall_flux",
]
