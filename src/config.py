r"""Configuration objects: the airfoil, the flow, and the mesh resolution.

Three frozen dataclasses hold everything a run needs, and nothing a run
computes.  They are frozen because a solved result keeps a reference to the
configuration that produced it, and a configuration that can be mutated
afterwards makes that reference a lie.

The non-dimensionalisation lives here
-------------------------------------
There is no SI anywhere in this code.  Every quantity is referred to the
free-stream state, exactly as ``legacy/initialstate.m`` did:

.. math::
    \rho_\infty = 1, \qquad
    |\vec{v}_\infty| = 1, \qquad
    R = \gamma - 1, \qquad
    c = 1 ,

with the chord :math:`c` as the length scale.  Two consequences are worth
stating because they are the ones students trip over:

* **Pressure is not 1.**  Fixing :math:`\rho_\infty` and :math:`|v_\infty|`
  fixes the free-stream Mach number through the pressure, so

  .. math::
      p_\infty = \frac{\rho_\infty}{\gamma}
                 \left(\frac{|v_\infty|}{M_\infty}\right)^2
               = \frac{1}{\gamma M_\infty^2} .

  At :math:`M_\infty = 0.8` that is :math:`p_\infty = 1.11607`, and the
  free-stream speed of sound is :math:`a_\infty = 1/M_\infty = 1.25`.

* **Dynamic pressure is exactly 1/2.**  :math:`q_\infty = \tfrac12 \rho_\infty
  |v_\infty|^2 = \tfrac12` for every operating point, so a force coefficient is
  just twice the force per unit span -- which is why :math:`c_\ell` and
  :math:`c_d` are comparable across Mach numbers without rescaling.

:math:`R = \gamma - 1` is the choice that makes :math:`c_v = 1`; it is the same
convention the companion nozzle code uses, and it means temperature and internal
energy per unit mass are numerically equal.  Nothing in the solver reads
:math:`R` -- the Euler equations in conservative form do not contain it -- so it
exists only for reporting a temperature.

See ``docs/theory.md`` for the full derivation and the invariance check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Backends the solver can execute on, fastest first.  See ``src.backends``.
BACKENDS = ("numba", "numpy")

#: Default :math:`M_{co}` for the AUSM family, from the sweep recorded in
#: :attr:`FlowConditions.reference_mach`.
AUSM_CUTOFF_MACH = 0.4

#: Mesh topologies.  See :mod:`src.mesh`.
TOPOLOGIES = ("ogrid", "cgrid")

#: Interface fluxes.  All four are approximate Riemann solvers in the sense that
#: they take a left and a right state and return a single upwind face flux; the
#: AUSM-family pair reach it by splitting rather than by linearising a Riemann
#: problem.  See ``docs/theory.md`` and ``docs/riemann_fluxes.md``.
FLUXES = ("roe", "hllc", "ausmup", "slau2")


@dataclass(frozen=True)
class FlowConditions:
    r"""The free-stream state and the gas.

    Parameters
    ----------
    mach
        Free-stream Mach number :math:`M_\infty`.
    flux
        Which approximate Riemann solver supplies the upwinding; see
        :data:`FLUXES` and :mod:`src.physics`.
    entropy_fix
        Roe only: the Harten-Hyman parameter :math:`\delta`, as a fraction of
        the Roe-averaged sound speed.  ``legacy/residual_calc.m`` used 0.01;
        0.05 is the usual choice for a transonic march, and it matters here
        because this flow has a sonic line.
    cutoff_mach
        AUSM-family only: Liou's :math:`M_{co}`.  ``None`` uses
        :data:`AUSM_CUTOFF_MACH`, which was measured rather than taken from the
        free stream -- see :attr:`reference_mach`.  It is a reference Mach
        number, **not** a regularisation epsilon: driving it to zero makes the
        scheme's pressure-diffusion coefficient diverge.
    alpha_deg
        Angle of attack in **degrees**, measured from the chord line, positive
        nose-up.  It enters the problem only through the free-stream velocity
        direction and the far-field boundary condition: the mesh does not
        rotate, so sweeping it costs nothing and reuses the same mesh.
    gamma
        Ratio of specific heats.
    """

    mach: float = 0.8
    alpha_deg: float = 1.25
    gamma: float = 1.4
    flux: str = "roe"
    entropy_fix: float = 0.05
    cutoff_mach: float | None = None

    def __post_init__(self) -> None:
        if self.mach <= 0.0:
            raise ValueError(f"mach must be positive, got {self.mach}")
        if self.gamma <= 1.0:
            raise ValueError(f"gamma must exceed 1, got {self.gamma}")
        if self.flux not in FLUXES:
            raise ValueError(f"unknown flux {self.flux!r}; choose from {FLUXES}")
        if self.entropy_fix < 0.0:
            raise ValueError(f"entropy_fix must be non-negative, got {self.entropy_fix}")
        if self.cutoff_mach is not None and not 0.0 < self.cutoff_mach <= 1.0:
            raise ValueError(
                f"cutoff_mach is a reference Mach number and must lie in (0, 1], got "
                f"{self.cutoff_mach}"
            )

    # -- derived free-stream quantities -----------------------------------
    @property
    def alpha(self) -> float:
        """Angle of attack in radians."""
        return math.radians(self.alpha_deg)

    @property
    def reference_mach(self) -> float:
        r"""The AUSM family's :math:`M_{co}`.

        The default is **measured, not inherited**.  Liou's guidance is
        :math:`M_{co} \sim M_\infty`, and the companion nozzle code followed it;
        on this problem it does not hold.  Swept on a NACA 0012 at two
        operating points, AUSM\ :sup:`+`-up converges only for
        :math:`0.30 \le M_{co} \le 0.45` and the window is the **same** at
        :math:`M_\infty = 0.5` and :math:`M_\infty = 0.8` -- so :math:`M_{co}`
        is not tracking the free stream here at all.  Outside it, below 0.2 the
        march dies in a handful of iterations as :math:`K_p/f_a` diverges, and
        at 0.5 and above it stalls in a limit cycle.

        0.4 is the middle of that window.  See ``docs/riemann_fluxes.md``; this
        is one section at two points, so treat it as a starting value rather
        than a constant of nature.
        """
        return self.cutoff_mach if self.cutoff_mach is not None else AUSM_CUTOFF_MACH

    def flux_options(self) -> dict:
        """The keyword arguments this operating point gives its flux."""
        if self.flux == "roe":
            return {"entropy_fix": self.entropy_fix}
        if self.flux == "ausmup":
            return {"cutoff_mach": self.reference_mach}
        return {}

    @property
    def gas_constant(self) -> float:
        r""":math:`R = \gamma - 1`, the choice that makes :math:`c_v = 1`."""
        return self.gamma - 1.0

    @property
    def density(self) -> float:
        r""":math:`\rho_\infty = 1` by construction."""
        return 1.0

    @property
    def speed(self) -> float:
        r""":math:`|\vec v_\infty| = 1` by construction."""
        return 1.0

    @property
    def pressure(self) -> float:
        r""":math:`p_\infty = 1/(\gamma M_\infty^2)`."""
        return self.density * self.speed**2 / (self.gamma * self.mach**2)

    @property
    def sound_speed(self) -> float:
        r""":math:`a_\infty = |v_\infty| / M_\infty = 1/M_\infty`."""
        return self.speed / self.mach

    @property
    def temperature(self) -> float:
        r""":math:`T_\infty = p_\infty / (\rho_\infty R)`."""
        return self.pressure / (self.density * self.gas_constant)

    @property
    def dynamic_pressure(self) -> float:
        r""":math:`q_\infty = \tfrac12 \rho_\infty |v_\infty|^2 = \tfrac12`."""
        return 0.5 * self.density * self.speed**2

    @property
    def velocity(self) -> tuple[float, float]:
        r""":math:`(u_\infty, v_\infty) = |v_\infty|(\cos\alpha, \sin\alpha)`."""
        a = self.alpha
        return self.speed * math.cos(a), self.speed * math.sin(a)

    @property
    def state(self) -> np.ndarray:
        r"""The conservative free-stream state :math:`(\rho, \rho u, \rho v, \rho E)`."""
        u, v = self.velocity
        rho = self.density
        rE = self.pressure / (self.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
        return np.array([rho, rho * u, rho * v, rE])

    @property
    def total_pressure(self) -> float:
        r"""Isentropic stagnation pressure, for reporting only."""
        g = self.gamma
        return self.pressure * (1.0 + 0.5 * (g - 1.0) * self.mach**2) ** (g / (g - 1.0))


@dataclass(frozen=True)
class Discretization:
    r"""Mesh resolution and the shape of the O-grid.

    The mesh is a body-fitted O-grid: ``n_surface`` points around the airfoil by
    ``n_radial`` cells out to a circular far field, each quadrilateral cell split
    into two triangles.  ``docs/theory.md`` describes the generator.

    Parameters
    ----------
    topology
        ``'ogrid'`` wraps a single closed loop around the section out to a
        circular far field.  ``'cgrid'`` instead cuts along the wake: the grid
        runs from the downstream outflow plane up the cut, around the section,
        and back down the cut, so cells **behind** the trailing edge are aligned
        with the wake rather than fanning out of it.  The C-grid costs
        ``2 * n_wake`` extra columns and is the better topology when the wake
        matters; the O-grid is smaller and simpler.  See :mod:`src.mesh`.
    n_wake
        C-grid only: cells along the wake cut, on each side of it.
    wake_length
        C-grid only: where the wake cut ends and the downstream outflow plane
        sits, in chords from the leading edge.
    n_surface
        Points around the airfoil.  Must be even so that a symmetric airfoil
        gets a point exactly at the leading edge.
    n_radial
        Cells from the wall to the far field.
    far_field_radius
        Radius of the far-field circle, in chords, centred on
        ``far_field_centre``.

        **Provisional default.**  The legacy MATLAB case used 50 chords because
        its far field imposed the full free-stream state, which reflects.  With
        the characteristic boundary condition this code uses, a much smaller
        domain is admissible, and 20 is a conservative starting point -- it will
        be set by measurement once the solver lands, and
        ``docs/verification.md`` will carry the sensitivity.
    far_field_centre
        Centre of the far-field circle, in chords.  The default puts it at
        mid-chord, which is what keeps the radial grid direction nearly parallel
        to the surface normal at both the leading and the trailing edge.
    first_cell_height
        Height of the first cell off the wall, in chords.  The radial spacing
        grows geometrically from here.
    growth_limit
        Largest cell-to-cell growth ratio the radial distribution may use.  If
        ``first_cell_height`` and ``n_radial`` cannot reach the far field within
        this ratio, the mesh builder raises rather than quietly returning a
        grid that stops short.
    orthogonality
        Exponent of the blend from wall-normal to radial marching.  Larger keeps
        cells wall-orthogonal further out; 0 marches radially from the start.
    clustering
        Surface point distribution: ``'cosine'`` clusters at both the leading and
        trailing edge, ``'uniform'`` does not.
    """

    topology: str = "ogrid"
    n_surface: int = 192
    n_radial: int = 64
    n_wake: int = 32
    wake_length: float = 20.0
    far_field_radius: float = 20.0
    far_field_centre: tuple[float, float] = (0.5, 0.0)
    first_cell_height: float = 2.0e-3
    growth_limit: float = 1.25
    orthogonality: float = 2.0
    clustering: str = "cosine"

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {self.topology!r}; use one of {TOPOLOGIES}")
        if self.topology == "cgrid":
            if self.n_wake < 2:
                raise ValueError(f"n_wake must be at least 2, got {self.n_wake}")
            if self.wake_length <= 2.0:
                raise ValueError(
                    f"wake_length must put the outflow plane more than 2 chords aft of "
                    f"the leading edge, got {self.wake_length}"
                )
        if self.n_surface < 16 or self.n_surface % 2:
            raise ValueError(f"n_surface must be even and at least 16, got {self.n_surface}")
        if self.n_radial < 4:
            raise ValueError(f"n_radial must be at least 4, got {self.n_radial}")
        if self.far_field_radius <= 2.0:
            raise ValueError(f"far_field_radius must exceed 2 chords, got {self.far_field_radius}")
        if not 0.0 < self.first_cell_height < 0.1:
            raise ValueError(
                f"first_cell_height must lie in (0, 0.1) chords, got {self.first_cell_height}"
            )
        if self.growth_limit <= 1.0:
            raise ValueError(f"growth_limit must exceed 1, got {self.growth_limit}")
        if self.orthogonality < 0.0:
            raise ValueError(f"orthogonality must be non-negative, got {self.orthogonality}")
        if self.clustering not in ("cosine", "uniform"):
            raise ValueError(f"unknown clustering {self.clustering!r}; use cosine or uniform")

    @property
    def n_columns(self) -> int:
        """Quadrilateral columns around the section.

        The O-grid's index wraps, so it has one column per surface point.  The
        C-grid's does not: it runs ``n_wake`` columns down the cut, ``n_surface``
        around the section and ``n_wake`` back up the other side of the cut.
        """
        if self.topology == "cgrid":
            return 2 * self.n_wake + self.n_surface
        return self.n_surface

    @property
    def n_elements(self) -> int:
        """Triangles in the mesh: two per quadrilateral cell."""
        return 2 * self.n_columns * self.n_radial

    @property
    def n_nodes(self) -> int:
        """Nodes in the mesh.

        The O-grid's surface index is periodic, so there is no seam.  The
        C-grid's wake-cut nodes at ``j = 0`` are **shared** between the two sides
        of the cut -- which is what makes the cut an ordinary interior interface
        rather than a boundary needing a condition -- so one row is short.
        """
        if self.topology == "cgrid":
            return (self.n_wake + self.n_surface) + self.n_radial * (self.n_columns + 1)
        return self.n_surface * (self.n_radial + 1)


#: Pseudo-time integrators.  ``euler`` is what ``legacy/main.m`` used.
SCHEMES = ("euler", "ssprk3", "rk4")


@dataclass(frozen=True)
class SolverOptions:
    r"""How the pseudo-time march is run.

    Parameters
    ----------
    cfl
        Courant number for the **local** time step.  Each cell advances at its
        own stable rate, which is legitimate because only the fixed point is
        wanted and it is what makes an explicit march to steady state
        affordable at all on a mesh whose cells span six orders of magnitude.
    tolerance
        Convergence threshold on the residual **scaled by its own initial
        value**, so the number means the same thing on every mesh and at every
        operating point.  ``legacy/main.m`` used an absolute
        :math:`L_\infty` residual of 1e-2, which is neither.
    max_iterations
        Give up after this many steps and return ``converged=False`` rather
        than raise -- a stalled run is a result worth looking at, and for the
        flux comparison it is *the* result.
    scheme
        ``euler`` is forward Euler, what the legacy used.  ``ssprk3`` and
        ``rk4`` cost three and four residuals per step and take a
        correspondingly larger one.
    backend
        ``auto`` takes the fastest installed, ``numpy`` the vectorised
        reference, ``numba`` the compiled one.  They agree to :math:`10^{-13}`
        and the answer does not depend on the choice.
    batch
        Steps the compiled backend runs per call, and so how often convergence
        is tested.  The whole point of that backend is not returning to Python
        every iteration, so this cannot be 1 without giving the speed back; it
        means a run can overshoot the tolerance by up to ``batch`` steps, which
        costs a fraction of a second and nothing else.  Ignored by ``numpy``,
        which checks every step.
    print_interval
        Iterations between progress lines; 0 is silent.
    divergence_factor
        Stop early if the scaled residual exceeds this.  A march that has
        blown up will not recover, and the sooner it says so the better.
    """

    backend: str = "auto"
    batch: int = 25
    cfl: float = 0.9
    tolerance: float = 1e-6
    max_iterations: int = 20_000
    scheme: str = "euler"
    print_interval: int = 500
    divergence_factor: float = 1e4

    def __post_init__(self) -> None:
        if self.backend not in ("auto", *BACKENDS):
            raise ValueError(f"unknown backend {self.backend!r}; choose auto or one of {BACKENDS}")
        if self.batch < 1:
            raise ValueError(f"batch must be at least 1, got {self.batch}")
        if self.cfl <= 0.0:
            raise ValueError(f"cfl must be positive, got {self.cfl}")
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {self.max_iterations}")
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {self.scheme!r}; choose from {SCHEMES}")
