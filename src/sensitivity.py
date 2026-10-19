r"""Exact design sensitivities by the discrete adjoint.

What this gives you
-------------------
For a scalar functional :math:`J` -- lift, drag, their ratio -- and design
variables :math:`\vec{a}`, this returns :math:`dJ/d\vec{a}` **exactly**, at a
cost that does not grow with how many design variables there are.  A
twenty-variable shape gradient costs one adjoint solve where finite differences
would cost twenty extra flow solves.

Why an adjoint rather than ``jax.grad`` through the solver
----------------------------------------------------------
Differentiating the march itself would tape thousands of iterations -- ruinous
in memory, and it gives the derivative of the *iteration history* rather than
of the converged answer.  Instead the converged state is defined implicitly by
:math:`R(\vec U, \vec a) = 0`, and the implicit function theorem gives

.. math::
    \frac{dJ}{d\vec a}
    = \frac{\partial J}{\partial \vec a}
      - \vec\lambda^{T} \frac{\partial R}{\partial \vec a},
    \qquad
    \left(\frac{\partial R}{\partial \vec U}\right)^{T}\!\vec\lambda
    = \left(\frac{\partial J}{\partial \vec U}\right)^{T} .

Only the converged state is needed, and both Jacobian actions come from JAX's
``vjp`` on the residual, so nothing is ever assembled as a matrix.

**The adjoint follows the scheme.**  It differentiates whatever
:func:`src.spatial.assembly.residual` computes, so changing the flux -- or adding a
higher-order reconstruction later -- changes the gradient with nothing to
re-derive by hand.

How the adjoint system is solved
--------------------------------
By a pseudo-time march mirroring the forward solver:

.. math::
    \frac{d\vec\lambda}{d\tau} = -\frac{1}{A}
    \left[\left(\frac{\partial R}{\partial \vec U}\right)^{T}\vec\lambda
          - \left(\frac{\partial J}{\partial \vec U}\right)^{T}\right].

The mass matrix here is just the cell area, a positive diagonal, so
:math:`A^{-1}(\partial R/\partial \vec U)^{T}` has the **same spectrum** as the
forward operator :math:`A^{-1}\partial R/\partial \vec U`.  The forward
solver's local time step therefore carries over unchanged, which is what makes
the adjoint as robust as the march and removes any need for a preconditioner.

Design variables, and the two routes to them
--------------------------------------------
``alpha`` and ``mach`` enter the problem through **one vector** -- the
free-stream state -- and not through the mesh, which does not rotate with
incidence.  Their derivatives are therefore exact with no geometric
differentiation whatever, and that is why they came first.

The shape variables move the mesh, and the mesh is the awkward part: making
the whole pipeline traceable means a differentiable surface generator, a
differentiable marcher including the bisection that solves for the growth
ratio, and a traced copy of every metric the residual reads.  That is a large
refactor for a small term, so this module takes the standard hybrid route
instead:

.. math::
    \frac{dJ}{da} \;\approx\;
      \underbrace{\frac{\partial J}{\partial a}\bigg|_{\vec U}}_{\text{finite difference}}
      - \;\vec\lambda^{T}
      \underbrace{\frac{\partial R}{\partial a}\bigg|_{\vec U}}_{\text{finite difference}}

**The expensive part stays exact.**  :math:`\vec\lambda` is the adjoint, solved
once, and it is what the flow's response to the shape costs.  The two
finite-differenced terms are evaluated at a **frozen state**: no iteration, no
convergence tolerance, just rebuild the mesh and evaluate a residual twice.
They are smooth and well conditioned, which is exactly what a finite difference
is good at and what a converged flow solve is not.

Cost: one adjoint, plus two residual evaluations per shape variable -- against
two full flow solves per variable for a naive finite difference.  On the mesh
used in the tests that is seconds against minutes, and the gap widens with the
number of variables.

Validity
--------
The gradient is of the **discrete** problem, so it matches a finite difference
of this solver's own output rather than of the exact Euler equations -- which
is precisely what gradient-based optimisation needs.  Two caveats worth saying
out loud:

* A **shock** makes the discrete objective only piecewise differentiable in the
  design variables: the shock jumps from one cell to the next rather than
  sliding. The gradient stays well defined and useful, but a finite-difference
  check at a transonic point will disagree at the level of that jump, and a
  step size below one cell's worth of shock movement measures noise.
  :func:`check_gradient` reports the comparison rather than asserting it.
* The march must have **converged**.  A gradient taken from a stalled state is
  the derivative of nothing in particular.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .api import Case
from .config import Discretization, FlowConditions, SolverOptions
from .geometry.mesh import BoundaryTag
from .physics import FLOOR
from .spatial import assembly as asm
from .spatial.metrics import Metrics
from .time import stepping as ts
from .time.march import SolveResult

#: Design variables that enter through the free stream alone, so their
#: derivatives need no geometric differentiation at all.
FLOW_VARIABLES = ("alpha", "mach")

#: Design variables that move the **mesh**.  The four-digit parameters, taken
#: continuously -- ``"0012"`` cannot be perturbed by :math:`10^{-6}`, which is
#: why :func:`src.geometry.naca_section` exists.
SHAPE_VARIABLES = ("camber", "camber_position", "thickness")

#: Everything :class:`DifferentiableCase` can differentiate with respect to.
DESIGN_VARIABLES = FLOW_VARIABLES + SHAPE_VARIABLES

#: Objectives it can differentiate.
OBJECTIVES = ("cl", "cd", "cm", "efficiency")


def freestream_state(mach, alpha_rad, gamma: float, xp):
    r"""The conservative free stream, written so it can be traced.

    Mirrors :attr:`src.config.FlowConditions.state` exactly -- and is checked
    against it -- but takes ``mach`` and ``alpha_rad`` as values rather than
    reading them off a frozen dataclass, which is what lets JAX follow them.
    """
    pressure = 1.0 / (gamma * mach * mach)
    u = xp.cos(alpha_rad)
    v = xp.sin(alpha_rad)
    return xp.stack([xp.ones_like(u), u, v, pressure / (gamma - 1.0) + 0.5 * (u * u + v * v)])


def wall_loads(U, alpha_rad, metrics: Metrics, gamma: float, xp):
    r"""``(cl, cd, cm)`` from the surface pressure integral, traceable.

    The same formula as :func:`src.post.postprocess.forces`, in the injected array
    module.  Kept separate rather than shared because that one takes a
    :class:`~src.time.march.SolveResult` and returns floats, which a tracer cannot
    pass through; ``tests/test_sensitivity.py`` pins the two together.
    """
    s = metrics.tag_slice(BoundaryTag.WALL)
    element = metrics.boundary_element[s]
    nx = xp.asarray(metrics.boundary_normal[s, 0])
    ny = xp.asarray(metrics.boundary_normal[s, 1])
    length = xp.asarray(metrics.boundary_length[s])
    mid = metrics.boundary_midpoint[s]

    face = U[element]
    rho = xp.maximum(face[:, 0], FLOOR)
    vx = face[:, 1] / rho
    vy = face[:, 2] / rho
    vn = vx * nx + vy * ny
    qt2 = (vx - vn * nx) ** 2 + (vy - vn * ny) ** 2
    p = xp.maximum((gamma - 1.0) * (face[:, 3] - 0.5 * rho * qt2), FLOOR)

    fx = xp.sum(p * nx * length)
    fy = xp.sum(p * ny * length)
    ca, sa = xp.cos(alpha_rad), xp.sin(alpha_rad)
    drag = fx * ca + fy * sa
    lift = -fx * sa + fy * ca

    arm_x = xp.asarray(mid[:, 0] - 0.25)
    arm_y = xp.asarray(mid[:, 1])
    moment = -xp.sum(p * (arm_x * ny - arm_y * nx) * length)
    return lift / 0.5, drag / 0.5, moment / 0.5


@dataclass
class DifferentiableCase:
    """A prepared case whose converged loads can be differentiated.

    Build one with :func:`differentiable_case`.
    """

    case: Case
    flow: FlowConditions
    options: SolverOptions
    shape: dict[str, float] | None = None
    shape_step: float = 1e-5
    #: Scaled defect the adjoint march stops at.  Measured, not guessed: the
    #: gradient stops moving at about :math:`10^{-8}`.  Going on to
    #: :math:`10^{-10}` costs a quarter more sweeps and changes
    #: :math:`d c_\ell/d\alpha` in its ninth significant figure, which is well
    #: below the discretisation error of the flow it is a derivative of.
    adjoint_tolerance: float = 1e-8
    adjoint_iterations: int = 40_000
    adjoint_batch: int = 50
    adjoint_sweeps: int = field(default=0, repr=False)
    result: SolveResult | None = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    # -- the pieces, all in jax.numpy --------------------------------------
    def _jnp(self):
        from .backends.jax_backend import enable_double_precision

        enable_double_precision()
        import jax.numpy as jnp

        return jnp

    def _residual(self, U, params: dict):
        jnp = self._jnp()
        gamma = self.flow.gamma
        U_inf = freestream_state(params["mach"], params["alpha"] * math.pi / 180.0, gamma, jnp)

        return asm.residual(U, self.case.metrics, self.flow, xp=jnp, U_inf=U_inf).R

    def _objective(self, U, params: dict, name: str):
        jnp = self._jnp()
        cl, cd, cm = wall_loads(
            U, params["alpha"] * math.pi / 180.0, self.case.metrics, self.flow.gamma, jnp
        )
        if name == "cl":
            return cl
        if name == "cd":
            return cd
        if name == "cm":
            return cm
        if name == "efficiency":
            return cl / cd
        raise ValueError(f"unknown objective {name!r}; choose from {OBJECTIVES}")

    def params(self, **overrides: float) -> dict:
        """The design-variable dictionary this case is currently at."""
        base = {"alpha": float(self.flow.alpha_deg), "mach": float(self.flow.mach)}
        if self.shape is not None:
            base.update({k: float(v) for k, v in self.shape.items()})
        unknown = set(overrides) - set(DESIGN_VARIABLES)
        if unknown:
            raise ValueError(
                f"unknown design variable(s) {sorted(unknown)}; choose from {DESIGN_VARIABLES}"
            )
        return {**base, **{k: float(v) for k, v in overrides.items()}}

    def flow_params(self, params: dict) -> dict:
        """Just the two the residual traces through."""
        return {"alpha": params["alpha"], "mach": params["mach"]}

    def _case_at(self, params: dict) -> Case:
        """Rebuild the mesh and metrics for a perturbed shape."""
        from .api import build_case
        from .geometry.section import naca_section

        section = naca_section(
            params["camber"],
            params["camber_position"],
            params["thickness"],
            self.case.discretization.n_surface,
            clustering=self.case.discretization.clustering,
        )
        return build_case(section, self.case.discretization)

    def _shape_partials(self, U, name: str, variables, step: float):
        r"""``dJ/da`` and ``dR/da`` at **frozen state**, by central difference.

        No flow solve happens in here: the mesh is rebuilt, the residual and
        the objective are evaluated once each side, and that is all.  Which is
        why a finite difference is the right tool for this term and the wrong
        one for the whole gradient.
        """

        out = {}
        for key in variables:
            dJ, dR = [], []
            for sign in (+1, -1):
                moved = self.params(**{key: self.params()[key] + sign * step})
                case = self._case_at(moved)
                flow = replace(self.flow, alpha_deg=moved["alpha"], mach=moved["mach"])
                dR.append(asm.residual(U, case.metrics, flow).R)
                cl, cd, cm = wall_loads(
                    U, moved["alpha"] * math.pi / 180.0, case.metrics, flow.gamma, np
                )
                dJ.append({"cl": cl, "cd": cd, "cm": cm, "efficiency": cl / cd}[name])
            out[key] = (
                (dJ[0] - dJ[1]) / (2.0 * step),
                (dR[0] - dR[1]) / (2.0 * step),
            )
        return out

    # -- the forward solve --------------------------------------------------
    def solve(self, *, force: bool = False) -> SolveResult:
        """Converge the flow, once, and keep it."""
        if self.result is None or force:
            self.result = self.case.solve(self.flow, self.options)
            if not self.result.converged:
                raise RuntimeError(
                    f"the flow solve did not converge ({self.result.message}), so a "
                    f"gradient taken from it would be the derivative of nothing in "
                    f"particular"
                )
        return self.result

    # -- the adjoint --------------------------------------------------------
    def adjoint(self, name: str = "cl") -> np.ndarray:
        r"""Solve :math:`(\partial R/\partial U)^{T}\lambda = (\partial J/\partial U)^{T}`.

        Marched in pseudo-time with the forward solver's own local time step,
        which is legitimate because the two operators share a spectrum.
        """
        import jax

        if name in self._cache:
            return self._cache[name]
        jnp = self._jnp()
        result = self.solve()
        metrics = self.case.metrics
        params = self.params()
        U = jnp.asarray(result.state)

        # dJ/dU, and the transpose of dR/dU as a function
        seed = jax.grad(lambda state: self._objective(state, params, name))(U)
        _, vjp = jax.vjp(lambda state: self._residual(state, params), U)

        wave = asm.residual(result.state, metrics, self.flow).wave
        dt = ts.local_time_step(wave, metrics, self.options.cfl)
        step = jnp.asarray(dt / metrics.area)[:, None]

        # Batched exactly like the forward march, and for the same reason: a
        # convergence test every sweep means a device-to-host transfer every
        # sweep, which on this problem costs more than the sweep does.  A block
        # of `adjoint_batch` sweeps runs inside one `fori_loop` as a single
        # compiled program, and only the block boundary is looked at.
        @jax.jit
        def block(lam):
            return jax.lax.fori_loop(
                0, self.adjoint_batch, lambda _, l: l - step * (vjp(l)[0] - seed), lam
            )

        @jax.jit
        def defect(lam):
            """How far the adjoint system is from being satisfied.

            The right measure, rather than how much the last sweep moved: this
            one does not depend on the time step or on the batch size, so the
            tolerance means the same thing however the march is arranged.
            """
            return jnp.linalg.norm(vjp(lam)[0] - seed)

        lam = jnp.zeros_like(U)
        reference = float(jnp.linalg.norm(seed)) or 1.0
        blocks = max(1, self.adjoint_iterations // self.adjoint_batch)
        for completed in range(1, blocks + 1):
            lam = block(lam)
            residual = float(defect(lam)) / reference
            if not np.isfinite(residual):
                raise RuntimeError(
                    f"the adjoint diverged within the first {completed * self.adjoint_batch} sweeps"
                )
            if residual < self.adjoint_tolerance:
                break
        else:
            raise RuntimeError(
                f"the adjoint did not converge in {blocks * self.adjoint_batch} sweeps "
                f"(scaled defect {residual:.3e}); the forward solve may be only "
                f"marginally converged"
            )
        self.adjoint_sweeps = completed * self.adjoint_batch
        self._cache[name] = np.asarray(lam)
        return self._cache[name]

    @property
    def variables(self) -> tuple[str, ...]:
        """What this case can be differentiated with respect to."""
        return DESIGN_VARIABLES if self.shape is not None else FLOW_VARIABLES

    def value_and_gradient(
        self, name: str = "cl", variables: tuple[str, ...] | None = None
    ) -> tuple[float, dict[str, float]]:
        r"""The objective and its exact gradient.

        .. math::
            \frac{dJ}{d\vec a} = \frac{\partial J}{\partial \vec a}
                               - \vec\lambda^{T}\frac{\partial R}{\partial \vec a}

        Both partials come from one reverse-mode pass each, so **every** design
        variable costs the same as one -- which is the whole point of the
        adjoint and the reason this scales to a shape.
        """
        import jax

        variables = tuple(variables) if variables is not None else self.variables
        unknown = set(variables) - set(DESIGN_VARIABLES)
        if unknown:
            raise ValueError(f"unknown design variable(s) {sorted(unknown)}")
        if self.shape is None and set(variables) & set(SHAPE_VARIABLES):
            raise ValueError(
                f"shape derivatives need a four-digit section, because the "
                f"parameters have to be exact; {self.case.airfoil.name!r} has none. "
                f"Differentiate {FLOW_VARIABLES} instead, or rebuild the case from a "
                f"NACA code."
            )
        jnp = self._jnp()
        U = jnp.asarray(self.solve().state)
        params = self.params()
        lam = jnp.asarray(self.adjoint(name))

        flow_vars = tuple(v for v in variables if v in FLOW_VARIABLES)
        shape_vars = tuple(v for v in variables if v in SHAPE_VARIABLES)

        traced = self.flow_params(params)
        value, direct = jax.value_and_grad(lambda a: self._objective(U, {**params, **a}, name))(
            traced
        )
        _, vjp = jax.vjp(lambda a: self._residual(U, {**params, **a}), traced)
        indirect = vjp(lam)[0]
        gradient = {k: float(direct[k] - indirect[k]) for k in flow_vars}

        if shape_vars:
            state = np.asarray(U)
            costate = np.asarray(lam)
            for key, (dJ, dR) in self._shape_partials(
                state, name, shape_vars, self.shape_step
            ).items():
                gradient[key] = float(dJ - np.sum(costate * dR))
        return float(value), {key: gradient[key] for key in variables}


def four_digit_parameters(spec: Any) -> dict[str, float] | None:
    r"""The exact :math:`(m, p, t)` behind a four-digit code, or ``None``.

    Recovering them by **sampling** the surface is the obvious shortcut and it
    is not good enough: a four-digit camber line is flat near its crest, so the
    position comes back as 0.441 for a NACA 2412 whose true value is 0.4.  The
    perturbed meshes would then surround a slightly different section from the
    one the flow was solved on, and the gradient would be of neither.

    So the parameters are parsed from the code when there is one, and shape
    derivatives are simply **unavailable** when there is not -- a section read
    from a file has no parameters to perturb, and inventing some would be worse
    than saying so.
    """
    import re

    text = re.fullmatch(r"(?:naca[\s-]*)?(\d{4})", str(spec).strip(), re.IGNORECASE)
    if text is None:
        return None
    digits = text.group(1)
    camber = int(digits[0]) / 100.0
    return {
        "camber": camber,
        "camber_position": int(digits[1]) / 10.0 if camber > 0.0 else 0.4,
        "thickness": int(digits[2:]) / 100.0,
    }


def differentiable_case(
    airfoil: str = "0012",
    *,
    case: Case | None = None,
    options: SolverOptions | None = None,
    **overrides: Any,
) -> DifferentiableCase:
    """Prepare a case for differentiation.  Takes the same options as
    :func:`~src.api.solve_airfoil`."""
    from .api import _split, build_case

    geometry, mesh_keys, flow_keys, solver_keys = _split(overrides)
    if case is None:
        case = build_case(airfoil, **geometry, **mesh_keys)
    base = options or SolverOptions(print_interval=0)
    if solver_keys:
        base = SolverOptions(**{**base.__dict__, **solver_keys})
    return DifferentiableCase(
        case=case,
        flow=FlowConditions(**flow_keys),
        options=base,
        shape=four_digit_parameters(geometry.get("airfoil", airfoil)),
    )


def four_digit_case(
    camber: float,
    camber_position: float,
    thickness: float,
    **overrides: Any,
) -> DifferentiableCase:
    r"""A differentiable case from **continuous** four-digit parameters.

    What an optimiser loop wants.  :func:`differentiable_case` takes a code --
    ``"2412"`` -- and recovers :math:`(m, p, t)` from the digits, which is right
    for a study at a named section and useless inside a design loop, where the
    parameters are the variables and ``camber = 0.0213`` has no code.

    >>> case = four_digit_case(0.0213, 0.37, 0.12, mach=0.5, alpha=1.0)  # doctest: +SKIP
    >>> value, gradient = case.value_and_gradient("cl")                  # doctest: +SKIP
    """
    from .geometry.section import naca_section

    geometry = {k: overrides.pop(k) for k in ("n_surface", "clustering") if k in overrides}
    section = naca_section(
        camber,
        camber_position,
        thickness,
        geometry.get("n_surface", Discretization().n_surface),
        clustering=geometry.get("clustering", Discretization().clustering),
    )
    case = differentiable_case(section, **geometry, **overrides)
    case.shape = {
        "camber": float(camber),
        "camber_position": float(camber_position),
        "thickness": float(thickness),
    }
    return case


def finite_difference_gradient(
    dc: DifferentiableCase,
    name: str = "cl",
    variables: tuple[str, ...] | None = None,
    step: float = 1e-4,
) -> dict[str, float]:
    """Central-difference gradient, for checking the adjoint.

    Two full flow solves per variable, which is exactly the cost the adjoint
    exists to avoid -- so this is a test tool, not a workflow.
    """
    from .post.postprocess import forces

    variables = tuple(variables) if variables is not None else dc.variables
    out = {}
    for key in variables:
        values = []
        for sign in (+1, -1):
            moved = dc.params(**{key: dc.params()[key] + sign * step})
            flow = FlowConditions(
                mach=moved["mach"],
                alpha_deg=moved["alpha"],
                gamma=dc.flow.gamma,
                flux=dc.flow.flux,
                entropy_fix=dc.flow.entropy_fix,
            )
            case = dc.case if key in FLOW_VARIABLES else dc._case_at(moved)
            result = case.solve(flow, dc.options)
            if not result.converged:
                raise RuntimeError(f"the {key}{sign:+} solve did not converge")
            load = forces(result)
            values.append(
                {"cl": load.cl, "cd": load.cd, "cm": load.cm, "efficiency": load.cl / load.cd}[name]
            )
        out[key] = (values[0] - values[1]) / (2.0 * step)
    return out


def check_gradient(
    dc: DifferentiableCase,
    name: str = "cl",
    variables: tuple[str, ...] | None = None,
    step: float = 1e-4,
) -> dict[str, dict[str, float]]:
    """Compare the adjoint against finite differences and **report**, not assert.

    At a transonic point the two legitimately disagree a little: the shock moves
    from one cell to the next rather than sliding, so the discrete objective is
    only piecewise differentiable and a small step measures that jump.  Judging
    the gradient wrong on that basis would be judging it against the wrong
    thing.
    """
    variables = tuple(variables) if variables is not None else dc.variables
    _, exact = dc.value_and_gradient(name, variables)
    approx = finite_difference_gradient(dc, name, variables, step)
    return {
        key: {
            "adjoint": exact[key],
            "finite_difference": approx[key],
            "relative_error": abs(exact[key] - approx[key]) / max(abs(approx[key]), 1e-12),
        }
        for key in variables
    }


__all__ = [
    "DESIGN_VARIABLES",
    "FLOW_VARIABLES",
    "SHAPE_VARIABLES",
    "OBJECTIVES",
    "DifferentiableCase",
    "check_gradient",
    "four_digit_parameters",
    "differentiable_case",
    "finite_difference_gradient",
    "four_digit_case",
    "freestream_state",
    "wall_loads",
]
