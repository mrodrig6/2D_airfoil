# Discrete adjoint

The discrete adjoint differentiates the converged residual equation. It avoids differentiating the full pseudo-time history.

## Why an adjoint, and not `jax.grad`

Differentiating the march itself is the obvious thing and the wrong thing. It
tapes every iteration — thousands of copies of the residual graph — and what it
returns is the derivative of the *iteration history*, which depends on where the
march started and when it was stopped.

The converged state is instead defined implicitly by $R(\vec U, \vec a) = 0$, and
the implicit function theorem gives the derivative of the **answer**:

```math
\frac{dJ}{d\vec a}
  = \frac{\partial J}{\partial \vec a}
  - \vec\lambda^{T}\frac{\partial R}{\partial \vec a},
\qquad
\left(\frac{\partial R}{\partial \vec U}\right)^{T}\!\vec\lambda
  = \left(\frac{\partial J}{\partial \vec U}\right)^{T}.
```

Only the converged state appears. Both Jacobian actions come from `jax.vjp` on
the residual, so no matrix is ever assembled, and — the part worth internalising
— **the adjoint follows the scheme**. It differentiates whatever
`src/assembly.py` computes, so changing the flux, or adding a higher-order
reconstruction later, changes the gradient with nothing to re-derive by hand.

The adjoint system itself is marched in pseudo-time, exactly like the flow:

```math
\frac{d\vec\lambda}{d\tau} = -\frac{1}{A}
  \left[\left(\frac{\partial R}{\partial \vec U}\right)^{T}\vec\lambda
        - \left(\frac{\partial J}{\partial \vec U}\right)^{T}\right].
```

The mass matrix is the cell area — a positive diagonal — so
$A^{-1}(\partial R/\partial \vec U)^{T}$ has the **same spectrum** as the forward
operator. The forward solver's local time step therefore carries over unchanged,
which is what makes the adjoint as robust as the march and removes any need for a
preconditioner.

## The shape variables, and the one compromise

`alpha` and `mach` enter through **one vector** — the free-stream state — and not
through the mesh. Their derivatives are exact with no geometric differentiation
whatever, which is why they came first.

The shape variables move the mesh, and making the whole pipeline traceable would
mean a differentiable surface generator, a differentiable marcher including the
bisection that solves for the growth ratio, and a traced copy of every metric the
residual reads. That is a large refactor for a small term, so the standard hybrid
route is taken instead:

```math
\frac{dJ}{da} \approx
  \underbrace{\frac{\partial J}{\partial a}\bigg|_{\vec U}}_{\text{finite difference}}
  - \vec\lambda^{T}
  \underbrace{\frac{\partial R}{\partial a}\bigg|_{\vec U}}_{\text{finite difference}}
```

**The expensive part stays exact.** $\vec\lambda$ is the adjoint, solved once,
and it is what the flow's response to the shape costs. The two
finite-differenced terms are evaluated at a **frozen state**: no iteration, no
convergence tolerance, just rebuild the mesh and evaluate a residual twice. They
are smooth and well conditioned, which is what a finite difference is good at and
what a converged flow solve is not. All three shape derivatives together add
**0.3 s** to the table above.
