# Reading results

Read convergence before loads or performance numbers. This page defines the reported fields, normalization and saved outputs.

## 8. Forces and coefficients

For an inviscid flow the only force on the section is the pressure integral over
the surface. With $\hat n$ the outward normal of the *flow domain* — which at the
wall points **into** the section — the force per unit span is

```math
\vec{F} = \oint_{\text{wall}} p\,\hat{n}\,ds .
```

The sign is worth deriving rather than remembering. The stress in a fluid is
$\sigma = -p\mathbf{I}$, so the traction it exerts on the body is
$-p\,\hat{n}_{\text{body}}$, with $\hat{n}_{\text{body}}$ pointing out of the
body into the flow. This code's $\hat{n}$ points out of the *flow domain*,
which at the wall is the opposite vector, so the two signs cancel. The check
that settles it: on the lower surface of a lifting section $\hat{n}$ points up
and the pressure is high, so the contribution must be upward — which
$+p\,\hat{n}$ gives and $-p\,\hat{n}$ does not.

and lift and drag are that vector resolved **perpendicular and parallel to the
free stream**, which is a rotation through $\alpha$:

```math
\begin{pmatrix} D \\ L \end{pmatrix}
= \begin{pmatrix} \cos\alpha & \sin\alpha \\ -\sin\alpha & \cos\alpha \end{pmatrix}
  \begin{pmatrix} F_x \\ F_y \end{pmatrix},
\qquad
c_\ell = \frac{L}{q_\infty c}, \quad c_d = \frac{D}{q_\infty c},
```

with $q_\infty = \tfrac12$ and $c = 1$. The pressure coefficient is

```math
c_p = \frac{p - p_\infty}{q_\infty} = 2\left(p - p_\infty\right),
```

plotted as $-c_p$ by convention, so suction is up.
