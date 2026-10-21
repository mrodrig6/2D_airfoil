# Euler equations

The compressible Euler equations describe the inviscid gas model. Their conservative flux and characteristic speeds determine the numerical and boundary fluxes.

## 3. Governing equations

The two-dimensional Euler equations in conservative form:

```math
\frac{\partial \vec{U}}{\partial t}
+ \frac{\partial \vec{F}(\vec{U})}{\partial x}
+ \frac{\partial \vec{G}(\vec{U})}{\partial y} = \vec{0},
```

```math
\vec{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{pmatrix},
\qquad
\vec{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ \rho u H \end{pmatrix},
\qquad
\vec{G} = \begin{pmatrix} \rho v \\ \rho u v \\ \rho v^2 + p \\ \rho v H \end{pmatrix},
```

closed by the perfect-gas relation

```math
p = (\gamma - 1)\left(\rho E - \tfrac12 \rho\,(u^2 + v^2)\right),
\qquad H = \frac{\rho E + p}{\rho} .
```

### The projected flux, and why only it is ever needed

A finite-volume scheme never needs $\vec F$ and $\vec G$ separately — only their
projection onto a face normal. With $v_n = u n_x + v n_y$,

```math
\vec{F}\,n_x + \vec{G}\,n_y
= \begin{pmatrix}
\rho v_n \\
\rho u v_n + p\,n_x \\
\rho v v_n + p\,n_y \\
\rho H v_n
\end{pmatrix} .
```

This form is **rotationally invariant**: the flux through a face depends on the
state and the normal only through $v_n$ and the tangential component, which is
what lets one Riemann solver serve every face of an unstructured mesh with no
per-face coordinate rotation.

### Characteristic structure

The normal flux Jacobian $\mathbf{A}_n = \partial(\vec F n_x + \vec G n_y) /
\partial \vec U$ has eigenvalues

```math
\lambda = \left\{\,v_n - a,\ \ v_n,\ \ v_n,\ \ v_n + a\,\right\},
```

an acoustic pair and a doubled convective value carrying entropy and the
tangential velocity. All four are real with a complete set of eigenvectors, so
the system is hyperbolic in time and the maximum signal speed at a face is
$|v_n| + a$ — the quantity the local time step is built from.

The sign pattern is what makes the problem transonic. Where $|v_n| < a$ the four
eigenvalues do not share a sign, information travels both ways through the face,
and the steady problem is elliptic; where $|v_n| > a$ they do, and it is
hyperbolic. Both regions exist in the same domain at $M_\infty = 0.8$, separated
by a sonic line that the solution itself determines.
