# Units and reference scales

The solver uses nondimensional values throughout. These reference scales define the meaning of its inputs and reported coefficients.

## 2. Non-dimensionalisation

There is no SI anywhere in this code. Every quantity is referred to the free
stream, with the chord as the length scale, exactly as `legacy/initialstate.m`
did:

```math
\rho_\infty = 1, \qquad |\vec{v}_\infty| = 1, \qquad c = 1,
\qquad R = \gamma - 1 .
```

Fixing the density and the speed means the free-stream **Mach number is carried
by the pressure**:

```math
M_\infty = \frac{|\vec v_\infty|}{a_\infty}
         = \frac{|\vec v_\infty|}{\sqrt{\gamma p_\infty / \rho_\infty}}
\qquad\Longrightarrow\qquad
\boxed{\ p_\infty = \frac{\rho_\infty}{\gamma}
        \left(\frac{|\vec v_\infty|}{M_\infty}\right)^{2}
      = \frac{1}{\gamma M_\infty^{2}}\ }
```

so $p_\infty$ is **not** 1. At $M_\infty = 0.8$ it is $1.116071\ldots$, and
$a_\infty = 1/M_\infty = 1.25$.

The free-stream conservative state is then

```math
\vec{U}_\infty = \left(1,\ \cos\alpha,\ \sin\alpha,\
\frac{p_\infty}{\gamma - 1} + \tfrac12\right)^{T} .
```

### Why this particular choice

Three properties make it the right one for a design study:

1. **Dynamic pressure is exactly $\tfrac12$** at every operating point, since
   $q_\infty = \tfrac12 \rho_\infty |\vec v_\infty|^2$. A force coefficient is
   therefore just twice the force per unit span, and $c_\ell$ at $M = 0.5$ is
   directly comparable with $c_\ell$ at $M = 0.8$ with no rescaling.
2. **The angle of attack enters in exactly one place** — the direction of
   $\vec v_\infty$. The mesh does not rotate with it, so an $\alpha$ sweep reuses
   one mesh and one set of metrics, which is what makes a sweep cheap and an
   $\alpha$ derivative meaningful.
3. **$R = \gamma - 1$ makes $c_v = 1$**, so temperature and internal energy per
   unit mass are numerically equal. This is the same convention the companion
   nozzle code uses, which keeps the two comparable. Nothing in the solver reads
   $R$ — the Euler equations in conservative form do not contain it — so it
   exists only for reporting a temperature.

### Reynolds number does not appear

Despite being a high-speed aerodynamics code, $Re$ is nowhere in it. The Euler
equations are the $Re \to \infty$ *limit*, not a high-$Re$ flow: there is no
length-to-viscosity ratio left to form. A result from this code is independent
of Reynolds number by construction, which is a modelling statement, not a
convenience.

`src/config.py` holds all of this, and `tests/test_flow.py` checks every claim
above rather than taking it on trust.
