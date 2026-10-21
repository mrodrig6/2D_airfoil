# Thin-airfoil lift

This comparison uses thin-airfoil theory with the Prandtl–Glauert correction. Finite thickness and discretization mean exact agreement is not expected.

## Lift against thin-airfoil theory

The one closed form available. For a thin section at small incidence,
$c_\ell = 2\pi\alpha/\sqrt{1 - M_\infty^2}$.

NACA 0012 at $M_\infty = 0.5$, $\alpha = 2^\circ$, far field 15 chords:

| mesh | cells | $c_\ell$ | theory | error |
|---|---|---|---|---|
| 96 × 40 | 7 680 | 0.2464 | 0.2532 | −2.7% |

Within the 12% a 12%-thick section at finite Mach number should differ from a
*thin*-airfoil result, and converging the right way.
