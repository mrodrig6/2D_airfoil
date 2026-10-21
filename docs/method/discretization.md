# Spatial discretization

The spatial scheme turns the Euler equations into a residual on each cell or element.

## 6. The finite-volume discretisation

Integrating the Euler equations over a fixed cell $e$ of area $A_e$ and applying
the divergence theorem gives the semi-discrete form the solver marches:

```math
A_e \frac{d\vec{U}_e}{dt}
+ \sum_{f \in \partial e} \hat{\vec{F}}_f\,\ell_f = \vec{0},
```

where $\vec U_e$ is the cell average and $\hat{\vec F}_f$ a single numerical flux
per face, shared by the two cells that meet there — which is what makes the
scheme conservative to machine precision regardless of the flux chosen.

The reconstruction is **piecewise constant**: $\vec U$ at a face is the
neighbouring cell average, first-order accurate, exactly as the legacy MATLAB.
The upwinding is supplied entirely by the choice of $\hat{\vec F}$.
