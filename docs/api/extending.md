# Extending the solver

Add a component at the existing source interfaces or postprocess a returned result. The worked metric below leaves the numerical implementation unchanged.

## Worked extension: a custom section

This creates a 10%-thicker version of a NACA 0012 and builds a mesh without editing the solver. `Airfoil.points` is a counter-clockwise loop with the trailing edge first; the last point is not repeated.

```python
from src import Airfoil, Discretization, build_case, naca4

baseline = naca4("0012", 48)
points = baseline.points.copy()
points[:, 1] *= 1.1
section = Airfoil(name="custom-thickness", points=points)
mesh = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)
case = build_case(section, mesh)
print(case.airfoil.name, case.metrics.n_elements)
```

For a new interface flux, start in `src/physics.py`, add dispatch in `src/assembly.py` and `src/config.py`, then implement the compiled counterpart in `src/backends/_numba_kernels.py`. Compare residuals and conservation with the existing backend tests before a full solve. This is source-level extension, not a plugin framework.

Shape gradients are unavailable for a custom section unless a parameterization is supplied through a separate implementation. Flow derivatives remain available; see [compatibility](../reference/compatibility.md).
