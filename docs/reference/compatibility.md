# Compatibility

Use this matrix to distinguish an implemented feature from a measured or untested combination.

| Combination | Status / restriction |
|---|---|
| NumPy reference solver | ✓ |
| Numba compiled solver | ✓; first run compiles kernels |
| JAX differentiation | ✓; CPU float64 is used for gradient checks |
| O-grid / C-grid | ✓; solver sees triangles from either |
| Imported SU2 / Gmsh 2.2 ASCII | ✓; triangles only, boundary markers must map |
| Gmsh 4 / quadrilateral solution cells | ✗; re-export Gmsh 2.2 and triangulate |
| Numba Euler | ✓ |
| SSP-RK3 / RK4 | ✗; removed, forward Euler is the only supported scheme |
| Four-digit shape gradients | ✓; mesh terms use frozen-state finite differences |
| Five-digit / file shape gradients | ✗; flow variables remain available |
| AUSM⁺-up at low Mach on fine grids | Restricted: reported limit cycles; check `converged` |
| SLAU2 | ✗; historical study only |
| Viscosity, turbulence, unsteady physical time | ✗; outside the model |
| Linux / Python 3.12 CPU | ✓; local validation environment |
| Python 3.10 / newest stable | CI targets; results depend on the workflow run |
| Windows / macOS / GPU | Not validated by this change |

See [assumptions](../model/problem.md), [parameters](../model/parameters.md), and [backend behavior](../method/backends.md).
