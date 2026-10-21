# Plotting

Use `src.plotting` to inspect pressure, fields and polars. Figures use nondimensional values and do not correct the solver’s first-order numerical error.

```bash
# Plot pressure, Mach and residual history from a transonic solve.
./fv2d.sh solve @transonic cp=cp.png figure=mach.png convergence=history.png
# Generate both documented transonic fields and an incidence polar.
python examples/06_postprocess.py figures
```

| Plot control | Interpretation |
|---|---|
| `cp=...` | Surface pressure, drawn with suction upward and a sonic reference |
| `figure=...` | Scalar field, Mach by default |
| `scalar=entropy` | Entropy: numerical dissipation plus physical shock contribution |
| `shading=flat` | Piecewise-constant cell values and the mesh |
| `shading=smooth` | Node-interpolated contours; the default |
| `convergence=...` | Residual history, which must be read with `converged` |

The library functions are `plot_surface_pressure`, `plot_field`, `plot_polar` and `plot_convergence`. [The postprocess example](../../examples/06_postprocess.py) is a complete Python recipe. The [figure inventory](../reference/figures.md) records source files and rebuild commands.

## Color maps

`src.plotting.field_cmap` refuses nonmonotonic maps such as `jet`, `rainbow` and `hsv`. The default `magma` increases in lightness, so a brighter patch represents a larger value and does not introduce a false shock edge. The original MATLAB plotted Mach with `jet`; that historical choice is not retained in current plots.
