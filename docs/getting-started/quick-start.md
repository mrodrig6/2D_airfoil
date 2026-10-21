# Quick start

Run this from the checkout root after installation. The displayed output was captured with Python 3.12 and the Numba backend.

```bash
# Solve one operating point and check the command's exit status.
./fv2d.sh solve @coarse mach=0.5 alpha=2 print_interval=0
```

```text
NACA 0012: 96 points, chord 1.000000, max thickness 0.12001, max camber -0.00000, trailing edge sharp
96x40 O-grid: 7680 triangles, 3936 nodes, 11424 interior edges, 192 boundary edges (96 wall, 96 far field)
7680 cells, area 706.272148, hydraulic diameter 4.831e-04 to 5.545e-01, closure error 4.44e-16
converged after 5550 iterations in 2.22 s (2495 it/s on numba), scaled residual 9.083e-07
  cl = +0.246450   cd = +0.023563   cm(c/4) = -0.004600   L/D = +10.5
  entropy error 1.7433e-04
  no upper-surface shock detected
  Mach number 0.0622 to 0.6624
```

- The first line describes the section and its trailing edge.
- The second line counts cells, nodes and boundary faces.
- The third reports area, hydraulic-diameter range and geometric closure; closure should be at round-off.
- `converged` confirms the residual crossed the requested tolerance; elapsed time and throughput depend on hardware and cached compilation.
- `cl`, `cd` and `cm(c/4)` are lift, drag and quarter-chord moment coefficients. First-order numerical dissipation contributes to drag.
- `entropy error` and Mach extrema describe the field. This subcritical run has no upper-surface shock.

`@coarse` is the shipped 96×40 mesh deck, not the production transonic mesh. Later arguments override its Mach number and incidence.

See [output definitions](../running/output.md), [configuration](../running/configuration.md), and [the complete examples](examples.md).
