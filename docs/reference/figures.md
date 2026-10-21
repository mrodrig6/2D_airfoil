# Figures

This inventory records image sources and rebuild commands. Run commands from the repository root unless stated otherwise.

## Rebuilding the figures

Every figure is produced by the code itself rather than drawn, so a figure that
stops matching the code is a one-command fix:

```bash
./fv2d.sh mesh @grid figure=docs/figures/mesh.png
./fv2d.sh geometry airfoil=0012 figure=docs/figures/section.png normals=yes
./fv2d.sh mesh @cgrid \
    figure=docs/figures/cgrid.png window=-0.6,2.6,-0.9,0.9
./fv2d.sh adapt naca=0012 ns=64 nr=24 R=10 h1=0.02 mach=0.8 alpha=1.25 \
    cycles=6 tolerance=1e-6 max_iterations=80000 print_interval=0 \
    figure=docs/figures/adapted.png window=-0.2,1.2,-0.5,0.5
```

The *result* figures — surface pressure, the Mach field and the entropy field,
at two operating points, plus the polar — all come from one script, because they
all come from the same three solves:

```bash
python examples/06_postprocess.py docs/figures
```

The `_transonic` figures are the legacy point, $M_\infty = 0.8$ at
$1.25^\circ$, which [tests](../validation/tests.md) tabulates. The `_shock`
figures are $M_\infty = 0.85$ at $1^\circ$, where both surfaces go supersonic
and each pocket ends in a shock; those are the ones in the top-level README,
because the shock is unmistakable in them.

**Field figures are smooth-shaded by default**, which interpolates cell values
onto nodes and contours them. That is a choice: `shading="flat"` draws one
colour per cell, which is exactly what the solver holds — and draws the mesh
with it, so the picture becomes as much about the grid as about the flow. Use
`flat` when the question is about the discretisation:

```bash
./fv2d.sh solve naca=0012 mach=0.85 alpha=1 figure=cells.png shading=flat
```

The documentation workflow builds the site from committed images. Figure refresh remains a separate reviewed step after geometry, plotting or solver changes.

## Adjacent source recipe

[rebuild.py](../figures/rebuild.py) rebuilds current section, mesh, pressure, field, polar and adaptation figures. Its inputs are the existing CLI and postprocess example; it does not change solver code.

```bash
# Rebuild current figures without overwriting committed images.
python docs/figures/rebuild.py --output /tmp/airfoil-figures
```

| Image | Source |
|---|---|
| `section.png`, `mesh.png`, `cgrid.png` | The geometry / `@grid` / `@cgrid` CLI recipes |
| `cp_*.png`, `mach_*.png`, `entropy_*.png`, `polar.png` | `examples/06_postprocess.py`, called by the adjacent recipe |
| `adapted.png` | Six Mach-jump adaptive cycles from the documented coarse grid |
| `convergence.png`, `flux_convergence.png` | Retained historical artifacts; the exact removed-code harness is unavailable |

The current recipe is reproducible, but no byte-for-byte match to historical exports is promised. The flux image includes obsolete implementations; its [study](../validation/flux-study.md) labels those results as historical. Missing historical provenance remains an explicit limitation, not an invented rebuild command.
