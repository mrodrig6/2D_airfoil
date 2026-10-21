# Examples

Run these scripts from the repository root with `[all]` installed. The scripts are the complete recipes; the linked workflow pages explain their choices.

| Script | Command | Additional requirements | Purpose |
|---|---|---|---|
| [01_solve.py](../../examples/01_solve.py) | `python examples/01_solve.py` | runtime dependencies | One operating point, start to finish: section, mesh, solve, loads. |
| [02_alpha_sweep.py](../../examples/02_alpha_sweep.py) | `python examples/02_alpha_sweep.py` | runtime dependencies | A lift curve from **one** mesh, and the slope it should have. |
| [03_sensitivity.py](../../examples/03_sensitivity.py) | `python examples/03_sensitivity.py` | jax | Exact design gradients, and the cost argument for taking them this way. |
| [04_optimise.py](../../examples/04_optimise.py) | `python examples/04_optimise.py` | jax, SciPy | Inverse design driven by the adjoint: which camber line gives cl = 0.6? |
| [05_adapt.py](../../examples/05_adapt.py) | `python examples/05_adapt.py` | runtime dependencies | Adaptive refinement against uniform refinement, on the same case. |
| [06_postprocess.py](../../examples/06_postprocess.py) | `python examples/06_postprocess.py` | plot | Every figure this code draws about a *result*, from one script. |

## Output and convergence

The first five scripts print results. `06_postprocess.py` writes pressure, field and polar images to `figures/`, or to the output directory supplied as its argument.

For headless runs, set `MPLBACKEND=Agg`. Treat reported nonconvergence as a failed operating point even when a study script exits zero. Full adaptation, optimization and convergence studies cost more than a single solve; hardware and first-use compilation affect wall time.

## Common recipes

```bash
# Run the test suite before changing a study.
./fv2d.sh test
# Inspect a geometry before a full solve.
./fv2d.sh geometry airfoil=2412
# Recheck tools after changing the environment.
./fv2d.sh check
```

See [solving](../running/solving.md), [sweeps](../running/sweeps.md), [sensitivities](../running/sensitivity.md), and [optimization](../running/optimization.md).
