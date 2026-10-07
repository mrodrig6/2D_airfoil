# Examples

Four scripts, in the order they are meant to be read. Each one runs on its own
with no arguments and prints what it found; none needs an install, because each
puts the repository root on the import path first.

```sh
python examples/01_solve.py
```

| | Script | Needs | Roughly |
|---|---|---|---|
| **01** | [`01_solve.py`](01_solve.py) — one operating point, start to finish | nothing | 15 s |
| **02** | [`02_alpha_sweep.py`](02_alpha_sweep.py) — a lift curve from one mesh | nothing | 30 s |
| **03** | [`03_sensitivity.py`](03_sensitivity.py) — exact gradients, and what they cost | `jax` | 45 s |
| **04** | [`04_optimise.py`](04_optimise.py) — a shape optimisation driven by them | `jax`, `scipy` | 1.5 min |
| **05** | [`05_adapt.py`](05_adapt.py) — adaptive against uniform refinement | nothing | 1.5 min |
| **06** | [`06_postprocess.py`](06_postprocess.py) — every result figure: $c_p$, the Mach and entropy fields, the polar | `matplotlib` | 2 min |

Install what the last two need with `./fv2d.sh install` (or
`pip install -e ".[all]"`).

Every script prints its numbers, and only `06` writes figures — into
`figures/`, or into a directory you name:

```sh
python examples/06_postprocess.py docs/figures
```

So the first five are safe to run over ssh with no display. `docs/workflows.md` explains what each one is doing
and why it is done that way.
