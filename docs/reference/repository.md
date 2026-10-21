# Repository map

This table describes the top-level files and folders. Generated study output should not replace committed scientific figures.

| Path | Purpose |
|---|---|
| [`src`](../../src) | Solver modules and public Python interface |
| [`tests`](../../tests) | Existing scientific and interface assertions |
| [`docs`](../../docs) | Topic pages, site hook, figures and figure helpers |
| [`examples`](../../examples) | Worked study scripts |
| [`cases`](../../cases) | Launcher argument decks |
| [`legacy`](../../legacy) | Original MATLAB source and geometry |
| [`fv2d.sh`](../../fv2d.sh) | Bash launcher |
| [`pyproject.toml`](../../pyproject.toml) | Package metadata and dependency extras |
| [`README.md`](../../README.md) | Short front door and source of the website home |
| [`LICENSE`](../../LICENSE) | MIT license |
| [`.gitignore`](../../.gitignore) | Generated-file exclusions |
| [`mkdocs.yml`](../../mkdocs.yml) | Site navigation and strict validation |
| [`.github`](../../.github) | Tests, quality checks and documentation deployment workflows |

The [code map](code-map.md) describes individual modules. The [figure inventory](figures.md) gives rebuild commands. `site/`, Python caches and local virtual environments are generated and ignored.

## Documentation migration

| Previous document | Canonical destinations |
|---|---|
| `README.md` | [problem](../model/problem.md) · [quick-start](../getting-started/quick-start.md) · [bibliography](bibliography.md) |
| `docs/README.md` | [figures](figures.md) · [plotting](../running/plotting.md) |
| `docs/adaptation.md` | [adaptation](../running/adaptation.md) · [adaptation](../method/adaptation.md) · [adaptation](../validation/adaptation.md) |
| `docs/lab.md` | [lab-guide](../running/lab-guide.md) |
| `docs/performance.md` | [backends](../method/backends.md) |
| `docs/riemann_fluxes.md` | [flux-study](../validation/flux-study.md) |
| `docs/roadmap.md` | [history](history.md) |
| `docs/theory.md` | [output](../running/output.md) · [problem](../model/problem.md) · [units](../model/units.md) · [discretization](../method/discretization.md) · [boundary-conditions](../method/boundary-conditions.md) · [euler](../theory/euler.md) · [bibliography](bibliography.md) · [time-stepping](../method/time-stepping.md) · [meshing](../method/meshing.md) · [airfoil-geometry](../theory/airfoil-geometry.md) · [legacy-forces](../validation/legacy-forces.md) |
| `docs/usage.md` | [configuration](../running/configuration.md) · [solving](../running/solving.md) · [package](../api/package.md) · [fluxes](../method/fluxes.md) · [geometry](../running/geometry.md) · [meshing](../running/meshing.md) · [mesh-import](../running/mesh-import.md) |
| `docs/verification.md` | [troubleshooting](../running/troubleshooting.md) · [tests](../validation/tests.md) · [gradients](../validation/gradients.md) · [thin-airfoil](../validation/thin-airfoil.md) · [transonic](../validation/transonic.md) · [numerical-drag](../validation/numerical-drag.md) · [far-field](../validation/far-field.md) · [legacy-forces](../validation/legacy-forces.md) |
| `docs/workflows.md` | [sweeps](../running/sweeps.md) · [sensitivity](../running/sensitivity.md) · [optimization](../running/optimization.md) · [backends](../method/backends.md) · [adjoint](../theory/adjoint.md) |
| `examples/README.md` | [examples](../getting-started/examples.md) |
| `legacy/README.md` | [legacy](legacy.md) |

Scientific derivations, lab exercises, numerical tables and research retractions were retained. Repeated introductions, manual tables of contents, duplicated parameter/gradient tables and obsolete present-tense status text were replaced by links or rewritten. Old documentation paths remain as pointer pages. No numerical implementation or existing tests were changed.

## Remaining limitations

- The source for the historical “published inviscid” NACA 0012 values is unavailable. The transonic comparison labels that column as unverified.
- Historical SLAU2 and old AUSM⁺-up experiments cannot all be reproduced from the current source. The study labels them as history.
- Some prose printed by the existing adaptation example overstates warm-start iteration savings. Example logic and printed strings remain unchanged in this documentation task.
