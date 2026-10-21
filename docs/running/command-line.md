# Command line

The Bash launcher expands decks and `name=value` arguments. The Python CLI uses ordinary flags; its parser is implemented in `src/cli.py`.

## Commands

| Python subcommand | Purpose |
|---|---|
| `check` | Report optional tools |
| `geometry` | Inspect geometry |
| `mesh` | Build, read or export a mesh |
| `solve` | Solve one point |
| `sweep` | Sweep incidence |
| `sensitivity` | Compute adjoint derivatives |
| `adapt` | Solve and refine |
| `bench` | Measure backend throughput |
| `flow` | Report free-stream scales |

| Launcher-only command | Purpose |
|---|---|
| `install [extra]` | Editable pip installation; defaults to `all` |
| `list` | List case decks |
| `help` | Show launcher help |
| `test [pytest arguments]` | Run existing tests |
| `verify [--fast]` | Tests, lint, format and spelling |
| `docs` | Print documentation entry points |


## Flags and positional arguments

Defaults below are parser defaults, which can differ from dataclass defaults or deck values. See [configuration precedence](configuration.md) and [parameters](../model/parameters.md).

| Flag / argument | Commands accepting it | Meaning / choices | Parser default |
|---|---|---|---|
| `--airfoil` | `geometry`, `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | a NACA four- or five-digit code, or a path to a Selig/Lednicer .dat file (default: %(default)s) | `'0012'` |
| `--n-surface` | `geometry`, `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | points around the section, even (default: %(default)s) | `192` |
| `--clustering` | `geometry`, `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | surface point distribution (default: %(default)s) Choices: `cosine`, `uniform`. | `'cosine'` |
| `--figure` | `geometry`, `mesh`, `solve`, `sweep`, `adapt` | write a PNG here | `None` |
| `--normals` | `geometry` | draw the outward normals on the figure | `False` |
| `--dat` | `geometry` | write the section as a Selig .dat file here | `None` |
| `--topology` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | 'ogrid' wraps a loop round the section; 'cgrid' cuts along the wake (default: %(default)s) Choices: `ogrid`, `cgrid`. | `'ogrid'` |
| `--n-wake` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | cgrid: cells along each side of the wake cut (default: %(default)s) | `32` |
| `--wake-length` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | cgrid: outflow plane, in chords (default: %(default)s) | `20.0` |
| `--n-radial` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | cells from the wall to the far field (default: %(default)s) | `64` |
| `--far-field-radius` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | far-field radius in chords (default: %(default)s) | `20.0` |
| `--first-cell-height` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | first cell height in chords (default: %(default)s) | `0.002` |
| `--growth-limit` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | largest radial growth ratio (default: %(default)s) | `1.25` |
| `--orthogonality` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | wall-normal to radial blend exponent (default: %(default)s) | `2.0` |
| `--mesh` | `mesh`, `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | read a .su2 or .msh mesh instead of generating an O-grid | `None` |
| `--window` | `mesh`, `adapt` | figure extent 'xmin,xmax,ymin,ymax', or 'all' (default: %(default)s) | `'-0.6,1.6,-0.7,0.7'` |
| `--su2` | `mesh` | write the mesh as SU2 here | `None` |
| `--mach` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench`, `flow` | Free-stream Mach number, M∞ | `0.8` |
| `--alpha` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench`, `flow` | angle of attack in degrees (default: %(default)s) | `1.25` |
| `--gamma` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench`, `flow` | Specific-heat ratio, γ | `1.4` |
| `--flux` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | the approximate Riemann solver (default: %(default)s) Choices: `roe`, `hllc`, `ausmup`. | `'roe'` |
| `--entropy-fix` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | roe only: Harten-Hyman delta (default: %(default)s) | `0.05` |
| `--cfl` | `solve`, `sweep`, `sensitivity`, `adapt`, `bench` | Courant multiplier; None selects the measured recommendation | `0.9` |
| `--tolerance` | `solve`, `sweep`, `sensitivity`, `adapt` | Convergence threshold on the scaled residual | `1e-06` |
| `--max-iterations` | `solve`, `sweep`, `sensitivity`, `adapt` | Maximum pseudo-time steps | `20000` |
| `--scheme` | `solve`, `sweep`, `sensitivity`, `adapt` |  Choices: `euler`, `ssprk3`, `rk4`. | `'euler'` |
| `--print-interval` | `solve`, `sweep`, `sensitivity`, `adapt` | Steps between progress lines; zero is silent | `500` |
| `--backend` | `solve`, `sweep`, `adapt` | execution strategy (default: %(default)s) Choices: `auto`, `numba`, `numpy`, `jax`. | `'auto'` |
| `--batch` | `solve` | compiled-backend steps per call (default: %(default)s) | `25` |
| `--cp` | `solve` | write the surface pressure distribution here | `None` |
| `--scalar` | `solve` | which field (default: %(default)s) | `'mach'` |
| `--shading` | `solve` | 'flat' draws one colour per cell, which is what the solver holds and also draws the mesh; 'smooth' contours it (default: %(default)s) Choices: `smooth`, `flat`. | `'smooth'` |
| `--convergence` | `solve` | write the residual history here | `None` |
| `--alphas` | `sweep` | angles in degrees: a list '-4,0,4' or a range 'start:stop:step' (default: %(default)s).  This replaces --alpha, which a sweep ignores | `'-4:8:2'` |
| `--objective` | `sensitivity` | cl, cd, cm or efficiency (default: %(default)s) | `'cl'` |
| `--variables` | `sensitivity` | comma-separated; default is everything the case offers | `None` |
| `--check` | `sensitivity` | also take the finite differences the adjoint replaces -- two full solves per variable | `False` |
| `--step` | `sensitivity` | finite-difference step (default: %(default)s) | `0.0001` |
| `--cycles` | `adapt` | solve-refine cycles (default: %(default)s) | `5` |
| `--fraction` | `adapt` | fraction of cells refined per cycle (default: %(default)s) | `0.05` |
| `--history` | `adapt` | write the loads against cell count here | `None` |
| `--steps` | `bench` | compiled-backend steps to time (default: %(default)s) | `2000` |

`-h` / `--help` works on every parser. The root `--version` flag reports the package version. Do not pass a solve deck to a geometry-only command merely because the deck names the same geometry.

## Launcher spelling

Underscores in `name=value` become hyphens in Python flags. Aliases are `naca=airfoil`, `ns=n_surface`, `nr=n_radial`, `R=far_field_radius`, `h1=first_cell_height`, and `aoa=alpha`. `M` is not an alias.

Boolean flags without value-taking parser actions should be passed as literal flags. The launcher also translates `check=yes` for the sensitivity gradient check.

`FV2D_PYTHON` selects the launcher's interpreter. Otherwise it looks for `python3`, then `python`. The installed `fvairfoil` command and `python -m src` use the Python subcommands above; launcher-only commands are not interchangeable with them.
