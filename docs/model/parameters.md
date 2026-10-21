# Parameters

These tables use the defaults in `src/config.py`.

## FlowConditions

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `mach` | Free-stream Mach number, M∞ | `0.8` | dimensionless |
| `alpha_deg` | Angle of attack, α; high-level API spells this alpha | `1.25` | degrees |
| `gamma` | Specific-heat ratio, γ | `1.4` | dimensionless |
| `flux` | Interface Riemann solver | `'roe'` | choice |
| `entropy_fix` | Roe entropy-fix coefficient | `0.05` | dimensionless |

## Discretization

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `topology` | O-grid or C-grid | `'ogrid'` | choice |
| `n_surface` | Surface point count; even and at least 16 | `192` | count |
| `n_radial` | Radial cell count | `64` | count |
| `n_wake` | Wake-cut cells per side, C-grid only | `32` | count |
| `wake_length` | Outflow-plane position, C-grid only | `20.0` | chords |
| `far_field_radius` | Far-field radius | `20.0` | chords |
| `far_field_centre` | Far-field center | `(0.5, 0.0)` | chords |
| `first_cell_height` | Requested wall spacing | `0.002` | chords |
| `growth_limit` | Largest radial growth ratio | `1.25` | dimensionless |
| `orthogonality` | Wall-normal/radial blend exponent | `2.0` | dimensionless |
| `clustering` | Surface point distribution | `'cosine'` | choice |

## SolverOptions

| Name in code | Meaning / symbol | Default | Units |
|---|---|---|---|
| `backend` | Execution backend; auto prefers Numba for Euler | `'auto'` | choice |
| `batch` | Compiled steps between Python calls | `25` | count |
| `cfl` | Courant multiplier | `0.9` | dimensionless |
| `tolerance` | Convergence threshold on the scaled residual | `1e-06` | dimensionless |
| `max_iterations` | Maximum pseudo-time steps | `20000` | count |
| `scheme` | Time integrator | `'euler'` | choice |
| `print_interval` | Steps between progress lines; zero is silent | `500` | count |
| `divergence_factor` | Residual growth threshold for aborting | `10000.0` | dimensionless |

## API and CLI names

Use `alpha` in `solve_airfoil` and the CLI; use `alpha_deg` when constructing `FlowConditions`. Passing `alpha_deg` to `solve_airfoil` is rejected. `airfoil="0012"` is the high-level geometry default; `build_case(mesh_file=...)` or `solve_airfoil(mesh_file=...)` imports a mesh, while the CLI spelling is `--mesh`.

The CLI exposes a subset of the Python configuration. [Command line](../running/command-line.md) records parser defaults separately; [configuration](../running/configuration.md) explains deck overrides. [Units](units.md) defines reference scales.
