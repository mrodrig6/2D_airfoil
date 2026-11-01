# 2D finite-volume airfoil solver

[![Tests](https://github.com/mrodrig6/2D_airfoil/actions/workflows/tests.yml/badge.svg)](https://github.com/mrodrig6/2D_airfoil/actions/workflows/tests.yml) [![Lint](https://github.com/mrodrig6/2D_airfoil/actions/workflows/lint.yml/badge.svg)](https://github.com/mrodrig6/2D_airfoil/actions/workflows/lint.yml) [![Formatting](https://github.com/mrodrig6/2D_airfoil/actions/workflows/format.yml/badge.svg)](https://github.com/mrodrig6/2D_airfoil/actions/workflows/format.yml) [![Spelling](https://github.com/mrodrig6/2D_airfoil/actions/workflows/spelling.yml/badge.svg)](https://github.com/mrodrig6/2D_airfoil/actions/workflows/spelling.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This Python program computes the airflow around the cross-section of a wing, including the shock waves that form when the air approaches the speed of sound. It is a teaching code for aerodynamics: students change the section shape or the angle of attack and see how lift, drag and the shock respond. It is a Python port of a MATLAB teaching code, whose original is kept in [legacy/](legacy/README.md).

- Generate NACA four- and five-digit sections, or read coordinates from a file.
- Build O- and C-shaped meshes, or import triangular SU2 and Gmsh meshes.
- Compute surface pressure, lift, drag and pitching moment, with shock positions and figures.
- Sweep the angle of attack, compute exact adjoint design gradients, optimise camber, and adapt the mesh.
- Run the same physics on a NumPy reference backend, a Numba compiled backend, or JAX for gradients.

To add your own geometry or numerical component, see [extending the solver](docs/api/extending.md).

## Install

Requires Python **3.10 or newer**. NumPy is installed automatically; Numba, JAX, Matplotlib and SciPy are optional.

```bash
python -m pip install -e ".[all]"
```

For isolated environments, platform notes and developer tools, see the [installation guide](docs/getting-started/installation.md).

## Quick start

The commands and the Python snippet below were run in this repository's development environment. Their output is not reproduced here.

```bash
# Check installed dependencies and backends.
./airfoil.sh check
# Inspect a section.
./airfoil.sh geometry airfoil=2412
# Build and plot the legacy mesh.
./airfoil.sh mesh cases/grid.json figure=mesh.png
# Solve the small case and save surface pressure.
./airfoil.sh run cases/coarse.json cp=cp.png
# Compute an incidence sweep on the small mesh.
./airfoil.sh sweep cases/coarse.json alphas=-2:2:2 figure=polar.png
```

```python
from src import solve_airfoil

result = solve_airfoil("0012", mach=0.5, alpha=2.0, n_surface=96, n_radial=40,
                       far_field_radius=15.0, first_cell_height=5e-3)
assert result.converged, result.message
print(result.report())
```

The command line and the Python API are two entry points to the same solver. The walkthrough, with output and how to read it, is in the [quick start](docs/getting-started/quick-start.md); the Python API is in [library usage](docs/api/package.md).

## Testing

From a fresh checkout, after the install step above:

```bash
python -m pytest
```

This runs every test, including the slow full-size solves. For the fast subset, use `python -m pytest --fast`. The suite stores no baseline or golden files, so no command regenerates or overwrites reference data, and there is no regression mode. For what the tests check and how to run a subset, see the [test-suite page](docs/validation/tests.md).

## Documentation

The documentation index is [docs/README.md](docs/README.md). It is in-repo only; no hosted site is published.

## Citation

No paper describes this code and no citation file exists yet. The methods are referenced in the [bibliography](docs/reference/bibliography.md).

## License

[MIT](LICENSE). Copyright (c) 2026 Mauro Rodriguez Jr.
