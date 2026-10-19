# Where everything lives

A map of `src/`, and the dependency tree underneath it. The point of the tree
is that it has a direction: every arrow points **down** the list below, so you
can read the code in that order and never need a module you have not met yet.

## Contents

- [The layers](#the-layers)
- [The tree](#the-tree)
- [Within each subpackage](#within-each-subpackage)
- [The deferred imports](#the-deferred-imports)
- [Where to add things](#where-to-add-things)

---

## The layers

Seven layers, each depending only on the ones above it.

| | layer | what it owns |
|---|---|---|
| 1 | `config.py` | the free stream, the gas, the mesh resolution, the solver options — and the non-dimensionalisation, documented in one place |
| 2 | `geometry/` | the section and the mesh: coordinates in, connectivity out |
| 3 | `physics/` | the equations. Pure functions of states and normals; nothing about meshes |
| 4 | `spatial/` | the right-hand side: metrics, the finite-volume residual, mesh adaptation |
| 5 | `time/` | how big a step to take, and the pseudo-time march to steady state |
| 6 | `backends/` | *how* the residual gets evaluated — NumPy, Numba or JAX |
| 7 | `post/`, `api.py`, `cli.py`, `sensitivity.py` | what a user actually calls |

Layers 3 and 4 are the split worth understanding. `physics/` answers *what is
the flux through this face*, and has never heard of a mesh. `spatial/` answers
*what is $dU/dt$ in this cell*, and does the mesh bookkeeping that turns the
first answer into the second. Changing a flux touches only `physics/`;
changing the discretisation touches only `spatial/`.

## The tree

Module-level imports only; `->` reads "imports". Deferred (function-level)
imports are listed [separately](#the-deferred-imports) because they are
what keeps this tree acyclic.

```
config                   -> (nothing)

geometry/section         -> (nothing)
geometry/mesh            -> config, geometry/section
geometry/meshio          -> geometry/mesh

physics/thermo           -> (nothing)
physics/fluxes           -> physics/thermo
physics/boundary         -> physics/thermo, physics/fluxes

spatial/metrics          -> geometry/mesh
spatial/assembly         -> config, geometry/mesh, physics, spatial/metrics

time/stepping            -> config, physics, spatial/metrics
time/march               -> config, spatial, spatial/metrics, time/stepping

backends/_numba_kernels  -> (nothing)
backends/base            -> config, geometry/mesh, spatial/metrics
backends/numpy_backend   -> backends/base, spatial, time
backends/numba_backend   -> backends/base, backends/_numba_kernels, config,
                            spatial, spatial/metrics, time
backends/jax_backend     -> backends/base, config, spatial, spatial/metrics, time
backends/__init__        -> backends/base, backends/numpy_backend, config,
                            spatial/metrics

post/postprocess         -> config, geometry/mesh, physics, time/march
post/plotting            -> geometry/mesh, geometry/section

spatial/adapt            -> config, geometry/{mesh,section}, physics,
                            post/postprocess, spatial/metrics, time/march
api                      -> config, geometry/{mesh,meshio,section},
                            post/postprocess, spatial/metrics, time/march
sensitivity              -> api, config, geometry/mesh, physics, spatial,
                            spatial/metrics, time, time/march
cli                      -> api, config, geometry/{mesh,meshio,section},
                            spatial/metrics
__init__                 -> api, config, geometry/*, physics,
                            post/postprocess, spatial/{assembly,metrics},
                            time/march
__main__                 -> cli
```

`backends/_numba_kernels.py` importing nothing is the one deliberate
duplication in the tree: Numba's `nopython` mode cannot call the `xp`-injected
physics, so those kernels restate it. `tests/test_backends.py` is what holds
the two copies in step — see [`docs/performance.md`](../docs/performance.md).

## Within each subpackage

Each `__init__.py` re-exports its subpackage's public names, so
`from src.physics import roe_flux` works without knowing that it lives in
`physics/fluxes.py`. Import from the subpackage when you only want the name,
and from the module when the file is the point.

```
geometry/   section.py      NACA four- and five-digit families, .dat files,
                            resampling, surface normals
            mesh.py         O-grid and C-grid generation, triangulation,
                            connectivity, boundary tags
            meshio.py       SU2 and Gmsh readers, SU2 writer

physics/    thermo.py       primitive <-> conservative, the ideal-gas
                            relations, the Euler flux itself
            fluxes.py       roe, hllc, ausmup, and the dispatch between them
            boundary.py     the inviscid wall and the characteristic far field

spatial/    metrics.py      areas, face lengths, face normals, edge lists
            assembly.py     the face loop: flux in, dU/dt out
            adapt.py        the error indicator and adaptive refinement

time/       stepping.py     local time step, residual norm, component weights
            march.py        the pseudo-time march, convergence, divergence

backends/   base.py         the Backend interface and element_faces
            numpy_backend.py    vectorised NumPy: the reference
            numba_backend.py    compiled, parallel: the default
            jax_backend.py      traceable: for sensitivity, not speed
            _numba_kernels.py   the nopython restatement of the physics

post/       postprocess.py  lift, drag, moment, surface pressure, fields
            plotting.py     matplotlib figures (optional dependency)
```

## The deferred imports

A number of imports sit inside functions rather than at module top level, for
two reasons: to break a cycle, or to keep an optional dependency optional.
These are the ones that matter.

| in | imports | why |
|---|---|---|
| `time/march.py` | `post.postprocess` | `SolveResult.report()` formats forces, but `post/postprocess` needs `SolveResult`. One of the two has to be deferred, and the march is the lower layer. |
| `post/plotting.py` | `post.postprocess` | the same pairing one layer up: a figure needs the field, the field does not need the figure. |
| `spatial/adapt.py` | `api` | refinement drives whole solves, which is what `api` is for — but `api` is layer 7 and `adapt` is layer 4. |
| `time/march.py`, `cli.py` | `backends` | importing a backend imports Numba or JAX. Deferring it means `import src` costs neither, and `fv2d.sh --help` does not pay for a JIT compiler. |
| `sensitivity.py` | `backends.jax_backend` | the adjoint needs JAX; nothing else does. |
| `src/__init__.py` | `post.plotting` | matplotlib is an optional dependency, so `src.plotting` is a lazy attribute rather than an import. |

Running

```bash
python - <<'EOF'
import ast, pathlib
for p in sorted(pathlib.Path("src").rglob("*.py")):
    tree = ast.parse(p.read_text())
    top = {id(n) for n in tree.body}
    for parent in ast.walk(tree):
        for n in ast.iter_child_nodes(parent):
            if isinstance(n, ast.ImportFrom) and n.level and id(n) not in top:
                print(p, "->", n.module)
EOF
```

prints the current list, which is the authority if this table has drifted.

## Where to add things

- **A new approximate Riemann solver** → `physics/fluxes.py`, then register it
  in `SOLVERS` and in `FLUXES` in `config.py`. If it should be fast, it also
  needs a kernel in `backends/_numba_kernels.py` and a row in
  `tests/test_backends.py`.
- **A new boundary condition** → `physics/boundary.py`, a tag in
  `geometry/mesh.py`, and a branch in `spatial/assembly.py`.
- **A new time stepper** → `time/stepping.py` for its stability limit,
  `time/march.py` for the dispatch.
- **A new mesh topology** → `geometry/mesh.py` and `MeshTopology`.
- **A new derived quantity** → `post/postprocess.py`, and a figure in
  `post/plotting.py` if it is worth looking at.

Anything that needs a *new* layer probably belongs in `spatial/` or `time/`;
the test for which is whether it would still make sense if the time stepper
changed.
