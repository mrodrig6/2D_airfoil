# Where everything lives

A map of `src/`, and the dependency tree underneath it. The point of the tree
is that it has a direction: every module depends only on modules **above** it
in the order below, so you can read the code in that order and never meet a
name you have not already seen.

## Contents

- [The layers](#the-layers)
- [The tree](#the-tree)
- [Within each subpackage](#within-each-subpackage)
- [The deferred imports](#the-deferred-imports)
- [Checking this page](#checking-this-page)
- [Where to add things](#where-to-add-things)

---

## The layers

| | layer | what it owns |
|---|---|---|
| 1 | `config.py` | the free stream, the gas, the mesh resolution, the solver options — and the non-dimensionalisation, documented in one place |
| 2 | `geometry/` | the section and the mesh: coordinates in, connectivity out |
| 3 | `physics/` | the equations. Pure functions of states and normals; nothing about meshes |
| 4 | `spatial/` | the right-hand side: metrics, and the finite-volume residual built on them |
| 5 | `timestepping/` | how big a step to take, and the pseudo-time march to steady state |
| 6 | `backends/` | *how* the residual gets evaluated — NumPy, Numba or JAX |
| 7 | `post/`, `api.py`, `cli.py`, `sensitivity.py`, `adapt.py` | the drivers: what a user actually calls |

Layers 3 and 4 are the split worth understanding. `physics/` answers *what is
the flux through this face*, and has never heard of a mesh. `spatial/` answers
*what is $dU/dt$ in this cell*, and does the mesh bookkeeping that turns the
first answer into the second. Changing a flux touches only `physics/`;
changing the discretisation touches only `spatial/`.

Layer 7 is not a subpackage, because its members have nothing in common except
being entry points. `adapt.py` is there rather than in `spatial/` for that
reason: refinement is a spatial question, but `adaptive_solve` drives whole
solves, so it sits with the other drivers rather than underneath them.

## The tree

Module-level imports only; `->` reads "imports". Deferred (function-level)
imports are listed [separately](#the-deferred-imports), because they are what
keeps this tree acyclic.

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

timestepping/step        -> config, physics, spatial/metrics
timestepping/march       -> config, spatial, spatial/metrics, timestepping/step

backends/_numba_kernels  -> (nothing)
backends/base            -> config, geometry/mesh, spatial/metrics,
                            timestepping/step
backends/numpy_backend   -> backends/base, spatial, timestepping
backends/numba_backend   -> backends/base, backends/_numba_kernels, config,
                            spatial, spatial/metrics, timestepping
backends/jax_backend     -> backends/base, config, spatial, spatial/metrics,
                            timestepping
backends/__init__        -> backends/base, backends/numpy_backend, config,
                            spatial/metrics

post/postprocess         -> config, geometry/mesh, physics, timestepping/march
post/plotting            -> geometry/mesh, geometry/section
api                      -> config, geometry/{mesh,meshio,section},
                            post/postprocess, spatial/metrics,
                            timestepping/march
adapt                    -> config, geometry/{mesh,section}, physics,
                            post/postprocess, spatial/metrics,
                            timestepping/march
sensitivity              -> api, config, geometry/mesh, physics, spatial,
                            spatial/metrics, timestepping, timestepping/march
cli                      -> api, config, geometry/{mesh,meshio,section},
                            spatial/metrics
__init__                 -> api, config, geometry/*, physics,
                            post/postprocess, spatial/{assembly,metrics},
                            timestepping/march
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
geometry/      section.py    NACA four- and five-digit families, .dat files,
                             resampling, surface normals
               mesh.py       O-grid and C-grid generation, triangulation,
                             connectivity, boundary tags
               meshio.py     SU2 and Gmsh readers, SU2 writer

physics/       thermo.py     primitive <-> conservative, the ideal-gas
                             relations, the Euler flux itself
               fluxes.py     roe, hllc, ausmup, and the dispatch between them
               boundary.py   the inviscid wall and the characteristic far field

spatial/       metrics.py    areas, face lengths, face normals, edge lists
               assembly.py   the face loop: flux in, dU/dt out

timestepping/  step.py       local time step, residual norm, component weights
               march.py      the pseudo-time march, convergence, divergence

backends/      base.py            the Backend interface and element_faces
               numpy_backend.py   vectorised NumPy: the reference
               numba_backend.py   compiled, parallel: the default
               jax_backend.py     traceable: for sensitivity, not speed
               _numba_kernels.py  the nopython restatement of the physics

post/          postprocess.py  lift, drag, moment, surface pressure, fields
               plotting.py     matplotlib figures (optional dependency)
```

`post/` re-exports `postprocess` but not `plotting`, because matplotlib is an
optional dependency and importing the package should not require it.
`src.plotting` is a lazy attribute on the top-level package for the same
reason.

## The deferred imports

Several imports sit inside functions rather than at module top level, for two
reasons: to break a cycle, or to keep an optional dependency optional. These
are the ones that matter.

| in | imports | why |
|---|---|---|
| `timestepping/march.py` | `post.postprocess` | `SolveResult.report()` formats forces, but `post/postprocess` needs `SolveResult`. One of the two has to be deferred, and the march is the lower layer. |
| `post/plotting.py` | `post.postprocess` | the same pairing one layer up: a figure needs the field, the field does not need the figure. |
| `adapt.py` | `api` | refinement drives whole solves, and `api` drives one. Both are layer 7, so the order between them is arbitrary; deferring keeps it from mattering. |
| `timestepping/march.py`, `cli.py` | `backends` | importing a backend imports Numba or JAX. Deferring it means `import src` costs neither, and `fv2d.sh --help` does not pay for a JIT compiler. |
| `sensitivity.py` | `backends.jax_backend` | the adjoint needs JAX; nothing else does. |
| `src/__init__.py` | `post.plotting` | matplotlib is optional, as above. |

`cli.py` defers most of its imports for the fourth reason.

## Checking this page

A dependency tree in prose rots. These two commands print the current truth,
and are what the tables above were generated from.

The tree, split into eager and deferred:

```bash
python - <<'PY'
import ast, pathlib
root = pathlib.Path("src")
for p in sorted(root.rglob("*.py")):
    rel = p.relative_to(root); pkg = list(rel.parts[:-1])
    name = str(rel.with_suffix("")).replace("/", ".").replace(".__init__", "")
    tree = ast.parse(p.read_text(encoding="utf-8")); top = {id(n) for n in tree.body}
    eager, lazy = set(), set()
    for parent in ast.walk(tree):
        for n in ast.iter_child_nodes(parent):
            if isinstance(n, ast.ImportFrom) and n.level:
                base = pkg[: len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                mods = ([".".join(base + (n.module or "").split("."))] if n.module
                        else [".".join(base + [a.name]) for a in n.names])
                (eager if id(n) in top else lazy).update(m for m in mods if m and m != name)
    print(f"{name:<24} eager: {', '.join(sorted(eager)) or '-'}")
    if lazy - eager:
        print(f"{'':<24} lazy : {', '.join(sorted(lazy - eager))}")
PY
```

And the layer invariant — that no module eagerly imports one below it. On a
tree that holds, it prints nothing:

```bash
python - <<'PY'
import ast, pathlib
LAYER = {"config": 1, "geometry": 2, "physics": 3, "spatial": 4,
         "timestepping": 5, "backends": 6}
def layer(m):
    return LAYER.get(m.split(".")[0], 7)
root = pathlib.Path("src")
for p in sorted(root.rglob("*.py")):
    rel = p.relative_to(root); pkg = list(rel.parts[:-1])
    name = str(rel.with_suffix("")).replace("/", ".").replace(".__init__", "")
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.level:
            base = pkg[: len(pkg) - (n.level - 1)] if n.level > 1 else pkg
            mods = ([".".join(base + (n.module or "").split("."))] if n.module
                    else [".".join(base + [a.name]) for a in n.names])
            for m in mods:
                if m and layer(m) > layer(name):
                    print(f"{name} (L{layer(name)}) -> {m} (L{layer(m)})")
PY
```

## Where to add things

- **A new approximate Riemann solver** → `physics/fluxes.py`, then register it
  in `SOLVERS` and in `FLUXES` in `config.py`. If it should be fast, it also
  needs a kernel in `backends/_numba_kernels.py` and a row in
  `tests/test_backends.py`.
- **A new boundary condition** → `physics/boundary.py`, a tag in
  `geometry/mesh.py`, and a branch in `spatial/assembly.py`.
- **A new time stepper** → `backends/base.py` for the stage combination (all
  three backends inherit it from there), `timestepping/step.py` for its
  measured stability limit, and `SCHEMES` in `config.py`. If it should be fast,
  the Numba backend's fused path needs a matching branch, which
  `tests/test_backends.py` holds in step by running every scheme on every
  backend.
- **A new mesh topology** → `geometry/mesh.py` and `MeshTopology`.
- **A new derived quantity** → `post/postprocess.py`, and a figure in
  `post/plotting.py` if it is worth looking at.
- **A new entry point** → the top level, next to `api.py`. If it drives
  solves, it belongs with the drivers rather than in the layer whose subject
  it happens to touch; that is why `adapt.py` is where it is.

Anything that needs a *new* layer probably belongs in `spatial/` or
`timestepping/`; the test for which is whether it would still make sense if
the time stepper changed.
