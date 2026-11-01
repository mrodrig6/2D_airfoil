# Troubleshooting

Use the error or convergence message to identify the failing condition. The table and cases below point to the relevant remedy and model limit.

## Troubleshooting

**The march diverges.** The solver stops and returns a result with `converged=False`; the CLI exits nonzero. Do not use the returned loads as a converged solution. In order of likelihood: the CFL is too high (the default 0.9 is near
the explicit limit, so try 0.5); or the mesh has a bad cell, which
`./airfoil.sh mesh` will report before you ever solve.

**It converges to the wrong answer.** Check free-stream preservation first —
`pytest tests/test_solver.py -k free_stream`. If that passes, the assembly and
the boundary conditions are sound and the problem is resolution.

**It will not converge but does not diverge either.** The residual falls a few
orders and then sits there. That is a limit cycle. AUSM⁺-up is shipped but can stall at low Mach on fine meshes. SLAU2 is not shipped. With Roe or HLLC, inspect mesh quality and compare the other supported flux.

**The mesh builder refuses.** The message names the four numbers that would fix
it. See [meshing](meshing.md#meshing-it).

**The drag looks far too large.** It probably is; see
[spurious drag](../validation/numerical-drag.md#spurious-drag-and-what-first-order-costs) above.

## Exception message reference

These are the explicit exception messages raised by the current source. Braced fields represent runtime values, not literal output. The linked line identifies the implementing function.

| Exception / message template | Implementation | Action |
|---|---|---|
| `AttributeError`: `module {__name__} has no attribute {name}` | [src/__init__.py](../../src/__init__.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `ValueError`: `fraction must be in (0, 1], got {fraction}` | [src/adapt.py](../../src/adapt.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `RuntimeError`: `element {element} descends from no solved cell` | [src/adapt.py](../../src/adapt.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `TypeError`: `build_case takes geometry and mesh options only; {sorted(flow_keys / solver_keys.keys())} belong to the solve` | [src/api.py](../../src/api.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `TypeError`: `{sorted(geometry / mesh_keys.keys())} would change the mesh, but a prepared case was given; build a new case instead` | [src/api.py](../../src/api.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `TypeError`: `unknown option {key}; known options are {known}` | [src/api.py](../../src/api.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `unknown backend {name}; choose from {available_backends()}` | [src/backends/__init__.py](../../src/backends/__init__.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ImportError`: `the 'numba' backend needs Numba installed: pip install numba` | [src/backends/__init__.py](../../src/backends/__init__.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ImportError`: `the 'jax' backend needs JAX installed: pip install jax` | [src/backends/__init__.py](../../src/backends/__init__.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ValueError`: `element {bad} has {slot[bad]} faces, not 3` | [src/backends/base.py](../../src/backends/base.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ValueError`: `an element has more than three faces; the mesh is not triangular` | [src/backends/base.py](../../src/backends/base.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ValueError`: `the jax backend implements forward Euler only, not {options.scheme}` | [src/backends/jax_backend.py](../../src/backends/jax_backend.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ValueError`: `the numba backend implements forward Euler only, not {options.scheme}; use backend='numpy' for the multi-stage schemes` | [src/backends/numba_backend.py](../../src/backends/numba_backend.py) | Install the named backend extra and select a supported scheme; see compatibility. |
| `ValueError`: `no angles to sweep; pass --alphas` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `unknown objective {args.objective}; choose from {OBJECTIVES}` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `SystemExit`: `--window needs four comma-separated numbers, or 'all'` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `expected start:stop:step with a non-zero step, got {text}` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `the step in {text} points away from {stop}` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `the adjoint needs JAX installed: pip install jax ({exc})` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `SystemExit`: `--window needs four comma-separated numbers, or 'all'` | [src/cli.py](../../src/cli.py) | Check the command/keyword spelling and rebuild the case if geometry or mesh changes. |
| `ValueError`: `mach must be positive, got {self.mach}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `gamma must exceed 1, got {self.gamma}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown flux {self.flux}; choose from {FLUXES}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `entropy_fix must be non-negative, got {self.entropy_fix}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown topology {self.topology}; use one of {TOPOLOGIES}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `n_surface must be even and at least 16, got {self.n_surface}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `n_radial must be at least 4, got {self.n_radial}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `far_field_radius must exceed 2 chords, got {self.far_field_radius}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `first_cell_height must lie in (0, 0.1) chords, got {self.first_cell_height}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `growth_limit must exceed 1, got {self.growth_limit}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `orthogonality must be non-negative, got {self.orthogonality}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown clustering {self.clustering}; use cosine or uniform` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown backend {self.backend}; choose auto or one of {BACKENDS}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `batch must be at least 1, got {self.batch}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `cfl must be positive, got {self.cfl}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `tolerance must be positive, got {self.tolerance}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `max_iterations must be at least 1, got {self.max_iterations}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown scheme {self.scheme}; choose from {SCHEMES}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `n_wake must be at least 2, got {self.n_wake}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `wake_length must put the outflow plane more than 2 chords aft of the leading edge, got {self.wake_length}` | [src/config.py](../../src/config.py) | Set this parameter within the range in the message; check the parameters table. |
| `ValueError`: `unknown clustering {clustering}; use one of {CLUSTERINGS}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `cannot resolve airfoil {spec}: it is neither a NACA four-digit code nor an existing coordinate file` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the path has a repeated point, so an edge has zero length` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the path doubles back on itself; cannot define a normal` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `points must have shape (n, 2), got {pts.shape}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the coordinates have zero chord` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `n_points must be even and at least 16, got {n_points}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `n_points must be even and at least 16, got {n_points}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `thickness must be positive, got {thickness}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `a cambered section needs 0 < camber_position < 1, got {camber_position}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `not a NACA four-digit code: {code}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `NACA {text} has zero thickness` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `NACA {text} is cambered but its camber position is {p}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `not a NACA five-digit code: {code}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `NACA {text} has zero thickness` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `NACA {text}: the third digit selects a plain (0) or reflexed (1) camber line; {reflexed} is neither` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `NACA {text}: camber position {position} is not tabulated for a {'reflexed' if reflexed else 'plain'} five-digit camber line.  Available: {sorted(table)}, putting the crest at {sorted((q / 20 for q in table))} chord.` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `not a NACA four- or five-digit code: {code}.  The six-series and the modified four-digit sections have no closed form, so read those from a coordinate file instead.` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{path}: found only {len(pairs)} coordinate lines` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `points must have shape (n, 2), got {pts.shape}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `an airfoil needs at least 16 points, got {pts.shape[0]}` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{path}: header asks for {n_upper}+{n_lower} points but only {len(body)} follow` | [src/geometry/section.py](../../src/geometry/section.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `every first cell height must be positive` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `cannot reach the outer boundary at station {worst} of {h1.size}: {n_cells} cells starting at {first_height[worst]} and growing by at most {growth_limit} span only {reach[worst] * total_length[worst]} of {total_length[worst]}.  Raise n_radial, raise first_cell_height, raise growth_limit, or shrink the domain.` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `n_cells must be positive, got {n_cells}` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `first_height must be positive, got {first_height}` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `cannot reach the far field: {n_cells} cells starting at {first_height} and growing by at most {growth_limit} span only {span(growth_limit) * total_length} of {total_length}.  Raise n_radial, raise first_cell_height, raise growth_limit, or shrink far_field_radius.` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the outer boundary touches the inner one somewhere` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the wall normal and the outward direction are opposed somewhere on the inner path; move far_field_centre inside the section` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `far_field_centre coincides with the trailing edge` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `an edge is shared by more than two elements; the mesh is not manifold` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{int(disagree.sum())} boundary edges join nodes with different tags; a boundary edge must lie wholly on one boundary` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `a boundary edge has untagged nodes, so it gets no boundary condition` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `wake_length {disc.wake_length} is not aft of the trailing edge at x = {trailing_edge[0]}` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `wake_length {disc.wake_length} must be aft of far_field_centre x = {centre[0]}, or the outer boundary turns back on itself` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `AssertionError`: `inner {inner.shape} and outer {outer.shape} must match` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `AssertionError`: `the two sides of the wake cut are not coincident` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{bad.size} of {self.n_elements} cells have non-positive area; the worst is element {worst} at ({centre[0]}, {centre[1]}) with area {area[worst]}.  The grid has folded: lower 'orthogonality', raise 'n_surface', or raise 'far_field_radius'.` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{self.n_nodes - used.size} of {self.n_nodes} nodes belong to no element` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `the {tag.name} boundary is not a closed loop` | [src/geometry/mesh.py](../../src/geometry/mesh.py) | Check geometry, resolution and the constraints named in the message before rebuilding the mesh. |
| `ValueError`: `{source}: no triangles found` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{source}: no boundary markers found, so no boundary conditions` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{source}: cannot map boundary marker(s) {unknown} onto a boundary condition.  Markers in this file: {sorted(lines)}.  Rename them, or pass wall_names=/farfield_names= naming them explicitly.` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{source}: no marker maps to {sorted(missing)}` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: no $MeshFormat section; is this a Gmsh file?` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: Gmsh format {version} is not supported, only 2.2 ASCII.  Re-export with:  gmsh -2 -format msh2 <input> -o {path.name}` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: unknown mesh extension {path.suffix}; expected .su2 or .msh` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{source}: a triangle refers to a node that does not exist` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: NDIME={value}; this solver is two-dimensional` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: element type {kind} is neither a triangle nor a boundary line; this solver needs a purely triangular mesh` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: element type {code} is not a triangle; this solver needs a purely triangular mesh` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `{path}: MARKER_ELEMS before any MARKER_TAG` | [src/geometry/meshio.py](../../src/geometry/meshio.py) | Use triangular SU2 or Gmsh 2.2 ASCII input, and map all boundary names. |
| `ValueError`: `a face has zero length, so it has no normal` | [src/spatial/metrics.py](../../src/spatial/metrics.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `ValueError`: `the discrete Gauss identity fails by {error} (scale {scale}); the faces and the elements disagree about this mesh` | [src/spatial/metrics.py](../../src/spatial/metrics.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `ValueError`: `unknown flux {name}; choose from {tuple(SOLVERS)}` | [src/physics/fluxes.py](../../src/physics/fluxes.py) | Check input shapes, mesh validity and supported options at the linked implementation. |
| `ValueError`: `{name} is not monotonic in lightness, so it manufactures edges that are not in the data -- on a transonic field that is indistinguishable from the shock.  Use {FIELD_CMAP}, 'viridis' or 'cividis'.` | [src/post/plotting.py](../../src/post/plotting.py) | Select a supported scalar or profile and provide a compatible converged result. |
| `ImportError`: `plotting needs matplotlib, which is optional: pip install matplotlib (or ./airfoil.sh install)` | [src/post/plotting.py](../../src/post/plotting.py) | Select a supported scalar or profile and provide a compatible converged result. |
| `ValueError`: `unknown shading {shading}; choose 'smooth' or 'flat'` | [src/post/plotting.py](../../src/post/plotting.py) | Select a supported scalar or profile and provide a compatible converged result. |
| `ValueError`: `unknown scalar {name}; choose from {SCALARS}` | [src/post/postprocess.py](../../src/post/postprocess.py) | Select a supported scalar or profile and provide a compatible converged result. |
| `ValueError`: `unknown objective {name}; choose from {OBJECTIVES}` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `ValueError`: `unknown design variable(s) {sorted(unknown)}; choose from {DESIGN_VARIABLES}` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `RuntimeError`: `the adjoint did not converge in {blocks * self.adjoint_batch} sweeps (scaled defect {residual}); the forward solve may be only marginally converged` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `ValueError`: `unknown design variable(s) {sorted(unknown)}` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `ValueError`: `shape derivatives need a four-digit section, because the parameters have to be exact; {self.case.airfoil.name} has none. Differentiate {FLOW_VARIABLES} instead, or rebuild the case from a NACA code.` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `RuntimeError`: `the flow solve did not converge ({self.result.message}), so a gradient taken from it would be the derivative of nothing in particular` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `RuntimeError`: `the adjoint diverged within the first {completed * self.adjoint_batch} sweeps` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |
| `RuntimeError`: `the {key}{sign} solve did not converge` | [src/sensitivity.py](../../src/sensitivity.py) | Use supported objectives/variables and a converged forward state; see sensitivity and gradient checks. |

For range details, use [parameters](../model/parameters.md); for combinations, use [compatibility](../reference/compatibility.md).
