# Code map

This map lists module responsibilities and entry points. Read the diagram as the direction of a normal solve.

## Module reference

| Module | Responsibility | Main functions / classes |
|---|---|---|


```mermaid
flowchart TD
  CLI[cli.main] --> API[api.solve_airfoil]
  API --> CASE[api.build_case]
  CASE --> GEOM[geometry.load_airfoil]
  CASE --> MESH[mesh.build_mesh / meshio.read_mesh]
  MESH --> METRICS[metrics.build_metrics]
  API --> SOLVER[solver.solve_steady]
  SOLVER --> BK[NumPy / Numba / JAX backend]
  BK --> PHYS[physics and assembly]
  SOLVER --> POST[postprocess.forces]
  ADAPT[adapt.adaptive_solve] --> SOLVER
  ADJ[sensitivity] --> SOLVER
```
