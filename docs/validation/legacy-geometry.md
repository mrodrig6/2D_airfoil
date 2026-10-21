# Legacy geometry

The Python NACA 0012 uses the same closed-trailing-edge polynomial as the MATLAB geometry. The comparison is geometric; it does not establish equivalence of the two solvers.

`tests/test_geometry.py:test_naca0012_reproduces_the_legacy_geometry` reads `legacy/nacageom.mat` with SciPy and compares surface coordinates against `src.geometry.naca4`.

| Reference | Reported agreement | Interpretation |
|---|---|---|
| Legacy closed thickness polynomial, coefficient 0.1036 | Maximum coordinate difference about 4e-17 | Round-off agreement |
| Original open polynomial, coefficient 0.1015 | Difference about 1.3e-3 | Different trailing-edge geometry |
| NACA 23012 camber crest | Generated x/c 0.1499 versus tabulated 0.15 | About 1e-4 chord difference on the sampled surface |

The five-digit reference is Abbott and von Doenhoff; see the [bibliography](../reference/bibliography.md). These are existing recorded measurements. The current geometry tests enforce their stated tolerances rather than a promise that every generated point equals the rounded table.

```bash
# Compare both analytic geometry and the MATLAB reference file.
./fv2d.sh test tests/test_geometry.py
```

See [geometry theory](../theory/airfoil-geometry.md) and [MATLAB provenance](../reference/legacy.md).
