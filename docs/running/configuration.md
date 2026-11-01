# Configuration

A case file is a JSON object that stores the arguments of one run. Pass it to `run`, `mesh`, `sweep` and the other commands in place of typing those arguments again.

## Case files

Each key is a name the Python API uses, and each value is the setting. A file in `cases/` looks like this:

```json
{
  "description": "the legacy case: NACA 0012 at M = 0.8, 1.25 deg -- a shock on the upper surface",
  "airfoil": "0012",
  "n_surface": 192,
  "n_radial": 64,
  "far_field_radius": 20.0,
  "first_cell_height": 0.002,
  "mach": 0.8,
  "alpha": 1.25
}
```

| Value type | Written as | Passed on as |
|---|---|---|
| Text | `"0012"` (quoted, so the leading zero survives) | `airfoil=0012` |
| Number | `192`, `1.25` | `n_surface=192`, `alpha=1.25` |
| Yes/no (flags such as `normals` and `check`) | `true`, `false` | `yes`, or left out for `no` |
| List | `[-4, 8, 2]` | `alphas=-4,8,2` |

The key `description` is for people. The launcher skips it and `./airfoil.sh list` prints it.

## Precedence

Arguments are read left to right, and the last value of a key wins:

```bash
./airfoil.sh run cases/transonic.json                 # the file's own values
./airfoil.sh run cases/transonic.json flux=hllc       # the file, with flux changed
./airfoil.sh run cases/transonic.json alpha=2 mach=0.7
```

Aliases such as `naca` or `aoa` work in a file exactly as they do on the command line.

## The bundled cases

| File | Use |
|---|---|
| [`cases/coarse.json`](../../cases/coarse.json) | a 96×40 mesh at M = 0.5, α = 2; solves in seconds |
| [`cases/grid.json`](../../cases/grid.json) | the legacy 192×64 mesh with no operating point; for `mesh` |
| [`cases/transonic.json`](../../cases/transonic.json) | the legacy mesh at M = 0.8, α = 1.25 |
| [`cases/subcritical.json`](../../cases/subcritical.json) | the same mesh at M = 0.5, α = 2; shock free |
| [`cases/cgrid.json`](../../cases/cgrid.json) | the legacy case on a C-grid |

Each file repeats the mesh keys it needs. Copy one to start a new case, and check the keys against [parameters](../model/parameters.md).

## Which commands read which keys

A command reports any key it does not use. The launcher passes the keys through, and the Python parser rejects the rest with `unrecognized arguments`. So `mesh` takes `cases/grid.json`, not `cases/transonic.json`, whose operating point only `run`, `sweep`, `sensitivity` and `adapt` read. This is the same check that catches a typo, so the error is the feedback you want.
