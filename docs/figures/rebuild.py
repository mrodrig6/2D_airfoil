"""Rebuild current figures beside this recipe; historical flux plots are excluded.

From the repository root: python docs/figures/rebuild.py --output /tmp/airfoil-figures
Use --geometry-only to omit the flow, polar and adaptive solves.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for command in (
        ["geometry", "airfoil=0012", f"figure={output / 'section.png'}", "normals=yes"],
        ["mesh", "@grid", f"figure={output / 'mesh.png'}"],
        ["mesh", "@cgrid", f"figure={output / 'cgrid.png'}", "window=-0.6,2.6,-0.9,0.9"],
    ):
        subprocess.run(["bash", str(ROOT / "fv2d.sh"), *command], cwd=ROOT, check=True)
    if not args.geometry_only:
        subprocess.run(
            [sys.executable, str(ROOT / "examples/06_postprocess.py"), str(output)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "bash",
                str(ROOT / "fv2d.sh"),
                "adapt",
                "naca=0012",
                "ns=64",
                "nr=24",
                "R=10",
                "h1=0.02",
                "mach=0.8",
                "alpha=1.25",
                "cycles=6",
                "tolerance=1e-6",
                "max_iterations=80000",
                "print_interval=0",
                f"figure={output / 'adapted.png'}",
                "window=-0.2,1.2,-0.5,0.5",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
