#!/usr/bin/env python3
"""Draw the actual NACA candidates used in the course handouts."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry.section import naca_section  # noqa: E402

out = Path(__file__).resolve().parent / "figures"
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7.0, 2.2), constrained_layout=True)
for code, color in (("0012", "#4c566a"), ("2412", "#268bd2"), ("4412", "#dc322f")):
    section = naca_section(int(code[0]) / 100, int(code[1]) / 10 if code[0] != "0" else .4, .12, 160)
    ax.plot(section.points[:, 0], section.points[:, 1], lw=2, label=f"NACA {code}", color=color)
ax.axhline(0, color="0.75", lw=.7)
ax.set(xlim=(-.03, 1.03), ylim=(-.16, .16), xlabel=r"$x/c$", ylabel=r"$y/c$")
ax.set_aspect("equal", adjustable="box")
ax.legend(ncol=3, frameon=False, loc="upper center")
fig.savefig(out / "airfoil_candidates.png", dpi=220)
