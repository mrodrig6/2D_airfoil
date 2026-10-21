"""The project's own consistency: the command line runs, and the documents link.

A broken cross-reference in the documentation is the failure this catches most
often, and it is invisible to every other test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.cli import main

ROOT = Path(__file__).resolve().parent.parent
MARKDOWN = (
    sorted(ROOT.glob("*.md"))
    + sorted(ROOT.glob("docs/**/*.md"))
    + sorted(ROOT.glob("src/**/*.md"))
    + [ROOT / "legacy" / "README.md"]
)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


# -- the command line -------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        ["check"],
        ["flow", "--mach", "0.8", "--alpha", "1.25"],
        ["geometry", "--airfoil", "2412", "--n-surface", "48"],
        [
            "mesh",
            "--n-surface",
            "48",
            "--n-radial",
            "24",
            "--far-field-radius",
            "10",
            "--first-cell-height",
            "0.02",
        ],
    ],
)
def test_the_command_line_runs(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_a_bad_request_exits_non_zero_with_a_message(capsys):
    assert main(["geometry", "--airfoil", "not-an-airfoil"]) == 2
    assert "cannot resolve" in capsys.readouterr().err


def test_mesh_writes_the_files_it_is_asked_for(tmp_path):
    pytest.importorskip("matplotlib")
    figure, su2 = tmp_path / "m.png", tmp_path / "m.su2"
    assert (
        main(
            [
                "mesh",
                "--n-surface",
                "48",
                "--n-radial",
                "24",
                "--far-field-radius",
                "10",
                "--first-cell-height",
                "0.02",
                "--figure",
                str(figure),
                "--su2",
                str(su2),
            ]
        )
        == 0
    )
    assert figure.stat().st_size > 0 and su2.stat().st_size > 0


# -- the documentation ----------------------------------------------------
# Link, anchor and maths checking lives in tests/test_docs.py, ported from the
# companion nozzle code.  It supersedes the link-only test that was here: it
# also resolves in-page anchors, and it caught three real defects in these
# documents that the old one could not see.


def test_nothing_outside_legacy_mentions_octave():
    """There is no Octave version of this code, so a stray mention would send a
    student looking for one that does not exist.

    ``legacy/`` is exempt because it is the MATLAB, and this file is exempt
    because it is the only place the word is supposed to appear.
    """
    offenders = []
    for path in list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("tests/*.py")) + MARKDOWN:
        if not path.exists() or "legacy" in path.parts or path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8").lower()
        for word in ("octave",):
            if word in text:
                offenders.append(f"{path.relative_to(ROOT)}: {word}")
    assert not offenders, offenders


# -- the solve command ------------------------------------------------------
@pytest.mark.slow
def test_the_solve_command_runs_and_writes_what_it_is_asked_for(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    argv = [
        "solve",
        "--n-surface",
        "48",
        "--n-radial",
        "24",
        "--far-field-radius",
        "10",
        "--first-cell-height",
        "0.02",
        "--mach",
        "0.5",
        "--alpha",
        "2.0",
        "--tolerance",
        "1e-5",
        "--print-interval",
        "0",
        "--max-iterations",
        "20000",
        "--cp",
        str(tmp_path / "cp.png"),
        "--figure",
        str(tmp_path / "mach.png"),
        "--convergence",
        str(tmp_path / "conv.png"),
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "converged" in out and "cl =" in out
    for name in ("cp.png", "mach.png", "conv.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_a_solve_that_does_not_converge_exits_non_zero(capsys):
    """A stalled march is a result, not a crash -- but the shell must know."""
    assert (
        main(
            [
                "solve",
                "--n-surface",
                "48",
                "--n-radial",
                "24",
                "--far-field-radius",
                "10",
                "--first-cell-height",
                "0.02",
                "--max-iterations",
                "5",
                "--print-interval",
                "0",
            ]
        )
        == 1
    )
    assert "NOT converged" in capsys.readouterr().out
