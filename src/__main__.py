"""``python -m src`` -- the command line, so the package runs without installing."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
