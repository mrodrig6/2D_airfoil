"""Shared fixtures and the ``--fast`` / ``--slow`` selectors.

Everything here is deliberately small: the unit tests must run in seconds so
they are actually run.  The few that build a production-sized mesh are marked
``slow``.
"""

from __future__ import annotations

import pytest

from src import Discretization, build_mesh, build_metrics, naca4

#: A mesh small enough to build many times, large enough to be a real O-grid.
#: A fixture rather than a module constant, so the test modules reach it through
#: pytest rather than through a relative import that only works as a package.
_TINY = Discretization(n_surface=48, n_radial=24, far_field_radius=10.0, first_cell_height=0.02)


@pytest.fixture(scope="session")
def tiny():
    return _TINY


@pytest.fixture(scope="session")
def section():
    return naca4("0012", _TINY.n_surface)


@pytest.fixture(scope="session")
def cambered():
    return naca4("2412", _TINY.n_surface)


@pytest.fixture(scope="session")
def tiny_mesh(section):
    mesh, _ = build_mesh(section, _TINY)
    return mesh


@pytest.fixture(scope="session")
def tiny_grid(section):
    return build_mesh(section, _TINY)[1]


@pytest.fixture(scope="session")
def tiny_metrics(tiny_mesh):
    return build_metrics(tiny_mesh)


def pytest_addoption(parser):
    """``--fast`` and ``--slow`` instead of ``-m "not slow"``.

    The quoted marker expression is easy to get wrong and easy to get wrong
    *silently*: a stray space or a smart quote selects a different set of tests
    and still exits zero.  A flag either exists or errors out.
    """
    group = parser.getgroup("fvairfoil")
    group.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="run only the fast tests (equivalent to -m 'not slow')",
    )
    group.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run only the slow tests (equivalent to -m slow)",
    )


def pytest_configure(config):
    """Turn the flags into a marker expression, so they *deselect* rather than skip."""
    fast, slow = config.getoption("--fast"), config.getoption("--slow")
    if fast and slow:
        raise pytest.UsageError("--fast and --slow select disjoint sets; pass neither to run both")
    if not (fast or slow):
        return
    wanted = "not slow" if fast else "slow"
    existing = config.option.markexpr
    config.option.markexpr = f"({existing}) and ({wanted})" if existing else wanted


def pytest_collection_modifyitems(config, items):
    """Skip backend-specific tests when the backend is not installed."""
    have = {}
    for name in ("numba", "jax"):
        try:
            __import__(name)
            have[name] = True
        except ImportError:
            have[name] = False
    for item in items:
        for name, installed in have.items():
            if name in item.keywords and not installed:
                item.add_marker(pytest.mark.skip(reason=f"{name} not installed"))
