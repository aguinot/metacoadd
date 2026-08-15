"""Shared pytest configuration for metacoadd tests."""

import os


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "coverage_all_params: run all parameters during coverage",
    )


def pytest_collection_modifyitems(config, items):
    """Keep one parametrization per test during coverage runs."""
    if os.environ.get("COVERAGE_MODE", "0") != "1":
        return

    selected = []
    deselected = []
    seen_parametrized_tests = set()

    for item in items:
        if not hasattr(item, "callspec"):
            selected.append(item)
            continue

        # Keep every parametrization for explicitly marked tests.
        if item.get_closest_marker("coverage_all_params") is not None:
            selected.append(item)
            continue

        test_id = item.nodeid.split("[", maxsplit=1)[0]

        if test_id in seen_parametrized_tests:
            deselected.append(item)
        else:
            seen_parametrized_tests.add(test_id)
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
