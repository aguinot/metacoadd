"""Shared pytest configuration for metacoadd tests."""

import os


def pytest_collection_modifyitems(config, items):
    """Keep one parametrization per test during coverage runs."""
    if os.environ.get("COVERAGE_MODE", "0") != "1":
        return

    selected = []
    deselected = []
    seen_parametrized_tests = set()

    for item in items:
        # Non-parametrized tests are always retained.
        if not hasattr(item, "callspec"):
            selected.append(item)
            continue

        # Remove the parameter ID from the pytest node ID.
        test_id = item.nodeid.split("[", maxsplit=1)[0]

        if test_id in seen_parametrized_tests:
            deselected.append(item)
        else:
            seen_parametrized_tests.add(test_id)
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
