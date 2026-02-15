from pathlib import Path


def pytest_collection_modifyitems(items):
    for item in items:
        path = Path(str(item.fspath)).as_posix()
        if '/tests/unit/' in path:
            item.add_marker('unit')
        elif '/tests/integration/' in path:
            item.add_marker('integration')
