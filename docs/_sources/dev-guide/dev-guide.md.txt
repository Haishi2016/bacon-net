# Developer Guide

## Rebuild Docs

The `docs/source` folder contains the source code of the documentation site. To rebuild:
```bash
# under docs/source folder
sphinx-build -b html . ../build/html
# to build without cache, add "-E" switch
sphinx-build -E -b html . ../build/html
```

## Build & Publish the Python Package

Follow these steps from the project root (where `pyproject.toml` lives).

1. Bump version

    Update the version in `bacon/__init__.py`:
    ```python
    __version__ = "0.1.1"
    ```

2. Build (sdist and wheel)

    ```powershell
    # from repo root folder
    python -m pip install --upgrade pip build twine
    python -m build
    ```

3. Validate artifacts

    ```powershell
    python -m twine check dist/*
    ```
    
4. Upload to PyPI (production)

    Create a PyPI token and set environment variables (PowerShell):
    ```powershell
    $env:TWINE_USERNAME="__token__"
    $env:TWINE_PASSWORD="<your-pypi-token>"
    python -m twine upload dist/*
    ```

Notes
- Always rebuild after changing the version.
- If re-building, consider cleaning old artifacts:
    ```powershell
    Remove-Item -Recurse -Force dist, build
    ```