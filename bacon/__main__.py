"""BACON __main__ - Allows running 'python -m bacon'.

This is a fallback for backwards compatibility.
The primary entry point is 'bacon' command installed via pyproject.toml.
"""

from bacon.cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())

