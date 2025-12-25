"""
BACON Distillation CLI - Make the bacon.distill module executable.

This allows running: python -m bacon.distill <args>
"""

from bacon.distill import main

if __name__ == '__main__':
    import sys
    sys.exit(main())
