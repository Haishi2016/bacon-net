#!/usr/bin/env python
"""
BACON Distillation Tool - Standalone CLI Wrapper

This script provides easy access to the BACON model distillation tool.

Usage:
    python bacon-distill.py <json_file> <output_file> [options]
    
See python bacon-distill.py --help for more information.
"""

import sys
from bacon.tools.distill import main

if __name__ == '__main__':
    sys.exit(main())
