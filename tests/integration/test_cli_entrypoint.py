import subprocess
import sys
import os

import pytest


@pytest.mark.integration
def test_module_cli_help_runs_successfully():
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(
        [sys.executable, '-m', 'bacon', '--help'],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert 'BACON' in result.stdout
