"""Camera command relocation checks; never open devices or start services."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
TOOLS = sorted(
    path.name for path in (ROOT / 'scripts/cameras').glob('*.py')
)


@pytest.mark.parametrize('name', TOOLS)
def test_camera_help_outside_repository(name, tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/cameras' / name), '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('relative,args', [
    ('src/camera_utils/camera_server.py', ['--help']),
    ('src/camera_utils/camera_server_realsense.py', ['--help']),
])
def test_direct_server_and_benchmark_help(relative, args, tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()
