#!/usr/bin/env python3
"""Export raw B601 episodes as a masked LeRobot dataset for Pi0.5."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    implementation = root / 'src/export/pi05_lerobot.py'
    interpreter_args = sys.orig_argv[1:len(sys.orig_argv) - len(sys.argv)]
    os.execv(sys.executable, [sys.executable, *interpreter_args, str(implementation), *sys.argv[1:]])
