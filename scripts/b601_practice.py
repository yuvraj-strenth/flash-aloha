#!/usr/bin/env python3
"""Gravity-compensated hand guiding for one reBot B601-DM.

The arm stays in compliant MIT (low kp, gravity torque). The gripper is sent
no torque, so it moves freely while its position is still recorded. Nothing is saved.
Ctrl+C damps until you press h, which returns to the task's folded pose.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import load_task, open_runtime


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', default=str(ROOT / 'src/config/tasks/b601_single.yaml'))
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    hardware = open_runtime(task)
    for arm in hardware.arms.values():
        arm.set_compliant(True)
    hardware.connect()
    print('Compliant MIT. Move the arm by hand. Ctrl+C damps; h homes.')
    try:
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            while True:
                state = hardware.read_state()
                print('qpos', ' '.join(f'{value:+.3f}' for value in state['qpos']), end='\r', flush=True)
                time.sleep(task.dt)
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
