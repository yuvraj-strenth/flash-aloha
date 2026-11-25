#!/usr/bin/env python3
"""Lock the measured B601 pose and save it as the task start pose.

Space locks the current 7-vector, n clears the lock, y writes
calibration/start_positions/<task>.npz with left_qpos. The arm stays compliant.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import apply_start_key, load_task, open_runtime, save_start_pose


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
    locked = None
    print('Space locks the pose, n clears it, y saves it, q quits.')
    try:
        from getkey import getkey
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            while True:
                measured = hardware.read_state()['qpos']
                key = getkey()
                if key == 'q':
                    break
                updated = apply_start_key(key, measured, locked)
                if key == 'y':
                    path = save_start_pose(task.name, updated)
                    print(f'Saved {path}')
                    break
                locked = updated
                if locked is None:
                    print('lock cleared')
                elif key == ' ':
                    print('locked', ' '.join(f'{value:+.3f}' for value in locked))
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
