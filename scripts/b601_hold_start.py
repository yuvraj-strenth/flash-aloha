#!/usr/bin/env python3
"""Move a B601 to its saved start pose with tracking MIT gains and hold it."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import load_start_pose, load_task, open_runtime


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', default=str(ROOT / 'src/config/tasks/b601_single.yaml'))
    parser.add_argument('--move-duration', type=float, default=3.0)
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    start = load_start_pose(task.name)
    hardware = open_runtime(task)
    hardware.connect()
    print('Moving to the saved start pose. Ctrl+C damps; h homes.')
    try:
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            arm = next(iter(hardware.arms.values()))
            arm.move_toward(start, duration_s=args.move_duration, interval_s=task.dt)
            while True:
                hardware.submit_target(start, interval_s=task.dt)
                time.sleep(task.dt)
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
