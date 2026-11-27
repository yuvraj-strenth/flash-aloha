#!/usr/bin/env python3
"""Low-friction hand-drag practice for one PiperX arm.

Enables drag-teach mode (or prompts you to press the physical teach button on
the arm) and prints live 7-DOF joint state (6 joints + gripper). Nothing is saved.
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from config.task_config import load_task_config
from hardware.arm_factory import build_arms
from hardware.runtime import HardwareRuntime


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', default=str(ROOT / 'src/config/tasks/piperx_single.yaml'))
    args = parser.parse_args(argv)
    task = load_task_config(args.task_config)
    doc = task.as_dict()
    if doc['robot']['type'] != 'piperx':
        raise ValueError(f'{args.task_config} is not a piperx task')
    hardware = HardwareRuntime(build_arms(doc['robot']))
    hardware.connect()
    try:
        for arm in hardware.arms.values():
            arm.set_teaching_mode(True)
        print('PiperX teaching mode active. Move the arm by hand. Ctrl+C exits.')
        while True:
            state = hardware.read_state()
            print('qpos', ' '.join(f'{v:+.3f}' for v in state['qpos']), end='\r', flush=True)
            time.sleep(task.dt)
    except KeyboardInterrupt:
        print()
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
