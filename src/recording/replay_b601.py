#!/usr/bin/env python3
"""Play saved absolute B601 actions at the task rate with tracking MIT gains."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import load_task, open_runtime


def resolve_episode(task, episode):
    path = Path(episode).expanduser()
    if path.exists():
        return path
    directory = Path(task.as_dict()['data']['directory'])
    index = int(episode)
    for name in (f'episode_{index}', f'episode_{index:05d}.zarr', f'episode_{index}.zarr'):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No B601 episode {episode} under {directory}')


def load_actions(path):
    import zarr
    root = zarr.open(str(path), mode='r')

    def find(group):
        for key in ('commanded_action', 'action'):
            if key in group:
                return group[key]
        for key in list(getattr(group, 'keys', lambda: [])()):
            child = group[key]
            if hasattr(child, 'keys'):
                for name in ('commanded_action', 'action'):
                    if name in child:
                        return child[name]
        return None

    dataset = find(root)
    if dataset is None:
        raise ValueError(f'{path} has no commanded_action or action array')
    actions = np.asarray(dataset, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f'{path} actions must be finite rows of 7 joint radians')
    return actions


def replay_actions(hardware, actions, dt):
    for arm in hardware.arms.values():
        arm.set_compliant(False)
    for row in actions:
        started = time.monotonic()
        hardware.submit_target(row, interval_s=dt)
        remaining = dt - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', required=True)
    parser.add_argument('episodes', nargs='+', help='Episode .zarr paths')
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    hardware = open_runtime(task)
    try:
        hardware.connect()
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            for episode in args.episodes:
                replay_actions(hardware, load_actions(resolve_episode(task, episode)), task.dt)
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
