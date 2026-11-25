#!/usr/bin/env python3
"""Tune B601 gravity compensation and MIT kp/kd, then save them in the stack profile.

    python scripts/b601_tune.py gravity
    python scripts/b601_tune.py pd

gravity holds the arm compliant (measured pose plus gravity torque). [ ] select an
arm joint, up/down change that joint's tau_scale, < > change that joint's compliant
kp, and , . change that joint's compliant kd.

pd holds the pose measured at start with tracking gains. [ ] select one of the
seven motors, up/down change its MIT kp, and < > change its MIT kd.

s writes joints[].MIT and gravity_compensation back to the task's hardware YAML.
Motor ids and POS_VEL are left as they are. Ctrl+C damps; h homes. A running
collector or policy keeps the gains it already loaded.
"""
import select
import sys
import termios
import tty
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hardware.backends.b601 import save_mit_profile
from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import load_task, open_runtime


ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
MOTOR_NAMES = ARM_JOINTS + ('gripper',)
UP = ('\x1b[A', '\x1bOA')
DOWN = ('\x1b[B', '\x1bOB')
TAU_STEP = 0.05
COMPLIANT_KP_STEP = 0.5
COMPLIANT_KD_STEP = 0.1
KP_STEP = 1.0
KD_STEP = 0.1


def tune_state(arm, mode):
    """Gains currently loaded on one arm."""
    if mode not in ('gravity', 'pd'):
        raise ValueError(f'Unknown tune mode {mode!r}')
    return {
        'mode': mode,
        'index': 0,
        'tau_scale': np.asarray(arm._tau_scale[:6], dtype=float).copy(),
        'compliant_kp': np.asarray(arm._compliant_kp, dtype=float).copy(),
        'compliant_kd': np.asarray(arm._compliant_kd, dtype=float).copy(),
        'kp': np.asarray(arm._track_kp, dtype=float).copy(),
        'kd': np.asarray(arm._track_kd, dtype=float).copy(),
    }


def _copy_state(state):
    return {
        'mode': state['mode'],
        'index': int(state['index']),
        'tau_scale': np.asarray(state['tau_scale'], dtype=float).copy(),
        'compliant_kp': np.asarray(state['compliant_kp'], dtype=float).copy(),
        'compliant_kd': np.asarray(state['compliant_kd'], dtype=float).copy(),
        'kp': np.asarray(state['kp'], dtype=float).copy(),
        'kd': np.asarray(state['kd'], dtype=float).copy(),
    }


def apply_tune_key(state, key):
    """Apply one tuner key. The second result is true when the operator pressed s."""
    state = _copy_state(state)
    count = 6 if state['mode'] == 'gravity' else 7
    index = state['index'] % count
    state['index'] = index
    if key == '[':
        state['index'] = (index - 1) % count
    elif key == ']':
        state['index'] = (index + 1) % count
    elif key in UP:
        _bump(state, +1)
    elif key in DOWN:
        _bump(state, -1)
    elif key == '<':
        _side(state, -1)
    elif key == '>':
        _side(state, +1)
    elif key == ',' and state['mode'] == 'gravity':
        state['compliant_kd'][index] = max(0.0, state['compliant_kd'][index] - COMPLIANT_KD_STEP)
    elif key == '.' and state['mode'] == 'gravity':
        state['compliant_kd'][index] = state['compliant_kd'][index] + COMPLIANT_KD_STEP
    return state, key == 's'


def _bump(state, sign):
    index = state['index']
    if state['mode'] == 'gravity':
        state['tau_scale'][index] = max(0.0, state['tau_scale'][index] + sign * TAU_STEP)
    else:
        state['kp'][index] = max(0.0, state['kp'][index] + sign * KP_STEP)


def _side(state, sign):
    if state['mode'] == 'gravity':
        index = state['index']
        state['compliant_kp'][index] = max(0.0, state['compliant_kp'][index] + sign * COMPLIANT_KP_STEP)
    else:
        index = state['index']
        state['kd'][index] = max(0.0, state['kd'][index] + sign * KD_STEP)


def apply_to_arm(arm, state):
    """Push the tuner state into the adapter. The next MIT frame sends it."""
    arm.set_tau_scale(state['tau_scale'])
    arm.set_compliant_gains(state['compliant_kp'], state['compliant_kd'])
    arm.set_tracking_gains(state['kp'], state['kd'])


def format_tune_line(state):
    names = ARM_JOINTS if state['mode'] == 'gravity' else MOTOR_NAMES
    name = names[state['index']]
    if state['mode'] == 'gravity':
        return (f'{name} tau_scale {state["tau_scale"][state["index"]]:.3f}  '
                f'compliant kp {state["compliant_kp"][state["index"]]:.2f} '
                f'kd {state["compliant_kd"][state["index"]]:.2f}')
    return f'{name} MIT kp {state["kp"][state["index"]]:.2f} kd {state["kd"][state["index"]]:.2f}'


def poll_key(timeout):
    """One key, or None when the wait expires. The terminal is already in cbreak."""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    key = sys.stdin.read(1)
    if key != '\x1b':
        return key
    for _ in range(2):
        extra, _, _ = select.select([sys.stdin], [], [], 0.02)
        if not extra:
            break
        key += sys.stdin.read(1)
    return key


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('gravity', 'pd'))
    parser.add_argument('--task-config', default=str(ROOT / 'src/config/tasks/b601_single.yaml'))
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    hardware = open_runtime(task)
    if len(hardware.arms) != 1:
        raise ValueError('b601_tune.py tunes one arm; the task has '
                         f'{len(hardware.arms)}')
    arm = next(iter(hardware.arms.values()))
    arm.set_compliant(args.mode == 'gravity')
    hardware.connect()
    state = tune_state(arm, args.mode)
    hold = None if args.mode == 'gravity' else hardware.read_state()['qpos'].copy()
    print('s saves the stack profile. Ctrl+C damps; h homes.')
    print('[ ] selects a joint. gravity: up/down tau_scale, < > kp, , . kd. '
          'pd: up/down kp, < > kd.')
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            while True:
                if hold is not None:
                    hardware.submit_target(hold, interval_s=task.dt)
                key = poll_key(task.dt)
                if key:
                    state, save = apply_tune_key(state, key)
                    apply_to_arm(arm, state)
                    if save:
                        path = save_mit_profile(
                            arm._hardware_yaml, tracking_kp=state['kp'], tracking_kd=state['kd'],
                            compliant_kp=state['compliant_kp'], compliant_kd=state['compliant_kd'],
                            tau_scale=state['tau_scale'])
                        print(f'\nSaved {path}')
                print(format_tune_line(state), end='\r', flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        hardware.close()


if __name__ == '__main__':
    main()
