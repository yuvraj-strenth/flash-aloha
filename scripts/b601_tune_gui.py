#!/usr/bin/env python3
"""Slider window for B601 gravity scale and per-joint MIT kp/kd.

    /home/thor/reBot-Isaacsim/.venv/bin/python scripts/b601_tune_gui.py

Gravity keeps the arm compliant: the measured pose plus gravity torque. Each
joint has its own tau_scale, kp, and kd. Joints 4-6 are the small motors, so
their gravity kp and kd start much lower. Hold freezes the measured pose and
switches the kp and kd sliders to the tracking gains. Save writes the stack
profile. Quit damps, then disconnects. Support the arm before starting.
"""
import os
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from b601_tune import MOTOR_NAMES, apply_to_arm, tune_state
from hardware.backends.b601 import save_mit_profile
from recording.b601_session import load_task, open_runtime


def open_window():
    """Open Tk on the running desktop. DISPLAY=:0 is not always the live screen."""
    authorities = []
    for candidate in (os.environ.get('XAUTHORITY'),
                      f'/run/user/{os.getuid()}/gdm/Xauthority',
                      str(Path.home() / '.Xauthority')):
        if candidate and candidate not in authorities and Path(candidate).is_file():
            authorities.append(candidate)
    displays = []
    if os.environ.get('DISPLAY'):
        displays.append(os.environ['DISPLAY'])
    socket_dir = Path('/tmp/.X11-unix')
    if socket_dir.is_dir():
        for sock in sorted(socket_dir.glob('X*')):
            name = f':{sock.name[1:]}'
            if name not in displays:
                displays.append(name)
    errors = []
    for display in displays or [':0']:
        for auth in authorities or [None]:
            os.environ['DISPLAY'] = display
            if auth:
                os.environ['XAUTHORITY'] = auth
            try:
                return tk.Tk()
            except tk.TclError as exc:
                errors.append(f'{display}: {exc}')
    raise SystemExit('Could not open a window.\n' + '\n'.join(errors))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', default=str(ROOT / 'src/config/tasks/b601_single.yaml'))
    args = parser.parse_args(argv)
    root = open_window()
    root.title('B601 gains')
    root.withdraw()
    hardware = None
    closed = {'value': False}
    try:
        task = load_task(args.task_config)
        hardware = open_runtime(task)
        if len(hardware.arms) != 1:
            raise ValueError(f'b601_tune_gui.py tunes one arm; the task has {len(hardware.arms)}')
        arm = next(iter(hardware.arms.values()))
        arm.set_compliant(True)
        hardware.connect()
    except BaseException:
        if hardware is not None:
            hardware.close()
        root.destroy()
        raise
    state = tune_state(arm, 'gravity')
    ready = {'value': False}
    showing = {'mode': 'gravity'}
    hold = {'qpos': None}
    status = tk.StringVar(value='Gravity: move the arm by hand. Each joint uses its own kp and kd.')

    def remember():
        state['tau_scale'] = np.array([var.get() for var in tau_vars], dtype=float)
        kp = np.array([var.get() for var in kp_vars], dtype=float)
        kd = np.array([var.get() for var in kd_vars], dtype=float)
        if showing['mode'] == 'gravity':
            state['compliant_kp'] = kp
            state['compliant_kd'] = kd
        else:
            state['kp'] = kp
            state['kd'] = kd

    def push(*_args):
        if closed['value'] or not ready['value']:
            return
        remember()
        apply_to_arm(arm, state)

    def show_mode(mode):
        ready['value'] = False
        kp_source = state['compliant_kp'] if mode == 'gravity' else state['kp']
        kd_source = state['compliant_kd'] if mode == 'gravity' else state['kd']
        for index, var in enumerate(kp_vars):
            var.set(float(kp_source[index]))
        for index, var in enumerate(kd_vars):
            var.set(float(kd_source[index]))
        showing['mode'] = mode
        ready['value'] = True

    def use_gravity():
        if showing['mode'] != 'gravity':
            remember()
        hold['qpos'] = None
        arm.set_compliant(True)
        show_mode('gravity')
        push()
        status.set('Gravity: each joint uses its own kp and kd, plus scaled gravity torque.')

    def use_hold():
        if showing['mode'] != 'pd':
            remember()
        hold['qpos'] = np.asarray(hardware.read_state()['qpos'], dtype=float).copy()
        arm.set_compliant(False)
        show_mode('pd')
        push()
        status.set('Hold: tracking the pose from the moment you pressed Hold.')

    def save():
        push()
        path = save_mit_profile(
            arm._hardware_yaml, tracking_kp=state['kp'], tracking_kd=state['kd'],
            compliant_kp=state['compliant_kp'], compliant_kd=state['compliant_kd'],
            tau_scale=state['tau_scale'])
        status.set(f'Saved {path.name}')

    def tick():
        if closed['value']:
            return
        if hold['qpos'] is not None:
            hardware.submit_target(hold['qpos'], interval_s=task.dt)
        root.after(50, tick)

    def on_close():
        if closed['value']:
            return
        closed['value'] = True
        try:
            arm.submit_safe_stop(interval_s=task.dt)
        finally:
            hardware.close()
            root.destroy()

    tau_vars, kp_vars, kd_vars = [], [], []
    header = tk.Frame(root)
    header.pack(fill='x', padx=8, pady=4)
    tk.Button(header, text='Gravity', command=use_gravity, width=10).pack(side='left')
    tk.Button(header, text='Hold', command=use_hold, width=10).pack(side='left', padx=6)
    tk.Label(header, text='Gravity and Hold both use these kp and kd sliders.').pack(side='left')

    table = tk.Frame(root)
    table.pack(padx=8, pady=4)
    for column, text in enumerate(('joint', 'tau_scale', 'kp', 'kd')):
        tk.Label(table, text=text, width=12, anchor='w').grid(row=0, column=column, sticky='w')
    for index, name in enumerate(MOTOR_NAMES):
        tk.Label(table, text=name, width=12, anchor='w').grid(row=index + 1, column=0, sticky='w')
        if index < 6:
            tau = tk.DoubleVar(value=float(state['tau_scale'][index]))
            tau_vars.append(tau)
            tk.Scale(table, variable=tau, from_=0.0, to=2.0, resolution=0.01,
                     orient='horizontal', length=160, command=push).grid(row=index + 1, column=1)
        kp = tk.DoubleVar(value=float(state['compliant_kp'][index]))
        kd = tk.DoubleVar(value=float(state['compliant_kd'][index]))
        kp_vars.append(kp)
        kd_vars.append(kd)
        tk.Scale(table, variable=kp, from_=0.0, to=200.0, resolution=0.05,
                 orient='horizontal', length=180, command=push).grid(row=index + 1, column=2)
        tk.Scale(table, variable=kd, from_=0.0, to=20.0, resolution=0.05,
                 orient='horizontal', length=160, command=push).grid(row=index + 1, column=3)

    buttons = tk.Frame(root)
    buttons.pack(fill='x', padx=8, pady=4)
    tk.Button(buttons, text='Save', command=save, width=10).pack(side='left')
    tk.Button(buttons, text='Quit', command=on_close, width=10).pack(side='left', padx=6)
    tk.Label(root, textvariable=status, anchor='w').pack(fill='x', padx=8, pady=(0, 8))
    ready['value'] = True

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.deiconify()
    tick()
    try:
        root.mainloop()
    finally:
        if not closed['value']:
            closed['value'] = True
            hardware.close()


if __name__ == '__main__':
    main()
