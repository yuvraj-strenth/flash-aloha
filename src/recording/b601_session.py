"""Shared B601 session helpers. They never connect unless a caller asks them to."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from config.paths import REPO_ROOT
from config.task_config import load_task_config, task_start_position_path
from hardware.arm_factory import build_arms
from hardware.runtime import HardwareRuntime


def load_task(path):
    task = load_task_config(path)
    if task.as_dict()['robot']['type'] != 'b601':
        raise ValueError(f'{path} is not a b601 task')
    return task


def open_runtime(task):
    return HardwareRuntime(build_arms(task.as_dict()['robot']))


def guided_sample(observation):
    """The recorded action is the measured pose. Hand guiding does not invent a command."""
    action = np.asarray(observation.qpos, dtype=float).copy()
    return action, {**observation.recording_fields(), 'commanded_action': action}


def collect_guided(hardware, camera_names, steps, dt, append, *, pace=False,
                   read_key=None, sleep=time.sleep, clock=time.monotonic):
    """Record up to ``steps`` samples. ``pace`` waits out ``dt`` so a full take lasts steps × dt.

    ``read_key(0)`` is polled after each sample. S ends the take early and the samples
    already appended are kept. The poll must not block.
    """
    for arm in hardware.arms.values():
        arm.set_compliant(True)
    next_tick = clock()
    recorded = 0
    for _ in range(steps):
        next_tick += dt
        observation = hardware.capture(camera_names, include_images=bool(camera_names))
        action, sample = guided_sample(observation)
        hardware.submit_target(action, interval_s=dt)
        append(sample)
        recorded += 1
        if read_key is not None and collect_command(read_key(0)) == 's':
            break
        if pace:
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
    return recorded


def collect_command(key):
    """S starts a take. D deletes the last saved one. Anything else is ignored."""
    if key is None:
        return None
    command = str(key).lower()
    if command in ('s', 'd'):
        return command
    return None


def delete_saved_episode(path, data_dir):
    """Remove one episode directory that lives under the task data directory."""
    episode = Path(path).resolve()
    root = Path(data_dir).resolve()
    if episode == root or root not in episode.parents or not episode.is_dir():
        raise ValueError(f'{episode} is not a saved episode under {root}')
    shutil.rmtree(episode)
    return episode


class QuitRequested(KeyboardInterrupt):
    """The key source ended the idle wait. This is not an arm fault."""


@contextmanager
def refresh_commands(hardware, dt, *, clock=time.monotonic):
    """Resubmit the measured pose while the caller blocks (episode start/stop).

    The host command watchdog expires if Python sends no target for
    ``command_timeout_s``. Camera warmup and drain are longer than that.
    """
    stop = threading.Event()
    errors = []

    def run():
        while not stop.is_set():
            started = clock()
            try:
                hardware.submit_target(hardware.read_state()['qpos'], interval_s=dt)
            except BaseException as exc:
                errors.append(exc)
                return
            remaining = dt - (clock() - started)
            if remaining > 0 and stop.wait(remaining):
                return

    worker = threading.Thread(target=run, daemon=True, name='b601-command-refresh')
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
    if errors:
        raise errors[0]


def collect_session(hardware, dt, read_key, record, delete):
    """Stay connected. S records one take, D deletes the last saved take."""
    for arm in hardware.arms.values():
        arm.set_compliant(True)
    last = None
    try:
        while True:
            hardware.submit_target(hardware.read_state()['qpos'], interval_s=dt)
            command = collect_command(read_key(dt))
            if command == 's':
                last = record()
            elif command == 'd':
                if last is None:
                    delete(None)
                else:
                    delete(last)
                    last = None
    except QuitRequested:
        return last
    return last


def start_pose_path(task_name):
    return REPO_ROOT / task_start_position_path(task_name)


def save_start_pose(task_name, qpos):
    qpos = np.asarray(qpos, dtype=float).reshape(-1)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise ValueError('A B601 start pose is 7 finite joint radians')
    path = start_pose_path(task_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, left_qpos=qpos)
    return path


def load_start_pose(task_name):
    path = start_pose_path(task_name)
    if not path.is_file():
        raise FileNotFoundError(f'B601 task {task_name} has no saved start pose at {path}')
    with np.load(path) as data:
        qpos = np.asarray(data['left_qpos'], dtype=float).reshape(-1)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise ValueError(f'{path} does not contain a 7-vector left_qpos')
    return qpos


def apply_start_key(key, measured, locked):
    """Space locks the measured pose, n clears it, y requires a lock."""
    if key == ' ':
        return np.asarray(measured, dtype=float).copy()
    if key in ('n', 'q'):
        return None
    if key == 'y':
        if locked is None:
            raise ValueError('Lock a pose with space before saving')
        return locked
    return locked
