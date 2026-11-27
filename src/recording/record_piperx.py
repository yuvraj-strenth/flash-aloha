#!/usr/bin/env python3
"""Hand-guided PiperX episode collector.

Reads joint state from a running ``scripts/piperx_joint_stream.py`` shared-memory
stream when available (or connects directly to the PiperX arm in teaching mode),
captures synchronized camera frames via ``standard_capture.capture_observation``,
and records episodes through ``DistributedRecorder``.

S starts/ends a take, D deletes the last saved take, Ctrl+C exits cleanly.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
import select
import shutil
import sys
import termios
import threading
import time
import tty
import uuid

import numpy as np

_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from config.task_config import load_task_config
from contracts import capture_gc, standard_capture
from contracts.robot_contract import normalize_robot_contract
from hardware.arm_factory import build_arms, hardware_robot_contract
from hardware.runtime import HardwareRuntime
from recording.piperx_shm import PiperXJointStreamReader, shm_stream_available


def _poll_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


@contextmanager
def _cbreak():
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield _poll_key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def _open_state_reader(doc, *, source: str = 'auto', shm_names: dict[str, str] | None = None):
    """Return (read_state_fn, metadata, cleanup_fn, mode_label)."""
    arms_cfg = doc['robot']['arms']
    roles = [role for role in ('left', 'right') if role in arms_cfg]
    use_shm = False
    if source == 'shm':
        use_shm = True
    elif source == 'auto':
        use_shm = all(
            shm_stream_available(
                arms_cfg[r]['can_interface'], r,
                name=(shm_names or {}).get(r),
            )
            for r in roles
        )

    if use_shm:
        readers = {
            r: PiperXJointStreamReader(
                arms_cfg[r]['can_interface'],
                r,
                max_age_s=max(0.2, float(arms_cfg[r].get('max_feedback_age_s', 0.1)) * 2),
                name=(shm_names or {}).get(r),
            )
            for r in roles
        }
        contract = hardware_robot_contract(doc['robot'])
        meta = {
            'robot_contract': contract,
            'control_freq': float(doc['control_freq']),
            'action_command_style': 'absolute',
            'sdk_controller_hz_by_role': {r: 200.0 for r in roles},
        }

        def read_shm_state():
            snaps = [readers[r].read() for r in roles]
            return {
                'qpos': np.concatenate([s['qpos'] for s in snaps]),
                'qvel': np.concatenate([s['qvel'] for s in snaps]),
            }

        def close_readers():
            for reader in readers.values():
                reader.close()

        return read_shm_state, meta, close_readers, 'shm'

    hardware = HardwareRuntime(build_arms(doc['robot']))
    hardware.connect()
    try:
        for arm in hardware.arms.values():
            arm.set_teaching_mode(True)
    except BaseException:
        hardware.close()
        raise
    meta = hardware.recording_metadata(doc['control_freq'])
    return hardware.read_state, meta, hardware.close, 'direct'


def collect_take(read_state, camera_names, steps, dt, append, *,
                 read_key=None, sleep=time.sleep, clock=time.monotonic,
                 snapshot_provider=None) -> int:
    next_tick = clock()
    recorded = 0
    for _ in range(steps):
        next_tick += dt
        observation = standard_capture.capture_observation(
            read_state, camera_names, include_images=bool(camera_names),
            clock=clock, snapshot_provider=snapshot_provider,
        )
        action = np.asarray(observation.qpos, dtype=float).copy()
        append({**observation.recording_fields(), 'commanded_action': action})
        recorded += 1
        if read_key is not None:
            key = read_key(0)
            if key is not None and str(key).lower() == 's':
                break
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)
    return recorded


def record_episode(task, *, source: str = 'auto', shm_names: dict[str, str] | None = None,
                   recorder_factory=None, camera_target=None, steps=None,
                   read_key=None, snapshot_provider=None):
    from camera_utils.camera_settings_contract import check_inference_camera_settings, snapshot_camera_settings
    from camera_utils.capture import capture_image_from_shared_memory
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    from recording.record_episode_single import wait_for_cameras

    doc = task.as_dict()
    if doc['robot']['type'] != 'piperx':
        raise ValueError('record_piperx.py only runs a piperx task')
    names = doc['cameras']['names']
    dt = task.dt
    take_steps = task.episode_steps if steps is None else steps
    data_dir = doc['data']['directory']

    with ExitStack() as cleanup:
        read_state, hw_meta, close_hw, mode = _open_state_reader(
            doc, source=source, shm_names=shm_names,
        )
        cleanup.callback(close_hw)
        print(f'[PiperX] Reading joint state via {mode} mode.', flush=True)

        if names:
            check_inference_camera_settings(task.legacy(), context='collection')
            stop = threading.Event()
            camera_thread = threading.Thread(
                target=camera_target or capture_image_from_shared_memory,
                args=(stop,), kwargs={'camera_names': names}, daemon=True,
            )
            camera_thread.start()
            cleanup.callback(lambda: (stop.set(), camera_thread.join(timeout=5)))
            wait_for_cameras(names, timeout_sec=15)

        recorder = (recorder_factory or DistributedRecorder)(camera_names=names, data_dir=data_dir)
        cleanup.callback(recorder.close)
        request = EpisodeRequest(
            session_id=uuid.uuid4().hex, data_dir=data_dir,
            camera_names=names, record_rollout=True, record_dagger=False,
            rollout_dir=data_dir, rollout_task_name=task.name,
            dagger_dir='', dagger_task_name='', dagger_camera_names=[],
            is_double_arm=len(doc['robot']['arms']) == 2, fps=task.control_freq,
            metadata={
                **hw_meta,
                'task_name': task.name,
                'camera_settings': snapshot_camera_settings(names) if names else {},
            },
        )

        last_episode = None
        print(
            f'S records up to {take_steps * dt:.0f}s. S again ends early. '
            'D deletes the last episode. Ctrl+C exits.',
            flush=True,
        )
        try:
            with (_cbreak() if read_key is None else nullcontext(read_key)) as keys:
                while True:
                    key = keys(dt)
                    cmd = None if key is None else str(key).lower()
                    if cmd == 's':
                        ep_req = replace(request, session_id=uuid.uuid4().hex)
                        recorder.start_episode(ep_req, timeout=15)
                        try:
                            saved_steps = collect_take(
                                read_state, names, take_steps, dt, recorder.append_sample,
                                read_key=keys, snapshot_provider=snapshot_provider,
                            )
                        except BaseException:
                            if recorder.active_session_id:
                                recorder.stop_episode(
                                    retained=False, incomplete=True,
                                    outputs={'rollout': {'requested': True, 'decision': 'failure'}},
                                )
                            raise
                        manifest = recorder.stop_episode(
                            retained=True,
                            outputs={'rollout': {'requested': True, 'decision': 'success'}},
                        )
                        capture_gc.collect_idle()
                        last_episode = (
                            Path(manifest).resolve().parent
                            if manifest else Path(data_dir) / ep_req.session_id
                        )
                        print(f'Saved {last_episode} ({saved_steps * dt:.2f}s)', flush=True)
                    elif cmd == 'd':
                        if last_episode is None:
                            print('No episode to delete.', flush=True)
                        else:
                            root = Path(data_dir).resolve()
                            ep = Path(last_episode).resolve()
                            if root in ep.parents and ep.is_dir():
                                shutil.rmtree(ep)
                                print(f'Deleted {ep.name}', flush=True)
                            last_episode = None
        except KeyboardInterrupt:
            print()
        finally:
            if recorder.active_session_id:
                recorder.stop_episode(
                    retained=False, incomplete=True,
                    outputs={'rollout': {'requested': True, 'decision': 'failure'}},
                )
            capture_gc.collect_idle()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', required=True)
    parser.add_argument(
        '--source', choices=('auto', 'shm', 'direct'), default='auto',
        help='auto uses piperx_joint_stream SHM if running, else connects directly',
    )
    args = parser.parse_args(argv)
    task = load_task_config(args.task_config)
    record_episode(task, source=args.source)


if __name__ == '__main__':
    main()
