#!/usr/bin/env python3
"""Hand-guided B601 collection. The arm stays in compliant MIT and the gripper is sent no torque. The action is the measured pose.

S records one take of up to collection.max_duration_seconds and saves it.
S again during the take ends it early and saves the samples already taken.
The script stays up for another S. D deletes the last saved episode. Ctrl+C damps; h homes.
The L5 recorder is unchanged and still refuses any model other than L5.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import select
import sys
import termios
import tty
import uuid

_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import (
    collect_guided, collect_session, delete_saved_episode, load_task, open_runtime,
    refresh_commands,
)


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


def record_episode(task, *, recorder_factory=None, camera_target=None, steps=None, read_key=None):
    from dataclasses import replace
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    from camera_utils.capture import capture_image_from_shared_memory
    from recording.record_episode_single import wait_for_cameras
    import threading
    doc = task.as_dict()
    if doc['robot']['type'] != 'b601':
        raise ValueError('record_b601.py only runs a b601 task')
    names = doc['cameras']['names']
    with ExitStack() as cleanup:
        hardware = open_runtime(task)
        cleanup.callback(hardware.close)
        dt = task.dt
        for arm in hardware.arms.values():
            arm.validate_target(arm._home_qpos, interval_s=dt)
            arm.set_compliant(True)
        from camera_utils.camera_settings_contract import check_inference_camera_settings, snapshot_camera_settings
        check_inference_camera_settings(task.legacy(), context='collection')
        stop = threading.Event()
        camera_thread = threading.Thread(target=camera_target or capture_image_from_shared_memory,
            args=(stop,), kwargs={'camera_names': names}, daemon=True)
        camera_thread.start()
        cleanup.callback(lambda: (stop.set(), camera_thread.join(timeout=5)))
        wait_for_cameras(names, timeout_sec=15)
        recorder = (recorder_factory or DistributedRecorder)(
            camera_names=names, data_dir=doc['data']['directory'])
        cleanup.callback(recorder.close)
        request = EpisodeRequest(
            session_id=uuid.uuid4().hex, data_dir=doc['data']['directory'],
            camera_names=names, record_rollout=True, record_dagger=False,
            rollout_dir=doc['data']['directory'], rollout_task_name=task.name,
            dagger_dir='', dagger_task_name='', dagger_camera_names=[],
            is_double_arm=len(hardware.arms) == 2, fps=task.control_freq,
            metadata={**hardware.recording_metadata(task.control_freq), 'task_name': task.name,
                      'camera_settings': snapshot_camera_settings(names)})
        hardware.connect()
        take_steps = task.episode_steps if steps is None else steps
        data_dir = doc['data']['directory']
        active_keys = None

        def record_take():
            episode_request = replace(request, session_id=uuid.uuid4().hex)
            print(f'Recording up to {take_steps * dt:.0f}s. S ends it early. Move the arm by the handle.',
                  flush=True)
            with refresh_commands(hardware, dt):
                recorder.start_episode(episode_request, timeout=15)
            try:
                saved_steps = collect_guided(
                    hardware, names, take_steps, dt, recorder.append_sample, pace=True, read_key=active_keys)
            except BaseException:
                if recorder.active_session_id:
                    with refresh_commands(hardware, dt):
                        recorder.stop_episode(retained=False, incomplete=True, outputs={'rollout': {
                            'requested': True, 'decision': 'failure'}})
                raise
            with refresh_commands(hardware, dt):
                manifest = recorder.stop_episode(retained=True, outputs={'rollout': {
                    'requested': True, 'decision': 'success'}})
            episode = Path(manifest).resolve().parent if manifest else Path(data_dir) / episode_request.session_id
            print(f'Saved {episode} ({saved_steps * dt:.2f}s)', flush=True)
            return episode

        def delete_take(path):
            if path is None:
                print('No episode to delete.', flush=True)
                return
            delete_saved_episode(path, data_dir)
            print(f'Deleted {path.name}', flush=True)

        print(f'S records up to {take_steps * dt:.0f}s and saves. S again ends the take early. '
              'D deletes the last episode. Ctrl+C damps; h folds the arm.', flush=True)
        try:
            with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
                with (_cbreak() if read_key is None else nullcontext(read_key)) as keys:
                    active_keys = keys
                    collect_session(hardware, dt, keys, record_take, delete_take)
        finally:
            if recorder.active_session_id:
                with refresh_commands(hardware, dt):
                    recorder.stop_episode(retained=False, incomplete=True, outputs={'rollout': {
                        'requested': True, 'decision': 'failure'}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-config', required=True)
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    record_episode(task)


if __name__ == '__main__':
    main()
