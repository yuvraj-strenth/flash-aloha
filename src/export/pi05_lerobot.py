"""Build a LeRobot dataset for Pi0.5 from raw B601 episodes.

Raw recordings stay as they were captured. This reads each robot sample's saved
camera frame IDs, applies the shared Masker to those BGR frames, and writes the
RGB images plus joint degrees that the reBot Pi0.5 rollout already uses
(d415 = overhead, d405 = wrist, public degrees, 15 Hz).

The recorded action on these takes is the measured pose, so each training row's
action is the next kept pose. A row whose referenced frame is missing is skipped
with the episode; frame IDs are never replaced by a timestamp neighbor.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _path in (_SRC, _SRC / 'kdl_parser' / 'kdl_parser_py'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse

import cv2
import numpy as np
import zarr

from camera_utils.camera_timing import CameraTimingError, selected_frame_indices
from data_processing.video_processing import ExactVideo
from utils.masking import Masker


JOINT_NAMES = (
    'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')
# LeRobot rollout camera names. Serials match src/config/cameras/cam_config.yaml.
CAMERA_KEYS = {'video_overhead': 'observation.images.d415', 'video_left': 'observation.images.d405'}
LEROBOT_TO_FLASH = {'d415': 'video_overhead', 'd405': 'video_left'}
DEFAULT_TASK = 'Pick up the blue box, then return the arm to the home position.'
DEFAULT_FPS = 15


def radians_to_public_degrees(qpos):
    """DM public degrees. Joint directions are +1, so this is the motor angle in degrees."""
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (len(JOINT_NAMES),) or not np.isfinite(values).all():
        raise ValueError(f'Expected 7 finite joint radians, got shape {values.shape}')
    return np.degrees(values).astype(np.float32)


def public_degrees_to_radians(observation):
    """The radian qpos Masker uses, from a LeRobot observation in public degrees."""
    try:
        degrees = [float(observation[f'{name}.pos']) for name in JOINT_NAMES]
    except KeyError as exc:
        raise ValueError('Observation is missing a reBot joint position') from exc
    values = np.asarray(degrees, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Observation joint positions are not finite')
    return np.radians(values)


def observation_action_rows(timestamps, fps):
    """Indices of samples on an ``fps`` grid, and the following sample used as the action.

    The grid is walked on the recorded robot clock. Each index is a real sample, so the
    camera frame is the one that sample already references.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Robot timestamps must be finite and strictly increasing')
    if fps <= 0:
        raise ValueError('fps must be positive')
    elapsed = times - times[0]
    grid = np.arange(0.0, elapsed[-1], 1.0 / float(fps))
    rows = np.unique(np.searchsorted(elapsed, grid, side='left'))
    rows = rows[rows < len(times)]
    if len(rows) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return rows[:-1].astype(np.int64), rows[1:].astype(np.int64)


def mask_lerobot_observation(observation, strategy, masker=None):
    """Apply ``strategy`` to a live reBot observation and return the masked copy.

    Images are RGB, matching the LeRobot RealSense default. Masker paints BGR frames
    from the radian joint state. Joint values are left in public degrees.
    """
    if masker is None:
        masker = Masker(masking_strategy=strategy)
    qpos = public_degrees_to_radians(observation)
    present = {}
    for camera, flash_name in LEROBOT_TO_FLASH.items():
        image = observation.get(camera)
        if image is None:
            continue
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f'{camera} image is not HWC RGB')
        present[flash_name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not present:
        raise ValueError('Observation has neither d415 nor d405')
    masked, _ = masker.mask(present, qpos)
    updated = dict(observation)
    for camera, flash_name in LEROBOT_TO_FLASH.items():
        if flash_name in masked:
            updated[camera] = cv2.cvtColor(masked[flash_name], cv2.COLOR_BGR2RGB)
    return updated


def _load_rows(episode):
    control = zarr.open_group(str(episode / 'control.zarr'), mode='r')
    try:
        timestamps = np.asarray(control['timestamps'], dtype=np.float64)
        qpos = np.asarray(control['qpos'], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f'episode is missing {exc}') from exc
    observations, actions = observation_action_rows(timestamps, DEFAULT_FPS)
    if len(observations) == 0:
        raise ValueError('episode is shorter than two samples at the training rate')
    if qpos.shape != (len(timestamps), len(JOINT_NAMES)):
        raise ValueError(f'qpos shape {qpos.shape} does not match the robot samples')
    references = {}
    for camera in CAMERA_KEYS:
        indices = selected_frame_indices(episode, camera, control)
        if indices is None:
            raise ValueError(f'{camera} has no saved frame references')
        references[camera] = indices
        video = episode / f'{camera}.zarr' / 'main.mp4'
        if not video.is_file():
            raise ValueError(f'missing {video}')
    return observations, actions, qpos, references


def export_episode(dataset, episode, masker, task):
    """Append one episode. Raises, without writing a partial episode, when references fail."""
    observations, actions, qpos, references = _load_rows(episode)
    videos = {camera: ExactVideo(episode / f'{camera}.zarr' / 'main.mp4') for camera in CAMERA_KEYS}
    try:
        for obs_index, action_index in zip(observations, actions):
            state = radians_to_public_degrees(qpos[obs_index])
            target = radians_to_public_degrees(qpos[action_index])
            bgr = {}
            for camera, video in videos.items():
                bgr[camera] = video.get(int(references[camera][obs_index]))
            masked, _ = masker.mask(bgr, qpos[obs_index])
            frame = {'task': task, 'observation.state': state, 'action': target}
            for camera, key in CAMERA_KEYS.items():
                frame[key] = cv2.cvtColor(masked[camera], cv2.COLOR_BGR2RGB)
            dataset.add_frame(frame)
        dataset.save_episode()
    except Exception:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        raise
    finally:
        for video in videos.values():
            video.close()
    return len(observations)


def _episode_directories(dataset_dir):
    return sorted(path for path in Path(dataset_dir).iterdir() if (path / 'control.zarr').is_dir())


def _image_shape(episode):
    observations, _, _, references = _load_rows(episode)
    video = ExactVideo(episode / 'video_overhead.zarr' / 'main.mp4')
    try:
        frame = video.get(int(references['video_overhead'][observations[0]]))
    finally:
        video.close()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f'Unexpected overhead frame shape {frame.shape}')
    return tuple(int(v) for v in frame.shape)


def export_dataset(dataset_dir, output, strategy, task, limit=None):
    episodes = _episode_directories(dataset_dir)
    if limit is not None:
        episodes = episodes[:limit]
    if not episodes:
        raise ValueError(f'No episodes under {dataset_dir}')
    shape = None
    first = None
    for episode in episodes:
        try:
            shape = _image_shape(episode)
            first = episode
            break
        except (OSError, ValueError, CameraTimingError):
            continue
    if shape is None:
        raise ValueError('No episode has usable camera references')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names = list(JOINT_NAMES)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (len(names),), 'names': [f'{name}.pos' for name in names]},
        'action': {'dtype': 'float32', 'shape': (len(names),), 'names': [f'{name}.pos' for name in names]},
    }
    for key in CAMERA_KEYS.values():
        features[key] = {'dtype': 'video', 'shape': shape, 'names': ['height', 'width', 'channels']}
    root = Path(output)
    if root.exists():
        raise FileExistsError(f'{root} already exists')
    dataset = LeRobotDataset.create(
        repo_id='local/b601_single_pi05', fps=DEFAULT_FPS, features=features, root=root,
        robot_type='rebot_b601_follower', use_videos=True, image_writer_threads=4)
    masker = Masker(masking_strategy=strategy)
    written, skipped = [], []
    # The probe episode is exported with the rest, in directory order.
    del first
    try:
        for episode in episodes:
            try:
                count = export_episode(dataset, episode, masker, task)
            except (OSError, ValueError, KeyError, CameraTimingError) as exc:
                skipped.append((episode.name, str(exc)))
                continue
            written.append((episode.name, count))
            print(f'[export] {episode.name}: {count} frames', flush=True)
    finally:
        dataset.finalize()
    if not written:
        raise ValueError('No episode could be exported')
    print(f'[export] wrote {len(written)} episodes, skipped {len(skipped)}, to {root}', flush=True)
    for name, reason in skipped:
        print(f'[export] skipped {name}: {reason}', flush=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export raw B601 episodes as a masked Pi0.5 LeRobot dataset')
    parser.add_argument('--dataset-dir', default='data/b601_single')
    parser.add_argument('--output', default='data/b601_single_pi05_end_effector')
    parser.add_argument('--masking-strategy', default='end_effector')
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args(argv)
    export_dataset(args.dataset_dir, args.output, args.masking_strategy, args.task, args.limit)


if __name__ == '__main__':
    main()
