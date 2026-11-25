"""B601 MIT adapter with a fake RebotArm. No serial port and no motor enable."""
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from config.run import build_parser, build_plan
from config.task_config import CONFIG_ROOT, default_start_position_path, load_task_config
from hardware.arm_factory import build_arms, normalize_hardware_config
from hardware.backends.b601 import (
    B601Adapter, REBOT_ARM_ROOT, resolve_hardware_yaml, save_mit_profile, urdf_arm_limits, b601_urdf_path)
from hardware.runtime import HardwareRuntime
from recording.b601_session import apply_start_key, collect_guided, save_start_pose, start_pose_path


ROOT = Path(__file__).resolve().parents[2]
URDF = b601_urdf_path()


class FakeGroup:
    def __init__(self, count, owner):
        self.pos = np.zeros(count)
        self.vel = np.zeros(count)
        self.owner = owner
        self.calls = []

    def mode_mit(self, kp=None, kd=None):
        return True

    def enable(self):
        pass

    def disable(self):
        pass

    def get_positions(self, request_feedback=True):
        return self.pos.copy()

    def get_velocities(self, request_feedback=True):
        return self.vel.copy()

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None):
        self.calls.append({
            'pos': np.asarray(pos, dtype=float).copy(),
            'kp': np.asarray(kp, dtype=float).copy(),
            'kd': np.asarray(kd, dtype=float).copy(),
            'tau': np.asarray(tau, dtype=float).copy(),
        })
        self.owner.frames.append(self.calls[-1])


class FakeRebotArm:
    def __init__(self, hardware_yaml):
        self.hardware_yaml = hardware_yaml
        self._channel = None
        self.frames = []
        self.arm = FakeGroup(6, self)
        self.gripper = FakeGroup(1, self)
        self.callback = None
        self.connected = False

    def connect(self):
        self.connected = True

    def enable_all(self):
        pass

    def disable_all(self):
        pass

    def disconnect(self):
        self.connected = False

    def start_control_loop(self, fn, rate=None):
        self.callback = fn

    def stop_control_loop(self):
        self.callback = None

    def pump(self):
        self.callback(self, 0.002)


def write_profile(tmp_path, urdf):
    document = {
        'rate': 500,
        'urdf_path': str(urdf),
        'gravity_compensation': {'kp': 2.0, 'kd': 1.5, 'tau_scale': [1, 1, 1, 1, 1, 1]},
        'joints': [
            {'name': f'joint{index}', 'MIT': {'kp': kp, 'kd': kd}}
            for index, kp, kd in (
                (1, 120.0, 8.0), (2, 120.0, 8.0), (3, 120.0, 8.0),
                (4, 18.0, 2.0), (5, 18.0, 2.0), (6, 18.0, 2.0),
            )
        ] + [{'name': 'gripper', 'MIT': {'kp': 8.0, 'kd': 1.0}}],
    }
    path = tmp_path / 'rebotarm_dm.yaml'
    path.write_text(yaml.safe_dump(document))
    return path


def make_arm(tmp_path, **kwargs):
    profile = write_profile(tmp_path, URDF)
    gravity = kwargs.pop('gravity', lambda _position: np.arange(1, 8, dtype=float))
    return B601Adapter(
        'left', 'B601_DM', '/dev/ttyACM0', str(profile),
        [0.0, -1.0, -1.2, 0.0, 0.0, 0.0, 0.0],
        robot_factory=FakeRebotArm, gravity=gravity,
        command_timeout_s=5.0, **kwargs)


def test_factory_accepts_b601_and_still_rejects_an_unknown_arx_model(tmp_path):
    profile = write_profile(tmp_path, URDF)
    config = {'type': 'b601', 'arms': {'left': {
        'model': 'B601_DM', 'serial_port': '/dev/ttyACM0', 'hardware_yaml': str(profile),
        'home_qpos': [0.0, -1.0, -1.2, 0.0, 0.0, 0.0, 0.0]}}}
    assert normalize_hardware_config(config)['type'] == 'b601'
    arm = build_arms(config)['left']
    try:
        assert arm.describe().model == 'B601_DM'
        assert arm._robot is None
    finally:
        arm.close()
    with pytest.raises(ValueError):
        normalize_hardware_config({'type': 'arx', 'arms': {'left': {'model': 'X5', 'can_interface': 'can0'}}})
    with pytest.raises(ValueError, match='Unknown robot.type'):
        load_task_config(CONFIG_ROOT / 'tasks' / 'single_arm.yaml', overrides={'robot': {'type': 'other'}})


def test_shipped_task_uses_the_dm_profile_and_has_no_arx_start_fallback():
    task = load_task_config(CONFIG_ROOT / 'tasks' / 'b601_single.yaml')
    assert task.as_dict()['robot']['type'] == 'b601'
    hardware_yaml = Path(task.as_dict()['robot']['arms']['left']['hardware_yaml'])
    assert hardware_yaml == (CONFIG_ROOT / 'hardware' / 'b601_dm.yaml').resolve()
    assert resolve_hardware_yaml('config/rebotarm_dm.yaml') == REBOT_ARM_ROOT / 'config/rebotarm_dm.yaml'
    arm = build_arms(task.as_dict()['robot'])['left']
    try:
        assert arm._track_kp.tolist() == [120.0, 120.0, 120.0, 18.0, 18.0, 18.0, 8.0]
        assert arm._track_kd.tolist() == [8.0, 8.0, 8.0, 2.0, 2.0, 2.0, 1.0]
        assert arm._tau_scale.tolist() == [1.0, 0.96, 0.91, 0.9, 0.6, 1.0, 1.0]
        assert arm._compliant_kp.tolist() == [1.35, 1.35, 1.35, 0.0, 0.0, 0.0, 0.0]
        assert arm._compliant_kd.tolist() == [0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.0]
    finally:
        arm.close()
    with pytest.raises(FileNotFoundError, match='no saved start pose'):
        default_start_position_path('b601_single', 'b601')


def test_hundred_hertz_is_legal_and_urdf_limits_reject_joint2(tmp_path):
    lower, upper = urdf_arm_limits(URDF)
    assert upper[1] == 0.0 and lower[1] < 0.0
    arm = make_arm(tmp_path)
    try:
        legal = np.array([0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0])
        assert arm.validate_target(legal, interval_s=0.01).shape == (7,)
        with pytest.raises(ValueError, match='URDF'):
            arm.validate_target(np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]), interval_s=0.01)
    finally:
        arm.close()


def test_tracking_compliant_and_safe_stop_frames(tmp_path):
    arm = make_arm(tmp_path)
    arm.connect()
    try:
        target = np.array([0.1, -0.4, -0.5, 0.2, -0.2, 0.3, 0.4])
        arm.submit_target(target, interval_s=0.01)
        arm._robot.pump()
        tracking = arm._robot.frames[-2:]
        assert np.allclose(np.r_[tracking[0]['kp'], tracking[1]['kp']],
                           [120, 120, 120, 18, 18, 18, 8])
        assert np.allclose(np.r_[tracking[0]['kd'], tracking[1]['kd']],
                           [8, 8, 8, 2, 2, 2, 1])
        assert np.allclose(np.r_[tracking[0]['tau'], tracking[1]['tau']], np.arange(1, 8))
        arm.set_compliant(True)
        arm._robot.pump()
        compliant = arm._robot.frames[-2:]
        assert np.allclose(np.r_[compliant[0]['kp'], compliant[1]['kp']], [2, 2, 2, 2, 2, 2, 0])
        assert np.allclose(np.r_[compliant[0]['kd'], compliant[1]['kd']], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0])
        assert np.allclose(np.r_[compliant[0]['tau'], compliant[1]['tau']], [1, 2, 3, 4, 5, 6, 0])
        arm.submit_safe_stop(interval_s=0.01)
        arm._robot.pump()
        stopped = arm._robot.frames[-2:]
        assert np.allclose(np.r_[stopped[0]['kp'], stopped[1]['kp']], 0.0)
        assert np.allclose(np.r_[stopped[0]['kd'], stopped[1]['kd']], [8, 8, 8, 2, 2, 2, 1])
        assert np.allclose(np.r_[stopped[0]['tau'], stopped[1]['tau']], np.arange(1, 8))
    finally:
        arm.close()


def test_guide_recorder_stores_the_measured_pose(tmp_path):
    arm = make_arm(tmp_path)
    arm.arm_wait = None
    measured = np.array([0.2, -0.3, -0.4, 0.1, -0.1, 0.2, 0.5])
    runtime = HardwareRuntime({'left': arm})
    runtime.connect()
    try:
        arm._robot.arm.pos[:] = measured[:6]
        arm._robot.gripper.pos[:] = measured[6:]
        saved = []
        collect_guided(runtime, [], 1, 0.01, saved.append)
        np.testing.assert_allclose(saved[0]['commanded_action'], measured)
        np.testing.assert_allclose(saved[0]['qpos'], measured)
    finally:
        runtime.close()


def test_start_pose_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr('recording.b601_session.REPO_ROOT', tmp_path)
    measured = np.array([0.2, -0.3, -0.4, 0.1, -0.1, 0.2, 0.5])
    assert apply_start_key(' ', measured, None).shape == (7,)
    locked = apply_start_key(' ', measured, None)
    with pytest.raises(ValueError, match='space'):
        apply_start_key('y', measured, None)
    path = save_start_pose('b601_single', apply_start_key('y', measured, locked))
    assert path == start_pose_path('b601_single')
    with np.load(path) as data:
        np.testing.assert_allclose(data['left_qpos'], measured)


def plan(*argv):
    args, extra = build_parser().parse_known_args(argv)
    command = build_plan(args, extra)['command']
    assert command[0] == sys.executable
    return [Path(part).name if str(part).startswith(str(ROOT)) else part for part in command[1:]]


def test_finger_joints_are_dropped_from_the_gravity_torque(tmp_path):
    arm = make_arm(tmp_path, gravity=lambda _position: np.arange(1, 9, dtype=float))
    arm.connect()
    try:
        arm.submit_target(np.array([0.1, -0.4, -0.5, 0.2, -0.2, 0.3, 0.4]), interval_s=0.01)
        arm._robot.pump()
        frame = arm._robot.frames[-2:]
        assert np.allclose(np.r_[frame[0]['tau'], frame[1]['tau']], [1, 2, 3, 4, 5, 6, 0])
    finally:
        arm.close()


def test_gain_updates_reach_the_next_mit_frame(tmp_path):
    arm = make_arm(tmp_path)
    arm.connect()
    try:
        target = np.array([0.1, -0.4, -0.5, 0.2, -0.2, 0.3, 0.4])
        arm.submit_target(target, interval_s=0.01)
        arm.set_tracking_gains([10, 11, 12, 13, 14, 15, 16], [1, 1, 1, 2, 2, 2, 3])
        arm._robot.pump()
        tracking = arm._robot.frames[-2:]
        assert np.allclose(np.r_[tracking[0]['kp'], tracking[1]['kp']],
                           [10, 11, 12, 13, 14, 15, 16])
        assert np.allclose(np.r_[tracking[0]['kd'], tracking[1]['kd']],
                           [1, 1, 1, 2, 2, 2, 3])
        arm.set_compliant(True)
        arm.set_compliant_gains(4.5, 0.4)
        arm.set_tau_scale([2, 2, 2, 2, 2, 2])
        arm._robot.pump()
        compliant = arm._robot.frames[-2:]
        assert np.allclose(np.r_[compliant[0]['kp'], compliant[1]['kp']], [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 0])
        assert np.allclose(np.r_[compliant[0]['kd'], compliant[1]['kd']], [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0])
        assert np.allclose(np.r_[compliant[0]['tau'], compliant[1]['tau']], [2, 4, 6, 8, 10, 12, 0])
        wrist = [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.4]
        arm.set_compliant_gains(wrist, [1.5, 1.5, 1.5, 0.2, 0.2, 0.2, 0.2])
        arm._robot.pump()
        per_joint = arm._robot.frames[-2:]
        assert np.allclose(np.r_[per_joint[0]['kp'], per_joint[1]['kp']], [1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0])
        assert np.allclose(np.r_[per_joint[0]['kd'], per_joint[1]['kd']],
                           [1.5, 1.5, 1.5, 0.2, 0.2, 0.2, 0])
        arm.submit_safe_stop(interval_s=0.01)
        arm._robot.pump()
        stopped = arm._robot.frames[-2:]
        assert np.allclose(np.r_[stopped[0]['kp'], stopped[1]['kp']], 0.0)
        assert np.allclose(np.r_[stopped[0]['kd'], stopped[1]['kd']], [1, 1, 1, 2, 2, 2, 3])
    finally:
        arm.close()


def test_save_rewrites_gains_and_keeps_motor_identity(tmp_path):
    joints = []
    for index in range(1, 8):
        joints.append({
            'name': f'joint{index}' if index < 7 else 'gripper',
            'motor_id': index, 'feedback_id': 16 + index, 'model': '4310',
            'MIT': {'kp': 1.0, 'kd': 1.0},
            'POS_VEL': {'vel_kp': 0.0125, 'vel_ki': 0.004, 'pos_kp': 150.0, 'pos_ki': 0.5, 'vlim': 5.0},
        })
    document = {
        'rate': 500, 'urdf_path': str(URDF), 'joints': joints,
        'gravity_compensation': {'kp': 2.0, 'kd': 1.5, 'tau_scale': [1, 1, 1, 1, 1, 1],
                                 'transition_duration': 1.0},
    }
    path = tmp_path / 'b601_dm.yaml'
    path.write_text(yaml.safe_dump(document))
    save_mit_profile(path, tracking_kp=[9, 8, 7, 6, 5, 4, 3], tracking_kd=[0.5] * 7,
                     compliant_kp=3.0, compliant_kd=0.8, tau_scale=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    saved = yaml.safe_load(path.read_text())
    assert [joint['motor_id'] for joint in saved['joints']] == list(range(1, 8))
    assert saved['joints'][0]['POS_VEL'] == document['joints'][0]['POS_VEL']
    assert saved['joints'][0]['MIT'] == {'kp': 9.0, 'kd': 0.5}
    assert saved['gravity_compensation']['kp'] == 3.0
    assert saved['gravity_compensation']['kd'] == 0.8
    assert saved['gravity_compensation']['tau_scale'] == [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    assert saved['gravity_compensation']['transition_duration'] == 1.0
    save_mit_profile(path, tracking_kp=[9, 8, 7, 6, 5, 4, 3], tracking_kd=[0.5] * 7,
                     compliant_kp=[2, 2, 2, 0.2, 0.2, 0.2, 0.4],
                     compliant_kd=[1.5, 1.5, 1.5, 0.2, 0.2, 0.2, 0.2],
                     tau_scale=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    saved = yaml.safe_load(path.read_text())
    assert saved['gravity_compensation']['kp'] == [2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.4]
    assert saved['gravity_compensation']['kd'] == [1.5, 1.5, 1.5, 0.2, 0.2, 0.2, 0.2]
    assert saved['joints'][0]['POS_VEL'] == document['joints'][0]['POS_VEL']


def test_tune_keys_change_gravity_and_pd_without_touching_the_other():
    import importlib.util
    spec = importlib.util.spec_from_file_location('b601_tune', ROOT / 'scripts' / 'b601_tune.py')
    tune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune)
    gravity = {
        'mode': 'gravity', 'index': 0,
        'tau_scale': np.ones(6), 'compliant_kp': np.full(7, 2.0), 'compliant_kd': np.full(7, 1.5),
        'kp': np.full(7, 18.0), 'kd': np.full(7, 2.0),
    }
    gravity, save = tune.apply_tune_key(gravity, '\x1b[A')
    assert not save and gravity['tau_scale'][0] == pytest.approx(1.0 + tune.TAU_STEP)
    assert np.all(gravity['kp'] == 18.0)
    gravity, _ = tune.apply_tune_key(gravity, '>')
    assert gravity['compliant_kp'][0] == pytest.approx(2.0 + tune.COMPLIANT_KP_STEP)
    assert np.all(gravity['compliant_kp'][1:] == 2.0)
    pd = dict(gravity, mode='pd', index=6)
    pd, save = tune.apply_tune_key(pd, '>')
    assert not save and pd['kd'][6] == pytest.approx(2.0 + tune.KD_STEP)
    assert pd['tau_scale'][0] == gravity['tau_scale'][0]
    pd, save = tune.apply_tune_key(pd, 's')
    assert save


def test_dry_run_uses_b601_entry_points_not_the_arx_ones(tmp_path):
    collect = plan('collect', 'b601_single')
    assert collect[0] == 'record_b601.py'
    assert 'record_episode_single' not in ' '.join(collect)
    replay = plan('replay', 'b601_single', '4')
    assert replay[0] == 'replay_b601.py'
    assert 'review_replay' not in replay
    view = plan('view', 'b601_single', '4')
    assert view[:4] == ['visualize.py', 'episode', '--robot', 'b601']
    recipe = yaml.safe_load((CONFIG_ROOT / 'training' / 'single_arm_act.yaml').read_text())
    recipe.update(task=str(CONFIG_ROOT / 'tasks' / 'b601_single.yaml'), model_name='b601_unit',
                  output_dir=str(tmp_path))
    path = tmp_path / 'b601_act.yaml'
    path.write_text(yaml.safe_dump(recipe))
    direct = plan('infer', str(path))
    assert direct[:2] == ['infer.py', 'b601']
    assert 'inference_direct' not in ' '.join(direct)
    remote_doc = {
        'schema_version': 1, 'task': str(CONFIG_ROOT / 'tasks' / 'b601_single.yaml'), 'backend': 'remote',
        'execution': {'chunk_steps': 50, 'async_inference': True, 'overlap_steps': 10},
        'rtc': {'training_time': False, 'enabled': False},
        'connection': {'host': 'localhost', 'port': 8000},
    }
    remote_path = tmp_path / 'b601_remote.yaml'
    remote_path.write_text(yaml.safe_dump(remote_doc))
    remote = plan('infer', str(path), '--inference-config', str(remote_path))
    assert remote[:2] == ['infer.py', 'remote']
    assert '--shared-hardware' in remote
    assert 'inference_direct' not in ' '.join(remote)
