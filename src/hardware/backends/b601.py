"""reBot B601-DM adapter: Damiao MIT through reBotArm_control_py.

Joint targets are the vendor's native radians (six arm joints plus the gripper
motor). They are not converted into the ARX metre gripper. The 500 Hz loop keeps
sending the latest MIT frame so a Damiao command timeout cannot drop the arm
between policy ticks. Gravity torque comes from the vendor dynamics, scaled by
the DM profile. Position-velocity mode is never selected.
"""
import os
from pathlib import Path
import platform
import struct
import time
import xml.etree.ElementTree as ET
import threading

import numpy as np
import yaml

from hardware.arm_interface import ArmDescription, ArmState, ManagedArm


REBOT_ARM_ROOT = Path(os.environ.get(
    'REBOT_ARM_ROOT', '/home/thor/strenth-flow/vendor/reBotArm_control_py'))
ARM_JOINTS = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
# The DM URDF models the fingers as prismatic joints. The gripper motor itself
# is commanded in radians over the Damiao MIT position span.
GRIPPER_MOTOR_LIMIT = (-12.5, 12.5)
MODEL = 'B601_DM'
# ELF e_machine values. The vendor checkout on this machine may contain a
# library built for a different CPU.
_ELF_MACHINE = {'x86_64': 62, 'aarch64': 183, 'arm64': 183}


def _library_machine(path):
    data = Path(path).read_bytes()[:20]
    if len(data) < 20 or data[:4] != b'\x7fELF':
        return None
    endian = '<' if data[5] == 1 else '>'
    return struct.unpack_from(endian + 'H', data, 18)[0]


def prepare_vendor_imports():
    """Put the reBot checkout, and a same-architecture motorbridge, on sys.path.

    motorbridge is not part of this repository. A copy that exists under the
    vendor virtualenv is used only when its shared library matches this CPU.
    Pinocchio is imported by the dynamics call, before any motor is enabled.
    """
    import sys
    root = str(REBOT_ARM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import motorbridge  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    wanted = _ELF_MACHINE.get(platform.machine())
    wrong = []
    lib = REBOT_ARM_ROOT / '.venv' / 'lib'
    for site in sorted(lib.glob('python*/site-packages')) if lib.is_dir() else []:
        library = site / 'motorbridge' / 'lib' / 'libmotor_abi.so'
        if not library.is_file():
            continue
        machine = _library_machine(library)
        if wanted is not None and machine not in (None, wanted):
            wrong.append(library)
            continue
        if str(site) not in sys.path:
            sys.path.append(str(site))
        return
    detail = ''
    if wrong:
        detail = (f' The library at {wrong[0]} is not built for {platform.machine()} '
                  'and was not added to the import path.')
    raise ModuleNotFoundError(
        'motorbridge is not installed for this Python '
        f'({platform.python_version()}, {platform.machine()}).{detail} '
        'Install motorbridge and pinocchio for this interpreter before connecting the arm.')


def b601_urdf_path():
    return REBOT_ARM_ROOT / 'urdf' / 'DM' / 'urdf' / 'ReBot_Arm_DM.urdf'


def resolve_hardware_yaml(hardware_yaml):
    path = Path(hardware_yaml).expanduser()
    if path.is_absolute():
        return path
    return REBOT_ARM_ROOT / path


def urdf_arm_limits(urdf_path):
    """Lower and upper bounds for joint1..joint6, in URDF order."""
    root = ET.parse(urdf_path).getroot()
    found = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in ARM_JOINTS:
            continue
        limit = joint.find('limit')
        if limit is None:
            raise ValueError(f'{urdf_path} joint {name} has no limit')
        found[name] = (float(limit.get('lower')), float(limit.get('upper')))
    missing = [name for name in ARM_JOINTS if name not in found]
    if missing:
        raise ValueError(f'{urdf_path} is missing joint limits for {missing}')
    lower = np.array([found[name][0] for name in ARM_JOINTS] + [GRIPPER_MOTOR_LIMIT[0]], dtype=float)
    upper = np.array([found[name][1] for name in ARM_JOINTS] + [GRIPPER_MOTOR_LIMIT[1]], dtype=float)
    return lower, upper


def load_mit_profile(hardware_yaml):
    """MIT gains, gravity scales, rate, and URDF limits from the DM hardware YAML."""
    path = resolve_hardware_yaml(hardware_yaml)
    document = yaml.safe_load(path.read_text())
    joints = document.get('joints') or []
    if len(joints) != 7:
        raise ValueError(f'{path} must list 6 arm joints and the gripper')
    kp, kd = [], []
    for joint in joints:
        mit = joint.get('MIT') or {}
        kp.append(float(mit['kp']))
        kd.append(float(mit['kd']))
    gravity = document.get('gravity_compensation') or {}
    compliant_kp = _compliant_vector(gravity.get('kp', 2.0), 'compliant kp')
    compliant_kd = _compliant_vector(gravity.get('kd', 1.5), 'compliant kd')
    tau_scale = np.asarray(gravity.get('tau_scale', [1.0] * 6), dtype=float).reshape(-1)
    if tau_scale.size == 6:
        tau_scale = np.r_[tau_scale, 1.0]
    if tau_scale.size != 7:
        raise ValueError(f'{path} gravity tau_scale must cover the arm joints')
    urdf = document.get('urdf_path')
    limits = None
    if urdf:
        urdf_path = Path(urdf).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = REBOT_ARM_ROOT / urdf_path
        if urdf_path.is_file():
            limits = urdf_arm_limits(urdf_path)
    return {
        'kp': np.asarray(kp, dtype=float),
        'kd': np.asarray(kd, dtype=float),
        'compliant_kp': compliant_kp,
        'compliant_kd': compliant_kd,
        'tau_scale': tau_scale,
        'rate': float(document.get('rate', 500.0)),
        'limits': limits,
        'urdf': None if not urdf else str(REBOT_ARM_ROOT / urdf if not Path(urdf).is_absolute() else Path(urdf)),
    }


def _gain_vector(values, size, name):
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all() or np.any(vector < 0):
        raise ValueError(f'{name} must be {size} finite nonnegative numbers')
    return vector


def _compliant_vector(value, name):
    """One gravity gain for every motor, or a gain for each of the seven motors."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.full(7, float(array[0]))
    return _gain_vector(array, 7, name)


def _stored_compliant(vector):
    """Keep a single number when every motor shares it. Otherwise store seven."""
    if np.allclose(vector, vector[0]):
        return float(vector[0])
    return [float(value) for value in vector]


def save_mit_profile(hardware_yaml, *, tracking_kp, tracking_kd, compliant_kp, compliant_kd, tau_scale):
    """Rewrite MIT kp/kd and gravity compensation. Motor ids and POS_VEL stay put."""
    path = resolve_hardware_yaml(hardware_yaml)
    document = yaml.safe_load(path.read_text())
    joints = document.get('joints') or []
    kp = _gain_vector(tracking_kp, 7, 'tracking kp')
    kd = _gain_vector(tracking_kd, 7, 'tracking kd')
    if len(joints) != 7:
        raise ValueError(f'{path} must list 6 arm joints and the gripper')
    for joint, joint_kp, joint_kd in zip(joints, kp, kd):
        mit = joint.setdefault('MIT', {})
        mit['kp'] = float(joint_kp)
        mit['kd'] = float(joint_kd)
    scale = _gain_vector(tau_scale, 6, 'tau_scale')
    compliant_kp = _compliant_vector(compliant_kp, 'compliant kp')
    compliant_kd = _compliant_vector(compliant_kd, 'compliant kd')
    gravity = document.setdefault('gravity_compensation', {})
    gravity['kp'] = _stored_compliant(compliant_kp)
    gravity['kd'] = _stored_compliant(compliant_kd)
    gravity['tau_scale'] = [float(value) for value in scale]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


class B601Adapter(ManagedArm):
    """One B601-DM arm. ``robot_factory`` and ``gravity`` are test injection points."""

    def __init__(self, role, model, serial_port, hardware_yaml, home_qpos, *,
                 robot_factory=None, gravity=None, limits=None, home_duration_s=3.0, **options):
        if model != MODEL:
            raise ValueError(f'Unsupported B601 robot model {model!r}')
        profile = load_mit_profile(hardware_yaml)
        self._serial = serial_port
        self._hardware_yaml = str(resolve_hardware_yaml(hardware_yaml))
        self._robot_factory = robot_factory
        self._gravity_fn = gravity
        self._track_kp = profile['kp']
        self._track_kd = profile['kd']
        self._compliant_kp = np.asarray(profile['compliant_kp'], dtype=float).copy()
        self._compliant_kd = np.asarray(profile['compliant_kd'], dtype=float).copy()
        self._tau_scale = profile['tau_scale']
        chosen = limits if limits is not None else profile['limits']
        if chosen is None:
            raise ValueError(f'{hardware_yaml} does not name a readable URDF for joint limits')
        self._lower, self._upper = (np.asarray(side, dtype=float).reshape(7) for side in chosen)
        self._home_qpos = np.asarray(home_qpos, dtype=float).reshape(7)
        if self._home_qpos.shape != (7,) or not np.isfinite(self._home_qpos).all():
            raise ValueError('B601 home_qpos must be 7 finite joint radians')
        if np.any(self._home_qpos < self._lower) or np.any(self._home_qpos > self._upper):
            raise ValueError('B601 home_qpos exceeds URDF joint limits')
        self._home_duration = float(home_duration_s)
        self._robot = None
        self._cmd_lock = threading.Lock()
        self._mode = 'tracking'
        self._target = None
        self._hold = None
        self._latest_q = np.zeros(7)
        self._latest_dq = np.zeros(7)
        self._latest_stamp = None
        super().__init__(ArmDescription(role, model, 'b601', 7, profile['rate']), **options)

    def set_compliant(self, enabled=True):
        """Hand guiding uses the DM gravity-compensation gains instead of tracking gains."""
        with self._cmd_lock:
            self._mode = 'compliant' if enabled else 'tracking'
            self._hold = None

    def set_tracking_gains(self, kp, kd):
        """MIT kp/kd for tracking and for the kd term of a safe stop. Used on the next frame."""
        kp = _gain_vector(kp, 7, 'tracking kp')
        kd = _gain_vector(kd, 7, 'tracking kd')
        with self._cmd_lock:
            self._track_kp = kp
            self._track_kd = kd

    def set_compliant_gains(self, kp, kd):
        """Gravity-compensation kp and kd. A single number applies to every motor."""
        kp = _compliant_vector(kp, 'compliant kp')
        kd = _compliant_vector(kd, 'compliant kd')
        with self._cmd_lock:
            self._compliant_kp = kp
            self._compliant_kd = kd

    def set_tau_scale(self, scale):
        """Per-arm-joint gravity scale. The gripper scale stays 1. Used on the next frame."""
        scale = _gain_vector(scale, 6, 'tau_scale')
        with self._cmd_lock:
            self._tau_scale = np.r_[scale, 1.0]

    def move_toward(self, target, *, duration_s, interval_s):
        """Interpolate to ``target`` under tracking gains. Used by hold and the start of inference."""
        goal = self.validate_target(target, interval_s=interval_s)
        start = self._positions()
        steps = max(1, int(round(float(duration_s) / interval_s)))
        for index in range(1, steps + 1):
            self.submit_target(start + (goal - start) * (index / steps), interval_s=interval_s)
            time.sleep(interval_s)

    def _connect(self):
        factory = self._robot_factory
        if factory is None:
            prepare_vendor_imports()
            from reBotArm_control_py.actuator import RebotArm
            factory = RebotArm
        robot = factory(self._hardware_yaml)
        # The task selects the serial bridge. The vendor YAML's channel is the default only.
        robot._channel = self._serial
        if self._gravity_fn is None:
            # Fail here, before enable, when pinocchio is missing.
            from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: F401
        robot.connect()
        robot.arm.mode_mit()
        robot.gripper.mode_mit()
        robot.enable_all()
        self._robot = robot
        self._sample(robot)
        robot.start_control_loop(self._on_control, rate=self.describe().motor_hz)

    def _on_control(self, robot, dt):
        try:
            self._send_frame(robot)
        except Exception as exc:
            self.fail(str(exc) or type(exc).__name__)

    def _sample(self, robot):
        position = np.r_[np.asarray(robot.arm.get_positions(), dtype=float).reshape(-1),
                         np.asarray(robot.gripper.get_positions(), dtype=float).reshape(-1)]
        velocity = np.r_[np.asarray(robot.arm.get_velocities(), dtype=float).reshape(-1),
                         np.asarray(robot.gripper.get_velocities(), dtype=float).reshape(-1)]
        if position.shape != (7,) or velocity.shape != (7,):
            raise RuntimeError('B601 feedback is not 6 joints plus the gripper')
        with self._cmd_lock:
            self._latest_q = position
            self._latest_dq = velocity
            self._latest_stamp = self._clock()
        return position, velocity

    def _gravity(self, position, tau_scale=None):
        if self._gravity_fn is not None:
            torque = np.asarray(self._gravity_fn(position), dtype=float).reshape(-1)
        else:
            prepare_vendor_imports()
            from reBotArm_control_py.dynamics import compute_generalized_gravity
            torque = np.asarray(compute_generalized_gravity(q=position[:6]), dtype=float).reshape(-1)
        # The DM URDF adds two finger joints after the six arm joints. The gripper
        # motor is not one of those, so its gravity term stays zero.
        if torque.size > 7:
            torque = np.r_[torque[:6], 0.0]
        elif torque.size == 6:
            torque = np.r_[torque, 0.0]
        if torque.shape != (7,):
            raise RuntimeError('B601 gravity torque must cover 6 joints plus the gripper')
        if tau_scale is None:
            with self._cmd_lock:
                tau_scale = self._tau_scale
        return torque * tau_scale

    def _send_frame(self, robot):
        position, _velocity = self._sample(robot)
        with self._cmd_lock:
            mode = self._mode
            target = None if self._target is None else self._target.copy()
            hold = None if self._hold is None else self._hold.copy()
            track_kp = self._track_kp
            track_kd = self._track_kd
            compliant_kp = self._compliant_kp
            compliant_kd = self._compliant_kd
            tau_scale = self._tau_scale
        if mode == 'safe':
            command = position if hold is None else hold
            kp = np.zeros(7)
            kd = track_kd
        elif mode == 'compliant':
            command = position if target is None else target
            kp = np.array(compliant_kp, copy=True)
            kd = np.array(compliant_kd, copy=True)
            # Hand guiding leaves the gripper free. Position feedback still arrives.
            kp[6] = 0.0
            kd[6] = 0.0
        else:
            command = position if target is None else target
            kp = track_kp
            kd = track_kd
        torque = self._gravity(position, tau_scale)
        if mode == 'compliant':
            torque = np.array(torque, copy=True)
            torque[6] = 0.0
        zeros6 = np.zeros(6)
        zeros1 = np.zeros(1)
        robot.arm.send_mit(command[:6], zeros6, kp[:6], kd[:6], torque[:6])
        robot.gripper.send_mit(command[6:], zeros1, kp[6:], kd[6:], torque[6:])

    def _positions(self):
        with self._cmd_lock:
            if self._latest_stamp is None:
                raise RuntimeError('B601 has no feedback yet')
            return self._latest_q.copy()

    def _validate_limits(self, target):
        if np.any(target < self._lower) or np.any(target > self._upper):
            raise ValueError('B601 target exceeds URDF joint limits')

    def _read_state(self):
        if self._robot is not None:
            self._sample(self._robot)
        with self._cmd_lock:
            if self._latest_stamp is None:
                raise RuntimeError('B601 has no feedback yet')
            position = self._latest_q.copy()
            velocity = self._latest_dq.copy()
            stamp = self._latest_stamp
        age = max(0.0, self._clock() - stamp)
        return ArmState(position, velocity, np.full(7, age), np.full(7, stamp))

    def _submit_target(self, target, *, interval_s):
        with self._cmd_lock:
            if self._mode == 'safe':
                self._mode = 'tracking'
                self._hold = None
            self._target = np.asarray(target, dtype=float).copy()

    def _submit_safe_stop(self, *, interval_s):
        with self._cmd_lock:
            self._mode = 'safe'
            self._hold = self._latest_q.copy()
            self._target = self._hold.copy()

    def _home(self):
        start = self._positions()
        goal = self._home_qpos.copy()
        duration = self._home_duration
        interval = 1.0 / self.describe().motor_hz
        steps = 1 if duration <= 0 else max(1, int(round(duration / interval)))
        for index in range(1, steps + 1):
            point = start + (goal - start) * (index / steps)
            with self._cmd_lock:
                self._mode = 'tracking'
                self._hold = None
                self._target = point
            if steps > 1:
                time.sleep(interval)

    def _close(self):
        robot = self._robot
        self._robot = None
        if robot is None:
            return
        stop = getattr(robot, 'stop_control_loop', None)
        if stop is not None:
            stop()
        disable = getattr(robot, 'disable_all', None)
        if disable is not None:
            disable()
        disconnect = getattr(robot, 'disconnect', None)
        if disconnect is not None:
            disconnect()
