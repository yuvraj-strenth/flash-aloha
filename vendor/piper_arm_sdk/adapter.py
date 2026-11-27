"""Named points and metre/radian TCP controls for a Piper robot.

Use ``with PiperAdapter(can="can0") as arm:`` to manage the CAN connection.
Cartesian points describe the TCP, with tool axes aligned to the flange and a
configurable flange-to-TCP offset.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from scipy.spatial.transform import Rotation


def _load_piper():
    """Load sibling robot.py instead of the agent robot package."""
    if __package__:
        from .robot import Piper
        return Piper
    path = Path(__file__).with_name("robot.py")
    spec = importlib.util.spec_from_file_location("piper_arm_sdk.robot", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Piper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Piper


Piper = _load_piper()

_POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
_TEACH_ARM_STATUS = (0x0B, 0x0C, 0x0D)


def _arm_is_teaching(status) -> bool:
    """Return whether this arm-status message is drag-teach control."""
    arm_status = int(status.arm_status)
    if arm_status != 0 and arm_status not in _TEACH_ARM_STATUS:
        return False
    teach = status.teach_status.value
    return (
        int(status.ctrl_mode) == 0x02
        or teach not in (0x00, 0x02)
        or arm_status in _TEACH_ARM_STATUS
    )


class PiperAdapter:
    def __init__(
        self, can="can0", json_path=None, tcp_offset_m=(0, 0, 0.145),
        rest_on_connect=True,
    ):
        """Connect and enable. Home only when ``rest_on_connect`` is true."""
        self.json_path = (
            Path(__file__).with_name("saved_positions.json")
            if json_path is None else Path(json_path).expanduser()
        ).resolve()
        self.tcp_offset_m = self._numbers(tcp_offset_m, 3)
        self._save_lock = threading.Lock()
        self.robot = Piper(can=can)
        try:
            self.enable()
            if rest_on_connect:
                self.move_to_rest()
        except BaseException:
            self.close()
            raise

    def close(self):
        return self.robot.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _numbers(values, count):
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(f"Expected {count} finite numbers")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for value in values):
            raise ValueError(f"Expected {count} finite numbers")
        try:
            result = tuple(float(value) for value in values)
        except OverflowError as exc:
            raise ValueError(f"Expected {count} finite numbers") from exc
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"Expected {count} finite numbers")
        return result

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Point name must be a nonblank string")

    @staticmethod
    def _speed(speed):
        if type(speed) is not int or not 1 <= speed <= 100:
            raise ValueError("speed must be an integer from 1 to 100")

    @staticmethod
    def _frame(frame):
        if frame not in ("base", "tool"):
            raise ValueError("frame must be 'base' or 'tool'")

    def get_joints(self) -> tuple[float, ...]:
        """Read six joint angles in radians."""
        return self._numbers(self.robot.get_joint_state(), 6)

    def get_tcp(self) -> tuple[float, ...]:
        """Read TCP XYZ in metres and roll/pitch/yaw in radians."""
        flange = self._numbers(self.robot.get_end_pose(), 6)
        rpy = tuple(map(math.radians, flange[3:]))
        offset = Rotation.from_euler("xyz", rpy).apply(self.tcp_offset_m)
        return tuple(flange[i] / 1000 + offset[i] for i in range(3)) + rpy

    def _flange_pose(self, pose):
        pose = self._numbers(pose, 6)
        offset = Rotation.from_euler("xyz", pose[3:]).apply(self.tcp_offset_m)
        return tuple((pose[i] - offset[i]) * 1000 for i in range(3)) + tuple(
            map(math.degrees, pose[3:])
        )

    def save_joints(self, name) -> tuple[float, ...]:
        self._name(name)
        with self._save_lock:
            data = self._read_store()
            joints = self.get_joints()
            data["piper"]["joints"][name] = list(joints)
            self._write_store(data)
        return joints

    def save_pose(self, name) -> tuple[float, ...]:
        self._name(name)
        with self._save_lock:
            data = self._read_store()
            pose = self.get_tcp()
            data["piper"]["cartesian"][name] = dict(zip(_POSE_FIELDS, pose))
            self._write_store(data)
        return pose

    def command_joints(self, joints, *, speed=100):
        """Command six absolute joint angles in radians."""
        joints = self._numbers(joints, 6)
        self._speed(speed)
        return self.robot.move_joints(joints, speed=speed)

    def start_joint_stream(self, *, speed=100):
        """Select position control for continuous arm playback."""
        self._speed(speed)
        self.robot.start_joint_stream(speed=speed)

    def stream_joints(self, joints):
        """Send one validated joint sample on the active stream."""
        self.robot.stream_joints(self._numbers(joints, 6))

    def command_pose(self, pose, *, speed=100, linear=False):
        """Command an absolute TCP pose in metres and radians.

        ``linear=False`` sends MOVE P. ``linear=True`` sends MOVE L.
        """
        self._speed(speed)
        flange = self._flange_pose(pose)
        if linear:
            return self.robot.move_linear(flange, speed=speed)
        return self.robot.move_pose(flange, speed=speed)

    def move_to_joint(self, name, *, speed=100):
        """Command a saved joint point."""
        self._name(name)
        self._speed(speed)
        joints = self._read_store()["piper"]["joints"][name]
        return self.robot.move_joints(joints, speed=speed)

    def move_to_pose(self, name, *, speed=100):
        """Command a saved TCP pose."""
        self._name(name)
        self._speed(speed)
        saved = self._read_store()["piper"]["cartesian"][name]
        pose = tuple(saved[field] for field in _POSE_FIELDS)
        return self.robot.move_pose(self._flange_pose(pose), speed=speed)

    def move_to_rest(self, *, speed=30):
        """Move all joints to zero and close the gripper."""
        self._speed(speed)
        return self.robot.move_zero(speed=speed)

    def move_translate(self, x, y, z, *, speed=100, frame="base"):
        """Translate the TCP by metres in base or tool coordinates."""
        delta = self._numbers((x, y, z), 3)
        self._speed(speed)
        self._frame(frame)
        current = self.get_tcp()
        if frame == "tool":
            delta = Rotation.from_euler("xyz", current[3:]).apply(delta)
        target = tuple(current[i] + delta[i] for i in range(3)) + current[3:]
        return self.robot.move_linear(self._flange_pose(target), speed=speed)

    def move_rotate(self, roll, pitch, yaw, *, speed=100, frame="base"):
        """Rotate around the TCP by radians in base or tool coordinates."""
        angles = self._numbers((roll, pitch, yaw), 3)
        self._speed(speed)
        self._frame(frame)
        current = self.get_tcp()
        rotation = Rotation.from_euler("xyz", current[3:])
        delta = Rotation.from_euler("xyz", angles)
        target_rotation = delta * rotation if frame == "base" else rotation * delta
        target = current[:3] + tuple(target_rotation.as_euler("xyz"))
        return self.robot.move_pose(self._flange_pose(target), speed=speed)

    def abort(self):
        """Send the Piper controller stop command."""
        return self.robot.stop()

    def set_mit_mode(self, speed=0):
        return self.robot.set_mit_mode(speed=speed)

    def enable(self, timeout: float = 5) -> None:
        """Enable all six Piper motors through the SDK."""
        self.robot.enable(timeout)

    def disable(self, timeout: float = 5) -> None:
        """Disable all six Piper motors through the SDK."""
        self.robot.disable(timeout=timeout)

    def teaching(self) -> bool:
        """Return whether fresh status shows the arm button's teach mode."""
        feedback = self.get_status()
        if feedback.time_stamp <= 0 or time.time() - feedback.time_stamp > 1:
            return False
        return _arm_is_teaching(feedback.arm_status)

    def enable_can(self, timeout: float = 5) -> None:
        """Hold the joints just measured and engage CAN command control.

        An active drag-teach state is ended before that hold, so the previous
        CAN target cannot run when teaching stops.
        """
        joints = self.get_joints()
        deadline = time.monotonic() + timeout
        commanded_at = None
        ready_since = None
        while True:
            feedback = self.robot.arm.GetArmStatus()
            stamp = feedback.time_stamp
            if stamp > 0 and time.time() - stamp <= 1:
                status = feedback.arm_status
                if _arm_is_teaching(status):
                    self.robot.arm.MotionCtrl_1(0x00, 0x00, 0x02)
                    self.command_joints(joints, speed=20)
                    commanded_at = time.time()
                    ready_since = None
                elif commanded_at is None:
                    self.command_joints(joints, speed=20)
                    commanded_at = time.time()
                elif (stamp > commanded_at and int(status.arm_status) == 0
                      and int(status.ctrl_mode) == 0x01
                      and int(status.mode_feed) == 0x01):
                    if ready_since is None:
                        ready_since = stamp
                    if stamp - ready_since >= 0.5:
                        return
                else:
                    ready_since = None
            if time.monotonic() >= deadline:
                status = self.robot.arm.GetArmStatus().arm_status
                raise TimeoutError(
                    "CAN control mode did not engage: ctrl={} teach={}".format(
                        status.ctrl_mode, status.teach_status
                    )
                )
            time.sleep(0.1)

    def clean_collisions(
        self, joints: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        timeout: float = 5,
    ) -> list[int]:
        """Clear collision and driver errors on the selected joints."""
        return self.robot.clear_collision(joints=joints, timeout=timeout)

    def clean_other_issues(self, width_m: float) -> None:
        """Clear an emergency stop and gripper error at the current width."""
        self.robot.clear_emergency_stop()

    def get_status(self):
        return self.robot.get_status()

    def control_gripper(self, width_m, *, effort=1000, clear_error=False):
        """Set an absolute gripper width in metres."""
        width_m = self._numbers((width_m,), 1)[0]
        if width_m < 0:
            raise ValueError("Gripper width must be nonnegative")
        return self.robot.set_gripper_width(
            width_m, effort=effort, clear_error=clear_error
        )

    def get_gripper_width(self) -> float:
        """Read gripper width in metres. Rejects stale feedback."""
        feedback = self.robot.get_gripper_status()
        if feedback.time_stamp <= 0 or time.time() - feedback.time_stamp > 1:
            raise TimeoutError("Gripper feedback is stale")
        try:
            raw = feedback.gripper_state.grippers_angle
        except AttributeError as exc:
            raise ValueError("Invalid gripper feedback") from exc
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("Invalid gripper feedback")
        try:
            width = float(raw) / 1_000_000
        except OverflowError as exc:
            raise ValueError("Invalid gripper feedback") from exc
        if not math.isfinite(width) or width < 0:
            raise ValueError("Invalid gripper feedback")
        return width

    def _read_store(self):
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Point file must contain a JSON object")
        points = data.setdefault("piper", {})
        if not isinstance(points, dict):
            raise ValueError("'piper' must be a JSON object")
        for category in ("joints", "cartesian"):
            if not isinstance(points.setdefault(category, {}), dict):
                raise ValueError(f"'piper.{category}' must be a JSON object")
        for name, joints in points["joints"].items():
            self._name(name)
            self._numbers(joints, 6)
        for name, pose in points["cartesian"].items():
            self._name(name)
            if not isinstance(pose, dict) or set(pose) != set(_POSE_FIELDS):
                raise ValueError(f"Invalid Cartesian point {name!r}")
            self._numbers(tuple(pose[field] for field in _POSE_FIELDS), 6)
        return data

    def _write_store(self, data):
        content = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.json_path.parent,
                prefix=f".{self.json_path.name}.", suffix=".tmp", delete=False,
            ) as target:
                temporary_path = Path(target.name)
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.json_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
