"""PiperX adapter: wraps vendor/piper_arm_sdk PiperAdapter for ManagedArm."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np

from hardware.arm_interface import ArmDescription, ArmState, ManagedArm

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / 'vendor'


def _load_piper_adapter():
    if str(_VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(_VENDOR_ROOT))
    from piper_arm_sdk.adapter import PiperAdapter
    return PiperAdapter


class PiperXAdapter(ManagedArm):
    """ManagedArm adapter for the PiperX 6-DOF arm + gripper over SocketCAN."""

    def __init__(self, role, model, can_interface, *, tcp_offset_m=(0.0, 0.0, 0.145),
                 adapter_cls=None, **options):
        if model != 'PiperX':
            raise ValueError(f'Unsupported PiperX robot model {model!r}')
        self._can = can_interface
        self._tcp_offset_m = tuple(float(v) for v in tcp_offset_m)
        self._adapter_cls = adapter_cls
        self._adapter = None
        self._teaching = False
        self._stream_started = False
        super().__init__(ArmDescription(role, model, 'piperx', 7, 200.0), **options)

    def _connect(self):
        adapter_cls = self._adapter_cls or _load_piper_adapter()
        self._adapter = adapter_cls(
            can=self._can,
            tcp_offset_m=self._tcp_offset_m,
            rest_on_connect=False,
        )
        self._teaching = bool(self._adapter.teaching())
        self._stream_started = False

    def set_teaching_mode(self, enabled: bool = True, *, timeout: float = 15.0,
                          soft_timeout_s: float = 1.0, prompt: bool = True,
                          poll_interval_s: float = 0.1) -> bool:
        """Enable or disable low-friction drag-teach mode.

        When enabling, sends ``MotionCtrl_1(0x00, 0x00, 0x01)`` first; if the arm
        does not enter teach mode within ``soft_timeout_s``, prompts the operator
        to press the physical teach button on the arm and waits up to ``timeout``.
        """
        if self._adapter is None:
            raise RuntimeError('PiperX arm is not connected')
        if not enabled:
            self._adapter.enable_can(timeout=timeout)
            self._teaching = False
            self._stream_started = False
            return False

        if self._adapter.teaching():
            self._teaching = True
            self._stream_started = False
            return True

        self._adapter.robot.arm.MotionCtrl_1(0x00, 0x00, 0x01)
        soft_deadline = time.monotonic() + min(soft_timeout_s, timeout)
        while time.monotonic() < soft_deadline:
            if self._adapter.teaching():
                self._teaching = True
                self._stream_started = False
                return True
            time.sleep(poll_interval_s)

        if prompt:
            print(
                f'[PiperX:{self._description.role}] Press the physical teach button '
                'on the arm to enter low-friction hand-drag mode...',
                flush=True,
            )

        deadline = time.monotonic() + max(0.0, timeout - min(soft_timeout_s, timeout))
        while time.monotonic() < deadline:
            if self._adapter.teaching():
                self._teaching = True
                self._stream_started = False
                return True
            time.sleep(poll_interval_s)

        raise TimeoutError(
            'PiperX did not enter teaching mode; press the physical teach button and retry'
        )

    @property
    def in_teaching_mode(self) -> bool:
        return self._teaching

    def _validate_limits(self, target):
        if target[6] < 0:
            raise ValueError('PiperX gripper width must be nonnegative')

    def _read_joint_velocities(self) -> np.ndarray:
        """Read 6 motor speeds (rad/s) from high-speed feedback + 0.0 for gripper."""
        try:
            high_spd = self._adapter.robot.arm.GetArmHighSpdInfoMsgs()
            speeds = [
                float(getattr(high_spd, f'motor_{i}').motor_speed) * 0.001
                for i in range(1, 7)
            ]
            if all(math.isfinite(v) for v in speeds):
                return np.array([*speeds, 0.0], dtype=float)
        except Exception:
            pass
        return np.zeros(7, dtype=float)

    def _read_state(self):
        joints = self._adapter.get_joints()
        gripper = self._adapter.get_gripper_width()
        joint_msg = self._adapter.robot.arm.GetArmJointMsgs()
        stamp = float(getattr(joint_msg, 'time_stamp', 0.0))
        if stamp <= 0:
            raise TimeoutError('PiperX joint feedback timestamp is missing')
        age = max(0.0, time.time() - stamp)
        host_time = self._clock() - age
        qpos = np.array([*joints, gripper], dtype=float)
        qvel = self._read_joint_velocities()
        return ArmState(qpos, qvel, np.full(7, age), np.full(7, host_time))

    def _submit_target(self, target, *, interval_s):
        if self._teaching:
            self.set_teaching_mode(False)
        if not self._stream_started:
            self._adapter.start_joint_stream(speed=100)
            self._stream_started = True
        self._adapter.stream_joints(tuple(float(v) for v in target[:6]))
        self._adapter.control_gripper(float(target[6]))

    def _submit_safe_stop(self, *, interval_s):
        if self._adapter is not None and not self._teaching:
            self._adapter.abort()
            self._stream_started = False

    def _home(self):
        self._adapter.move_to_rest()
        self._teaching = False
        self._stream_started = False

    def _close(self):
        if self._adapter is not None:
            adapter, self._adapter = self._adapter, None
            self._teaching = False
            self._stream_started = False
            adapter.close()
