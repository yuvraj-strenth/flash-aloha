"""Callable versions of the examples in :mod:`piper_sdk.demo.V3`.

Values passed to motion methods use radians for joints, metres for gripper
width, and millimetres/degrees for Cartesian poses. Read methods return one
snapshot; callers decide how often to poll. Motion targets must be explicit.
"""

import math
import sys
import time
from pathlib import Path


def _prefer_vendored_piper_sdk():
    """Use vendor/piper_sdk/piper_sdk, not the repo folder on sys.path.

    Python started in vendor/ otherwise treats the repo directory as a
    namespace package and C_PiperInterface_V3 is missing.
    """
    package_root = Path(__file__).resolve().parents[1] / "piper_sdk"
    if not (package_root / "piper_sdk" / "__init__.py").is_file():
        return
    root = str(package_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


_prefer_vendored_piper_sdk()

from piper_sdk import C_PiperInterface_V3, LogLevel, LogManager
from piper_sdk.utils import global_area


class Piper:
    def __init__(self, can="can0"):
        self.can = can
        self.arm = C_PiperInterface_V3(can_name=can,
                                       start_sdk_joint_limit=True,
                                       start_sdk_gripper_limit=True)
        self.arm.ConnectPort()

    def close(self):
        self.arm.DisconnectPort()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _enable(self, timeout=5):
        deadline = time.monotonic() + timeout
        while not self.arm.EnablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError("Arm did not enable")
            time.sleep(0.01)

    def _mode(self, move=0x01, speed=100, mit=0x00):
        self.arm.MotionCtrl_2(0x01, move, speed, mit)

    @staticmethod
    def _joint_values(joints):
        if len(joints) != 6:
            raise ValueError("Expected six joint angles in radians")
        return tuple(round(angle * 180000 / math.pi) for angle in joints)

    @staticmethod
    def _pose_values(pose):
        if len(pose) != 6:
            raise ValueError("Expected (X, Y, Z, RX, RY, RZ)")
        return tuple(round(value * 1000) for value in pose)

    @staticmethod
    def _gripper_width(width):
        if width < 0:
            raise ValueError("Gripper width must be nonnegative")
        return round(width * 1000000)

    # V2-compatible examples that use the V3 interface.
    def set_motor_max_speed(self, speed=300, delay=0.1):
        """Set every V3 joint limit in raw 0.01 rad/s units (0..300)."""
        if not isinstance(speed, int) or not 0 <= speed <= 300:
            raise ValueError("speed must be an integer from 0 to 300 (0.00 to 3.00 rad/s)")
        self._enable()
        for joint in range(1, 7):
            self.arm.MotorMaxSpdSet(joint, speed)
            time.sleep(delay)
        return self.get_motor_angle_speed_limits()

    def get_gripper_parameters(self):
        self.arm.ArmParamEnquiryAndConfig(4)
        return self.arm.GetGripperTeachingPendantParamFeedback()

    def set_gripper_parameters(self, travel=100, max_range=70, friction=1):
        self.arm.GripperTeachingPendantParamConfig(travel, max_range, friction)
        time.sleep(0.05)
        return self.get_gripper_parameters()

    def set_installation_position(self, position=0x01):
        """Set mounting: 0x01 upright, 0x02/0x03 side, 0x04 inverted."""
        self.arm.MotionCtrl_2(0x01, 0x01, 0, 0, 0, position)

    # V3-specific examples.
    def set_gripper_angle(self, angle=90, effort=1000):
        """Command the angle-mode gripper in degrees."""
        self._enable()
        self.arm.GripperCtrl_V3(0, effort, 0x06, 0)
        self.arm.GripperCtrl_V3(round(angle * 1000), effort, 0x05, 0)

    def move_joint_mit(self, joint=6, position=0, velocity=0,
                                kp=10, kd=0.8, torque=0):
        """Send one joint MIT target; position is in radians."""
        self._enable()
        self._mode(0x06, 0, 0xAD)
        self.arm.JointMitCtrl(joint, position, velocity, kp, kd, torque)

    def move_pose_and_get_ik(self, pose, timeout=2):
        previous = self.arm.GetArmIKJointMsgs().time_stamp
        self._enable()
        self._mode(0x00)
        self.arm.EndPoseCtrl(*self._pose_values(pose))
        deadline = time.monotonic() + timeout
        while True:
            feedback = self.arm.GetArmIKJointMsgs()
            if feedback.time_stamp > previous:
                return feedback
            if time.monotonic() >= deadline:
                raise TimeoutError("No new IK joint feedback")
            time.sleep(0.01)

    def get_ik_joints(self):
        return self.arm.GetArmIKJointMsgs()

    def set_gripper_angle_limits(self, minimum=-180, maximum=180):
        self.arm.SetSDKGripperAngleLimitParam(minimum, maximum)
        return self.arm.GetSDKGripperAngleLimitParam()

    # Recovery and basic control.
    def get_driver_codes(self):
        """Return all six driver status bytes, or None for incomplete feedback."""
        feedback = self.arm.GetArmLowSpdInfoMsgs()
        if feedback.time_stamp <= 0 or time.time() - feedback.time_stamp > 1:
            return None
        motors = [getattr(feedback, "motor_{}".format(i)) for i in range(1, 7)]
        if any(m.can_id != 0x260 + i for i, m in enumerate(motors, 1)):
            return None
        return [m.foc_status_code for m in motors]

    def clear_collision(self, joints=(1, 2, 3, 4, 5, 6), timeout=5):
        """Clear driver faults, then enable all joints after feedback is clear."""
        deadline = time.monotonic() + timeout
        while self.get_driver_codes() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("No feedback from all six motor drivers")
            time.sleep(0.1)
        for joint in joints:
            self.arm.JointConfig(joint_num=joint, set_zero=0,
                                 acc_param_is_effective=0,
                                 max_joint_acc=0x7FFF, clear_err=0xAE)
        while True:
            state = self.get_driver_codes()
            if state is not None and all(code & 0x30 == 0 for code in state):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Driver fault or feedback remains: {}".format(state))
            time.sleep(0.1)
        self.arm.EnableArm(7)
        while True:
            state = self.get_driver_codes()
            if state is not None and all(code & 0x40 for code in state):
                return state
            if time.monotonic() >= deadline:
                raise TimeoutError("One or more motors did not enable")
            time.sleep(0.1)

    def disable(self, timeout=5):
        deadline = time.monotonic() + timeout
        while self.arm.DisablePiper():
            if time.monotonic() >= deadline:
                raise TimeoutError("Arm did not disable")
            time.sleep(0.01)

    def enable(self, timeout=5):
        self._enable(timeout)

    def move_pose(self, pose, speed=100, gripper_width=None):
        """Send one MOVE P target (mm, degrees); gripper width is optional."""
        self._enable()
        self._mode(0x00, speed, 0x00)
        self.arm.EndPoseCtrl(*self._pose_values(pose))
        if gripper_width is not None:
            self.arm.GripperCtrl(self._gripper_width(gripper_width), 1000, 0x01, 0)

    def move_zero(self, speed=30):
        self._enable()
        self._mode(0x01, speed, 0x00)
        self.arm.JointCtrl(0, 0, 0, 0, 0, 0)
        self.arm.GripperCtrl(0, 1000, 0x01, 0)

    def set_gripper_width(self, width=0.05, effort=1000, clear_error=False):
        """Command gripper width in metres without resetting it each time.

        Set ``clear_error=True`` only when recovering from a gripper fault;
        the target is then sent once with enable-and-clear-error (0x03).
        """
        self._enable()
        code = 0x03 if clear_error else 0x01
        self.arm.GripperCtrl(self._gripper_width(width), effort, code, 0)

    def move_joints(self, joints, speed=100, gripper_width=None):
        """Send one MOVE J target with joint angles in radians."""
        self._enable()
        self._mode(0x01, speed, 0x00)
        self.arm.JointCtrl(*self._joint_values(joints))
        if gripper_width is not None:
            self.arm.GripperCtrl(self._gripper_width(gripper_width), 1000, 0x01, 0)

    def start_joint_stream(self, speed=100):
        """Select MOVE J once before sending closely spaced arm targets."""
        self._mode(0x01, speed, 0x00)

    def stream_joints(self, joints):
        """Send one six-joint target on the selected MOVE J stream."""
        self.arm.JointCtrl(*self._joint_values(joints))

    def move_linear(self, pose, speed=100, gripper_width=None):
        self._enable()
        self._mode(0x02, speed, 0x00)
        self.arm.EndPoseCtrl(*self._pose_values(pose))
        if gripper_width is not None:
            self.arm.GripperCtrl(self._gripper_width(gripper_width), 1000, 0x01, 0)

    def move_arc(self, points, speed=30, delay=0.001):
        """Send three supplied Cartesian poses as an arc."""
        if len(points) != 3:
            raise ValueError("MOVE C requires three poses")
        self._enable()
        self._mode(0x03, speed, 0x00)
        for index, pose in enumerate(points, 1):
            self.arm.EndPoseCtrl(*self._pose_values(pose))
            self.arm.MoveCAxisUpdateCtrl(index)
            time.sleep(delay)
        self._mode(0x03, speed, 0x00)

    def move_linear_path(self, points, speed=100, dwell=2):
        """Move to the first pose, then follow the remaining linear targets."""
        if len(points) < 2:
            raise ValueError("MOVE L requires a start and at least one target")
        self._enable()
        self._mode(0x00, speed, 0x00)
        self.arm.EndPoseCtrl(*self._pose_values(points[0]))
        time.sleep(dwell)
        for pose in points[1:]:
            self._mode(0x02, speed, 0x00)
            self.arm.EndPoseCtrl(*self._pose_values(pose))
            time.sleep(dwell)

    def reset(self):
        """Resume/reset motion; the arm can lose motor power."""
        self.arm.MotionCtrl_1(0x02, 0, 0)

    def stop(self):
        self.arm.MotionCtrl_1(0x01, 0, 0)

    def clear_emergency_stop(self, timeout=5):
        """Resume a latched stop and wait for fresh normal arm feedback."""
        deadline = time.monotonic() + timeout
        while True:
            feedback = self.arm.GetArmStatus()
            if feedback.time_stamp > 0 and time.time() - feedback.time_stamp <= 1:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("No arm status feedback")
            time.sleep(0.1)
        # STOP can leave NORMAL status with standby control, but still needs
        # the 0x150 resume command before CAN control can be restored.
        self.arm.EmergencyStop(0x02)
        sent_at = time.time()
        while True:
            feedback = self.arm.GetArmStatus()
            if (feedback.time_stamp > sent_at
                    and time.time() - feedback.time_stamp <= 1):
                status = feedback.arm_status
                if status.arm_status.value == 0x00:
                    return status
                if status.arm_status.value != 0x01:
                    raise RuntimeError("Arm fault remains: {}".format(status.arm_status))
            if time.monotonic() >= deadline:
                raise TimeoutError("Arm did not resume to normal status")
            time.sleep(0.1)

    # Feedback examples. Each method returns one reading instead of printing forever.
    def get_message_rates(self):
        a = self.arm
        return {"is_ok": a.isOk(), "can": a.GetCanFps(),
                "status": a.GetArmStatus().Hz,
                "end_pose": a.GetArmEndPoseMsgs().Hz,
                "joints": a.GetArmJointMsgs().Hz,
                "gripper": a.GetArmGripperMsgs().Hz,
                "high_speed": a.GetArmHighSpdInfoMsgs().Hz,
                "low_speed": a.GetArmLowSpdInfoMsgs().Hz,
                "joint_ctrl": a.GetArmJointCtrl().Hz,
                "gripper_ctrl": a.GetArmGripperCtrl().Hz,
                "mode_ctrl": a.GetArmModeCtrl().Hz}

    def get_motor_acceleration_limits(self):
        self.arm.SearchAllMotorMaxAccLimit()
        return self.arm.GetAllMotorMaxAccLimit()

    def get_motor_angle_speed_limits(self, timeout=1):
        """Return one dict per motor with limits and max speed in rad/s.

        The SDK's aggregate feedback object reuses one mutable message, so each
        motor is queried and copied before requesting the next one.
        """
        limits = []
        for joint in range(1, 7):
            previous = self.arm.GetAllMotorAngleLimitMaxSpd().time_stamp
            self.arm.SearchMotorMaxAngleSpdAccLimit(joint, 0x01)
            deadline = time.monotonic() + timeout
            while True:
                feedback = self.arm.GetAllMotorAngleLimitMaxSpd()
                motor = feedback.all_motor_angle_limit_max_spd.motor[joint]
                if feedback.time_stamp > previous and motor.motor_num == joint:
                    limits.append({"joint": joint,
                                   "min_angle_deg": motor.min_angle_limit / 10,
                                   "max_angle_deg": motor.max_angle_limit / 10,
                                   "max_speed_rad_s": motor.max_joint_spd / 100})
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("No limit feedback from joint {}".format(joint))
                time.sleep(0.01)
        return limits

    def get_collision_protection(self, level=None):
        """Read protection levels; optionally set all six to ``level`` first."""
        if level is not None:
            self.arm.CrashProtectionConfig(*([level] * 6))
        self.arm.ArmParamEnquiryAndConfig(0x02, 0, 0, 0, 0x03)
        return self.arm.GetCrashProtectionLevelFeedback()

    def get_end_pose(self):
        """Return [X, Y, Z, RX, RY, RZ] in mm and degrees."""
        pose = self.arm.GetArmEndPoseMsgs().end_pose
        return [pose.X_axis / 1000, pose.Y_axis / 1000, pose.Z_axis / 1000,
                pose.RX_axis / 1000, pose.RY_axis / 1000, pose.RZ_axis / 1000]

    def get_firmware_version(self):
        return self.arm.GetPiperFirmwareVersion()

    def get_forward_kinematics(self):
        self.arm.EnableFkCal()
        return {"feedback": self.arm.GetFK("feedback"),
                "control": self.arm.GetFK("control")}

    def get_gripper_status(self):
        return self.arm.GetArmGripperMsgs()

    def get_high_speed_motor_status(self):
        return self.arm.GetArmHighSpdInfoMsgs()

    def get_joint_command(self):
        return self.arm.GetArmJointCtrl()

    def get_joint_state(self):
        """Return [j1, j2, j3, j4, j5, j6] in radians."""
        joints = self.arm.GetArmJointMsgs().joint_state
        return [math.radians(angle / 1000) for angle in
                (joints.joint_1, joints.joint_2, joints.joint_3,
                 joints.joint_4, joints.joint_5, joints.joint_6)]

    def get_low_speed_motor_status(self):
        return self.arm.GetArmLowSpdInfoMsgs()

    def get_motion_mode_command(self):
        return self.arm.GetArmModeCtrl()

    def get_control_frame_151(self):
        return self.arm.GetArmCtrlCode151()

    def get_instruction_response(self, enable=False, clear=False):
        """Read the last response; optionally send enable or clear it."""
        if enable:
            self.arm.EnableArm()
        response = self.arm.GetRespInstruction()
        if clear:
            self.arm.ClearRespSetInstruction()
        return response

    def get_status(self):
        return self.arm.GetArmStatus()

    @staticmethod
    def apply_tool_offset(pose, tool_offset):
        """Return TCP XYZ in 0.001 mm from a J6 pose and metre offset."""
        x, y, z, rx, ry, rz = pose
        ax, ay, az = (math.radians(angle / 1000) for angle in (rx, ry, rz))
        cx, sx = math.cos(ax), math.sin(ax)
        cy, sy = math.cos(ay), math.sin(ay)
        cz, sz = math.cos(az), math.sin(az)
        ox, oy, oz = (value * 1000000 for value in tool_offset)
        return (round(x + (cy * cz) * ox + (sx * sy * cz - cx * sz) * oy
                      + (cx * sy * cz + sx * sz) * oz),
                round(y + (cy * sz) * ox + (sx * sy * sz + cx * cz) * oy
                      + (cx * sy * sz - sx * cz) * oz),
                round(z - sy * ox + sx * cy * oy + cx * cy * oz))

    def get_tcp_pose(self, tool_offset=(0, 0, 0.145)):
        pose = self.arm.GetArmEndPoseMsgs().end_pose
        raw = (pose.X_axis, pose.Y_axis, pose.Z_axis,
               pose.RX_axis, pose.RY_axis, pose.RZ_axis)
        return self.apply_tool_offset(raw, tool_offset)

    def get_versions(self):
        return {"interface": self.arm.GetCurrentInterfaceVersion(),
                "protocol": self.arm.GetCurrentProtocolVersion(),
                "sdk": self.arm.GetCurrentSDKVersion(),
                "firmware": self.arm.GetPiperFirmwareVersion()}

    # Configuration examples.
    def use_can_port(self, channel="/dev/ttyACM0", bustype="slcan",
                      bitrate=1000000):
        """Replace the current connection with a manually created CAN bus."""
        self.close()
        self.can = channel
        self.arm = C_PiperInterface_V3(can_name=channel, can_auto_init=False,
                                       judge_flag=False,
                                       start_sdk_joint_limit=True,
                                       start_sdk_gripper_limit=True)
        self.arm.CreateCanBus(channel, bustype, bitrate, False)
        self.arm.ConnectPort(piper_init=False)
        return self.arm.GetCanFps()

    def zero_gripper(self):
        self.arm.GripperCtrl(0, 1000, 0x00, 0)
        time.sleep(1.5)
        self.arm.GripperCtrl(0, 1000, 0x00, 0xAE)

    def restore_default_limits(self):
        self.arm.ArmParamEnquiryAndConfig(0x01, 0x02, 0, 0, 0x02)

    def zero_joint(self, joint, confirm=None):
        """Disable a joint, wait for manual positioning, set zero, re-enable.

        ``confirm`` receives the joint number and should return True only once
        the operator has positioned the joint. With no callback, prompt on stdin.
        """
        if joint not in range(1, 8):
            raise ValueError("joint must be 1..7 (7 means all joints)")
        self.arm.DisableArm(joint)
        try:
            ready = (confirm(joint) if confirm is not None else
                     input("Position joint {}, then press Enter (q cancels): ".format(joint)).lower() != "q")
            if not ready:
                return False
            self.arm.JointConfig(joint, 0xAE)
            return True
        finally:
            self.arm.EnableArm(joint)

    def calibrate_cpv_zero(self, target="j456", confirm=None):
        """Send the V3 CPV calibration frame after manual positioning.

        ``target`` is ``j456`` or ``gripper``. A cancelled operation restores
        enable state; a sent calibration frame leaves CPV handling to firmware.
        """
        if target not in ("j456", "gripper"):
            raise ValueError("target must be 'j456' or 'gripper'")
        if target == "j456":
            for joint in (4, 5, 6):
                self.arm.DisableArm(joint)
            joints = (4, 5, 6)
        else:
            self.arm.GripperCtrl(0, 1000, 0x00, 0)
            joints = (7,)
        try:
            ready = (confirm(target) if confirm is not None else
                     input("Position {}, then press Enter (q cancels): ".format(target)).lower() != "q")
        except BaseException:
            self._restore_cpv_enable(target, joints)
            raise
        if not ready:
            self._restore_cpv_enable(target, joints)
            return False
        self.arm.MotionCtrl_2(0x01, 0x05)
        bus = self.arm.GetCanBus()
        for joint in joints:
            result = bus.SendCanMessage(0x180 + joint,
                                        [0x77, 0x65, 0x65, 0x65, 0, 0, 0], 7)
            if result != bus.CAN_STATUS.SEND_MESSAGE_SUCCESS:
                raise RuntimeError("CPV calibration frame failed for joint {}".format(joint))
        return True

    def _restore_cpv_enable(self, target, joints):
        if target == "j456":
            for joint in joints:
                self.arm.EnableArm(joint)
        else:
            self.arm.GripperCtrl(0, 1000, 0x02, 0)
            self.arm.GripperCtrl(0, 1000, 0x01, 0)

    def set_payload(self, load=2):
        """Set end payload: 0 empty, 1 half, 2 full."""
        self.arm.ArmParamEnquiryAndConfig(0, 0, 0, 0xAE, load)

    def set_log_level(self, level=LogLevel.DEBUG, log_to_file=True):
        LogManager.update_logger(global_area=global_area,
                                 local_area="InterfaceV2", level=level,
                                 log_to_file=log_to_file)
        return LogManager.get_log_file_path(global_area)

    def set_master_mode(self):
        self.arm.MasterSlaveConfig(0xFA, 0, 0, 0)

    def set_mit_mode(self, speed=0):
        """Select V3 MOVE J with the MIT flag, as in the SDK demo."""
        self._mode(0x01, speed, 0xAD)

    def set_motion_mode(self, move_mode=0x06, speed=50, mit=0xAD):
        self._mode(move_mode, speed, mit)

    def set_motor_angle_limits(self, limits=None):
        """Write six (max, min) angle limits in 0.1-degree units."""
        if limits is None:
            limits = ((1500, -1500), (1800, 0), (0, -1700),
                      (1000, -1000), (700, -700), (1700, -1700))
        if len(limits) != 6:
            raise ValueError("Expected limits for six joints")
        self._enable()
        for joint, (maximum, minimum) in enumerate(limits, 1):
            self.arm.MotorAngleLimitMaxSpdSet(joint, maximum, minimum)

    def set_motor_acceleration_limit(self, acceleration=500, delay=0.5):
        """Write six raw acceleration limits (500 means 5 rad/s²)."""
        self._enable()
        for joint in range(1, 7):
            self.arm.JointMaxAccConfig(joint, acceleration)
            time.sleep(delay)
        return self.get_motor_acceleration_limits()

    def set_software_limits(self, gripper=(0, 0.05),
                            joint="j6", joint_limits=(-2.09, 2.09)):
        self.arm.SetSDKGripperRangeParam(*gripper)
        self.arm.SetSDKJointLimitParam(joint, *joint_limits)
        return {"gripper": self.arm.GetSDKGripperRangeParam(),
                "joint": self.arm.GetSDKJointLimitParam(joint)}

    def set_slave_mode(self):
        self.arm.MasterSlaveConfig(0xFC, 0, 0, 0)

    # Interface test examples.
    def reconnect(self):
        self.arm.DisconnectPort()
        self.arm.ConnectPort(can_init=True)
        return self.arm.isOk()

    def connect_second_arm(self, other_can="can_1"):
        """Return a connected second V3 interface; caller closes it."""
        if other_can == self.can:
            raise ValueError("The second interface needs a different CAN port")
        other = C_PiperInterface_V3(can_name=other_can)
        other.ConnectPort()
        return other


# Lowercase import alias; constructing an instance still requires piper().
piper = Piper
