"""Piper adapter checks using fake feedback; no CAN connection is opened."""

import json
import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call, sentinel

import pytest
from scipy.spatial.transform import Rotation

from vendor.piper_arm_sdk import adapter as adapter_module
from vendor.piper_arm_sdk.adapter import PiperAdapter
from vendor.piper_arm_sdk.robot import Piper


def test_direct_import_from_adapter_directory():
    subprocess.run(
        [sys.executable, "-c", "import adapter; assert adapter.PiperAdapter"],
        cwd=Path(adapter_module.__file__).parent,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def robot(monkeypatch):
    fake = Mock()
    fake.get_joint_state.return_value = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    fake.get_end_pose.return_value = [100, 200, 300, 0, 0, 0]
    fake.move_joints.return_value = sentinel.joint_motion
    fake.move_zero.return_value = sentinel.rest_motion
    fake.move_pose.return_value = sentinel.pose_motion
    fake.move_linear.return_value = sentinel.linear_motion
    fake.stop.return_value = sentinel.stop
    fake.set_gripper_width.return_value = sentinel.gripper_motion
    monkeypatch.setattr(adapter_module, "Piper", Mock(return_value=fake))
    return fake


@pytest.fixture
def adapter(robot, tmp_path):
    return PiperAdapter(json_path=tmp_path / "points.json")


def test_adapter_creates_and_closes_robot(robot, tmp_path):
    with PiperAdapter(can="can1", json_path=tmp_path / "points.json") as arm:
        assert arm.robot is robot
        adapter_module.Piper.assert_called_once_with(can="can1")
    robot.close.assert_called_once_with()


def test_save_and_named_moves_round_trip(adapter, robot):
    joints = adapter.save_joints("home")
    pose = adapter.save_pose("home")
    assert joints == pytest.approx((0.1, -0.2, 0.3, -0.4, 0.5, -0.6))
    assert pose == pytest.approx((0.1, 0.2, 0.445, 0, 0, 0))
    assert json.loads(adapter.json_path.read_text()) == {
        "piper": {
            "joints": {"home": list(joints)},
            "cartesian": {"home": dict(zip(
                ("x", "y", "z", "roll", "pitch", "yaw"), pose
            ))},
        },
    }

    robot.get_joint_state.return_value = [0] * 6
    robot.get_end_pose.return_value = [0] * 6
    assert adapter.move_to_joint("home", speed=35) is sentinel.joint_motion
    robot.move_joints.assert_called_once_with(list(joints), speed=35)
    assert adapter.move_to_pose("home", speed=40) is sentinel.pose_motion
    commanded = robot.move_pose.call_args.args[0]
    assert commanded == pytest.approx((100, 200, 300, 0, 0, 0))
    assert robot.move_pose.call_args.kwargs == {"speed": 40}


def test_move_to_rest_commands_zero_and_closes_gripper(adapter, robot):
    robot.move_zero.reset_mock()
    assert adapter.move_to_rest(speed=25) is sentinel.rest_motion
    robot.move_zero.assert_called_once_with(speed=25)
    with pytest.raises(ValueError):
        adapter.move_to_rest(speed=0)
    robot.move_zero.assert_called_once()


def test_robot_zero_command_uses_joint_mode():
    interface = Mock()
    interface.EnablePiper.return_value = True
    robot = Piper.__new__(Piper)
    robot.arm = interface

    robot.move_zero(speed=25)
    assert interface.mock_calls == [
        call.EnablePiper(),
        call.MotionCtrl_2(0x01, 0x06, 0, 0xAD),
        call.MotionCtrl_2(0x01, 0x01, 25, 0x00),
        call.JointCtrl(0, 0, 0, 0, 0, 0),
        call.GripperCtrl(0, 1000, 0x01, 0),
    ]


def test_tcp_offset_and_existing_json_are_preserved(robot, tmp_path):
    path = tmp_path / "points.json"
    path.write_text(json.dumps({"other_robot": {"home": [1, 2, 3]}}))
    adapter = PiperAdapter(json_path=path, tcp_offset_m=(0, 0, 0.2))
    robot.get_end_pose.return_value = [100, 0, 300, 0, 90, 0]
    assert adapter.get_tcp() == pytest.approx(
        (0.3, 0, 0.3, 0, math.pi / 2, 0)
    )
    assert adapter.save_pose("side") == pytest.approx(adapter.get_tcp())
    assert json.loads(path.read_text())["other_robot"] == {"home": [1, 2, 3]}
    adapter.move_to_pose("side")
    assert robot.move_pose.call_args.args[0] == pytest.approx(
        (100, 0, 300, 0, 90, 0)
    )


def test_replacing_a_point_preserves_other_points(adapter, robot):
    adapter.save_joints("home")
    adapter.save_joints("other")
    adapter.save_pose("home")
    robot.get_joint_state.return_value = [0] * 6
    adapter.save_joints("home")
    saved = json.loads(adapter.json_path.read_text())["piper"]
    assert saved["joints"]["home"] == [0] * 6
    assert saved["joints"]["other"] == [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    assert "home" in saved["cartesian"]


def test_relative_translation_uses_base_or_tool_axes(adapter, robot):
    robot.get_end_pose.return_value = [100, 0, 200, 0, 0, 90]
    assert adapter.move_translate(0.01, 0, 0, speed=25) is sentinel.linear_motion
    assert robot.move_linear.call_args.args[0] == pytest.approx(
        (110, 0, 200, 0, 0, 90)
    )
    assert adapter.move_translate(
        0.01, 0, 0, frame="tool"
    ) is sentinel.linear_motion
    assert robot.move_linear.call_args.args[0] == pytest.approx(
        (100, 10, 200, 0, 0, 90)
    )


@pytest.mark.parametrize("frame", ["base", "tool"])
def test_relative_rotation_composes_in_frame_and_holds_tcp(adapter, robot, frame):
    robot.get_end_pose.return_value = [100, 20, 200, 30, 0, 0]
    before = adapter.get_tcp()
    delta = Rotation.from_euler("xyz", (0, 0, 0.4))
    current = Rotation.from_euler("xyz", before[3:])
    expected = delta * current if frame == "base" else current * delta

    assert adapter.move_rotate(0, 0, 0.4, frame=frame) is sentinel.pose_motion
    flange = robot.move_pose.call_args.args[0]
    commanded_rotation = Rotation.from_euler("xyz", flange[3:], degrees=True)
    assert (commanded_rotation.inv() * expected).magnitude() < 1e-10
    commanded_tcp = [
        flange[i] / 1000 + commanded_rotation.apply(adapter.tcp_offset_m)[i]
        for i in range(3)
    ]
    assert commanded_tcp == pytest.approx(before[:3])


def test_abort_and_absolute_gripper_width(adapter, robot):
    assert adapter.abort() is sentinel.stop
    robot.stop.assert_called_once_with()
    assert adapter.control_gripper(
        0.03, effort=800, clear_error=True
    ) is sentinel.gripper_motion
    robot.set_gripper_width.assert_called_once_with(
        0.03, effort=800, clear_error=True
    )


def test_mode_collision_and_status_forwarding(adapter, robot):
    robot.set_mit_mode.return_value = sentinel.mode
    robot.clear_collision.return_value = sentinel.cleared
    robot.get_status.return_value = sentinel.status

    assert adapter.set_mit_mode(speed=20) is sentinel.mode
    robot.set_mit_mode.assert_called_once_with(speed=20)
    assert adapter.clean_collisions(joints=(1, 2), timeout=3) is sentinel.cleared
    robot.clear_collision.assert_called_once_with(joints=(1, 2), timeout=3)
    assert adapter.get_status() is sentinel.status
    robot.get_status.assert_called_once_with()


def test_missing_or_malformed_points_never_command_motion(adapter, robot):
    with pytest.raises(KeyError):
        adapter.move_to_joint("missing")
    with pytest.raises(KeyError):
        adapter.move_to_pose("missing")
    assert not adapter.json_path.exists()

    bad = {"piper": {"joints": {"bad": [0] * 5}, "cartesian": {}}}
    adapter.json_path.write_text(json.dumps(bad))
    for action in (adapter.move_to_joint, adapter.move_to_pose,
                   adapter.save_joints, adapter.save_pose):
        with pytest.raises(ValueError):
            action("bad")
    assert json.loads(adapter.json_path.read_text()) == bad
    robot.move_joints.assert_not_called()
    robot.move_pose.assert_not_called()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 10**400])
def test_nonfinite_saved_joints_are_rejected(adapter, robot, bad_value):
    bad = {"piper": {"joints": {"bad": [bad_value] * 6}, "cartesian": {}}}
    adapter.json_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        adapter.move_to_joint("bad")
    robot.move_joints.assert_not_called()


def test_invalid_relative_options_do_not_read_feedback_or_move(adapter, robot):
    with pytest.raises(ValueError):
        adapter.move_translate(0.01, 0, 0, frame="unknown")
    with pytest.raises(ValueError):
        adapter.move_rotate(0, 0, 0.1, speed=0)
    robot.get_end_pose.assert_not_called()
    robot.move_linear.assert_not_called()
    robot.move_pose.assert_not_called()


def test_failed_write_preserves_existing_file(adapter, monkeypatch):
    adapter.save_joints("home")
    before = adapter.json_path.read_bytes()

    def fail_replace(*_args):
        raise OSError("write failed")

    monkeypatch.setattr(adapter_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="write failed"):
        adapter.save_pose("other")
    assert adapter.json_path.read_bytes() == before
    assert list(adapter.json_path.parent.iterdir()) == [adapter.json_path]


@pytest.mark.parametrize("name", [None, "", "  ", 5])
def test_invalid_names_do_not_read_feedback_or_write(adapter, robot, name):
    for action in (adapter.save_joints, adapter.save_pose,
                   adapter.move_to_joint, adapter.move_to_pose):
        with pytest.raises(ValueError):
            action(name)
    robot.get_joint_state.assert_not_called()
    robot.get_end_pose.assert_not_called()
    assert not adapter.json_path.exists()
