"""PiperX backend, task config, and joint-stream unit tests (no live CAN)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from config.task_config import CONFIG_ROOT, load_task_config
from hardware.arm_factory import build_arms, normalize_hardware_config
from hardware.backends.piperx import PiperXAdapter
import scripts.piperx_joint_stream as joint_stream


def _fake_adapter_cls(*, teaching_seq=(True,)):
    seq = iter(teaching_seq)
    last_teach = teaching_seq[-1]

    def next_teach():
        nonlocal last_teach
        try:
            last_teach = next(seq)
        except StopIteration:
            pass
        return last_teach

    arm_sdk = Mock()
    arm_sdk.GetArmJointMsgs.return_value = SimpleNamespace(time_stamp=1e12)
    high_spd = SimpleNamespace(**{
        f'motor_{i}': SimpleNamespace(motor_speed=100 * i) for i in range(1, 7)
    })
    arm_sdk.GetArmHighSpdInfoMsgs.return_value = high_spd

    instance = Mock()
    instance.robot = SimpleNamespace(arm=arm_sdk)
    instance.teaching.side_effect = next_teach
    instance.get_joints.return_value = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6)
    instance.get_gripper_width.return_value = 0.04
    cls = Mock(return_value=instance)
    return cls, instance, arm_sdk


def test_piperx_task_config_and_arm_factory():
    task = load_task_config(CONFIG_ROOT / 'tasks' / 'piperx_single.yaml')
    doc = task.as_dict()
    assert doc['robot']['type'] == 'piperx'
    normalized = normalize_hardware_config(doc['robot'])
    assert normalized['arms']['left']['model'] == 'PiperX'
    arms = build_arms(doc['robot'])
    assert isinstance(arms['left'], PiperXAdapter)
    assert arms['left'].describe().action_dim == 7


def test_piperx_read_state_and_submit_target(monkeypatch):
    monkeypatch.setattr('hardware.backends.piperx.time.time', lambda: 1e12 + 0.01)
    cls, instance, _ = _fake_adapter_cls(teaching_seq=(False,))
    arm = PiperXAdapter('left', 'PiperX', 'can0', adapter_cls=cls)
    arm.connect()
    try:
        state = arm.read_state()
        assert state.qpos == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.04])
        assert state.qvel == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0])

        target = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.3, 0.02])
        arm.submit_target(target, interval_s=0.01)
        instance.start_joint_stream.assert_called_once_with(speed=100)
        instance.stream_joints.assert_called_once()
        instance.control_gripper.assert_called_once_with(0.02)
    finally:
        arm.close()
    instance.close.assert_called_once_with()


def test_piperx_teaching_mode_software_and_physical_fallback(capsys):
    cls, instance, arm_sdk = _fake_adapter_cls(teaching_seq=(False, False, True))
    arm = PiperXAdapter('left', 'PiperX', 'can0', adapter_cls=cls)
    arm.connect()
    try:
        assert arm.set_teaching_mode(True, timeout=2.0, poll_interval_s=0.0) is True
        arm_sdk.MotionCtrl_1.assert_called_once_with(0x00, 0x00, 0x01)
        assert arm.in_teaching_mode is True

        arm.set_teaching_mode(False, timeout=2.0)
        instance.enable_can.assert_called_once_with(timeout=2.0)
        assert arm.in_teaching_mode is False
    finally:
        arm.close()

    # Physical button prompt when software teach doesn't engage within soft_timeout_s
    cls2, _, arm_sdk2 = _fake_adapter_cls(teaching_seq=(False, False, True))
    arm2 = PiperXAdapter('left', 'PiperX', 'can0', adapter_cls=cls2)
    arm2.connect()
    try:
        assert arm2.set_teaching_mode(
            True, timeout=0.5, soft_timeout_s=0.0, prompt=True, poll_interval_s=0.0
        ) is True
        out = capsys.readouterr().out
        assert 'physical teach button' in out
    finally:
        arm2.close()


def test_joint_stream_emits_json_and_closes(monkeypatch):
    monkeypatch.setattr('hardware.backends.piperx.time.time', lambda: 1e12 + 0.01)
    cls, instance, _ = _fake_adapter_cls(teaching_seq=(True, True))
    arm = PiperXAdapter('left', 'PiperX', 'can0', adapter_cls=cls)
    lines = []
    count = joint_stream.stream_joint_angles(
        arm,
        hz=100.0,
        enable_teach=True,
        as_json=True,
        max_samples=2,
        sleep=lambda _: None,
        emit=lambda msg, **_: lines.append(msg),
    )
    assert count == 2
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload['qpos'] == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.04])
    assert payload['teaching'] is True
    instance.close.assert_called_once_with()


def test_shm_writer_reader_and_staleness():
    from recording.piperx_shm import PiperXJointStreamReader, PiperXJointStreamWriter, shm_stream_available

    shm_name = 'test_piperx_shm_unit'
    now = [100.0]
    writer = PiperXJointStreamWriter('can0', 'left', name=shm_name)
    try:
        qpos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05]
        qvel = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.0]
        writer.write(qpos, qvel, teaching=True, timestamp=now[0])

        reader = PiperXJointStreamReader('can0', 'left', max_age_s=0.2, name=shm_name, clock=lambda: now[0])
        try:
            snap = reader.read()
            assert snap['qpos'] == pytest.approx(qpos)
            assert snap['qvel'] == pytest.approx(qvel)
            assert snap['teaching'] is True

            now[0] += 0.5
            with pytest.raises(TimeoutError, match='stale'):
                reader.read()
        finally:
            reader.close()
    finally:
        writer.close(unlink=True)
    assert shm_stream_available('can0', 'left', name=shm_name) is False


def test_record_piperx_reads_from_shm_stream(monkeypatch):
    from recording.piperx_shm import PiperXJointStreamReader, PiperXJointStreamWriter
    from recording.record_piperx import collect_take

    monkeypatch.setattr('hardware.backends.piperx.time.time', lambda: 1e12 + 0.01)
    cls, instance, _ = _fake_adapter_cls(teaching_seq=(True, True))
    arm = PiperXAdapter('left', 'PiperX', 'can0', adapter_cls=cls)
    shm_name = 'test_piperx_stream_to_collector'

    # Keep SHM alive after stream_joint_angles by testing writer -> reader -> collect_take
    writer = PiperXJointStreamWriter('can0', 'left', name=shm_name)
    try:
        arm.connect()
        state = arm.read_state()
        writer.write(state.qpos, state.qvel, teaching=True)
        reader = PiperXJointStreamReader('can0', 'left', max_age_s=1.0, name=shm_name)
        try:
            samples = []
            saved = collect_take(
                reader.read, [], steps=2, dt=0.01, append=samples.append,
                sleep=lambda _: None,
            )
            assert saved == 2
            assert len(samples) == 2
            assert samples[0]['qpos'] == pytest.approx([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.04])
            assert samples[0]['commanded_action'] == pytest.approx(samples[0]['qpos'])
            assert samples[0]['capture_contract_version'] == 1
        finally:
            reader.close()
    finally:
        writer.close(unlink=True)
        arm.close()


def test_run_collect_piperx_dispatches_to_record_piperx():
    from config.run import build_plan, build_parser

    args = build_parser().parse_args(['collect', 'piperx_single'])
    plan = build_plan(args)
    assert Path(plan['command'][1]).name == 'record_piperx.py'
    assert '--task-config' in plan['command']
