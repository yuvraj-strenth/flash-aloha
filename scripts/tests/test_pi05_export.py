import numpy as np
import pytest

from export.pi05_lerobot import (
    observation_action_rows, public_degrees_to_radians, radians_to_public_degrees)


def test_training_rows_follow_the_recorded_clock_and_use_the_next_pose():
    timestamps = np.arange(0.0, 1.0, 0.01)  # 100 Hz for one second
    observations, actions = observation_action_rows(timestamps, 15)
    assert len(observations) == len(actions) == 14
    assert np.all(actions > observations)
    assert np.diff(timestamps[observations]) == pytest.approx(1 / 15, abs=0.011)
    assert actions[-1] < len(timestamps)


def test_public_degrees_match_the_dm_motor_angle():
    radians = np.array([0.0, -1.0, -1.2, 0.5, 0.0, 0.0, 0.0])
    degrees = radians_to_public_degrees(radians)
    assert degrees == pytest.approx(np.degrees(radians))
    observation = {f'{name}.pos': float(value) for name, value in zip(
        ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper'),
        degrees)}
    assert public_degrees_to_radians(observation) == pytest.approx(radians)


def test_a_short_recording_exports_no_rows():
    observations, actions = observation_action_rows(np.array([1.0, 1.01]), 15)
    assert len(observations) == len(actions) == 0
