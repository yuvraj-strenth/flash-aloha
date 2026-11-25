from pathlib import Path

import pytest

import time

from recording.b601_session import (
    QuitRequested, collect_command, collect_session, delete_saved_episode,
    refresh_commands,
)


class _Arm:
    def set_compliant(self, enabled=True):
        self.compliant = enabled


class _Hardware:
    def __init__(self):
        self.arms = {'left': _Arm()}
        self.targets = []

    def read_state(self):
        return {'qpos': [0.0]}

    def submit_target(self, target, *, interval_s):
        self.targets.append((target, interval_s))


def test_s_during_a_take_saves_the_samples_already_taken():
    from types import SimpleNamespace
    import numpy as np
    from recording.b601_session import collect_guided

    class Hardware(_Hardware):
        def capture(self, _names, include_images=False):
            qpos = np.zeros(7)
            return SimpleNamespace(qpos=qpos, recording_fields=lambda: {'qpos': qpos.copy()})

    saved = []
    keys = iter(['s'])

    def read_key(_timeout):
        return next(keys)

    recorded = collect_guided(Hardware(), [], 10, 0.01, saved.append, read_key=read_key)
    assert recorded == 1 and len(saved) == 1


def test_s_and_d_are_the_only_collect_commands():
    assert collect_command('S') == 's'
    assert collect_command('d') == 'd'
    assert collect_command('h') is None
    assert collect_command(None) is None


def test_session_records_then_deletes_only_the_last_episode():
    saved = []
    hardware = _Hardware()
    keys = iter(['x', 's', 's', 'd'])

    def read_key(_timeout):
        try:
            return next(keys)
        except StopIteration:
            raise QuitRequested

    def record():
        episode = Path(f'/tmp/episode-{len(saved)}')
        saved.append(episode)
        return episode

    deleted = []
    last = collect_session(hardware, 0.01, read_key, record, deleted.append)
    assert saved == [Path('/tmp/episode-0'), Path('/tmp/episode-1')]
    assert deleted == [Path('/tmp/episode-1')]
    assert last is None
    assert hardware.arms['left'].compliant is True
    assert hardware.targets


def test_refresh_commands_keeps_submitting_while_the_caller_blocks():
    hardware = _Hardware()
    with refresh_commands(hardware, 0.01):
        time.sleep(0.05)
    assert len(hardware.targets) >= 2


def test_delete_refuses_a_path_outside_the_data_directory(tmp_path):
    data = tmp_path / 'data'
    episode = data / 'abc'
    episode.mkdir(parents=True)
    other = tmp_path / 'other'
    other.mkdir()
    assert delete_saved_episode(episode, data) == episode.resolve()
    assert not episode.exists()
    with pytest.raises(ValueError):
        delete_saved_episode(other, data)
