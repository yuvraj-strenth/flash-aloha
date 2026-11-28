from pathlib import Path
import struct

import numpy as np
import pytest

from camera_utils.camera_server_realsense import (
    HostClockAlignment, color_capture_size, load_cam_config, logger_endpoint_port,
    monotonic_capture_time, publish_captured_frame, serial_for_camera)
from camera_utils.camera_timing import MONOTONIC_CLOCK_FLAGS, CameraTimingError


def test_config_maps_top_and_wrist_to_the_existing_stream_names(tmp_path):
    path = tmp_path / 'cam_config.yaml'
    path.write_text('cameras:\n  top: "111"\n  wrist: "222"\nfps: 30\n')
    specs, fps = load_cam_config(path)
    by_role = {spec['role']: spec for spec in specs}
    assert by_role['top'] == {'role': 'top', 'serial': '111', 'stream': 'overhead'}
    assert by_role['wrist'] == {'role': 'wrist', 'serial': '222', 'stream': 'left'}
    assert fps == 30


def test_calibration_opens_overhead_and_left_at_640x480(tmp_path):
    path = tmp_path / 'cam_config.yaml'
    path.write_text('cameras:\n  top: "111"\n  wrist: "222"\nfps: 30\n')
    assert color_capture_size() == (640, 480)
    assert serial_for_camera('video_overhead', path) == ('111', 30)
    assert serial_for_camera('video_left', path) == ('222', 30)
    with pytest.raises(ValueError, match='video_right'):
        serial_for_camera('video_right', path)


def test_config_refuses_a_missing_serial(tmp_path):
    path = tmp_path / 'cam_config.yaml'
    path.write_text('cameras:\n  top: ""\n  wrist: "222"\n')
    with pytest.raises(ValueError, match='top'):
        load_cam_config(path)


def test_global_time_converts_onto_the_monotonic_clock():
    stamp = monotonic_capture_time(999_900.0, 'global_time', realtime=1000.0, monotonic=50.0)
    assert stamp == pytest.approx(49.9)


def test_a_constant_global_time_offset_is_removed_and_frame_spacing_stays():
    clock = HostClockAlignment(smooth=1.0)
    period = 1.0 / 30.0
    overhead_bias = 0.024
    stamps = {'left': [], 'overhead': []}
    for index in range(40):
        realtime = 1_000.0 + index * period
        monotonic = 50.0 + index * period
        for name, bias in (('left', 0.0), ('overhead', overhead_bias)):
            captured = realtime - 0.010 - bias
            raw = monotonic_capture_time(captured * 1000.0, 'global_time', realtime=realtime, monotonic=monotonic)
            stamps[name].append(clock.correct(name, raw, realtime - captured))
    left = np.diff(stamps['left'][-5:])
    overhead = np.diff(stamps['overhead'][-5:])
    assert left == pytest.approx(period, abs=1e-6)
    assert overhead == pytest.approx(period, abs=1e-6)
    assert stamps['overhead'][-1] - stamps['left'][-1] == pytest.approx(0.0, abs=0.002)


def test_a_hardware_clock_stamp_is_not_published():
    with pytest.raises(CameraTimingError, match='global_time'):
        monotonic_capture_time(10.0, 'hardware_clock', realtime=1000.0, monotonic=50.0)


def test_logger_ports_match_the_collector():
    assert logger_endpoint_port('left') == 55551
    assert logger_endpoint_port('overhead') == 55552


def test_published_frame_is_submitted_with_the_shared_memory_stamp():
    from camera_utils.camera_server import SharedFrameBuffer
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (9, 8, 7)
    submitted = []

    class Writer:
        def submit_frame(self, image, timestamp, frame_number, frame_ready_time, *, raw_timestamp=None, **kwargs):
            submitted.append((image, timestamp, frame_number, raw_timestamp, kwargs))

    buffer = SharedFrameBuffer('realsense_logger_test', 2, 2)
    try:
        publish_captured_frame(buffer, Writer(), frame, 7, 12.5)
        number, stamp = struct.unpack('Qd', bytes(buffer.shm.buf[:16]))
    finally:
        shm_name = buffer.shm_name
        buffer.cleanup()
        Path('/dev/shm', f'{shm_name}.clock.json').unlink(missing_ok=True)
    assert number == 7
    assert stamp == pytest.approx(12.5)
    image, timestamp, frame_number, raw_timestamp, kwargs = submitted[0]
    assert image is frame
    assert timestamp == 12.5
    assert frame_number == 7
    assert raw_timestamp == 12.5
    assert kwargs['buffer_flags'] == MONOTONIC_CLOCK_FLAGS
    assert kwargs['driver_sequence'] == 7
