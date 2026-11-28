"""Publish RealSense color frames into the same shared memory as camera_server.py.

    /home/thor/miniforge3/envs/lerobot_env2/bin/python src/camera_utils/camera_server_realsense.py
    /home/thor/miniforge3/envs/lerobot_env2/bin/python src/camera_utils/camera_server_realsense.py --list-devices

The config names each camera by role. top is the overhead camera and is published
as video_overhead. wrist is the left-arm wrist camera and is published as
video_left. Depth is not started. Each color frame is written BGR, with the same
16-byte header the V4L2 server uses, and that same frame and monotonic stamp are
submitted to the video logger the collector connects to. left listens on
tcp://127.0.0.1:55551 and overhead on 55552. The RealSense global timestamp is
converted into CLOCK_MONOTONIC seconds so the existing capture-clock check can
read it. Each camera's constant global-time offset is removed so both stamps
share one host clock; the spacing between a camera's frames stays the sensor interval.
"""
import argparse
from collections import deque
import os
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_utils.camera_timing import MONOTONIC_CLOCK_FLAGS, CameraTimingError
from contracts import capture_gc


DEFAULT_CONFIG = SRC / 'config' / 'cameras' / 'cam_config.yaml'
ROLE_TO_STREAM = {'top': 'overhead', 'wrist': 'left'}
# Calibration opens these published names. There is no right-wrist serial.
CAMERA_NAME_TO_ROLE = {'video_overhead': 'top', 'video_left': 'wrist'}
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
LOGGER_ENDPOINT_BASE = 55551
# This host's ffmpeg provides libx264, not h264_nvenc. The writer's NVENC
# fallback only runs after a resource error, so request the CPU encoder directly.
LOGGER_CODEC = 'libx264'


def load_cam_config(path):
    """Roles and the color rate. Empty serials are refused."""
    document = yaml.safe_load(Path(path).read_text()) or {}
    cameras = document.get('cameras') or {}
    if set(cameras) != set(ROLE_TO_STREAM):
        raise ValueError(
            f'{path} cameras must be exactly {sorted(ROLE_TO_STREAM)}, got {sorted(cameras)}')
    specs = []
    for role, stream in ROLE_TO_STREAM.items():
        serial = str(cameras[role] or '').strip()
        if not serial:
            raise ValueError(f'{path} has no serial number for {role}')
        specs.append({'role': role, 'serial': serial, 'stream': stream})
    fps = float(document.get('fps', 30))
    if fps <= 0:
        raise ValueError(f'{path} fps must be positive, got {fps}')
    return specs, fps


def monotonic_capture_time(timestamp_ms, domain_name, *, realtime, monotonic):
    """Convert a RealSense global-time stamp into CLOCK_MONOTONIC seconds."""
    if domain_name != 'global_time':
        raise CameraTimingError(f'RealSense timestamp domain is {domain_name}, not global_time')
    captured = float(timestamp_ms) / 1000.0
    if not np.isfinite(captured) or captured <= 0:
        raise CameraTimingError('RealSense returned no valid capture timestamp')
    stamp = float(monotonic) - (float(realtime) - captured)
    if not np.isfinite(stamp) or stamp <= 0:
        raise CameraTimingError('RealSense capture time is not on CLOCK_MONOTONIC')
    return stamp


class HostClockAlignment:
    """Remove each camera's constant global-time offset from the host clock.

    Two RealSense devices map their clocks onto the host independently, so one
    camera's stamps can sit tens of milliseconds behind the other while both
    still deliver every frame. The correction is the difference of those
    offsets. It changes slowly, and a camera's frame spacing stays the interval
    the sensor reported.
    """

    def __init__(self, window=90, minimum_samples=15, percentile=0.1, smooth=0.15):
        self.window = int(window)
        self.minimum_samples = int(minimum_samples)
        self.percentile = float(percentile)
        self.smooth = float(smooth)
        self._samples = {}
        self._applied = {}
        self._lock = threading.Lock()

    def correct(self, name, raw_stamp, residual):
        with self._lock:
            series = self._samples.setdefault(name, deque(maxlen=self.window))
            series.append(float(residual))
            ready = [values for values in self._samples.values() if len(values) >= self.minimum_samples]
            if len(series) < self.minimum_samples or not ready:
                return float(raw_stamp)
            target = self._low(series) - min(self._low(values) for values in ready)
            applied = self._applied.get(name, 0.0)
            applied += self.smooth * (target - applied)
            self._applied[name] = applied
        return float(raw_stamp) + applied

    def _low(self, values):
        ordered = sorted(values)
        return ordered[int(self.percentile * (len(ordered) - 1))]


_HOST_CLOCK = HostClockAlignment()


def _domain_name(domain):
    text = str(domain).rsplit('.', 1)[-1].lower()
    return text


def _stream_size(stream_name):
    """The width and height the existing readers will attach with."""
    from camera_utils.camera_server import resolve_camera_config
    config = resolve_camera_config(
        name=stream_name, device_path=f'/dev/video_{stream_name}')
    return int(config.width), int(config.height)


def logger_endpoint_port(stream_name, endpoint_base=LOGGER_ENDPOINT_BASE):
    """TCP port DistributedRecorder uses for this stream's video logger."""
    from camera_utils.camera_server import CAMERA_ENDPOINT_OFFSETS
    if stream_name not in CAMERA_ENDPOINT_OFFSETS:
        raise ValueError(f'Unknown camera name for endpoint mapping: {stream_name}')
    return int(endpoint_base) + CAMERA_ENDPOINT_OFFSETS[stream_name]


def _prefer_env_ffmpeg():
    """Find ffmpeg beside this interpreter when the env is not activated."""
    bindir = Path(sys.executable).resolve().parent
    if (bindir / 'ffmpeg').is_file():
        os.environ['PATH'] = str(bindir) + os.pathsep + os.environ.get('PATH', '')


def start_recording_writer(stream_name, width, height, fps):
    """Open the same per-camera logger the V4L2 server hosts. Failure raises."""
    from camera_utils.camera_server import CameraRecordingWriter, _build_camera_video_logger
    _prefer_env_ffmpeg()
    endpoint_port = logger_endpoint_port(stream_name)
    writer = None
    try:
        writer = CameraRecordingWriter(
            lambda: _build_camera_video_logger(
                cam_name=stream_name, width=int(width), height=int(height), fps=float(fps),
                endpoint_base=LOGGER_ENDPOINT_BASE, codec=LOGGER_CODEC)[0],
            stream_name, float(fps))
        writer.start()
        writer.wait_initialized()
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        raise
    print(
        f'[RoboLogger] {stream_name} logger ready at '
        f'tcp://0.0.0.0:{endpoint_port} (name=video_{stream_name})',
        flush=True)
    return writer


def publish_captured_frame(buffer, writer, frame, frame_number, stamp):
    """Write one BGR frame, then hand that same frame and monotonic stamp to the logger.

    ``monotonic_capture_time`` already rejected any stamp that is not on CLOCK_MONOTONIC.
    The logger records that fact as the monotonic clock flag. Alignment refuses a row
    whose flag was never written.
    """
    buffer.write_frame(frame, frame_number, stamp)
    writer.submit_frame(
        frame, stamp, frame_number, time.monotonic(), raw_timestamp=stamp,
        buffer_flags=MONOTONIC_CLOCK_FLAGS, driver_sequence=int(frame_number))


def _enable_global_time(profile, rs):
    """Global time is a sensor option. device.supports() only accepts camera info."""
    enabled = False
    for sensor in profile.get_device().query_sensors():
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1)
            enabled = True
    if not enabled:
        raise CameraTimingError('This RealSense cannot enable global_time')


def _start_color_pipeline(rs, serial, width, height, fps):
    last_error = None
    for pixel_format, convert_to_bgr in ((rs.format.bgr8, False), (rs.format.rgb8, True)):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, pixel_format, int(fps))
        try:
            profile = pipeline.start(config)
            _enable_global_time(profile, rs)
        except RuntimeError as exc:
            last_error = exc
            try:
                pipeline.stop()
            except Exception:
                pass
            if isinstance(exc, CameraTimingError):
                raise
            continue
        return pipeline, convert_to_bgr
    raise RuntimeError(
        f'{serial} has no {width}x{height} color stream at {fps:g} fps ({last_error})')


def color_capture_size():
    """Width and height of a calibration color stream."""
    return COLOR_WIDTH, COLOR_HEIGHT


def serial_for_camera(camera_name, config_path=None):
    """The RealSense serial and color rate for a published camera name.

    ``video_overhead`` is ``top`` and ``video_left`` is ``wrist``. Any other name
    is refused: this rig has no right-wrist camera.
    """
    role = CAMERA_NAME_TO_ROLE.get(camera_name)
    if role is None:
        known = ', '.join(sorted(CAMERA_NAME_TO_ROLE))
        raise ValueError(
            f'No RealSense is configured for camera {camera_name!r}. This rig captures {known}.')
    specs, fps = load_cam_config(config_path or DEFAULT_CONFIG)
    serial = next(spec['serial'] for spec in specs if spec['role'] == role)
    return serial, fps


class RealSenseColorCapture:
    """One 640x480 BGR color stream. Stop the camera server before opening it."""

    def __init__(self, camera_name, config_path=None):
        import pyrealsense2 as rs
        self.camera_name = camera_name
        self.width, self.height = color_capture_size()
        serial, fps = serial_for_camera(camera_name, config_path)
        try:
            self._pipeline, self._convert_to_bgr = _start_color_pipeline(
                rs, serial, self.width, self.height, fps)
        except RuntimeError as exc:
            raise RuntimeError(
                f'Cannot open {camera_name} ({serial}) at {self.width}x{self.height}: {exc}. '
                'Stop the camera server before calibrating.') from exc
        self._open = True

    def read(self):
        """``(ok, bgr_frame)``, the same pair ``cv2.VideoCapture.read`` returns."""
        if not self._open:
            return False, None
        try:
            frames = self._pipeline.wait_for_frames(1000)
        except RuntimeError as exc:
            raise RuntimeError(
                f'Failed to read a frame from {self.camera_name}. '
                'Stop the camera server before calibrating.') from exc
        color = frames.get_color_frame()
        if not color:
            return False, None
        frame = np.array(np.asanyarray(color.get_data()), dtype=np.uint8, copy=True)
        if self._convert_to_bgr:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape != (self.height, self.width, 3):
            raise RuntimeError(
                f'{self.camera_name} frame is {frame.shape}, expected {(self.height, self.width, 3)}')
        return True, frame

    def release(self):
        if self._open:
            self._open = False
            self._pipeline.stop()


def _publish_loop(spec, width, height, fps, stop):
    import cv2
    import pyrealsense2 as rs
    from camera_utils.camera_server import SharedFrameBuffer

    pipeline, convert_to_bgr = _start_color_pipeline(rs, spec['serial'], width, height, fps)
    buffer = None
    writer = None
    frame_number = 0
    try:
        buffer = SharedFrameBuffer(spec['stream'], width, height)
        writer = start_recording_writer(spec['stream'], width, height, fps)
        print(
            f"[RealSense] {spec['role']} serial {spec['serial']} -> "
            f"video_{spec['stream']} {buffer.shm_name} {width}x{height}@{fps:g}",
            flush=True)
        while not stop.is_set():
            try:
                frames = pipeline.wait_for_frames(500)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            realtime = time.time()
            monotonic = time.monotonic()
            try:
                captured = float(color.get_timestamp()) / 1000.0
                stamp = monotonic_capture_time(
                    color.get_timestamp(), _domain_name(color.get_frame_timestamp_domain()),
                    realtime=realtime, monotonic=monotonic)
                stamp = _HOST_CLOCK.correct(spec['stream'], stamp, realtime - captured)
            except CameraTimingError as exc:
                print(f"[RealSense] {spec['role']} skipped a frame: {exc}", flush=True)
                continue
            image = np.asanyarray(color.get_data())
            frame = np.array(image, dtype=np.uint8, copy=True)
            if convert_to_bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame.shape != (height, width, 3):
                raise RuntimeError(
                    f"{spec['role']} frame is {frame.shape}, expected {(height, width, 3)}")
            publish_captured_frame(buffer, writer, frame, frame_number, stamp)
            frame_number += 1
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            pipeline.stop()
            if buffer is not None:
                buffer.cleanup()


def list_devices():
    import pyrealsense2 as rs
    devices = rs.context().query_devices()
    if len(devices) == 0:
        print('No RealSense cameras found.')
        return 1
    for device in devices:
        print(f"{device.get_info(rs.camera_info.serial_number)}  {device.get_info(rs.camera_info.name)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--list-devices', action='store_true',
                        help='Print connected serial numbers and exit')
    args = parser.parse_args(argv)
    if args.list_devices:
        return list_devices()
    specs, fps = load_cam_config(args.config)
    sizes = {spec['stream']: _stream_size(spec['stream']) for spec in specs}
    stop = threading.Event()

    def _stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    capture_gc.pause_for_stream()
    errors = []

    def _run(spec, width, height):
        try:
            _publish_loop(spec, width, height, fps, stop)
        except Exception as exc:
            errors.append(exc)
            print(f"[RealSense] {spec['role']} stopped: {exc}", flush=True)
            stop.set()

    threads = []
    for spec in specs:
        width, height = sizes[spec['stream']]
        thread = threading.Thread(
            target=_run, name=spec['stream'], args=(spec, width, height), daemon=True)
        thread.start()
        threads.append(thread)
    try:
        while not stop.is_set() and any(thread.is_alive() for thread in threads):
            time.sleep(0.2)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=20)
    if errors:
        raise errors[0]
    return 0


if __name__ == '__main__':
    sys.exit(main())
