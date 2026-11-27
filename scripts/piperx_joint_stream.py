#!/usr/bin/env python3
"""Continuously fetch PiperX joint angles (6 joints in rad + gripper width in m) for the recorder.

Enables low-friction drag-teach mode via software (or prompts the operator to
press the physical teach button on the arm), publishes samples to POSIX shared
memory for ``record_piperx.py`` / ``run.py collect``, and optionally prints them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from contracts import capture_gc
from hardware.backends.piperx import PiperXAdapter
from recording.piperx_shm import PiperXJointStreamWriter


def stream_joint_angles(
    arm: PiperXAdapter,
    *,
    hz: float = 100.0,
    enable_teach: bool = True,
    as_json: bool = False,
    publish_shm: bool = True,
    shm_name: str | None = None,
    max_samples: int | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
    emit=print,
) -> int:
    """Connect ``arm``, optionally enter teach mode, and stream joint angles."""
    if hz <= 0:
        raise ValueError('hz must be positive')
    dt = 1.0 / hz
    capture_gc.pause_for_stream()
    writer = (
        PiperXJointStreamWriter(arm._can, role=arm.describe().role, name=shm_name)
        if publish_shm else None
    )
    arm.connect()
    count = 0
    try:
        if enable_teach:
            arm.set_teaching_mode(True)
        if writer is not None:
            print(f'[PiperX] Publishing joint stream to shm://{writer.name} at {hz:g} Hz',
                  file=sys.stderr, flush=True)
        while max_samples is None or count < max_samples:
            started = clock()
            state = arm.read_state()
            qpos = [float(v) for v in state.qpos]
            qvel = [float(v) for v in state.qvel]
            if writer is not None:
                writer.write(qpos, qvel, teaching=arm.in_teaching_mode, timestamp=started)
            if as_json:
                emit(json.dumps({
                    'timestamp': time.time(),
                    'qpos': qpos,
                    'qvel': qvel,
                    'teaching': arm.in_teaching_mode,
                }), flush=True)
            else:
                formatted = ' '.join(f'{v:+.4f}' for v in qpos)
                emit(f'qpos {formatted}', end='\r', flush=True)
            count += 1
            remaining = dt - (clock() - started)
            if remaining > 0:
                sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close(unlink=True)
        arm.close()
        capture_gc.collect_idle()
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--can', default='can0', help='SocketCAN interface (default: can0)')
    parser.add_argument('--role', default='left', choices=('left', 'right'), help='Arm role')
    parser.add_argument('--hz', type=float, default=100.0, help='Polling rate in Hz (default: 100)')
    parser.add_argument('--no-teach', action='store_true', help='Do not enter drag-teach mode')
    parser.add_argument('--no-shm', action='store_true', help='Disable shared-memory publishing')
    parser.add_argument('--json', action='store_true', help='Emit newline-delimited JSON records')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    arm = PiperXAdapter(role=args.role, model='PiperX', can_interface=args.can)
    stream_joint_angles(
        arm,
        hz=args.hz,
        enable_teach=not args.no_teach,
        as_json=args.json,
        publish_shm=not args.no_shm,
    )


if __name__ == '__main__':
    main()
