"""POSIX shared-memory transport for continuous PiperX joint-state streaming."""

from __future__ import annotations

from multiprocessing import shared_memory
import re
import struct
import time

import numpy as np

# seq (Q), mono_ts (d), wall_ts (d), qpos[7] (7d), qvel[7] (7d), teaching (B), pad (7x)
_HEADER_STRUCT = struct.Struct('<Qdd14dB7x')
SHM_SIZE = _HEADER_STRUCT.size


def shm_name_for_arm(can_interface: str, role: str = 'left') -> str:
    """Return a valid POSIX shared-memory name for a PiperX arm stream."""
    safe_can = re.sub(r'[^A-Za-z0-9_]', '_', str(can_interface))
    safe_role = re.sub(r'[^A-Za-z0-9_]', '_', str(role))
    return f'piperx_joint_{safe_role}_{safe_can}'[:60]


class PiperXJointStreamWriter:
    """Single-writer seqlock publisher of 7-DOF PiperX joint state to shared memory."""

    def __init__(self, can_interface: str, role: str = 'left', *, name: str | None = None):
        self.name = name or shm_name_for_arm(can_interface, role)
        try:
            self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=SHM_SIZE)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.name, create=False)
            existing.close()
            existing.unlink()
            self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=SHM_SIZE)
        self._seq = 0
        self._shm.buf[:SHM_SIZE] = b'\x00' * SHM_SIZE

    def write(self, qpos, qvel, *, teaching: bool = False,
              timestamp: float | None = None, wall_time: float | None = None) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(7)
        qvel_arr = np.asarray(qvel, dtype=np.float64).reshape(7)
        mono = time.monotonic() if timestamp is None else float(timestamp)
        wall = time.time() if wall_time is None else float(wall_time)
        self._seq += 1
        struct.pack_into('<Q', self._shm.buf, 0, self._seq)  # odd = write in progress
        _HEADER_STRUCT.pack_into(
            self._shm.buf, 0,
            self._seq, mono, wall,
            *qpos_arr.tolist(), *qvel_arr.tolist(),
            1 if teaching else 0,
        )
        self._seq += 1
        struct.pack_into('<Q', self._shm.buf, 0, self._seq)  # even = stable

    def close(self, *, unlink: bool = True) -> None:
        if self._shm is not None:
            shm, self._shm = self._shm, None
            try:
                shm.close()
            finally:
                if unlink:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass


class PiperXJointStreamReader:
    """Lock-free seqlock reader for a PiperX shared-memory joint stream."""

    def __init__(self, can_interface: str, role: str = 'left', *,
                 max_age_s: float = 0.2, name: str | None = None, clock=time.monotonic):
        self.name = name or shm_name_for_arm(can_interface, role)
        self.max_age_s = float(max_age_s)
        self._clock = clock
        self._shm = shared_memory.SharedMemory(name=self.name, create=False)

    def read(self) -> dict:
        for _ in range(16):
            seq1 = struct.unpack_from('<Q', self._shm.buf, 0)[0]
            if seq1 == 0 or (seq1 & 1):
                continue
            unpacked = _HEADER_STRUCT.unpack_from(self._shm.buf, 0)
            seq2 = struct.unpack_from('<Q', self._shm.buf, 0)[0]
            if seq1 == seq2:
                mono = unpacked[1]
                age = self._clock() - mono
                if age > self.max_age_s:
                    raise TimeoutError(
                        f'PiperX joint stream {self.name} is stale ({age:.3f}s > {self.max_age_s:.3f}s)'
                    )
                qpos = np.array(unpacked[3:10], dtype=np.float64)
                qvel = np.array(unpacked[10:17], dtype=np.float64)
                return {
                    'qpos': qpos,
                    'qvel': qvel,
                    'timestamp': mono,
                    'wall_time': unpacked[2],
                    'teaching': bool(unpacked[17]),
                }
        raise RuntimeError(f'PiperX joint stream {self.name} did not yield a consistent sample')

    def close(self) -> None:
        if self._shm is not None:
            shm, self._shm = self._shm, None
            shm.close()


def shm_stream_available(can_interface: str, role: str = 'left', *,
                         max_age_s: float = 0.5, name: str | None = None) -> bool:
    """Return True if a live, fresh shared-memory joint stream exists for this arm."""
    try:
        reader = PiperXJointStreamReader(can_interface, role, max_age_s=max_age_s, name=name)
    except FileNotFoundError:
        return False
    try:
        reader.read()
        return True
    except Exception:
        return False
    finally:
        reader.close()
