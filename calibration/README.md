# Calibration files

Measured files are not checked in. The cameras on this rig are the RealSense
serials in `src/config/cameras/cam_config.yaml`, and capture runs at 640×480.
Produce the files below for this robot before a projected mask, a tuned exposure,
or a saved start pose is used.

| File | What it is | How to produce your own |
| --- | --- | --- |
| `camera_params.npz` | Saved camera controls (exposure, gain, ...) per camera | `python scripts/cameras/tune_cameras.py` |
| `start_position.npz` | Shared start pose (SDK joint radians, gripper width in metres) | `uv run scripts/setup/set_up_recording.py` (per-task poses go to `start_positions/<task>.npz`) |
| `masking/` | Masking config, camera intrinsics and extrinsics for projected masks | See [`masking/README.md`](masking/README.md) |

Gravity masses and friction terms are not stored here: the ARX tuners
(`uv run scripts/tune.py gravity L5 can0`, then `friction`) save them in the model's
URDF under `src/hardware/arx5-sdk/models/`; see
[`docs/arm_bringup.md`](../docs/arm_bringup.md#tuning-details).
