# Scripts Inventory (`pi_rollout` & PiperX / RealSense Updates)

| Script / File | What it does |
| --- | --- |
| [pi_rollout/smoothing.py](file:///home/strenth/flash-aloha/pi_rollout/smoothing.py) | Pure-NumPy smoothing primitives for Pi0.5 rollouts: `blend_actions` (`linear`, `exponential`, `sigmoid`), `blend_rtc_boundary`, `TemporalAggregator` (`k = 0.005`), and `ActionGapInterpolator` (Bezier sub-step interpolation). |
| [pi_rollout/pi05_rollout.py](file:///home/strenth/flash-aloha/pi_rollout/pi05_rollout.py) | Pi0.5 remote rollout runner (`SmoothedPolicySession`, `--dry-run`, and live hardware loop) supporting chunk blending, RTC boundary blending, temporal aggregation, action-gap interpolation, and optional `Masker` masking before `openpi_request_images`. |
| [pi_rollout/configs/piperx_pi05_remote.yaml](file:///home/strenth/flash-aloha/pi_rollout/configs/piperx_pi05_remote.yaml) | PiperX Pi0.5 rollout configuration (`localhost:8000`, `chunk_steps: 25`, `overlap_steps: 20`, `sigmoid` blending, RTC, and optional smoothing/masking fields). |
| [pi_rollout/tests/test_pi05_smoothing.py](file:///home/strenth/flash-aloha/pi_rollout/tests/test_pi05_smoothing.py) | Unit tests for smoothing parity against `inference_direct.py`, `TemporalAggregator`, `ActionGapInterpolator`, `SmoothedPolicySession` (sync/async/RTC/temporal_agg), and `Masker` integration. |
| [scripts/setup/calibrate_masking.py](file:///home/strenth/flash-aloha/scripts/setup/calibrate_masking.py) | Extrinsic camera calibration (`wrist-camera` and `overhead`) updated with `--backend {arx,piperx}` and `capture_piperx()` using `PiperXAdapter` + RealSense cameras. |
| [calibration/urdf/piper_description.urdf](file:///home/strenth/flash-aloha/calibration/urdf/piper_description.urdf) | Official AgileX Piper URDF (`link6`, `gripper_base`) used for forward kinematics in masking and calibration. |
| [calibration/masking/masking.yaml](file:///home/strenth/flash-aloha/calibration/masking/masking.yaml) | Masking config updated to reference `../urdf/piper_description.urdf` and `links: {wrist: link6, end_effector: gripper_base}`. |
| [src/recording/distributed_recording.py](file:///home/strenth/flash-aloha/src/recording/distributed_recording.py) | Distributed Zarr+MP4 recorder updated in `publish_recording()` to remove the hardcoded `robot_type='arx', robot_model='L5'` metadata override. |
