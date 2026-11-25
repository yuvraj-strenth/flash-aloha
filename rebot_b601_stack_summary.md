# Chat summary: this stack, the ARX arm, and a reBot B601-DM

Notes from a read of the Flash ALOHA repo. Nothing here was changed in the robot, recording, or inference code.

## What robot this repo is for

Flash ALOHA is a data-collection, training, and deployment stack for **ARX arms**, models **L5** (the default) and **Trossen_L5**.

- Tasks set `robot.type: arx`.
- `src/config/tasks/single_arm.yaml` is one L5 arm. `src/config/tasks/bimanual.yaml` is two.
- Control goes through the vendored ARX SDK in `src/hardware/arx5-sdk` (`arx5_interface` / `Arx5JointController`) over CAN.
- The SDK also knows X5, X5_umi, L5_umi, and X7, but this repo's arm factory only accepts `L5` and `Trossen_L5`.

## Will a Seeed reBot B601-DM work?

No. A B601-DM will not run on this stack as it is.

The motors look similar and are not interchangeable.

| | reBot B601-DM | ARX L5 (this repo) |
| --- | --- | --- |
| Joints 1–3 | DM4340P | DM-J4340 |
| Joints 4–6 and gripper | DM4310 | DM-J4310 |
| CAN IDs | `0x01`–`0x07`, replies `0x11`–`0x17` | motors `1, 2, 4, 5, 6, 7`, gripper `8` |
| Control | Damiao MIT on Seeed's USB-CAN bridge | ARX joint controller at 500 Hz |
| Model | Seeed URDF, about 767 mm reach, gripper about 0–0.05 m | `L5.urdf`, gripper width 0.100 m, L5 gravity and velocity scaling |

`src/hardware/arm_factory.py` and `src/hardware/backends/arx.py` reject any model other than `L5` and `Trossen_L5`. There is no B601 config in `src/hardware/arx5-sdk/include/app/config.h`.

Pointing the L5 profile at a B601 would use the wrong CAN IDs, velocity scaling, kinematics, gravity compensation, and gripper mapping. Do not run `scripts/tune.py`, practice, collect, or infer against that arm.

## Which motor mode the ARX controller uses

**MIT mode**, through the ARX joint controller. This stack does not use Damiao position-velocity mode.

Collection, replay, and inference open `Arx5JointController` with the SDK config `joint_controller`. That loop runs at 500 Hz, interpolates the joint target, and each cycle calls `send_DM_motor_cmd(id, kp, kd, position, velocity, torque)`. That is the 8-byte Damiao MIT frame.

What goes in the frame:

- **kp / kd** are the joint PD gains. L5 defaults are kp `80, 70, 70, 30, 30, 20`. The motor runs that PD. Flash ALOHA then overwrites the active gains in `src/utils/hardware_utils.py` (`ACTIVE_KP` / `ACTIVE_KD`) before constructing the controller.
- **position** is the interpolated joint target.
- **velocity** is the command velocity, scaled for the L5 Damiao VMAX settings (DM4340 at 10 rad/s, DM4310 at 30 rad/s, while `libhardware.so` decodes DaMiao velocity as VMAX 45 rad/s).
- **torque** is host-side feedforward. Gravity compensation is on for normal control. Friction feedforward is on for hand guiding and off while a policy is running.

Practice sets kp to 0, so the arm stays in MIT mode with no position stiffness: gravity torque plus damping.

The SDK also has `cartesian_controller`. The Flash ALOHA arm adapter does not select it.

## Scripts

Most files under `scripts/` only dispatch into `src/`. `scripts/tests/` is offline and never starts a camera or an arm.

### Daily workflow

| Script | What it does |
| --- | --- |
| `scripts/run.py` | One command for the pipeline. `collect`, `train`, `infer`, `serve`, `view`, and `replay` turn a task or training-config name into the real command. `--dry-run` prints it and starts nothing. |
| `scripts/record.py` | `single` records hand-guided demonstrations, including review replay. `publish` turns finished rollout or DAgger recordings into episodes and does not touch the robot. |
| `scripts/policy/imitate_episodes_train.py` | Trains ACT or Flow from a training YAML. |
| `scripts/infer.py` | `direct` loads a checkpoint and runs the arm in this process. `serve` hosts that checkpoint over WebSocket. `remote` is the robot-side client. |
| `scripts/visualize.py` | `episode` opens a recording in Rerun or video. `--review_replay` physically replays episodes on the arm. |
| `scripts/_launcher.py` | Shared dispatcher for `record.py`, `infer.py`, `visualize.py`, and `tune.py`. |

### Arm bring-up, tuning, and practice

These connect to or move an ARX arm.

| Script | What it does |
| --- | --- |
| `arx_setup.sh` | Activates the venv and loads the ARX library paths. |
| `scripts/setup/paths.sh` | Puts the ARX SDK and its native libraries on `PYTHONPATH` and `LD_LIBRARY_PATH`. |
| `scripts/setup/packages.sh` | Creates the `arx-py310` Conda env and builds the ARX SDK. |
| `scripts/setup/activate_arm` | Brings up `can0` or `can1` at 1 Mbit/s. Run it with that arm's controller stopped. |
| `scripts/setup/arx_arms.sh` | Same bring-up for SLCAN adapters `/dev/arxcan0` and `/dev/arxcan1`. |
| `src/hardware/check_gain.py` | Prints the L5 default joint and gripper gains. It does not connect to an arm. |
| `scripts/tune.py` | First step on a new ARX arm. `gravity` tunes link masses so the arm floats. `friction` tunes joint and gripper friction. Both enable the motors and write into the ARX URDF. |
| `scripts/setup/calibrate_arx_arm.py` | Interactive gain calibration, one motor at a time. |
| `scripts/practice.py` | Gravity-compensated hand guiding. Position gains are zero, nothing is saved, and the loop only reads the arm. |
| `src/recording/replay_practice.py` | Record a hand-guided motion and replay it. No task, no cameras, nothing written. This is the tool to run after practice. |
| `scripts/setup/set_up_recording.py` | Hand guiding with a camera preview. Space locks the current arm and gripper pose. `y` saves it as that task's start pose. |
| `scripts/setup/hold_start_position.py` | Moves the arm to the saved start pose and holds it until interrupted. |

### Collection, inference, and replay that move the arm

| Script | What it does |
| --- | --- |
| `scripts/record.py single` | Hand-guided demonstration recording, including the review replay that moves the arm. The implementation is `src/recording/record_episode_single.py`, which constructs `Arx5JointController("L5")` and rejects any other model. |
| `scripts/infer.py direct` | Loads a checkpoint in this process and executes it on the arm, starting from the saved pose. The implementation is `src/inference/inference_direct.py`, which also opens the ARX SDK directly. |
| `scripts/infer.py remote` | Robot-side client. A normal rollout requires `robot.type: arx`. `--shared_hardware` is the path that goes through `build_arms` (`src/inference/shared_hardware.py`). `serve` stays on the GPU machine and does not touch the arm. |
| `scripts/visualize.py episode` | With `--review_replay`, physically replays saved episodes. Replay control is in `src/utils/replay_utils.py` and is hardcoded to L5. |
| `scripts/run.py` | Resolves `collect`, `infer`, and `replay` into the commands above. |

### Cameras, masks, and checks

| Script | What it does |
| --- | --- |
| `scripts/cameras/camera_servers_tmux.sh` | Starts, stops, and tails the per-camera servers (`left`, `right`, `overhead`) in tmux. |
| `scripts/cameras/configure_camera_controls.sh` | Turns off dynamic exposure framerate on the V4L2 devices. |
| `scripts/cameras/tune_cameras.py` | Interactive V4L2 controls. `y` saves an NPZ. |
| `scripts/cameras/livestream_camera.py` | Preview of a camera server that is already running. |
| `scripts/cameras/camera_int.py` | Checkerboard capture and the camera intrinsic matrix. |
| `scripts/setup/calibrate_masking.py` | Wrist-camera and overhead extrinsic calibration. Homes the arm, then holds it passive while you guide it. |
| `scripts/setup/tune_masking.py` | Adjust mask polygons on a saved episode, or live (`--live`) on an ARX arm. |
| `scripts/config/create_config.py` | Writes a new task or training YAML from a template. |
| `scripts/config/check_task_config.py` | Checks a task YAML against recordings and a checkpoint. |
| `scripts/config/check_training_config.py` | Checks a training recipe. |
| `scripts/config/check_inference_config.py` | Checks an inference profile against its checkpoint. |
| `scripts/diagnostics/write_recording_rate_metadata.py` | Writes source recording-rate metadata for a LeRobot export. |

## What it would take to run a reBot B601-DM

There is no B601 driver in this repo. Wrap Seeed's Damiao MIT driver and add one backend behind the existing arm interface. Do not reuse the ARX SDK profile.

Outside this tree:

- Motor bus and MIT frames: Seeed MotorBridge / `can-motor-control` (`send_mit(pos, vel, kp, kd, tau)`, serial `/dev/ttyACM*` or SocketCAN).
- Geometry: [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) `hardware/reBot_B601_DM` URDF.
- Gravity torque is not in that bus library. Compute it from the B601 URDF. `scripts/tune.py` writes the ARX URDF and cannot be reused.

The arm has no brakes. A MIT timeout or a process exit drops it. The new backend has to match `src/hardware/safe_exit.py`: stay in damping until `h`, then home.

### Write these

| New file | What it replaces |
| --- | --- |
| `src/hardware/backends/b601.py` | `src/hardware/backends/arx.py`. A `ManagedArm`: 6 joints plus gripper, 7-D state, absolute joint targets in radians, gripper in metres. Each cycle sends MIT (`kp`, `kd`, position, velocity, gravity torque). `damp()` is zero kp, damping kd, gravity on, gripper held. `go_home()` moves to the B601 folded pose. Keep a background refresh so a Damiao command timeout does not disable the motors. Declare `motor_hz` as the rate that loop actually runs. |
| `src/hardware/b601/` (URDF plus a small config module) | `config.h` and `models/L5.urdf`. Copy the Seeed B601-DM URDF. Put joint limits, the CAN map, and gripper metres here. |
| A hand-guide provider, for example `src/hardware/b601/guide.py` | The passive mode inside `record_episode_single.py`. MIT with kp 0 and gravity torque. Return a 7-vector target. |
| `src/config/tasks/b601_single.yaml` | A copy of `single_arm.yaml` with `robot.type: b601`, model `B601_DM`, and the serial port or `can0`. Leave the ARX task files as they are. `state_dim` and `action_dim` stay 7. |
| `calibration/start_positions/<task>.npz` | The L5 start pose. Capture it on the B601. An L5 pose is a different arm. |

### Edit only these

| File | Change |
| --- | --- |
| `src/hardware/arm_factory.py` | It allows only `type: arx` and models `L5` / `Trossen_L5`. Add `b601` and construct the new adapter. This is the switch `HardwareRuntime` already uses. |
| `scripts/record.py` and `src/config/run.py` | `collect` launches `record_episode_single.py`, which rejects any model other than L5. B601 collection has to call `src/recording/record_hardware.py` with the new guide provider. |
| `src/inference/inference_remote.py` | A normal remote rollout requires `robot.type: arx` and then uses `inference_direct.py`. B601 rollouts go through the existing `--shared_hardware` path, which calls `build_arms`. |

`src/inference/shared_hardware.py`, `src/recording/record_hardware.py`, and `src/hardware/runtime.py` already talk to whatever adapter the factory builds. They need no B601-specific code if the new adapter matches `ManagedArm`: `read_state()`, `validate_target()`, `submit_target()`, `stop()`, `close()`.

### Leave these alone

Cameras, `src/contracts/standard_capture.py`, `standard_recording.py`, `standard_image_processing.py`, and the ACT/Flow trainers. They never open the arm. They work once the adapter returns one 7-D `qpos` / `qvel` and accepts absolute 7-D targets at the task rate.

`src/contracts/robot_contract.py` only stores the model name. A checkpoint tagged `B601_DM` will refuse an L5 runtime. Collect new demonstrations. An L5 dataset will not run on this arm.

### Do not retarget these at the B601

| File | Why it stays ARX |
| --- | --- |
| `src/hardware/arx5-sdk/**` | L5 MIT profile, CAN map, URDF, gravity, and friction. |
| `src/hardware/backends/arx.py` | The L5 / Trossen adapter. |
| `src/utils/hardware_utils.py` | L5 kp/kd and the 500 Hz SDK loop. Seeed's MIT gains are different and have to be tuned on the B601. |
| `scripts/tune.py` | Writes gravity and friction into the ARX URDF. |
| `scripts/practice.py`, `src/recording/replay_practice.py` | Open `Arx5JointController` directly. |
| `scripts/setup/set_up_recording.py`, `hold_start_position.py`, `calibrate_arx_arm.py` | Same. Home, hold, and gain calibration need the new backend before these can move a B601. |
| `src/inference/inference_direct.py`, `src/utils/replay_utils.py` | Hardcoded L5 controllers and `robot_model: L5`. |
| `src/visualization/visualize_episode.py`, `rerun_utils.py` | Draw `L5.urdf`. Swap the URDF path only for the on-screen robot. |
| `scripts/setup/calibrate_masking.py`, `tune_masking.py` | Home an ARX arm. The saved masks were measured on the L5. Recalibrate after the B601 backend can hold the arm. |
| `scripts/setup/activate_arm`, `arx_arms.sh` | Bring up the ARX USB-CAN adapters. The Seeed bridge is a serial port. Use SocketCAN setup only if the B601 is actually on `can0`. |
