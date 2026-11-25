# Camera calibration and start pose (operator-run)

Run these commands yourself from the repository root. Stop the camera server before
the camera steps: each calibration script opens the RealSense color stream itself,
at 640×480. Stop every arm controller before connecting to CAN. A normal finish
returns the arm home. Ctrl+C or a failure enters damping until you press `h`, which
homes the arm ([safe exit](safe_exit.md)).

This checkout has two RealSense cameras, named in
`src/config/cameras/cam_config.yaml`: `top` is published as `video_overhead`, and
`wrist` is published as `video_left`. There is no right-wrist camera.
`calibration/masking/masking.yaml` lists the left arm only.

Do the camera steps in this order. The start pose is independent and can be saved
before or after the cameras.

1. Intrinsics for `video_overhead`, then `video_left`.
2. Wrist-camera pose on the left end effector.
3. Left arm base in the overhead camera.
4. Start pose for the task you will record.

## What to have ready

| Item | Used for | Exact pattern the scripts assume |
| --- | --- | --- |
| ChArUco board | Intrinsics only | 12 × 8 **squares**, square side **20 mm**, ArUco marker side **15 mm**, dictionary **DICT_5X5_250**. One A4 sheet, landscape |
| Checkerboard | Wrist and overhead extrinsics | **10 × 7 inner corners** (11 × 8 squares), square side **20 mm**. One A4 sheet, landscape |
| Both RealSense cameras connected | All camera steps | Serials already in `cam_config.yaml` |
| Flat, rigid backing | Both boards | Tape each print to a stiff panel so it cannot bend |
| Left arm, motor power on | Extrinsics and the start pose | This arm is the reBot B601-DM on the Damiao serial adapter `/dev/ttyACM0`, not the Thor `can0` port |
| A display | Every step that opens a window | Click the OpenCV window before pressing keys |

Print each board at 100% scale on one A4 sheet in landscape. Do not fit to page.
Measure one square with a ruler: both squares must be 20 mm. The two prints are
different boards; one cannot stand in for the other.

Gravity compensation should already be saved for this arm
([tuning](arm_bringup.md#tuning-details)). The extrinsic and start-pose tools hold
the arm with gravity compensation while you move it by hand. Support the arm at
startup and shutdown.

A camera-settings file (`calibration/camera_params.npz`) is not an input to these
steps. Recording still runs at 640×480 without it.

## 1. Camera intrinsics

Stop the camera server. The ChArUco board is the only target in this step. The arm
stays still; this script does not connect to it.

```bash
python scripts/cameras/camera_int.py --camera_name video_overhead
python scripts/cameras/camera_int.py --camera_name video_left
```

Each command opens that RealSense at 640×480, shows `Checkerboard Capture`, then
solves. In the window:

| Key | Action |
| --- | --- |
| `s` | Save the current view |
| `q` | Stop capturing and solve the views already saved |

Save up to 20 views. Solving needs at least 4 views that each contain more than 3
detected ChArUco corners. Hold the board at different distances and angles, and
place it toward the edges and corners of the image as well as the center. Keep the
whole board sharp and inside the frame. A view the detector misses is printed and
skipped.

Outputs, next to the images that produced them:

- `calibration/masking/video_overhead_intri_imgs/video_overhead_intrinsics.npy`
- `calibration/masking/video_left_intri_imgs/video_left_intrinsics.npy`

`video_right` is refused. Width and height must stay 640×480.

## 2. Wrist camera on the end effector

Stop the camera server. Fix the **checkerboard** in the scene and leave it there
for every view. The script homes the left arm, then switches to gravity-compensated
passive mode.

```bash
/home/thor/miniforge3/envs/lerobot_env2/bin/python scripts/setup/calibrate_masking.py wrist-camera --arm left
```

That opens the B601 through `/dev/ttyACM0`, the same Damiao adapter `lerobot-calibrate` used. It does not use `can0`.

Guide the arm by hand so the wrist camera sees the full checkerboard. Click the
`Masking calibration` window:

| Key | Action |
| --- | --- |
| `s` | Save this view and the arm pose |
| `q` | Finish capturing and solve |

The default is 15 views, and the solver needs at least 3 in which the board is
found. Rotate the wrist about more than one axis between views, and change the
distance. A view whose board is not detected is skipped. When the window closes,
the arm returns home.

Output: `calibration/masking/robot_video_left_hand/video_left_ee_to_camera.npy`

## 3. Arm base in the overhead camera

Stop the camera server. Step 2 must already have written the wrist-camera file.
Use the same checkerboard. Each saved view must show that board, complete, in the
overhead camera and the wrist camera at the same time.

```bash
/home/thor/miniforge3/envs/lerobot_env2/bin/python scripts/setup/calibrate_masking.py overhead --arm left
```

The arm homes, then goes passive. The window shows both cameras side by side.
`s` saves both images and the arm pose; `q` finishes. The default is 5 views, and
each view needs the board in both cameras. Move the arm between views and keep the
board still. The printout includes the translation spread across views, in metres.
The arm returns home when the window closes.

Output: `calibration/masking/head-to-robot_video_left_hand/head_to_robot_base.npz`

To recompute either extrinsic result from views already on disk, add `--solve-only`.
That path does not open a camera or an arm.

## 4. Start pose

No board. This saves the joint angles and the gripper opening that recording and
inference move to before a take. Joints are SDK radians. The gripper value is a
width in metres.

CAN must be up and this script must be the only controller on that interface.
The camera preview inside this tool opens V4L2 devices such as `/dev/video_left`.
These RealSense cameras are not those devices, so pass `--cameras none` and type
the keys in the terminal.

For the task you will record (the task YAML supplies the arm and `can0`):

```bash
uv run scripts/setup/set_up_recording.py --task_name single_arm --cameras none
```

That writes `calibration/start_positions/single_arm.npz`. Use your own task name
in place of `single_arm`. Without `--task_name`, the same tool writes the shared
`calibration/start_position.npz`:

```bash
uv run scripts/setup/set_up_recording.py --arm left --can-interface can0 --cameras none
```

The arm starts in practice mode: position gains are zero and gravity compensation
holds it, so you can pose the arm and the gripper by hand.

| Key | Action |
| --- | --- |
| Space | Lock the pose you are holding |
| `y` | Save the locked pose and exit |
| `n` | Unlock and keep posing |
| `q` | Quit without saving |

A normal exit returns the arm home. Ctrl+C enters damping until `h`.

`--reset-home` homes the arm before practice mode. `--gripper-only` moves to a
pose already saved, holds the six arm joints, and lets you change only the gripper
before saving again.

## After calibration

Projected masks stay unavailable until both intrinsic files and both extrinsic
files exist. Recording and inference use `calibration/start_positions/<task>.npz`
when that file exists, and otherwise `calibration/start_position.npz`.

To look at a projected mask on a recorded episode, set `data.masking_strategy` in
a training recipe and run
`python scripts/visualize.py episode --data-dir EPISODE --training-config RECIPE`.
The file list and the mask conventions are in
[calibration/masking/README.md](../calibration/masking/README.md).
