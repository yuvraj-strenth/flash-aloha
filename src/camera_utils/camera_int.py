
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))
_SCRIPT_ROOT = _ACT_ROOT.parent / "scripts"
if str(_SCRIPT_ROOT) not in _path_sys.path:
    _path_sys.path.append(str(_SCRIPT_ROOT))
# get the camera intrinsic matrix

import cv2
import numpy as np
import os
import glob
import argparse

from camera_utils.camera_server_realsense import RealSenseColorCapture, color_capture_size

calib_path = './calibration/masking/'  # the masking calibration set (utils.masking)


def capture_checkerboard_images(camera_name, num_images=20):
    """Capture ChArUco views from the RealSense named ``camera_name``.

    The camera server must be stopped. Frames are the native 640x480 color stream.
    ``s`` saves a view and ``q`` finishes.
    """
    path = os.path.join(calib_path, camera_name + '_intri_imgs')
    if not os.path.exists(path):
        os.makedirs(path)

    width, height = color_capture_size()
    cap = RealSenseColorCapture(camera_name)
    img_count = 0
    try:
        print(f'Capturing {camera_name} at {width}x{height}. s saves a view, q finishes.')
        while img_count < num_images:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f'Failed to grab a frame from {camera_name}')
            cv2.imshow('Checkerboard Capture', frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                img_name = f"{path}/checkerboard_{img_count}.png"
                cv2.imwrite(img_name, frame)
                print(f"Saved {img_name}")
                img_count += 1
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return img_count


def calibrate_camera(camera_name, checkerboard_size=(12, 8), square_size=0.02, resolution=(640, 480)):
    """
    Calibrate camera using saved charuco board images and save intrinsic parameters.

    Images are solved at the size they were captured. A frame of another size is refused
    instead of being resized, so the saved intrinsics match ``resolution``.

    Args:
        checkerboard_size (tuple): Internal corners of the checkerboard (width, height)
        square_size (float): Size of each square on the checkerboard in meters
    """
    all_object_points = []
    all_image_points = []
    image_size = None

    path = os.path.join(calib_path, camera_name + '_intri_imgs')
    images = glob.glob(os.path.join(path, '*.png'))

    board = cv2.aruco.CharucoBoard(
        (checkerboard_size[0], checkerboard_size[1]), square_size, 0.015,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250))
    charuco_detector = cv2.aruco.CharucoDetector(board)

    if not images:
        raise ValueError("No images found in {} directory".format(path))

    expected = (int(resolution[0]), int(resolution[1]))
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            raise ValueError(f'Could not read {fname}')
        actual = (img.shape[1], img.shape[0])
        if actual != expected:
            raise ValueError(f'{fname} is {actual[0]}x{actual[1]}, expected {expected[0]}x{expected[1]}')
        cv2.imshow('Image', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) > 3:
            obj_points = board.getChessboardCorners()
            detected_obj_points = np.array([obj_points[id[0]] for id in charuco_ids], dtype=np.float32)

            all_object_points.append(detected_obj_points)
            all_image_points.append(charuco_corners)

            print(f"Image {fname}: Detected {len(charuco_corners)} charuco corners")

            img_display = img.copy()
            if marker_corners is not None:
                cv2.aruco.drawDetectedMarkers(img_display, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(img_display, charuco_corners, charuco_ids)
            cv2.imshow('Corners', img_display)
            cv2.waitKey(500)
        else:
            corners_found = len(charuco_corners) if charuco_corners is not None else 0
            print(f"Image {fname}: Only detected {corners_found} charuco corners (need > 3)")

    cv2.destroyAllWindows()

    print(f"Total images processed: {len(images)}")
    print(f"Images with valid charuco detections: {len(all_image_points)}")

    if len(all_image_points) < 4:
        raise ValueError(
            f"Not enough valid charuco detections for calibration. Found {len(all_image_points)}, "
            "need at least 4 images with detected corners.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None
    )

    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist
    }
    np.save(os.path.join(path, camera_name + '_intrinsics.npy'), calibration_data)
    print("Camera calibration completed and saved to camera_intrinsics.npy")
    print(f"{calibration_data=}")

    return mtx, dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration (RealSense, 640x480)")
    parser.add_argument('--camera_name', type=str, default="video_overhead",
                        help="video_overhead or video_left")
    parser.add_argument('--width', type=int, default=color_capture_size()[0],
                        help="Captured frame width. Must be the RealSense color width, 640.")
    parser.add_argument('--height', type=int, default=color_capture_size()[1],
                        help="Captured frame height. Must be the RealSense color height, 480.")
    args = parser.parse_args()
    expected = color_capture_size()
    if (args.width, args.height) != expected:
        raise SystemExit(
            f'RealSense calibration captures at {expected[0]}x{expected[1]}, '
            f'not {args.width}x{args.height}.')
    capture_checkerboard_images(args.camera_name)
    calibrate_camera(args.camera_name, resolution=(args.width, args.height))
