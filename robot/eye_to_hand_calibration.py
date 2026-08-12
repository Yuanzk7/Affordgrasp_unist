"""Run fixed-D435 (eye-to-hand) calibration for an xArm7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from ..camera import CameraCaptureError, capture_realsense
from .transforms import (
    TransformError,
    invert_transform,
    make_transform,
    pose_aa_to_transform,
    rotation_distance_rad,
    validate_transform,
)
from .xarm_connection import connect_xarm, read_tcp_offset


class CalibrationError(RuntimeError):
    """Raised when a calibration sample or solution is unsafe to use."""


_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

_BOARD_SQUARES_X = 4
_BOARD_SQUARES_Y = 5
_BOARD_DICTIONARY = "DICT_4X4_50"
_MARKER_TO_SQUARE_RATIO = 22.0 / 30.0
_MIN_SAMPLES = 12
_MIN_TRANSLATION_SPAN_M = 0.08
_MIN_ROTATION_SPAN_DEG = 25.0
_MAX_TRANSLATION_RMS_M = 0.01
_MAX_ROTATION_RMS_DEG = 2.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_samples(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 2,
            "calibration_type": "eye-to-hand",
            "calibration_target": "charuco",
            "matrix_conventions": {
                "robot_pose": "T_base_tcp",
                "target_pose": "T_camera_charuco_board",
            },
            "samples": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"could not read sample file: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise CalibrationError("calibration sample JSON is malformed")
    if payload.get("calibration_type") != "eye-to-hand":
        raise CalibrationError("sample file is not eye-to-hand calibration data")
    if payload.get("calibration_target") != "charuco":
        raise CalibrationError(
            "sample file is not ChArUco calibration data; do not mix old "
            "single-ArUco samples with ChArUco samples"
        )
    return payload


def _dictionary(name: str) -> Any:
    constant = getattr(cv2.aruco, name, None)
    if constant is None:
        raise CalibrationError(f"unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(constant)


def _charuco_board(
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
) -> Any:
    if squares_x < 3 or squares_y < 3:
        raise CalibrationError("ChArUco board needs at least 3x3 squares")
    if square_length_m <= 0.0 or marker_length_m <= 0.0:
        raise CalibrationError("ChArUco square and marker lengths must be positive")
    if marker_length_m >= square_length_m:
        raise CalibrationError(
            "ChArUco marker length must be smaller than square length"
        )
    try:
        return cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            float(square_length_m),
            float(marker_length_m),
            _dictionary(dictionary_name),
        )
    except cv2.error as exc:
        raise CalibrationError(f"could not create ChArUco board: {exc}") from exc


def generate_charuco_board(
    output_path: Path,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
    pixels_per_square: int,
    margin_pixels: int,
) -> Dict[str, Any]:
    if output_path.suffix.lower() != ".png":
        raise CalibrationError("ChArUco board output must be a PNG file")
    if pixels_per_square < 80:
        raise CalibrationError("pixels per square must be at least 80")
    if margin_pixels < 20:
        raise CalibrationError("board margin must be at least 20 pixels")
    board = _charuco_board(
        squares_x,
        squares_y,
        square_length_m,
        marker_length_m,
        dictionary_name,
    )
    width = squares_x * pixels_per_square + 2 * margin_pixels
    height = squares_y * pixels_per_square + 2 * margin_pixels
    image = board.generateImage(
        (width, height),
        marginSize=margin_pixels,
        borderBits=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels_per_metre = pixels_per_square / square_length_m
    print_dpi = pixels_per_metre * 0.0254
    try:
        Image.fromarray(image).save(
            output_path,
            format="PNG",
            dpi=(print_dpi, print_dpi),
        )
    except OSError as exc:
        raise CalibrationError(
            f"could not save ChArUco board: {output_path}"
        ) from exc
    margin_length_m = margin_pixels / pixels_per_square * square_length_m
    metadata = {
        "target_type": "charuco",
        "image": str(output_path),
        "dictionary": dictionary_name,
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_m": square_length_m,
        "marker_length_m": marker_length_m,
        "printed_pattern_width_m": squares_x * square_length_m,
        "printed_pattern_height_m": squares_y * square_length_m,
        "printed_page_width_m": squares_x * square_length_m + 2 * margin_length_m,
        "printed_page_height_m": squares_y * square_length_m + 2 * margin_length_m,
        "embedded_print_dpi": print_dpi,
        "image_width_px": width,
        "image_height_px": height,
        "print_note": (
            "Measure a chessboard square after printing and pass its actual "
            "side length to --square-length-m. Print at actual size/100%, "
            "disable fit-to-page, and keep the white margin."
        ),
    }
    metadata_path = output_path.with_suffix(".json")
    _write_json(metadata_path, metadata)
    metadata["metadata"] = str(metadata_path)
    return metadata


def _camera_matrix(camera_payload: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    intrinsics = camera_payload.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise CalibrationError("capture camera JSON has no intrinsics")
    camera_matrix = np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(
        intrinsics.get("distortion_coefficients", [0, 0, 0, 0, 0]),
        dtype=np.float64,
    )
    return camera_matrix, coefficients


def _detect_charuco_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    marker_length_m: float,
    dictionary_name: str,
    minimum_corners: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Sequence[np.ndarray],
    np.ndarray,
    float,
]:
    board = _charuco_board(
        squares_x,
        squares_y,
        square_length_m,
        marker_length_m,
        dictionary_name,
    )
    available_corners = (squares_x - 1) * (squares_y - 1)
    if minimum_corners < 4 or minimum_corners > available_corners:
        raise CalibrationError(
            "minimum ChArUco corners must be between 4 and "
            f"{available_corners} for this board"
        )
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_parameters.cameraMatrix = camera_matrix
    charuco_parameters.distCoeffs = distortion
    charuco_parameters.minMarkers = 2
    charuco_parameters.tryRefineMarkers = True
    detector_parameters = cv2.aruco.DetectorParameters()
    detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(
        board,
        charuco_parameters,
        detector_parameters,
    )
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        image
    )
    marker_count = 0 if marker_ids is None else len(marker_ids)
    if marker_count == 0:
        raise CalibrationError("no ArUco markers from the ChArUco board were detected")
    corner_count = 0 if charuco_ids is None else len(charuco_ids)
    if corner_count < minimum_corners:
        raise CalibrationError(
            f"only {corner_count} ChArUco corners were detected; "
            f"at least {minimum_corners} are required"
        )
    if board.checkCharucoCornersCollinear(charuco_ids):
        raise CalibrationError("detected ChArUco corners are collinear")
    object_points, image_points = board.matchImagePoints(
        charuco_corners, charuco_ids
    )
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or float(tvec.reshape(3)[2]) <= 0.0:
        raise CalibrationError("could not estimate a positive-depth ChArUco pose")
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec,
            tvec,
        )
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    reprojection = float(
        np.sqrt(
            np.mean(
                np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)
            )
        )
    )
    transform = make_transform(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
    return (
        transform,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
        reprojection,
    )


def _read_robot_pose(arm: Any) -> np.ndarray:
    code, pose = arm.get_position_aa(is_radian=True)
    if code != 0:
        raise CalibrationError(f"xArm get_position_aa failed with code {code}")
    return pose_aa_to_transform(pose)


def _connect_robot(robot_ip: str) -> Any:
    arm = connect_xarm(robot_ip, CalibrationError)
    if int(getattr(arm, "error_code", 0)) != 0:
        arm.disconnect()
        raise CalibrationError(f"xArm has error code {arm.error_code}; clear it in Studio")
    return arm


def capture_sample(arguments: argparse.Namespace) -> Dict[str, Any]:
    board_metadata = {
        "target_type": "charuco",
        "dictionary": arguments.dictionary,
        "squares_x": arguments.squares_x,
        "squares_y": arguments.squares_y,
        "square_length_m": arguments.square_length_m,
        "marker_length_m": arguments.marker_length_m,
    }
    _charuco_board(
        arguments.squares_x,
        arguments.squares_y,
        arguments.square_length_m,
        arguments.marker_length_m,
        arguments.dictionary,
    )
    sample_file = arguments.samples.expanduser().resolve()
    payload = _load_samples(sample_file)
    samples = payload["samples"]
    # Curated calibration files can contain sparse original capture indices.
    # Continue after the largest one so a later capture never overwrites a
    # retained image or creates a duplicate sample index.
    index = max((int(sample["index"]) for sample in samples), default=0) + 1
    output_dir = arguments.image_dir.expanduser().resolve()
    prefix = f"calib_{index:03d}"

    arm = _connect_robot(arguments.robot_ip)
    try:
        required_tcp_offset = np.asarray(
            arguments.required_tcp_offset_mm, dtype=np.float64
        )
        current_tcp_offset = read_tcp_offset(
            arm,
            CalibrationError,
            expected=required_tcp_offset,
            tolerance=arguments.tcp_offset_tolerance_mm,
        )
        if current_tcp_offset.shape != (6,) or not np.allclose(
            current_tcp_offset,
            required_tcp_offset,
            atol=arguments.tcp_offset_tolerance_mm,
        ):
            raise CalibrationError(
                "xArm TCP offset does not match the required xArm Gripper "
                f"offset: current={current_tcp_offset.tolist()}, "
                f"required={required_tcp_offset.tolist()}"
            )
        before = _read_robot_pose(arm)
        capture = capture_realsense(
            output_dir=output_dir,
            prefix=prefix,
            warmup_frames=arguments.warmup_frames,
        )
        after = _read_robot_pose(arm)
    finally:
        arm.disconnect()

    translation_motion = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    rotation_motion = rotation_distance_rad(after[:3, :3], before[:3, :3])
    if translation_motion > 0.0005 or rotation_motion > np.deg2rad(0.25):
        raise CalibrationError(
            "robot moved during capture "
            f"({translation_motion * 1000:.2f} mm, {np.rad2deg(rotation_motion):.2f} deg)"
        )
    base_to_tcp = before.copy()

    try:
        camera_payload = json.loads(
            capture.camera_info_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError("could not read captured camera metadata") from exc
    image = cv2.imread(str(capture.rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise CalibrationError(f"could not read captured RGB: {capture.rgb_path}")
    camera_matrix, distortion = _camera_matrix(camera_payload)
    (
        camera_to_board,
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
        reprojection,
    ) = _detect_charuco_pose(
        image,
        camera_matrix,
        distortion,
        arguments.squares_x,
        arguments.squares_y,
        arguments.square_length_m,
        arguments.marker_length_m,
        arguments.dictionary,
        arguments.min_charuco_corners,
    )
    if reprojection > arguments.max_reprojection_error_px:
        raise CalibrationError(
            f"ChArUco reprojection error is too high: {reprojection:.2f} px"
        )

    overlay = image.copy()
    cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(
        overlay,
        charuco_corners,
        charuco_ids,
        (0, 255, 0),
    )
    cv2.drawFrameAxes(
        overlay,
        camera_matrix,
        distortion,
        cv2.Rodrigues(camera_to_board[:3, :3])[0],
        camera_to_board[:3, 3],
        arguments.square_length_m * 2.0,
    )
    overlay_path = output_dir / f"{prefix}_charuco_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise CalibrationError(f"could not save ChArUco overlay: {overlay_path}")

    serial = str(camera_payload.get("serial_number", ""))
    existing_serial = payload.get("camera_serial_number")
    if existing_serial and serial != existing_serial:
        raise CalibrationError(
            f"D435 serial changed from {existing_serial} to {serial}"
        )
    existing_tcp_offset = payload.get("tcp_offset_mm_rad")
    if existing_tcp_offset is not None and not np.allclose(
        np.asarray(existing_tcp_offset, dtype=np.float64),
        current_tcp_offset,
        atol=arguments.tcp_offset_tolerance_mm,
    ):
        raise CalibrationError("xArm TCP offset changed between calibration samples")
    existing_board = payload.get("charuco_board")
    if existing_board is not None and existing_board != board_metadata:
        raise CalibrationError(
            "ChArUco board settings changed between calibration samples"
        )
    payload.update(
        {
            "robot_model": "xArm7",
            "robot_ip": arguments.robot_ip,
            "camera_model": str(camera_payload.get("camera_name", "D435")),
            "camera_serial_number": serial,
            "tcp_offset_mm_rad": current_tcp_offset.tolist(),
            "charuco_board": board_metadata,
            "updated_at": _utc_now(),
        }
    )
    sample = {
        "index": index,
        "captured_at": _utc_now(),
        "T_base_tcp": base_to_tcp.tolist(),
        "T_camera_charuco_board": camera_to_board.tolist(),
        "charuco_corner_count": int(len(charuco_ids)),
        "aruco_marker_count": int(len(marker_ids)),
        "charuco_reprojection_error_px": reprojection,
        "robot_motion_during_capture_m": translation_motion,
        "robot_rotation_during_capture_deg": float(np.rad2deg(rotation_motion)),
        "rgb_path": str(capture.rgb_path),
        "charuco_overlay_path": str(overlay_path),
        "camera_info_path": str(capture.camera_info_path),
    }
    samples.append(sample)
    _write_json(sample_file, payload)
    return sample


def _pairwise_quality(
    base_to_camera: np.ndarray,
    base_to_tcp: Sequence[np.ndarray],
    camera_to_board: Sequence[np.ndarray],
) -> Dict[str, float]:
    tcp_to_board = [
        invert_transform(robot_pose) @ base_to_camera @ board_pose
        for robot_pose, board_pose in zip(base_to_tcp, camera_to_board)
    ]
    translation_errors = []
    rotation_errors = []
    for left in range(len(tcp_to_board)):
        for right in range(left + 1, len(tcp_to_board)):
            translation_errors.append(
                float(
                    np.linalg.norm(
                        tcp_to_board[left][:3, 3] - tcp_to_board[right][:3, 3]
                    )
                )
            )
            rotation_errors.append(
                rotation_distance_rad(
                    tcp_to_board[left][:3, :3], tcp_to_board[right][:3, :3]
                )
            )
    translation_array = np.asarray(translation_errors, dtype=np.float64)
    rotation_array = np.asarray(rotation_errors, dtype=np.float64)
    return {
        "pairwise_translation_rms_m": float(
            np.sqrt(np.mean(translation_array**2))
        ),
        "pairwise_translation_max_m": float(np.max(translation_array)),
        "pairwise_rotation_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean(rotation_array**2)))
        ),
        "pairwise_rotation_max_deg": float(np.rad2deg(np.max(rotation_array))),
    }


def _sample_diversity(base_to_tcp: Sequence[np.ndarray]) -> Dict[str, float]:
    translations = np.stack([pose[:3, 3] for pose in base_to_tcp])
    maximum_translation = 0.0
    maximum_rotation = 0.0
    for left in range(len(base_to_tcp)):
        for right in range(left + 1, len(base_to_tcp)):
            maximum_translation = max(
                maximum_translation,
                float(np.linalg.norm(translations[left] - translations[right])),
            )
            maximum_rotation = max(
                maximum_rotation,
                rotation_distance_rad(
                    base_to_tcp[left][:3, :3], base_to_tcp[right][:3, :3]
                ),
            )
    return {
        "translation_span_m": maximum_translation,
        "rotation_span_deg": float(np.rad2deg(maximum_rotation)),
    }


def solve_calibration(arguments: argparse.Namespace) -> Dict[str, Any]:
    payload = _load_samples(arguments.samples.expanduser().resolve())
    samples = payload["samples"]
    if len(samples) < arguments.min_samples:
        raise CalibrationError(
            f"need at least {arguments.min_samples} samples, found {len(samples)}"
        )
    base_to_tcp = [
        validate_transform(np.asarray(sample["T_base_tcp"], dtype=np.float64))
        for sample in samples
    ]
    camera_to_board = [
        validate_transform(
            np.asarray(sample["T_camera_charuco_board"], dtype=np.float64)
        )
        for sample in samples
    ]
    diversity = _sample_diversity(base_to_tcp)
    diversity_passed = (
        diversity["translation_span_m"] >= arguments.min_translation_span_m
        and diversity["rotation_span_deg"] >= arguments.min_rotation_span_deg
    )
    if not diversity_passed and not arguments.allow_low_diversity:
        raise CalibrationError(
            "sample poses are not diverse enough: "
            f"translation={diversity['translation_span_m']:.3f} m, "
            f"rotation={diversity['rotation_span_deg']:.1f} deg"
        )

    # Eye-to-hand conversion for calibrateHandEye: passing T_tcp_base
    # (the inverse robot pose) makes the returned camera-to-pseudo-gripper
    # transform exactly T_base_camera.
    tcp_to_base = [invert_transform(pose) for pose in base_to_tcp]
    candidates = []
    for method_name, method in _METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [pose[:3, :3] for pose in tcp_to_base],
                [pose[:3, 3] for pose in tcp_to_base],
                [pose[:3, :3] for pose in camera_to_board],
                [pose[:3, 3] for pose in camera_to_board],
                method=method,
            )
            base_to_camera = make_transform(rotation, np.asarray(translation).reshape(3))
            quality = _pairwise_quality(
                base_to_camera, base_to_tcp, camera_to_board
            )
            score = (
                quality["pairwise_translation_rms_m"]
                + np.deg2rad(quality["pairwise_rotation_rms_deg"]) * 0.05
            )
        except (cv2.error, TransformError, ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(base_to_camera)):
            candidates.append((score, method_name, base_to_camera, quality))
    if not candidates:
        raise CalibrationError("all OpenCV hand-eye calibration methods failed")
    candidates.sort(key=lambda item: item[0])
    _, method_name, base_to_camera, quality = candidates[0]
    quality_passed = (
        quality["pairwise_translation_rms_m"]
        <= arguments.max_translation_rms_m
        and quality["pairwise_rotation_rms_deg"]
        <= arguments.max_rotation_rms_deg
        and diversity_passed
    )
    result = {
        "schema_version": 2,
        "calibration_type": "eye-to-hand",
        "matrix_convention": "p_base = T_base_camera @ p_camera",
        "T_base_camera": base_to_camera.tolist(),
        "robot": {
            "model": "xArm7",
            "ip": payload.get("robot_ip"),
            "pose_frame": "configured xArm TCP",
            "tcp_offset_mm_rad": payload.get("tcp_offset_mm_rad"),
        },
        "camera": {
            "model": payload.get("camera_model"),
            "serial_number": payload.get("camera_serial_number"),
            "frame": "RealSense color optical frame",
        },
        "calibration_target": payload.get("charuco_board"),
        "method": method_name,
        "sample_count": len(samples),
        "diversity": {**diversity, "passed": diversity_passed},
        "quality": {**quality, "passed": quality_passed},
        "created_at": _utc_now(),
        "source_samples": str(arguments.samples.expanduser().resolve()),
    }
    _write_json(arguments.output.expanduser().resolve(), result)
    if not quality_passed:
        raise CalibrationError(
            "calibration was saved for diagnosis but failed quality thresholds: "
            f"translation RMS={quality['pairwise_translation_rms_m'] * 1000:.1f} mm, "
            f"rotation RMS={quality['pairwise_rotation_rms_deg']:.2f} deg"
        )
    return result


def _load_robot_settings(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"could not read robot config: {path}") from exc
    if not isinstance(payload, dict):
        raise CalibrationError("robot config must contain a JSON object")
    try:
        tcp_offset = [float(value) for value in payload["required_tcp_offset_mm"]]
        tolerance = float(payload["tcp_offset_tolerance_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            "robot config needs required_tcp_offset_mm and "
            "tcp_offset_tolerance_mm"
        ) from exc
    if len(tcp_offset) != 6 or tolerance < 0.0:
        raise CalibrationError("robot config has invalid TCP offset settings")
    robot_ip = os.environ.get("AFFORDGRASP_ROBOT_IP") or payload.get("robot_ip")
    if not isinstance(robot_ip, str) or not robot_ip.strip():
        raise CalibrationError(
            "set AFFORDGRASP_ROBOT_IP or robot_ip in robot_config.json"
        )
    return {
        "robot_ip": robot_ip.strip(),
        "required_tcp_offset_mm": tcp_offset,
        "tcp_offset_tolerance_mm": tolerance,
    }


def _prepare_arguments(arguments: argparse.Namespace) -> bool:
    if arguments.square_length_m <= 0.0:
        raise CalibrationError("square length must be positive")
    settings = _load_robot_settings(arguments.robot_config)
    for name, value in settings.items():
        setattr(arguments, name, value)

    arguments.squares_x = _BOARD_SQUARES_X
    arguments.squares_y = _BOARD_SQUARES_Y
    arguments.marker_length_m = (
        arguments.square_length_m * _MARKER_TO_SQUARE_RATIO
    )
    arguments.dictionary = _BOARD_DICTIONARY
    arguments.min_charuco_corners = 8
    arguments.warmup_frames = 30
    arguments.max_reprojection_error_px = 2.0
    arguments.min_samples = _MIN_SAMPLES
    arguments.min_translation_span_m = _MIN_TRANSLATION_SPAN_M
    arguments.min_rotation_span_deg = _MIN_ROTATION_SPAN_DEG
    arguments.max_translation_rms_m = _MAX_TRANSLATION_RMS_M
    arguments.max_rotation_rms_deg = _MAX_ROTATION_RMS_DEG
    arguments.allow_low_diversity = False

    board_path = arguments.board.expanduser().resolve()
    if board_path.exists():
        return False
    generate_charuco_board(
        board_path,
        arguments.squares_x,
        arguments.squares_y,
        arguments.square_length_m,
        arguments.marker_length_m,
        arguments.dictionary,
        pixels_per_square=300,
        margin_pixels=100,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one eye-to-hand sample. Repeat at varied robot poses; "
            "the calibration is solved automatically after 12 samples."
        )
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path(
            os.environ.get("AFFORDGRASP_ROBOT_CONFIG", "robot_config.json")
        ),
        help="robot configuration containing the IP and required TCP offset",
    )
    parser.add_argument(
        "--square-length-m",
        type=float,
        default=0.030,
        help="measured side length of one printed ChArUco square",
    )
    parser.add_argument(
        "--board",
        type=Path,
        default=Path("calibration/charuco_4x5.png"),
        help="generated printable ChArUco board",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("calibration/eye_to_hand_charuco_samples.json"),
        help="accumulated calibration samples",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("calibration/captures"),
        help="captured calibration images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "AFFORDGRASP_EYE_TO_HAND_CALIBRATION",
                "calibration/eye_to_hand.json",
            )
        ),
        help="validated eye-to-hand calibration",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if _prepare_arguments(arguments):
            print(
                json.dumps(
                    {
                        "status": "board_created",
                        "board": str(arguments.board.expanduser().resolve()),
                        "next": (
                            "Print at 100%, attach it rigidly to the robot TCP, "
                            "then run this command again."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        sample = capture_sample(arguments)
        sample_count = len(
            _load_samples(arguments.samples.expanduser().resolve())["samples"]
        )
        summary: Dict[str, Any] = {
            "status": "sample_captured",
            "sample_index": sample["index"],
            "sample_count": sample_count,
        }
        if sample_count < arguments.min_samples:
            summary["remaining_samples"] = arguments.min_samples - sample_count
            summary["next"] = (
                "Move the robot to a substantially different position and "
                "orientation, then run this command again."
            )
        else:
            result = solve_calibration(arguments)
            summary.update(
                {
                    "status": "calibration_complete",
                    "output": str(arguments.output.expanduser().resolve()),
                    "method": result["method"],
                    "diversity": result["diversity"],
                    "quality": result["quality"],
                }
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (CalibrationError, CameraCaptureError, TransformError) as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
