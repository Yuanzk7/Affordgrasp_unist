"""Collect and solve fixed-D435 (eye-to-hand) calibration for an xArm7."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

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
            "schema_version": 1,
            "calibration_type": "eye-to-hand",
            "matrix_conventions": {
                "robot_pose": "T_base_tcp",
                "marker_pose": "T_camera_marker",
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
    return payload


def _dictionary(name: str) -> Any:
    constant = getattr(cv2.aruco, name, None)
    if constant is None:
        raise CalibrationError(f"unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(constant)


def generate_marker(
    output_path: Path,
    marker_id: int,
    dictionary_name: str,
    pixels: int,
) -> Path:
    if marker_id < 0 or pixels < 200:
        raise CalibrationError("marker id must be non-negative and pixels >= 200")
    marker = cv2.aruco.generateImageMarker(
        _dictionary(dictionary_name), marker_id, pixels
    )
    border = max(20, pixels // 8)
    page = np.full((pixels + 2 * border, pixels + 2 * border), 255, np.uint8)
    page[border:-border, border:-border] = marker
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), page):
        raise CalibrationError(f"could not save marker image: {output_path}")
    return output_path


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


def _detect_marker_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_id: int,
    marker_length_m: float,
    dictionary_name: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    detector = cv2.aruco.ArucoDetector(
        _dictionary(dictionary_name), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        raise CalibrationError("ArUco marker was not detected")
    matches = np.flatnonzero(ids.reshape(-1) == marker_id)
    if len(matches) != 1:
        raise CalibrationError(
            f"expected exactly one marker id {marker_id}, found {len(matches)}"
        )
    image_points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
    half = marker_length_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success or float(tvec.reshape(3)[2]) <= 0.0:
        raise CalibrationError("could not estimate a positive-depth marker pose")
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    reprojection = float(
        np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1)))
    )
    transform = make_transform(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
    return transform, image_points, reprojection


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
    if arguments.marker_length_m <= 0.0:
        raise CalibrationError("marker length must be positive")
    sample_file = arguments.samples.expanduser().resolve()
    payload = _load_samples(sample_file)
    samples = payload["samples"]
    index = len(samples) + 1
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
    camera_to_marker, image_points, reprojection = _detect_marker_pose(
        image,
        camera_matrix,
        distortion,
        arguments.marker_id,
        arguments.marker_length_m,
        arguments.dictionary,
    )
    if reprojection > arguments.max_reprojection_error_px:
        raise CalibrationError(
            f"marker reprojection error is too high: {reprojection:.2f} px"
        )

    overlay = image.copy()
    cv2.polylines(
        overlay,
        [np.round(image_points).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
    )
    cv2.drawFrameAxes(
        overlay,
        camera_matrix,
        distortion,
        cv2.Rodrigues(camera_to_marker[:3, :3])[0],
        camera_to_marker[:3, 3],
        arguments.marker_length_m * 0.5,
    )
    overlay_path = output_dir / f"{prefix}_marker_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise CalibrationError(f"could not save marker overlay: {overlay_path}")

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
    payload.update(
        {
            "robot_model": "xArm7",
            "robot_ip": arguments.robot_ip,
            "camera_model": str(camera_payload.get("camera_name", "D435")),
            "camera_serial_number": serial,
            "tcp_offset_mm_rad": current_tcp_offset.tolist(),
            "marker": {
                "dictionary": arguments.dictionary,
                "id": arguments.marker_id,
                "black_square_length_m": arguments.marker_length_m,
            },
            "updated_at": _utc_now(),
        }
    )
    sample = {
        "index": index,
        "captured_at": _utc_now(),
        "T_base_tcp": base_to_tcp.tolist(),
        "T_camera_marker": camera_to_marker.tolist(),
        "marker_reprojection_error_px": reprojection,
        "robot_motion_during_capture_m": translation_motion,
        "robot_rotation_during_capture_deg": float(np.rad2deg(rotation_motion)),
        "rgb_path": str(capture.rgb_path),
        "marker_overlay_path": str(overlay_path),
        "camera_info_path": str(capture.camera_info_path),
    }
    samples.append(sample)
    _write_json(sample_file, payload)
    return sample


def read_robot_status(robot_ip: str) -> Dict[str, Any]:
    """Read controller state without enabling motion or changing configuration."""

    arm = _connect_robot(robot_ip)
    try:
        code, pose = arm.get_position_aa(is_radian=True)
        if code != 0:
            raise CalibrationError(f"xArm get_position_aa failed with code {code}")
        return {
            "robot_ip": robot_ip,
            "connected": bool(arm.connected),
            "state": int(getattr(arm, "state", -1)),
            "mode": int(getattr(arm, "mode", -1)),
            "error_code": int(getattr(arm, "error_code", 0)),
            "warn_code": int(getattr(arm, "warn_code", 0)),
            "tcp_pose_mm_rad": list(map(float, pose[:6])),
            "tcp_offset_mm_rad": read_tcp_offset(
                arm, CalibrationError
            ).tolist(),
        }
    finally:
        arm.disconnect()


def _pairwise_quality(
    base_to_camera: np.ndarray,
    base_to_tcp: Sequence[np.ndarray],
    camera_to_marker: Sequence[np.ndarray],
) -> Dict[str, float]:
    tcp_to_marker = [
        invert_transform(robot_pose) @ base_to_camera @ marker_pose
        for robot_pose, marker_pose in zip(base_to_tcp, camera_to_marker)
    ]
    translation_errors = []
    rotation_errors = []
    for left in range(len(tcp_to_marker)):
        for right in range(left + 1, len(tcp_to_marker)):
            translation_errors.append(
                float(
                    np.linalg.norm(
                        tcp_to_marker[left][:3, 3] - tcp_to_marker[right][:3, 3]
                    )
                )
            )
            rotation_errors.append(
                rotation_distance_rad(
                    tcp_to_marker[left][:3, :3], tcp_to_marker[right][:3, :3]
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
    camera_to_marker = [
        validate_transform(np.asarray(sample["T_camera_marker"], dtype=np.float64))
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
                [pose[:3, :3] for pose in camera_to_marker],
                [pose[:3, 3] for pose in camera_to_marker],
                method=method,
            )
            base_to_camera = make_transform(rotation, np.asarray(translation).reshape(3))
            quality = _pairwise_quality(
                base_to_camera, base_to_tcp, camera_to_marker
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
        "schema_version": 1,
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
        "marker": payload.get("marker"),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--robot-ip", default="192.168.1.216")

    marker = subparsers.add_parser("generate-marker")
    marker.add_argument("--output", type=Path, required=True)
    marker.add_argument("--marker-id", type=int, default=0)
    marker.add_argument("--dictionary", default="DICT_4X4_50")
    marker.add_argument("--pixels", type=int, default=1200)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--robot-ip", default="192.168.1.216")
    capture.add_argument("--marker-id", type=int, default=0)
    capture.add_argument("--marker-length-m", type=float, required=True)
    capture.add_argument("--dictionary", default="DICT_4X4_50")
    capture.add_argument(
        "--samples", type=Path, default=Path("calibration/eye_to_hand_samples.json")
    )
    capture.add_argument(
        "--image-dir", type=Path, default=Path("calibration/captures")
    )
    capture.add_argument("--warmup-frames", type=int, default=30)
    capture.add_argument("--max-reprojection-error-px", type=float, default=2.0)
    capture.add_argument(
        "--required-tcp-offset-mm",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 172.0, 0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    capture.add_argument("--tcp-offset-tolerance-mm", type=float, default=3.0)

    solve = subparsers.add_parser("solve")
    solve.add_argument(
        "--samples", type=Path, default=Path("calibration/eye_to_hand_samples.json")
    )
    solve.add_argument(
        "--output", type=Path, default=Path("calibration/eye_to_hand.json")
    )
    solve.add_argument("--min-samples", type=int, default=12)
    solve.add_argument("--min-translation-span-m", type=float, default=0.08)
    solve.add_argument("--min-rotation-span-deg", type=float, default=25.0)
    solve.add_argument("--max-translation-rms-m", type=float, default=0.01)
    solve.add_argument("--max-rotation-rms-deg", type=float, default=2.5)
    solve.add_argument("--allow-low-diversity", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "status":
            print(
                json.dumps(
                    read_robot_status(arguments.robot_ip),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif arguments.command == "generate-marker":
            path = generate_marker(
                arguments.output.expanduser().resolve(),
                arguments.marker_id,
                arguments.dictionary,
                arguments.pixels,
            )
            print(json.dumps({"marker": str(path)}, indent=2))
        elif arguments.command == "capture":
            sample = capture_sample(arguments)
            print(json.dumps(sample, ensure_ascii=False, indent=2))
        else:
            result = solve_calibration(arguments)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (CalibrationError, CameraCaptureError, TransformError) as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
