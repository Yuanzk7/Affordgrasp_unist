"""Capture aligned metric RGB-D data from an Intel RealSense D435."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np


class CameraCaptureError(RuntimeError):
    """Raised when metric RGB-D capture cannot be completed safely."""


@dataclass(frozen=True)
class RealSenseCapture:
    """Files and calibration values saved from one synchronized capture."""

    rgb_path: Path
    depth_raw_path: Path
    depth_filtered_path: Path
    depth_preview_path: Path
    camera_info_path: Path
    depth_scale: float
    image_width: int
    image_height: int
    raw_valid_ratio: float
    filtered_valid_ratio: float
    filled_from_invalid_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rgb_path": str(self.rgb_path),
            "depth_raw_path": str(self.depth_raw_path),
            "depth_filtered_path": str(self.depth_filtered_path),
            "depth_preview_path": str(self.depth_preview_path),
            "camera_info_path": str(self.camera_info_path),
            "depth_scale": self.depth_scale,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "raw_valid_ratio": self.raw_valid_ratio,
            "filtered_valid_ratio": self.filtered_valid_ratio,
            "filled_from_invalid_ratio": self.filled_from_invalid_ratio,
        }


def _load_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraCaptureError(
            "원본 depth 촬영에는 pyrealsense2가 필요합니다. "
            "현재 Python 환경에 'python -m pip install pyrealsense2'로 "
            "설치하세요."
        ) from exc
    return rs


def _set_option_if_supported(
    sensor: Any,
    option: Any,
    value: float,
) -> bool:
    """Set a RealSense sensor option when the connected device supports it."""

    try:
        supported_options = sensor.get_supported_options()
        if option not in supported_options:
            return False
        option_range = sensor.get_option_range(option)
        clamped = max(option_range.min, min(option_range.max, value))
        sensor.set_option(option, float(clamped))
        return True
    except RuntimeError:
        return False


def _configure_depth_sensor(rs: Any, device: Any) -> Dict[str, Any]:
    """Apply D435 settings that favor point-cloud fill in tabletop scenes."""

    depth_sensor = device.first_depth_sensor()
    applied: Dict[str, Any] = {}

    preset = getattr(getattr(rs, "rs400_visual_preset", None), "high_density", None)
    if preset is not None:
        preset_value = getattr(preset, "value", preset)
        try:
            preset_number = float(preset_value)
        except (TypeError, ValueError):
            preset_number = 4.0
        applied["high_density_preset"] = _set_option_if_supported(
            depth_sensor,
            rs.option.visual_preset,
            preset_number,
        )

    applied["emitter_enabled"] = _set_option_if_supported(
        depth_sensor,
        rs.option.emitter_enabled,
        1.0,
    )
    applied["depth_auto_exposure"] = _set_option_if_supported(
        depth_sensor,
        rs.option.enable_auto_exposure,
        1.0,
    )

    color_auto_exposure = False
    for sensor in device.query_sensors():
        try:
            sensor_name = sensor.get_info(rs.camera_info.name).casefold()
        except RuntimeError:
            continue
        if "rgb" not in sensor_name and "color" not in sensor_name:
            continue
        color_auto_exposure = (
            _set_option_if_supported(
                sensor,
                rs.option.enable_auto_exposure,
                1.0,
            )
            or color_auto_exposure
        )
    applied["color_auto_exposure"] = color_auto_exposure
    return applied


def _create_depth_filters(rs: Any) -> Tuple[Any, Any, Any, Any]:
    """Create conservative D4xx spatial-temporal depth post-processing."""

    depth_to_disparity = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)

    _set_option_if_supported(spatial, rs.option.filter_magnitude, 2.0)
    _set_option_if_supported(spatial, rs.option.filter_smooth_alpha, 0.5)
    _set_option_if_supported(spatial, rs.option.filter_smooth_delta, 20.0)
    # Only fill very small horizontal gaps during edge-preserving filtering.
    _set_option_if_supported(spatial, rs.option.holes_fill, 1.0)

    _set_option_if_supported(temporal, rs.option.filter_smooth_alpha, 0.4)
    _set_option_if_supported(temporal, rs.option.filter_smooth_delta, 20.0)
    _set_option_if_supported(temporal, rs.option.holes_fill, 3.0)

    return (
        depth_to_disparity,
        spatial,
        temporal,
        disparity_to_depth,
    )


def _apply_depth_filters(depth_frame: Any, filters: Tuple[Any, ...]) -> Any:
    filtered = depth_frame
    for depth_filter in filters:
        filtered = depth_filter.process(filtered)
    return filtered.as_depth_frame()


def _intrinsics_to_dict(intrinsics: Any) -> Dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "distortion_coefficients": [
            float(value) for value in intrinsics.coeffs
        ],
    }


def _device_info(device: Any, camera_info: Any) -> str:
    try:
        return str(device.get_info(camera_info))
    except RuntimeError:
        return ""


def _create_depth_preview(
    depth_raw: np.ndarray,
    depth_scale: float,
    minimum_meters: float = 0.15,
    maximum_meters: float = 2.0,
) -> np.ndarray:
    """Colorize metric depth while keeping invalid pixels black."""

    meters = depth_raw.astype(np.float32) * depth_scale
    normalized = (meters - minimum_meters) / (maximum_meters - minimum_meters)
    normalized = np.clip(normalized, 0.0, 1.0)
    color_values = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
    preview = cv2.applyColorMap(color_values, cv2.COLORMAP_TURBO)
    preview[depth_raw == 0] = 0
    return preview


def _write_png(path: Path, image: np.ndarray, label: str) -> None:
    if not cv2.imwrite(str(path), image):
        raise CameraCaptureError(f"{label} 저장에 실패했습니다: {path}")


def capture_realsense(
    output_dir: Union[str, Path] = "captures",
    prefix: str = "scene",
    warmup_frames: int = 30,
    frame_timeout_ms: int = 5000,
) -> RealSenseCapture:
    """Capture RGB plus aligned raw/filtered metric depth from a D435.

    Depth is captured natively at 848x480, aligned into the 640x480 color
    coordinate system, and saved losslessly as uint16 PNG. The JSON record
    stores the depth scale and color intrinsics required to reconstruct 3D
    points from the aligned depth image.
    """

    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("capture prefix must not be empty")
    if Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("capture prefix must be a plain file-name prefix")
    if warmup_frames <= 0:
        raise ValueError("warmup_frames must be positive")
    if frame_timeout_ms <= 0:
        raise ValueError("frame_timeout_ms must be positive")

    rs = _load_realsense()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rgb_path = destination / f"{prefix}_rgb.png"
    depth_raw_path = destination / f"{prefix}_depth_raw.png"
    depth_filtered_path = destination / f"{prefix}_depth_filtered.png"
    depth_preview_path = destination / f"{prefix}_depth_preview.png"
    camera_info_path = destination / f"{prefix}_camera.json"

    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_stream(
        rs.stream.depth,
        848,
        480,
        rs.format.z16,
        30,
    )
    stream_config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30,
    )

    profile = None
    try:
        profile = pipeline.start(stream_config)
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        applied_options = _configure_depth_sensor(rs, device)
        align_to_color = rs.align(rs.stream.color)
        filters = _create_depth_filters(rs)

        color_frame = None
        raw_depth_frame = None
        filtered_depth_frame = None
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames(frame_timeout_ms)
            aligned_frames = align_to_color.process(frames)
            current_color = aligned_frames.get_color_frame()
            current_depth = aligned_frames.get_depth_frame()
            if not current_color or not current_depth:
                continue
            current_filtered = _apply_depth_filters(current_depth, filters)
            if not current_filtered:
                continue
            color_frame = current_color
            raw_depth_frame = current_depth
            filtered_depth_frame = current_filtered

        if color_frame is None or raw_depth_frame is None:
            raise CameraCaptureError(
                "동기화된 RGB 및 aligned depth 프레임을 받지 못했습니다."
            )
        if filtered_depth_frame is None:
            raise CameraCaptureError("후처리된 depth 프레임을 만들지 못했습니다.")

        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(raw_depth_frame.get_data()).copy()
        depth_filtered = np.asanyarray(filtered_depth_frame.get_data()).copy()
        if color_bgr.dtype != np.uint8 or color_bgr.ndim != 3:
            raise CameraCaptureError("RGB 프레임 형식이 BGR8이 아닙니다.")
        if depth_raw.dtype != np.uint16 or depth_filtered.dtype != np.uint16:
            raise CameraCaptureError("Depth 프레임 형식이 Z16이 아닙니다.")
        if depth_raw.shape != color_bgr.shape[:2]:
            raise CameraCaptureError(
                "Aligned depth와 RGB 해상도가 일치하지 않습니다: "
                f"depth={depth_raw.shape}, RGB={color_bgr.shape[:2]}"
            )
        if depth_filtered.shape != depth_raw.shape:
            raise CameraCaptureError(
                "후처리 depth와 raw depth 해상도가 일치하지 않습니다."
            )

        intrinsics = (
            color_frame.profile.as_video_stream_profile().get_intrinsics()
        )
        raw_valid_ratio = float(np.count_nonzero(depth_raw) / depth_raw.size)
        filtered_valid_ratio = float(
            np.count_nonzero(depth_filtered) / depth_filtered.size
        )
        filled_from_invalid_ratio = float(
            np.count_nonzero((depth_raw == 0) & (depth_filtered > 0))
            / depth_raw.size
        )
        preview = _create_depth_preview(depth_filtered, depth_scale)

        _write_png(rgb_path, color_bgr, "RGB")
        _write_png(depth_raw_path, depth_raw, "aligned raw depth")
        _write_png(
            depth_filtered_path,
            depth_filtered,
            "aligned filtered depth",
        )
        _write_png(depth_preview_path, preview, "depth preview")

        metadata = {
            "camera_name": _device_info(device, rs.camera_info.name),
            "serial_number": _device_info(
                device,
                rs.camera_info.serial_number,
            ),
            "firmware_version": _device_info(
                device,
                rs.camera_info.firmware_version,
            ),
            "coordinate_system": "depth aligned to RGB/color pixels",
            "depth_format": "uint16 PNG (Z16 sensor units)",
            "depth_scale_meters_per_unit": depth_scale,
            "intrinsics": _intrinsics_to_dict(intrinsics),
            "capture": {
                "native_depth_stream": {
                    "width": 848,
                    "height": 480,
                    "fps": 30,
                },
                "color_stream": {
                    "width": int(color_bgr.shape[1]),
                    "height": int(color_bgr.shape[0]),
                    "fps": 30,
                },
                "warmup_frames": warmup_frames,
                "applied_options": applied_options,
                "raw_valid_ratio": raw_valid_ratio,
                "filtered_valid_ratio": filtered_valid_ratio,
                "filled_from_invalid_ratio": filled_from_invalid_ratio,
            },
            "files": {
                "rgb": str(rgb_path),
                "depth_raw": str(depth_raw_path),
                "depth_filtered": str(depth_filtered_path),
                "depth_preview": str(depth_preview_path),
            },
        }
        camera_info_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        return RealSenseCapture(
            rgb_path=rgb_path,
            depth_raw_path=depth_raw_path,
            depth_filtered_path=depth_filtered_path,
            depth_preview_path=depth_preview_path,
            camera_info_path=camera_info_path,
            depth_scale=depth_scale,
            image_width=int(color_bgr.shape[1]),
            image_height=int(color_bgr.shape[0]),
            raw_valid_ratio=raw_valid_ratio,
            filtered_valid_ratio=filtered_valid_ratio,
            filled_from_invalid_ratio=filled_from_invalid_ratio,
        )
    except CameraCaptureError:
        raise
    except RuntimeError as exc:
        raise CameraCaptureError(f"RealSense RGB-D 촬영 실패: {exc}") from exc
    except OSError as exc:
        raise CameraCaptureError(f"RGB-D 파일 저장 실패: {exc}") from exc
    finally:
        if profile is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
