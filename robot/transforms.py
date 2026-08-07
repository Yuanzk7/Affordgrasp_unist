"""Small, explicit homogeneous-transform helpers used by robot code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class TransformError(ValueError):
    """Raised when a transform is malformed or uses an unexpected convention."""


def validate_rotation(rotation: np.ndarray, label: str = "rotation") -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise TransformError(f"{label} must be a finite 3x3 matrix")
    orthogonality = np.linalg.norm(value.T @ value - np.eye(3), ord="fro")
    determinant = float(np.linalg.det(value))
    if orthogonality > 1e-4 or abs(determinant - 1.0) > 1e-4:
        raise TransformError(
            f"{label} is not a proper rotation "
            f"(orthogonality={orthogonality:.3g}, det={determinant:.6f})"
        )
    return value


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = validate_rotation(rotation)
    vector = np.asarray(translation, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise TransformError("translation must contain three finite values")
    transform[:3, 3] = vector
    return transform


def validate_transform(transform: np.ndarray, label: str = "transform") -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise TransformError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise TransformError(f"{label} must end with [0, 0, 0, 1]")
    validate_rotation(value[:3, :3], f"{label} rotation")
    return value


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = validate_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = value[:3, :3].T
    inverse[:3, 3] = -(value[:3, :3].T @ value[:3, 3])
    return inverse


def pose_aa_to_transform(pose: Sequence[float]) -> np.ndarray:
    """Convert xArm [x,y,z mm, rx,ry,rz rad] into ``T_base_tcp``."""

    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.size < 6 or not np.all(np.isfinite(values[:6])):
        raise TransformError("xArm axis-angle pose must contain six finite values")
    rotation = cv2.Rodrigues(values[3:6])[0]
    return make_transform(rotation, values[:3] / 1000.0)


def transform_to_pose_aa(transform: np.ndarray) -> list[float]:
    """Convert a metric transform into xArm mm + radians axis-angle pose."""

    value = validate_transform(transform)
    rotation_vector = cv2.Rodrigues(value[:3, :3])[0].reshape(3)
    return [
        *(value[:3, 3] * 1000.0).tolist(),
        *rotation_vector.tolist(),
    ]


def rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = validate_rotation(first) @ validate_rotation(second).T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransformError(f"could not read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TransformError(f"JSON root must be an object: {path}")
    return payload


def load_base_to_camera(path: Path) -> np.ndarray:
    payload = load_json_object(path)
    if payload.get("calibration_type") != "eye-to-hand":
        raise TransformError("calibration_type must be 'eye-to-hand'")
    convention = payload.get("matrix_convention")
    if convention != "p_base = T_base_camera @ p_camera":
        raise TransformError(f"unexpected calibration convention: {convention!r}")
    quality = payload.get("quality")
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        raise TransformError("calibration quality has not passed")
    return validate_transform(
        np.asarray(payload.get("T_base_camera"), dtype=np.float64),
        "T_base_camera",
    )
