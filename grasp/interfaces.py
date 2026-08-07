"""Backend-neutral data contracts for grasp pose generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class GraspBackendError(RuntimeError):
    """Raised when a grasp backend cannot produce valid candidates."""


@dataclass(frozen=True)
class GraspInput:
    """Scene point cloud plus the affordance region passed to grasp backends.

    Coordinates are metres in the aligned RealSense color-camera frame:
    +x right, +y down and +z forward. Colors are RGB floats in [0, 1].
    ``affordance_region_mask`` maps element-for-element to the scene points.
    """

    scene_points_xyz_m: np.ndarray
    scene_colors_rgb: np.ndarray
    affordance_region_mask: np.ndarray
    affordance_centroid_xyz_m: np.ndarray

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(
            self.scene_points_xyz_m,
            dtype=np.float32,
        )
        colors = np.ascontiguousarray(
            self.scene_colors_rgb,
            dtype=np.float32,
        )
        region = np.ascontiguousarray(
            self.affordance_region_mask,
            dtype=bool,
        )
        centroid = np.asarray(self.affordance_centroid_xyz_m, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(
                "scene_points_xyz_m must have shape (N, 3) with N > 0"
            )
        if colors.shape != points.shape:
            raise ValueError(
                "scene_colors_rgb must have the same shape as scene_points_xyz_m"
            )
        if region.shape != (len(points),):
            raise ValueError(
                "affordance_region_mask must have shape (N,) matching scene points"
            )
        if not np.any(region):
            raise ValueError("affordance_region_mask must select at least one point")
        if centroid.shape != (3,):
            raise ValueError("affordance_centroid_xyz_m must have shape (3,)")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(centroid)):
            raise ValueError("grasp input contains non-finite coordinates")
        if not np.all(np.isfinite(colors)) or np.any(colors < 0) or np.any(colors > 1):
            raise ValueError(
                "scene_colors_rgb must contain finite values in [0, 1]"
            )
        object.__setattr__(self, "scene_points_xyz_m", points)
        object.__setattr__(self, "scene_colors_rgb", colors)
        object.__setattr__(self, "affordance_region_mask", region)
        object.__setattr__(self, "affordance_centroid_xyz_m", centroid)

    @property
    def affordance_points_xyz_m(self) -> np.ndarray:
        """Return the scene points selected by the affordance region."""

        return self.scene_points_xyz_m[self.affordance_region_mask]

    @property
    def affordance_colors_rgb(self) -> np.ndarray:
        """Return RGB values for the affordance-selected scene points."""

        return self.scene_colors_rgb[self.affordance_region_mask]


@dataclass(frozen=True)
class GraspCandidate:
    """One backend-neutral parallel-jaw grasp candidate.

    ``rotation_matrix_camera`` stores the gripper axes as columns in the camera
    frame: approach, jaw-closing and the remaining right-handed axis.
    """

    rotation_matrix_camera: np.ndarray
    translation_xyz_m: np.ndarray
    width_m: float
    score: float
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_matrix_camera, dtype=np.float64)
        translation = np.asarray(self.translation_xyz_m, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_matrix_camera must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation_xyz_m must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("grasp pose contains non-finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2):
            raise ValueError("rotation_matrix_camera must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("rotation_matrix_camera must be right-handed")
        width = float(self.width_m)
        score = float(self.score)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("width_m must be positive")
        if not np.isfinite(score):
            raise ValueError("score must be finite")
        if not self.backend:
            raise ValueError("backend must not be empty")
        object.__setattr__(self, "rotation_matrix_camera", rotation)
        object.__setattr__(self, "translation_xyz_m", translation)
        object.__setattr__(self, "width_m", width)
        object.__setattr__(self, "score", score)


class GraspBackend(Protocol):
    """Interface implemented by grasp pose backends such as AnyGrasp."""

    @property
    def name(self) -> str:
        """Stable backend identifier used in JSON output."""

    def generate(self, grasp_input: GraspInput) -> Sequence[GraspCandidate]:
        """Generate zero or more normalized grasp candidates."""


@dataclass(frozen=True)
class PreparedSample:
    """Validated image-derived data plus diagnostic output paths."""

    grasp_input: GraspInput
    image_width: int
    image_height: int
    valid_depth_pixels: int
    affordance_pixels: int
    camera_serial_number: str
    mask_path: Path
    overlay_path: Path
