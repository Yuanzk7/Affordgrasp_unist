"""Grasp backend implementations behind the common interface."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import numpy as np

from .interfaces import (
    GraspBackendError,
    GraspCandidate,
    GraspInput,
)


class PcaBaselineBackend:
    """Deterministic geometry baseline for pipeline and visualization checks.

    This is not a learned grasp detector. It estimates the local surface frame
    with PCA and places a parallel-jaw grasp at the affordance-cloud centroid.
    """

    name = "pca-baseline"

    def __init__(
        self,
        max_gripper_width_m: float = 0.10,
        minimum_gripper_width_m: float = 0.01,
        width_margin_m: float = 0.01,
    ) -> None:
        if not 0.0 < minimum_gripper_width_m <= max_gripper_width_m:
            raise ValueError(
                "gripper widths must satisfy 0 < minimum <= maximum"
            )
        if width_margin_m < 0.0:
            raise ValueError("width_margin_m must not be negative")
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.minimum_gripper_width_m = float(minimum_gripper_width_m)
        self.width_margin_m = float(width_margin_m)

    def generate(self, grasp_input: GraspInput) -> List[GraspCandidate]:
        points = grasp_input.affordance_points_xyz_m.astype(
            np.float64,
            copy=False,
        )
        if len(points) < 10:
            raise GraspBackendError(
                "PCA baseline requires at least 10 affordance depth points"
            )

        centroid = points.mean(axis=0)
        centered = points - centroid
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        axes = eigenvectors[:, order]

        # PCA: major axis follows the part, middle axis crosses it, and the
        # smallest-variance axis approximates the visible surface normal.
        closing = axes[:, 1]
        approach = axes[:, 2]
        if np.dot(approach, centroid) < 0.0:
            approach = -approach
        if closing[0] < 0.0:
            closing = -closing
        remaining = np.cross(approach, closing)
        remaining /= np.linalg.norm(remaining)
        closing = np.cross(remaining, approach)
        closing /= np.linalg.norm(closing)
        rotation = np.column_stack((approach, closing, remaining))

        closing_coordinates = centered @ closing
        lower, upper = np.percentile(closing_coordinates, (2.5, 97.5))
        measured_width = float(upper - lower)
        width = float(
            np.clip(
                measured_width + self.width_margin_m,
                self.minimum_gripper_width_m,
                self.max_gripper_width_m,
            )
        )

        total_variance = float(eigenvalues.sum())
        planar_fraction = (
            float((eigenvalues[0] + eigenvalues[1]) / total_variance)
            if total_variance > 0.0
            else 0.0
        )
        candidate = GraspCandidate(
            rotation_matrix_camera=rotation,
            translation_xyz_m=grasp_input.affordance_centroid_xyz_m,
            width_m=width,
            score=planar_fraction,
            backend=self.name,
            metadata={
                "warning": "geometry-only test pose; do not execute on a robot",
                "pca_eigenvalues_m2": eigenvalues.tolist(),
                "measured_closing_extent_m": measured_width,
            },
        )
        return [candidate]


def _resolve_anygrasp_detection_dir(anygrasp_sdk: Optional[Path]) -> Optional[Path]:
    if anygrasp_sdk is None:
        return None

    directory = anygrasp_sdk.expanduser().resolve()
    if (directory / "grasp_detection").is_dir():
        directory = directory / "grasp_detection"
    if not directory.is_dir():
        raise GraspBackendError(
            f"AnyGrasp grasp_detection directory does not exist: {directory}"
        )
    return directory


@contextmanager
def _anygrasp_sdk_context(detection_dir: Optional[Path]) -> Iterator[None]:
    if detection_dir is None:
        yield
        return

    module_path = str(detection_dir)
    previous_directory = Path.cwd()
    sys.path.insert(0, module_path)
    os.chdir(detection_dir)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        try:
            sys.path.remove(module_path)
        except ValueError:
            pass


class AnyGraspBackend:
    """Adapter that normalizes AnyGrasp SDK output to ``GraspCandidate``."""

    name = "anygrasp"

    def __init__(
        self,
        checkpoint_path: Optional[Path],
        anygrasp_sdk: Optional[Path],
        max_gripper_width_m: float,
        gripper_height_m: float,
    ) -> None:
        self.detection_dir = _resolve_anygrasp_detection_dir(anygrasp_sdk)
        if checkpoint_path is None:
            if self.detection_dir is None:
                raise GraspBackendError(
                    "AnyGrasp requires --checkpoint or --anygrasp-sdk"
                )
            checkpoint_path = (
                self.detection_dir / "log" / "checkpoint_detection.tar"
            )
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise GraspBackendError(
                f"AnyGrasp checkpoint does not exist: {self.checkpoint_path}"
            )
        if self.detection_dir is not None and not (
            self.detection_dir / "license"
        ).is_dir():
            raise GraspBackendError(
                "AnyGrasp license directory does not exist: "
                f"{self.detection_dir / 'license'}"
            )
        if not 0.0 < max_gripper_width_m <= 0.10:
            raise ValueError("max_gripper_width must be in (0, 0.10]")
        if gripper_height_m <= 0.0:
            raise ValueError("gripper_height must be positive")
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.gripper_height_m = float(gripper_height_m)

    def _create_detector(self) -> Any:
        try:
            from gsnet import create_detector
        except Exception as exc:
            raise GraspBackendError(
                "AnyGrasp SDK is missing; install gsnet and its CUDA "
                "dependencies, then complete license registration: "
                f"{exc}"
            ) from exc

        config = SimpleNamespace(
            checkpoint_path=str(self.checkpoint_path),
            max_gripper_width=self.max_gripper_width_m,
            gripper_height=self.gripper_height_m,
        )
        try:
            detector = create_detector(config)
            if detector is None:
                raise RuntimeError(
                    "AnyGrasp returned no detector instance; license validation failed"
                )
        except Exception as exc:
            raise GraspBackendError(
                "AnyGrasp detector initialization failed; verify checkpoint and "
                f"license: {exc}"
            ) from exc
        return detector

    def generate(self, grasp_input: GraspInput) -> List[GraspCandidate]:
        with _anygrasp_sdk_context(self.detection_dir):
            detector = self._create_detector()
            scene_points = np.ascontiguousarray(
                grasp_input.scene_points_xyz_m,
                dtype=np.float32,
            )
            affordance_region = np.ascontiguousarray(
                grasp_input.affordance_region_mask,
                dtype=bool,
            )
            inference_options = {
                "dense_grasp": False,
                "collision_detection": True,
                "region_steering": affordance_region,
                "approach_steering": None,
                "approach_thresh": float(np.pi),
            }
            try:
                grasps = detector.get_grasp(
                    scene_points,
                    inference_options,
                )
            except Exception as exc:
                raise GraspBackendError(
                    f"AnyGrasp inference failed: {exc}"
                ) from exc
        if grasps is None or len(grasps) == 0:
            raise GraspBackendError(
                "AnyGrasp produced no collision-free grasp in the affordance region"
            )
        grasps = grasps.nms().sort_by_score()

        rotations = np.asarray(grasps.rotation_matrices)
        translations = np.asarray(grasps.translations)
        widths = np.asarray(grasps.widths)
        scores = np.asarray(grasps.scores)
        heights = np.asarray(grasps.heights)
        depths = np.asarray(grasps.depths)
        object_ids = np.asarray(grasps.object_ids)
        if not (
            len(rotations)
            == len(translations)
            == len(widths)
            == len(scores)
            == len(heights)
            == len(depths)
            == len(object_ids)
        ):
            raise GraspBackendError("AnyGrasp returned inconsistent candidate arrays")

        candidates: List[GraspCandidate] = []
        for index in range(len(grasps)):
            insertion_depth = float(depths[index])
            tip_xyz = (
                translations[index]
                + insertion_depth * rotations[index][:, 0]
            )
            candidates.append(
                GraspCandidate(
                    rotation_matrix_camera=rotations[index],
                    translation_xyz_m=translations[index],
                    width_m=float(widths[index]),
                    score=float(scores[index]),
                    backend=self.name,
                    metadata={
                        "anygrasp_candidate_index": index,
                        "height_m": float(heights[index]),
                        "insertion_depth_m": insertion_depth,
                        "object_id": int(object_ids[index]),
                        "gripper_tip_xyz_m": np.asarray(tip_xyz).tolist(),
                        "region_steering": True,
                        "collision_detection": True,
                        "dense_grasp": False,
                    },
                )
            )
        return candidates
