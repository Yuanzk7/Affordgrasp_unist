"""Grasp backend implementations behind the common interface."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from .interfaces import (
    GraspBackendError,
    GraspCandidate,
    GraspInput,
)


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
        max_scene_points: int = 80_000,
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
        if max_scene_points <= 0:
            raise ValueError("max_scene_points must be positive")
        self.max_gripper_width_m = float(max_gripper_width_m)
        self.gripper_height_m = float(gripper_height_m)
        self.max_scene_points = int(max_scene_points)

    def _subsample_input(
        self,
        scene_points: np.ndarray,
        affordance_region: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Limit GPU memory while retaining target-region representation."""

        point_count = len(scene_points)
        if point_count <= self.max_scene_points:
            return scene_points, affordance_region

        region_indices = np.flatnonzero(affordance_region)
        background_indices = np.flatnonzero(~affordance_region)
        # Reserve up to one quarter for affordance points, then use the rest
        # for scene context and model-free collision detection.
        region_quota = min(len(region_indices), max(1, self.max_scene_points // 4))
        background_quota = min(
            len(background_indices), self.max_scene_points - region_quota
        )
        region_quota = min(
            len(region_indices), self.max_scene_points - background_quota
        )

        random = np.random.default_rng(0)
        if len(region_indices) > region_quota:
            region_indices = random.choice(
                region_indices, region_quota, replace=False
            )
        if len(background_indices) > background_quota:
            background_indices = random.choice(
                background_indices, background_quota, replace=False
            )
        selected = np.sort(np.concatenate((region_indices, background_indices)))
        return (
            np.ascontiguousarray(scene_points[selected], dtype=np.float32),
            np.ascontiguousarray(affordance_region[selected], dtype=bool),
        )

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
            original_point_count = len(scene_points)
            scene_points, affordance_region = self._subsample_input(
                scene_points,
                affordance_region,
            )
            if len(scene_points) < original_point_count:
                print(
                    "AnyGrasp scene subsampling: "
                    f"{original_point_count} -> {len(scene_points)} points "
                    f"(affordance={int(np.count_nonzero(affordance_region))})"
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
                        "original_scene_point_count": original_point_count,
                        "backend_input_point_count": len(scene_points),
                    },
                )
            )
        return candidates
