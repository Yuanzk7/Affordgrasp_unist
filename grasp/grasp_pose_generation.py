"""Generate a backend-neutral grasp pose from an affordance RGB-D sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .backends import AnyGraspBackend
from .interfaces import (
    GraspBackend,
    GraspBackendError,
    GraspCandidate,
    GraspInput,
    PreparedSample,
)
from .visualization import (
    save_grasp_visualization,
    save_point_cloud_ply,
    save_scene_point_cloud_ply,
)


class GraspPoseGenerationError(RuntimeError):
    """Raised when RGB-D preparation or grasp output cannot finish."""


# Two polygons cover the upper and lower handles in the bundled pliers sample.
_SAMPLE_PLIERS_HANDLE_POLYGONS: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    (
        (285, 230),
        (299, 226),
        (390, 245),
        (414, 249),
        (433, 258),
        (427, 276),
        (408, 270),
        (388, 260),
        (300, 252),
        (286, 244),
    ),
    (
        (260, 279),
        (273, 270),
        (403, 271),
        (430, 278),
        (431, 297),
        (407, 300),
        (274, 301),
        (260, 292),
    ),
)


def _load_camera(camera_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(camera_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraspPoseGenerationError(
            f"could not read camera metadata: {camera_path}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("intrinsics"), dict
    ):
        raise GraspPoseGenerationError("camera metadata has no intrinsics object")
    return payload


def _positive_number(payload: Dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraspPoseGenerationError(f"camera field {field!r} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise GraspPoseGenerationError(f"camera field {field!r} must be positive")
    return value


def _load_images(rgb_path: Path, depth_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise GraspPoseGenerationError(
            "could not read the sample RGB-D images"
        ) from exc

    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise GraspPoseGenerationError(
            "sample depth must be a single-channel integer PNG"
        )
    if depth.size == 0 or int(depth.max()) > np.iinfo(np.uint16).max:
        raise GraspPoseGenerationError("sample depth is not valid Z16 data")
    depth = depth.astype(np.uint16, copy=False)
    if rgb.shape[:2] != depth.shape:
        raise GraspPoseGenerationError(
            "RGB and depth dimensions differ: "
            f"{rgb.shape[:2]} vs {depth.shape}"
        )
    return rgb, depth


def _create_sample_affordance_mask(width: int, height: int) -> np.ndarray:
    if (width, height) != (640, 480):
        raise GraspPoseGenerationError(
            "the bundled manual pliers mask requires the 640x480 sample image"
        )
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in _SAMPLE_PLIERS_HANDLE_POLYGONS:
        draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def _load_affordance_mask(
    mask_source: Optional[Path],
    width: int,
    height: int,
) -> np.ndarray:
    if mask_source is None:
        return _create_sample_affordance_mask(width, height)
    try:
        with Image.open(mask_source) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except (OSError, ValueError) as exc:
        raise GraspPoseGenerationError(
            f"could not read affordance mask: {mask_source}"
        ) from exc
    if mask.shape != (height, width):
        raise GraspPoseGenerationError(
            f"affordance mask dimensions differ: {mask.shape} vs {(height, width)}"
        )
    if not np.any(mask):
        raise GraspPoseGenerationError("affordance mask is empty")
    return mask


def _save_mask_outputs(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "affordance_mask_input.png"
    overlay_path = output_dir / "affordance_overlay.png"

    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    overlay = rgb.astype(np.float32)
    cyan = np.array([0.0, 255.0, 255.0], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.40 + cyan * 0.60
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
        overlay_path
    )
    return mask_path, overlay_path


def prepare_sample(
    sample_dir: Path,
    output_dir: Path,
    depth_source: str = "filtered",
    minimum_depth_m: float = 0.10,
    maximum_depth_m: float = 2.00,
    mask_source: Optional[Path] = None,
    rgb_source: Optional[Path] = None,
    depth_image_source: Optional[Path] = None,
    camera_source: Optional[Path] = None,
) -> PreparedSample:
    """Build the backend-neutral, affordance-filtered camera point cloud.

    By default this reads the conventional filenames in ``sample_dir``. Passing
    all three explicit RGB, depth and camera sources connects a real pipeline
    run without renaming or copying its capture files.
    """

    if depth_source not in {"raw", "filtered"}:
        raise ValueError("depth_source must be raw or filtered")
    if not 0.0 <= minimum_depth_m < maximum_depth_m:
        raise ValueError("depth range must satisfy 0 <= minimum < maximum")

    output_dir = output_dir.expanduser().resolve()
    explicit_sources = (rgb_source, depth_image_source, camera_source)
    if any(source is not None for source in explicit_sources) and not all(
        source is not None for source in explicit_sources
    ):
        raise ValueError("explicit input requires --rgb, --depth and --camera together")
    if rgb_source is not None:
        assert depth_image_source is not None and camera_source is not None
        rgb_path = rgb_source.expanduser().resolve()
        depth_path = depth_image_source.expanduser().resolve()
        camera_path = camera_source.expanduser().resolve()
    else:
        sample_dir = sample_dir.expanduser().resolve()
        rgb_path = sample_dir / "sample_rgb.png"
        depth_path = sample_dir / f"sample_depth_{depth_source}.png"
        camera_path = sample_dir / "sample_camera.json"
    for path in (rgb_path, depth_path, camera_path):
        if not path.is_file():
            raise GraspPoseGenerationError(f"required sample file is missing: {path}")
    if mask_source is not None:
        mask_source = mask_source.expanduser().resolve()
        if not mask_source.is_file():
            raise GraspPoseGenerationError(
                f"affordance mask does not exist: {mask_source}"
            )

    rgb, depth = _load_images(rgb_path, depth_path)
    camera = _load_camera(camera_path)
    intrinsics = camera["intrinsics"]
    depth_scale = _positive_number(camera, "depth_scale_meters_per_unit")
    fx = _positive_number(intrinsics, "fx")
    fy = _positive_number(intrinsics, "fy")
    cx = float(intrinsics.get("cx"))
    cy = float(intrinsics.get("cy"))
    if not np.isfinite(cx) or not np.isfinite(cy):
        raise GraspPoseGenerationError("camera principal point must be finite")

    height, width = depth.shape
    if int(intrinsics.get("width", -1)) != width or int(
        intrinsics.get("height", -1)
    ) != height:
        raise GraspPoseGenerationError(
            "camera intrinsics dimensions do not match the sample depth image"
        )

    affordance_mask = _load_affordance_mask(mask_source, width, height)
    mask_path, overlay_path = _save_mask_outputs(rgb, affordance_mask, output_dir)

    z = depth.astype(np.float32) * np.float32(depth_scale)
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x = (u - np.float32(cx)) / np.float32(fx) * z
    y = (v - np.float32(cy)) / np.float32(fy) * z
    point_image = np.stack((x, y, z), axis=-1)

    valid = (z > minimum_depth_m) & (z < maximum_depth_m)
    scene_points = point_image[valid].astype(np.float32, copy=False)
    scene_colors = (rgb[valid].astype(np.float32) / 255.0).astype(
        np.float32,
        copy=False,
    )
    affordance_region = np.ascontiguousarray(affordance_mask[valid], dtype=bool)
    affordance_points = scene_points[affordance_region]
    if len(affordance_points) == 0:
        raise GraspPoseGenerationError(
            "the affordance mask contains no valid depth points"
        )

    centroid = affordance_points.mean(axis=0, dtype=np.float64).astype(np.float32)
    grasp_input = GraspInput(
        scene_points_xyz_m=scene_points,
        scene_colors_rgb=scene_colors,
        affordance_region_mask=affordance_region,
        affordance_centroid_xyz_m=centroid,
    )

    return PreparedSample(
        grasp_input=grasp_input,
        image_width=width,
        image_height=height,
        valid_depth_pixels=int(np.count_nonzero(valid)),
        affordance_pixels=int(np.count_nonzero(affordance_region)),
        camera_serial_number=str(camera.get("serial_number", "")),
        mask_path=mask_path,
        overlay_path=overlay_path,
    )


def _select_candidate(
    candidates: Sequence[GraspCandidate],
    centroid_xyz_m: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    if not candidates:
        raise GraspPoseGenerationError("grasp backend returned no candidates")
    translations = np.stack(
        [candidate.translation_xyz_m for candidate in candidates], axis=0
    )
    scores = np.asarray([candidate.score for candidate in candidates], dtype=np.float64)
    distances = np.linalg.norm(
        translations - centroid_xyz_m.astype(np.float64), axis=1
    )
    safe_distances = np.maximum(distances, 1e-6)
    objectives = scores / safe_distances
    return int(np.argmax(objectives)), distances, objectives


def _write_result(
    output_dir: Path,
    prepared: PreparedSample,
    candidates: Sequence[GraspCandidate],
    selected_index: int,
    distances: np.ndarray,
    objectives: np.ndarray,
    visualization_path: Path,
    affordance_point_cloud_path: Path,
    scene_point_cloud_path: Path,
) -> Path:
    selected = candidates[selected_index]

    def candidate_payload(index: int, candidate: GraspCandidate) -> Dict[str, Any]:
        distance = float(distances[index])
        objective = float(objectives[index])
        return {
            "index": index,
            "score": candidate.score,
            "distance_to_affordance_centroid_m": distance,
            "selection_objective": objective,
            "R": candidate.rotation_matrix_camera.tolist(),
            "t": candidate.translation_xyz_m.tolist(),
            "w": candidate.width_m,
            "gripper_tip_xyz_m": candidate.metadata.get(
                "gripper_tip_xyz_m"
            ),
            "metadata": dict(candidate.metadata),
        }

    payload = {
        "backend": selected.backend,
        "coordinate_frame": "RealSense color camera: +x right, +y down, +z forward",
        "rotation_convention": (
            "R columns are gripper approach, jaw-closing, remaining right-handed axis"
        ),
        "selection_formula": "argmax(score(g) / max(||t(g) - c||_2, 1e-6))",
        "candidate_count": len(candidates),
        "affordance_centroid_xyz_m": (
            prepared.grasp_input.affordance_centroid_xyz_m.tolist()
        ),
        "selected_candidate_index": selected_index,
        "selected_grasp": {
            "g": "[R, t, w]",
            **candidate_payload(selected_index, selected),
        },
        "candidates": [
            candidate_payload(index, candidate)
            for index, candidate in enumerate(candidates)
        ],
        "inputs": {
            "camera_serial_number": prepared.camera_serial_number,
            "partial_view_point_count": prepared.valid_depth_pixels,
            "scene_point_count": len(prepared.grasp_input.scene_points_xyz_m),
            "affordance_filtered_point_count": len(
                prepared.grasp_input.affordance_points_xyz_m
            ),
            "affordance_region_fraction": float(
                np.mean(prepared.grasp_input.affordance_region_mask)
            ),
            "affordance_mask": str(prepared.mask_path),
            "affordance_overlay": str(prepared.overlay_path),
        },
        "visualization": {
            "scene_point_cloud_ply": str(scene_point_cloud_path),
            "affordance_point_cloud_ply": str(affordance_point_cloud_path),
            "grasp_pose_png": str(visualization_path),
        },
    }
    result_path = output_dir / "grasp_pose_result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def _create_backend(arguments: argparse.Namespace) -> GraspBackend:
    return AnyGraspBackend(
        checkpoint_path=arguments.checkpoint,
        anygrasp_sdk=arguments.anygrasp_sdk,
        max_gripper_width_m=arguments.max_gripper_width,
        gripper_height_m=arguments.gripper_height,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D435 RGB-D와 affordance mask에서 grasp 입력을 생성하고 "
            "AnyGrasp를 실행합니다."
        )
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("examples/d435_sample"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/sample_grasp_pose"),
    )
    parser.add_argument(
        "--depth-source",
        choices=("raw", "filtered"),
        default="filtered",
    )
    parser.add_argument("--rgb", type=Path, help="실제 pipeline RGB PNG")
    parser.add_argument("--depth", type=Path, help="실제 pipeline uint16 depth PNG")
    parser.add_argument("--camera", type=Path, help="실제 pipeline camera JSON")
    parser.add_argument(
        "--mask",
        type=Path,
        help="외부 affordance mask PNG; 생략하면 번들 pliers 수동 mask 사용",
    )
    parser.add_argument("--checkpoint", type=Path, help="AnyGrasp checkpoint.tar")
    parser.add_argument(
        "--anygrasp-sdk",
        type=Path,
        help=(
            "AnyGrasp SDK 루트 또는 grasp_detection 경로; AnyGrasp에서만 사용"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="mask와 point cloud 입력만 검증하고 grasp backend는 실행하지 않음",
    )
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=2.00)
    parser.add_argument("--max-gripper-width", type=float, default=0.10)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument(
        "--max-vis-points",
        type=int,
        default=20_000,
        help="3D PNG에 표시할 최대 point 수",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="호환성 옵션; grasp 3D PNG는 추론 시 항상 저장됨",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not arguments.prepare_only:
        if arguments.checkpoint is None and arguments.anygrasp_sdk is None:
            parser.error("AnyGrasp requires --checkpoint or --anygrasp-sdk")
    explicit_sources = (arguments.rgb, arguments.depth, arguments.camera)
    if any(source is not None for source in explicit_sources):
        if not all(source is not None for source in explicit_sources):
            parser.error("explicit input requires --rgb, --depth and --camera together")
        if arguments.mask is None:
            parser.error("explicit RGB-D input requires --mask")

    try:
        output_dir = arguments.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_sample(
            sample_dir=arguments.sample_dir,
            output_dir=output_dir,
            depth_source=arguments.depth_source,
            minimum_depth_m=arguments.min_depth,
            maximum_depth_m=arguments.max_depth,
            mask_source=arguments.mask,
            rgb_source=arguments.rgb,
            depth_image_source=arguments.depth,
            camera_source=arguments.camera,
        )
        affordance_point_cloud_path = save_point_cloud_ply(
            prepared.grasp_input,
            output_dir / "affordance_point_cloud.ply",
        )
        scene_point_cloud_path = save_scene_point_cloud_ply(
            prepared.grasp_input,
            output_dir / "scene_point_cloud.ply",
        )
        if arguments.prepare_only:
            print(
                json.dumps(
                    {
                        "partial_view_point_count": prepared.valid_depth_pixels,
                        "scene_point_count": len(
                            prepared.grasp_input.scene_points_xyz_m
                        ),
                        "affordance_filtered_point_count": len(
                            prepared.grasp_input.affordance_points_xyz_m
                        ),
                        "affordance_centroid_xyz_m": (
                            prepared.grasp_input.affordance_centroid_xyz_m.tolist()
                        ),
                        "affordance_mask": str(prepared.mask_path),
                        "affordance_overlay": str(prepared.overlay_path),
                        "scene_point_cloud_ply": str(scene_point_cloud_path),
                        "affordance_point_cloud_ply": str(
                            affordance_point_cloud_path
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        backend = _create_backend(arguments)
        candidates = list(backend.generate(prepared.grasp_input))
        selected_index, distances, objectives = _select_candidate(
            candidates,
            prepared.grasp_input.affordance_centroid_xyz_m,
        )
        visualization_path = save_grasp_visualization(
            grasp_input=prepared.grasp_input,
            candidate=candidates[selected_index],
            output_path=output_dir / "grasp_pose_3d.png",
            max_points=arguments.max_vis_points,
        )
        result_path = _write_result(
            output_dir=output_dir,
            prepared=prepared,
            candidates=candidates,
            selected_index=selected_index,
            distances=distances,
            objectives=objectives,
            visualization_path=visualization_path,
            affordance_point_cloud_path=affordance_point_cloud_path,
            scene_point_cloud_path=scene_point_cloud_path,
        )
        print(
            json.dumps(
                {
                    "backend": backend.name,
                    "result": str(result_path),
                    "visualization": str(visualization_path),
                    "scene_point_cloud": str(scene_point_cloud_path),
                    "affordance_point_cloud": str(
                        affordance_point_cloud_path
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (
        GraspBackendError,
        GraspPoseGenerationError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Grasp Pose Generation 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
