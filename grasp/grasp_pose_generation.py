"""Generate an affordance-steered AnyGrasp pose from the bundled D435 sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


class GraspPoseGenerationError(RuntimeError):
    """Raised when sample preparation or AnyGrasp inference cannot finish."""


@dataclass(frozen=True)
class PreparedSample:
    points: np.ndarray
    colors: np.ndarray
    affordance_centroid: np.ndarray
    image_width: int
    image_height: int
    valid_depth_pixels: int
    affordance_pixels: int
    mask_path: Path
    overlay_path: Path


# Two polygons cover the upper and lower handles of the pliers in sample_rgb.png.
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
        raise GraspPoseGenerationError("could not read the sample RGB-D images") from exc

    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.integer):
        raise GraspPoseGenerationError("sample depth must be a single-channel integer PNG")
    if depth.size == 0 or int(depth.max()) > np.iinfo(np.uint16).max:
        raise GraspPoseGenerationError("sample depth is not valid Z16 data")
    depth = depth.astype(np.uint16, copy=False)
    if rgb.shape[:2] != depth.shape:
        raise GraspPoseGenerationError(
            f"RGB and depth dimensions differ: {rgb.shape[:2]} vs {depth.shape}"
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


def _save_mask_outputs(
    rgb: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "manual_affordance_mask.png"
    overlay_path = output_dir / "manual_affordance_overlay.png"

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
) -> PreparedSample:
    """Build a camera-frame point cloud and pliers-handle steering mask."""

    if depth_source not in {"raw", "filtered"}:
        raise ValueError("depth_source must be raw or filtered")
    if not 0.0 <= minimum_depth_m < maximum_depth_m:
        raise ValueError("depth range must satisfy 0 <= minimum < maximum")

    sample_dir = sample_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rgb_path = sample_dir / "sample_rgb.png"
    depth_path = sample_dir / f"sample_depth_{depth_source}.png"
    camera_path = sample_dir / "sample_camera.json"
    for path in (rgb_path, depth_path, camera_path):
        if not path.is_file():
            raise GraspPoseGenerationError(f"required sample file is missing: {path}")

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

    affordance_mask = _create_sample_affordance_mask(width, height)
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
    affordance_valid = valid & affordance_mask
    affordance_points = point_image[affordance_valid]
    if len(affordance_points) == 0:
        raise GraspPoseGenerationError(
            "the manual affordance mask contains no valid depth points"
        )

    # AffordGrasp Sec. III-C: filter the partial-view point cloud with the
    # affordance mask, then give only that object-part cloud to AnyGrasp.
    points = affordance_points.astype(np.float32, copy=False)
    colors = (rgb[affordance_valid].astype(np.float32) / 255.0).astype(
        np.float32, copy=False
    )
    centroid = affordance_points.mean(axis=0, dtype=np.float64).astype(np.float32)

    return PreparedSample(
        points=points,
        colors=colors,
        affordance_centroid=centroid,
        image_width=width,
        image_height=height,
        valid_depth_pixels=int(np.count_nonzero(valid)),
        affordance_pixels=int(np.count_nonzero(affordance_valid)),
        mask_path=mask_path,
        overlay_path=overlay_path,
    )


def _resolve_anygrasp_detection_dir(anygrasp_sdk: Optional[Path]) -> Optional[Path]:
    if anygrasp_sdk is None:
        return None

    directory = anygrasp_sdk.expanduser().resolve()
    if (directory / "grasp_detection").is_dir():
        directory = directory / "grasp_detection"
    if not directory.is_dir():
        raise GraspPoseGenerationError(
            f"AnyGrasp grasp_detection directory does not exist: {directory}"
        )
    return directory


@contextmanager
def _anygrasp_sdk_context(detection_dir: Optional[Path]):
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


def _run_anygrasp(
    prepared: PreparedSample,
    checkpoint_path: Optional[Path],
    anygrasp_sdk: Optional[Path],
    max_gripper_width: float,
    gripper_height: float,
) -> Tuple[Any, int, np.ndarray, np.ndarray]:
    detection_dir = _resolve_anygrasp_detection_dir(anygrasp_sdk)
    if checkpoint_path is None:
        if detection_dir is None:
            raise GraspPoseGenerationError(
                "provide --checkpoint or --anygrasp-sdk"
            )
        checkpoint_path = detection_dir / "log" / "checkpoint_detection.tar"
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise GraspPoseGenerationError(
            f"AnyGrasp checkpoint does not exist: {checkpoint_path}"
        )
    if detection_dir is not None and not (detection_dir / "license").is_dir():
        raise GraspPoseGenerationError(
            f"AnyGrasp license directory does not exist: {detection_dir / 'license'}"
        )
    if not 0.0 < max_gripper_width <= 0.10:
        raise ValueError("max_gripper_width must be in (0, 0.10]")
    if gripper_height <= 0:
        raise ValueError("gripper_height must be positive")

    with _anygrasp_sdk_context(detection_dir):
        try:
            from gsnet import create_detector
        except ImportError as exc:
            raise GraspPoseGenerationError(
                "AnyGrasp SDK is missing; install gsnet, its CUDA dependencies, "
                "and complete AnyGrasp license registration"
            ) from exc

        config = SimpleNamespace(
            checkpoint_path=str(checkpoint_path),
            max_gripper_width=max_gripper_width,
            gripper_height=gripper_height,
        )
        detector = create_detector(config)
        if detector is None:
            raise GraspPoseGenerationError(
                "AnyGrasp detector initialization failed; verify checkpoint and license"
            )

        grasps = detector.get_grasp(prepared.points, {})
    if grasps is None or len(grasps) == 0:
        raise GraspPoseGenerationError(
            "AnyGrasp produced no grasp from the affordance-filtered point cloud"
        )

    scores = np.asarray(grasps.scores, dtype=np.float64)
    translations = np.asarray(grasps.translations, dtype=np.float64)
    distances = np.linalg.norm(
        translations - prepared.affordance_centroid.astype(np.float64),
        axis=1,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        paper_objective = np.where(distances == 0.0, np.inf, scores / distances)
    selected_index = int(np.argmax(paper_objective))
    return grasps, selected_index, distances, paper_objective


def _write_result(
    output_dir: Path,
    prepared: PreparedSample,
    grasps: Any,
    selected_index: int,
    distances: np.ndarray,
    paper_objective: np.ndarray,
) -> Path:
    rotations = np.asarray(grasps.rotation_matrices)
    translations = np.asarray(grasps.translations)
    widths = np.asarray(grasps.widths)
    scores = np.asarray(grasps.scores)

    rotation = rotations[selected_index]
    translation = translations[selected_index]
    payload = {
        "coordinate_frame": "RealSense color camera: +x right, +y down, +z forward",
        "selection_formula": "argmax(score(g) / ||t(g) - c||_2)",
        "candidate_count": int(len(grasps)),
        "affordance_centroid_xyz_m": prepared.affordance_centroid.tolist(),
        "selected_candidate_index": selected_index,
        "selected_grasp": {
            "g": "[R, t, w]",
            "score": float(scores[selected_index]),
            "distance_to_affordance_centroid_m": float(distances[selected_index]),
            "paper_objective": float(paper_objective[selected_index]),
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "w": float(widths[selected_index]),
        },
        "inputs": {
            "partial_view_point_count": prepared.valid_depth_pixels,
            "affordance_filtered_point_count": int(len(prepared.points)),
            "manual_affordance_mask": str(prepared.mask_path),
            "manual_affordance_overlay": str(prepared.overlay_path),
        },
    }
    result_path = output_dir / "grasp_pose_result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_path


def _visualize(prepared: PreparedSample, grasps: Any, selected_index: int) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise GraspPoseGenerationError(
            "--vis requires the open3d package"
        ) from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(prepared.points)
    cloud.colors = o3d.utility.Vector3dVector(prepared.colors)
    grippers = grasps[selected_index : selected_index + 1].to_open3d_geometry_list()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    o3d.visualization.draw_geometries([cloud, frame, *grippers])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bundled D435 sample에서 수동 pliers-handle mask로 "
            "AffordGrasp Sec. III-C의 grasp pose를 생성합니다."
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
    parser.add_argument("--checkpoint", type=Path, help="AnyGrasp checkpoint.tar")
    parser.add_argument(
        "--anygrasp-sdk",
        type=Path,
        help=(
            "anygrasp_sdk 루트 또는 grasp_detection 경로; 지정하면 SDK import, "
            "license/ 및 기본 log/checkpoint_detection.tar를 자동으로 사용"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="mask와 point cloud 입력만 검증하고 AnyGrasp는 실행하지 않음",
    )
    parser.add_argument("--max-gripper-width", type=float, default=0.10)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--vis", action="store_true", help="최종 grasp를 Open3D로 표시")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if (
        not arguments.prepare_only
        and arguments.checkpoint is None
        and arguments.anygrasp_sdk is None
    ):
        parser.error(
            "--checkpoint or --anygrasp-sdk is required unless --prepare-only is used"
        )
    if arguments.prepare_only and arguments.vis:
        parser.error("--vis requires AnyGrasp inference")

    try:
        output_dir = arguments.output_dir.expanduser().resolve()
        prepared = prepare_sample(
            sample_dir=arguments.sample_dir,
            output_dir=output_dir,
            depth_source=arguments.depth_source,
        )
        if arguments.prepare_only:
            print(
                json.dumps(
                    {
                        "point_count": int(len(prepared.points)),
                        "partial_view_point_count": prepared.valid_depth_pixels,
                        "affordance_filtered_point_count": int(len(prepared.points)),
                        "affordance_centroid_xyz_m": (
                            prepared.affordance_centroid.tolist()
                        ),
                        "manual_affordance_mask": str(prepared.mask_path),
                        "manual_affordance_overlay": str(prepared.overlay_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        grasps, selected_index, distances, objective = _run_anygrasp(
            prepared=prepared,
            checkpoint_path=arguments.checkpoint,
            anygrasp_sdk=arguments.anygrasp_sdk,
            max_gripper_width=arguments.max_gripper_width,
            gripper_height=arguments.gripper_height,
        )
        result_path = _write_result(
            output_dir,
            prepared,
            grasps,
            selected_index,
            distances,
            objective,
        )
        print(result_path)
        if arguments.vis:
            _visualize(prepared, grasps, selected_index)
        return 0
    except (GraspPoseGenerationError, OSError, ValueError) as exc:
        print(f"Grasp Pose Generation 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
