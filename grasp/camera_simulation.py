"""Simulate an AnyGrasp end-effector path entirely in the D435 camera frame."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class CameraSimulationError(RuntimeError):
    """Raised when a camera-frame trajectory cannot be generated."""


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraSimulationError(f"could not read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CameraSimulationError(f"JSON root must be an object: {path}")
    return payload


def _positive(value: float, name: str, allow_zero: bool = False) -> float:
    number = float(value)
    minimum_ok = number >= 0.0 if allow_zero else number > 0.0
    if not np.isfinite(number) or not minimum_ok:
        operator = ">=" if allow_zero else ">"
        raise CameraSimulationError(f"{name} must be finite and {operator} 0")
    return number


def _rotation(value: Any) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise CameraSimulationError("grasp rotation R must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2):
        raise CameraSimulationError("grasp rotation R is not orthonormal")
    if float(np.linalg.det(rotation)) < 0.0:
        raise CameraSimulationError("grasp rotation R is not right-handed")
    return rotation


def _vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise CameraSimulationError(f"{name} must contain three finite values")
    return vector


def _candidate(
    grasp_result: Mapping[str, Any], candidate_index: Optional[int]
) -> Mapping[str, Any]:
    if candidate_index is None:
        selected = grasp_result.get("selected_grasp")
        if not isinstance(selected, dict):
            raise CameraSimulationError("grasp result has no selected_grasp object")
        return selected
    candidates = grasp_result.get("candidates")
    if not isinstance(candidates, list):
        raise CameraSimulationError("grasp result has no candidates array")
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("index") == candidate_index:
            return candidate
    raise CameraSimulationError(
        f"candidate index {candidate_index} does not exist in grasp result"
    )


def build_camera_trajectory(
    grasp_result: Mapping[str, Any],
    candidate_index: Optional[int],
    pregrasp_offset_m: float,
    retreat_offset_m: float,
    lift_offset_m: float,
    lift_axis_camera: Sequence[float],
    maximum_gripper_width_m: float,
) -> Dict[str, Any]:
    """Create TCP waypoints without a robot/base transform."""

    candidate = _candidate(grasp_result, candidate_index)
    rotation = _rotation(candidate.get("R"))
    grasp_origin = _vector(candidate.get("t"), "grasp translation t")
    width = _positive(candidate.get("w"), "grasp width")
    maximum_width = _positive(maximum_gripper_width_m, "maximum gripper width")
    if width > maximum_width + 1e-9:
        raise CameraSimulationError(
            f"candidate width {width:.4f} m exceeds configured maximum "
            f"{maximum_width:.4f} m"
        )

    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    tip_value = candidate.get("gripper_tip_xyz_m")
    if tip_value is None:
        tip_value = metadata.get("gripper_tip_xyz_m")
    if tip_value is not None:
        grasp_tcp = _vector(tip_value, "gripper tip")
    else:
        insertion_depth = float(metadata.get("insertion_depth_m", 0.0))
        if not np.isfinite(insertion_depth):
            raise CameraSimulationError("insertion depth must be finite")
        grasp_tcp = grasp_origin + insertion_depth * rotation[:, 0]

    approach = rotation[:, 0]
    lift_axis = _vector(lift_axis_camera, "camera lift axis")
    lift_axis_norm = float(np.linalg.norm(lift_axis))
    if lift_axis_norm <= 1e-9:
        raise CameraSimulationError("camera lift axis must not be zero")
    lift_axis /= lift_axis_norm

    pregrasp_offset = _positive(
        pregrasp_offset_m, "pregrasp offset", allow_zero=True
    )
    retreat_offset = _positive(
        retreat_offset_m, "retreat offset", allow_zero=True
    )
    lift_offset = _positive(lift_offset_m, "lift offset", allow_zero=True)
    pregrasp = grasp_tcp - approach * pregrasp_offset
    retreat = grasp_tcp - approach * retreat_offset
    lift = retreat + lift_axis * lift_offset
    open_width = min(maximum_width, max(width + 0.015, width * 1.15))

    def waypoint(name: str, position: np.ndarray, jaw_width: float) -> Dict[str, Any]:
        return {
            "name": name,
            "tcp_position_xyz_m": position.tolist(),
            "R_camera_gripper": rotation.tolist(),
            "jaw_width_m": float(jaw_width),
        }

    return {
        "schema_version": 1,
        "simulation_type": "camera_frame_gripper_trajectory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": (
            "RealSense color camera: +x right, +y down, +z forward"
        ),
        "uses_eye_to_hand_calibration": False,
        "connects_to_robot": False,
        "candidate": {
            "index": int(candidate.get("index", -1)),
            "backend": str(grasp_result.get("backend", "unknown")),
            "score": float(candidate.get("score", 0.0)),
            "grasp_origin_xyz_m": grasp_origin.tolist(),
            "gripper_tip_xyz_m": grasp_tcp.tolist(),
            "approach_axis_camera": approach.tolist(),
            "jaw_closing_axis_camera": rotation[:, 1].tolist(),
            "required_width_m": width,
        },
        "parameters": {
            "pregrasp_offset_m": pregrasp_offset,
            "retreat_offset_m": retreat_offset,
            "lift_offset_m": lift_offset,
            "lift_axis_camera": lift_axis.tolist(),
            "open_width_m": open_width,
        },
        "waypoint_order": ["pregrasp", "grasp", "retreat", "lift"],
        "waypoints": {
            "pregrasp": waypoint("pregrasp", pregrasp, open_width),
            "grasp": waypoint("grasp", grasp_tcp, width),
            "retreat": waypoint("retreat", retreat, width),
            "lift": waypoint("lift", lift, width),
        },
        "limitations": [
            "This is a gripper/TCP path in the camera frame, not an xArm joint trajectory.",
            "No xArm inverse kinematics or whole-arm collision checking is performed.",
            "Camera -Y is used as visual lift by default and may not equal gravity-up if the D435 is tilted.",
            "The path must never be sent to a real robot without a measured base-to-camera transform.",
        ],
    }


def _resolve_cloud_path(
    result_path: Path,
    grasp_result: Mapping[str, Any],
    explicit_path: Optional[Path],
    key: str,
) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
    else:
        visualization = grasp_result.get("visualization")
        if not isinstance(visualization, dict) or not visualization.get(key):
            raise CameraSimulationError(
                f"grasp result has no visualization.{key}; pass its path explicitly"
            )
        path = Path(str(visualization[key])).expanduser()
        if not path.is_absolute():
            path = result_path.parent / path
        path = path.resolve()
    if not path.is_file():
        raise CameraSimulationError(f"point cloud does not exist: {path}")
    return path


def _load_ascii_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="ascii") as file:
        first = file.readline().strip()
        if first != "ply":
            raise CameraSimulationError(f"not a PLY file: {path}")
        vertex_count: Optional[int] = None
        format_ascii = False
        properties = []
        in_vertex = False
        while True:
            line = file.readline()
            if not line:
                raise CameraSimulationError(f"truncated PLY header: {path}")
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                format_ascii = True
            elif fields[:2] == ["element", "vertex"] and len(fields) == 3:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element" and fields[1:2] != ["vertex"]:
                in_vertex = False
            elif fields[:1] == ["property"] and in_vertex:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break
        if not format_ascii:
            raise CameraSimulationError("only ASCII PLY point clouds are supported")
        if vertex_count is None or vertex_count <= 0:
            raise CameraSimulationError(f"PLY contains no vertices: {path}")
        required = ("x", "y", "z", "red", "green", "blue")
        if any(name not in properties for name in required):
            raise CameraSimulationError(f"PLY lacks XYZ/RGB properties: {path}")
        try:
            values = np.loadtxt(file, dtype=np.float64, max_rows=vertex_count)
        except ValueError as exc:
            raise CameraSimulationError(f"could not parse PLY vertices: {path}") from exc
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != (vertex_count, len(properties)):
        raise CameraSimulationError(f"PLY vertex count/properties are inconsistent: {path}")
    indices = [properties.index(name) for name in required]
    points = values[:, indices[:3]]
    colors = np.clip(values[:, indices[3:]] / 255.0, 0.0, 1.0)
    if not np.all(np.isfinite(points)):
        raise CameraSimulationError(f"PLY contains non-finite points: {path}")
    return points, colors


def _subsample(
    points: np.ndarray, colors: np.ndarray, maximum: int
) -> Tuple[np.ndarray, np.ndarray]:
    if maximum <= 0:
        raise CameraSimulationError("max visualization points must be positive")
    if len(points) <= maximum:
        return points, colors
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices], colors[indices]


def _gripper_segments(
    tcp: np.ndarray,
    rotation: np.ndarray,
    width: float,
    insertion_depth: float,
    finger_length_m: float,
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    approach = rotation[:, 0]
    closing = rotation[:, 1]
    origin = tcp - insertion_depth * approach
    half_width = width * 0.5
    left_front = origin - closing * half_width
    right_front = origin + closing * half_width
    left_back = left_front - approach * finger_length_m
    right_back = right_front - approach * finger_length_m
    wrist = (left_back + right_back) * 0.5 - approach * finger_length_m * 0.35
    return (
        (left_front, left_back),
        (right_front, right_back),
        (left_back, right_back),
        ((left_back + right_back) * 0.5, wrist),
    )


def _equal_limits(axis: Any, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.58, 0.06)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _setup_axis(
    axis: Any,
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    affordance_points: np.ndarray,
    context: np.ndarray,
) -> None:
    axis.scatter(
        scene_points[:, 0],
        scene_points[:, 1],
        scene_points[:, 2],
        c=scene_colors,
        s=1.2,
        alpha=0.24,
        depthshade=False,
        label="visible scene",
    )
    axis.scatter(
        affordance_points[:, 0],
        affordance_points[:, 1],
        affordance_points[:, 2],
        color="cyan",
        s=3.0,
        alpha=0.78,
        depthshade=False,
        label="affordance region",
    )
    _equal_limits(axis, context)
    axis.set_xlabel("camera x (m, right)")
    axis.set_ylabel("camera y (m, down)")
    axis.set_zlabel("camera z (m, forward)")
    axis.view_init(elev=24, azim=-64)


def _waypoint_arrays(plan: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    waypoints = plan["waypoints"]
    names = plan["waypoint_order"]
    positions = np.asarray(
        [waypoints[name]["tcp_position_xyz_m"] for name in names],
        dtype=np.float64,
    )
    rotation = np.asarray(waypoints["grasp"]["R_camera_gripper"], dtype=np.float64)
    widths = np.asarray([waypoints[name]["jaw_width_m"] for name in names])
    return positions, rotation, widths


def save_static_visualization(
    plan: Mapping[str, Any],
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    affordance_points: np.ndarray,
    output_path: Path,
    finger_length_m: float,
) -> Path:
    from matplotlib import pyplot as plt

    positions, rotation, widths = _waypoint_arrays(plan)
    grasp_origin = np.asarray(plan["candidate"]["grasp_origin_xyz_m"])
    grasp_tip = np.asarray(plan["candidate"]["gripper_tip_xyz_m"])
    insertion_depth = float(np.dot(grasp_tip - grasp_origin, rotation[:, 0]))
    nearby = np.linalg.norm(scene_points - grasp_tip, axis=1) <= 0.28
    local_scene = scene_points[nearby]
    if len(local_scene) == 0:
        local_scene = affordance_points
    context = np.vstack((local_scene, positions))

    figure = plt.figure(figsize=(11, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    _setup_axis(axis, scene_points, scene_colors, affordance_points, context)
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="crimson",
        linewidth=2.5,
        linestyle="--",
        label="TCP path",
    )
    styles = (
        ("pregrasp", "deepskyblue", 0.60),
        ("grasp", "magenta", 1.00),
        ("retreat", "orange", 0.70),
        ("lift", "limegreen", 0.90),
    )
    for index, (name, color, alpha) in enumerate(styles):
        for start, end in _gripper_segments(
            positions[index],
            rotation,
            float(widths[index]),
            insertion_depth,
            finger_length_m,
        ):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                alpha=alpha,
                linewidth=3.0,
            )
        axis.scatter(*positions[index], color=color, s=35)
        axis.text(*positions[index], f"  {index + 1}. {name}", color=color)

    axis.set_title("Camera-frame grasp trajectory (no robot calibration)")
    axis.legend(loc="upper left", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _animation_states(
    positions: np.ndarray, widths: np.ndarray, frames_per_segment: int
) -> Tuple[np.ndarray, np.ndarray]:
    if frames_per_segment < 2:
        raise CameraSimulationError("animation frames per segment must be >= 2")
    state_positions = []
    state_widths = []
    open_grasp_width = widths[0]
    closed_grasp_width = widths[1]
    key_positions = (
        positions[0],
        positions[1],
        positions[1],
        positions[2],
        positions[3],
    )
    key_widths = (
        open_grasp_width,
        open_grasp_width,
        closed_grasp_width,
        widths[2],
        widths[3],
    )
    for segment in range(len(key_positions) - 1):
        fractions = np.linspace(
            0.0,
            1.0,
            frames_per_segment,
            endpoint=segment == len(key_positions) - 2,
        )
        for fraction in fractions:
            state_positions.append(
                key_positions[segment] * (1.0 - fraction)
                + key_positions[segment + 1] * fraction
            )
            state_widths.append(
                key_widths[segment] * (1.0 - fraction)
                + key_widths[segment + 1] * fraction
            )
    return np.asarray(state_positions), np.asarray(state_widths)


def save_animation(
    plan: Mapping[str, Any],
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    affordance_points: np.ndarray,
    output_path: Path,
    finger_length_m: float,
    frames_per_segment: int,
) -> Path:
    from matplotlib import animation, pyplot as plt

    positions, rotation, widths = _waypoint_arrays(plan)
    moving_positions, moving_widths = _animation_states(
        positions, widths, frames_per_segment
    )
    grasp_origin = np.asarray(plan["candidate"]["grasp_origin_xyz_m"])
    grasp_tip = np.asarray(plan["candidate"]["gripper_tip_xyz_m"])
    insertion_depth = float(np.dot(grasp_tip - grasp_origin, rotation[:, 0]))
    nearby = np.linalg.norm(scene_points - grasp_tip, axis=1) <= 0.28
    local_scene = scene_points[nearby]
    if len(local_scene) == 0:
        local_scene = affordance_points
    context = np.vstack((local_scene, positions))

    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    _setup_axis(axis, scene_points, scene_colors, affordance_points, context)
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="crimson",
        linewidth=2.0,
        linestyle="--",
    )
    for index, name in enumerate(plan["waypoint_order"]):
        axis.scatter(*positions[index], color="crimson", s=22)
        axis.text(*positions[index], f"  {name}", fontsize=8)
    lines = [axis.plot([], [], [], color="black", linewidth=4.0)[0] for _ in range(4)]
    tcp_marker = axis.plot([], [], [], marker="o", color="magenta", markersize=6)[0]
    axis.set_title("Camera-frame trajectory simulation")

    def update(frame_index: int) -> Sequence[Any]:
        tcp = moving_positions[frame_index]
        segments = _gripper_segments(
            tcp,
            rotation,
            float(moving_widths[frame_index]),
            insertion_depth,
            finger_length_m,
        )
        for line, (start, end) in zip(lines, segments):
            line.set_data_3d(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
            )
        tcp_marker.set_data_3d([tcp[0]], [tcp[1]], [tcp[2]])
        return (*lines, tcp_marker)

    simulation = animation.FuncAnimation(
        figure,
        update,
        frames=len(moving_positions),
        interval=160,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation.save(output_path, writer=animation.PillowWriter(fps=6), dpi=110)
    plt.close(figure)
    return output_path


def run_simulation(arguments: argparse.Namespace) -> Dict[str, Any]:
    result_path = arguments.grasp_result.expanduser().resolve()
    if not result_path.is_file():
        raise CameraSimulationError(f"grasp result does not exist: {result_path}")
    grasp_result = _load_json(result_path)
    if grasp_result.get("coordinate_frame") != (
        "RealSense color camera: +x right, +y down, +z forward"
    ):
        raise CameraSimulationError(
            "grasp result is not expressed in the expected D435 color-camera frame"
        )
    plan = build_camera_trajectory(
        grasp_result=grasp_result,
        candidate_index=arguments.candidate_index,
        pregrasp_offset_m=arguments.pregrasp_offset,
        retreat_offset_m=arguments.retreat_offset,
        lift_offset_m=arguments.lift_offset,
        lift_axis_camera=arguments.lift_axis_camera,
        maximum_gripper_width_m=arguments.max_gripper_width,
    )
    scene_path = _resolve_cloud_path(
        result_path,
        grasp_result,
        arguments.scene_point_cloud,
        "scene_point_cloud_ply",
    )
    affordance_path = _resolve_cloud_path(
        result_path,
        grasp_result,
        arguments.affordance_point_cloud,
        "affordance_point_cloud_ply",
    )
    scene_points, scene_colors = _load_ascii_ply(scene_path)
    affordance_points, affordance_colors = _load_ascii_ply(affordance_path)
    scene_points, scene_colors = _subsample(
        scene_points, scene_colors, arguments.max_vis_points
    )
    affordance_points, _ = _subsample(
        affordance_points, affordance_colors, arguments.max_vis_points
    )

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(tempfile.gettempdir()) / "affordgrasp-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    static_path = save_static_visualization(
        plan,
        scene_points,
        scene_colors,
        affordance_points,
        output_dir / "camera_trajectory_3d.png",
        arguments.finger_length,
    )
    animation_path = save_animation(
        plan,
        scene_points,
        scene_colors,
        affordance_points,
        output_dir / "camera_trajectory.gif",
        arguments.finger_length,
        arguments.frames_per_segment,
    )
    plan.update(
        {
            "source_grasp_result": str(result_path),
            "source_scene_point_cloud": str(scene_path),
            "source_affordance_point_cloud": str(affordance_path),
            "outputs": {
                "static_visualization": str(static_path),
                "animation": str(animation_path),
            },
        }
    )
    plan_path = output_dir / "camera_trajectory.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "trajectory": str(plan_path),
        "visualization": str(static_path),
        "animation": str(animation_path),
        "connects_to_robot": False,
        "uses_eye_to_hand_calibration": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-point-cloud", type=Path)
    parser.add_argument("--affordance-point-cloud", type=Path)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--pregrasp-offset", type=float, default=0.12)
    parser.add_argument("--retreat-offset", type=float, default=0.12)
    parser.add_argument("--lift-offset", type=float, default=0.08)
    parser.add_argument(
        "--lift-axis-camera",
        type=float,
        nargs=3,
        default=[0.0, -1.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="camera-frame visual lift axis; default -Y (up in the RGB image)",
    )
    parser.add_argument("--max-gripper-width", type=float, default=0.085)
    parser.add_argument("--finger-length", type=float, default=0.04)
    parser.add_argument("--max-vis-points", type=int, default=15_000)
    parser.add_argument("--frames-per-segment", type=int, default=6)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        result = run_simulation(build_parser().parse_args(argv))
    except (CameraSimulationError, OSError, ValueError) as exc:
        print(f"Camera Simulation 오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
