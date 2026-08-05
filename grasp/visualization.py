"""Backend-neutral static 3D visualization for grasp candidates."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .interfaces import GraspCandidate, GraspInput


def _set_equal_3d_limits(axis: object, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.55, 0.03)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _draw_segment(
    axis: object,
    start: np.ndarray,
    end: np.ndarray,
    **kwargs: object,
) -> None:
    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        **kwargs,
    )


def save_grasp_visualization(
    grasp_input: GraspInput,
    candidate: GraspCandidate,
    output_path: Path,
    max_points: int = 20_000,
    finger_length_m: float = 0.04,
) -> Path:
    """Save a point cloud and normalized gripper pose as a 3D PNG."""

    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if finger_length_m <= 0.0:
        raise ValueError("finger_length_m must be positive")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(tempfile.gettempdir()) / "affordgrasp-matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = grasp_input.points_xyz_m
    colors = grasp_input.colors_rgb
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points_to_draw = points[indices]
        colors_to_draw = colors[indices]
    else:
        points_to_draw = points
        colors_to_draw = colors

    center = candidate.translation_xyz_m
    approach = candidate.rotation_matrix_camera[:, 0]
    closing = candidate.rotation_matrix_camera[:, 1]
    remaining = candidate.rotation_matrix_camera[:, 2]
    half_width = candidate.width_m * 0.5
    left_tip = center - closing * half_width
    right_tip = center + closing * half_width
    left_back = left_tip - approach * finger_length_m
    right_back = right_tip - approach * finger_length_m

    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points_to_draw[:, 0],
        points_to_draw[:, 1],
        points_to_draw[:, 2],
        c=colors_to_draw,
        s=2,
        alpha=0.65,
        depthshade=False,
    )

    gripper_style = {"color": "black", "linewidth": 4, "solid_capstyle": "round"}
    _draw_segment(axis, left_tip, left_back, **gripper_style)
    _draw_segment(axis, right_tip, right_back, **gripper_style)
    _draw_segment(axis, left_back, right_back, **gripper_style)
    axis.scatter(*center, color="magenta", s=45, label="grasp center")

    axis_length = max(0.025, min(candidate.width_m * 0.6, 0.05))
    for direction, color, label in (
        (approach, "red", "approach (+x grasp)"),
        (closing, "green", "jaw closing (+y grasp)"),
        (remaining, "blue", "gripper +z"),
    ):
        axis.quiver(
            center[0],
            center[1],
            center[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            normalize=True,
            color=color,
            linewidth=2,
            label=label,
        )

    context_points = np.vstack((points_to_draw, left_back, right_back))
    _set_equal_3d_limits(axis, context_points)
    axis.set_xlabel("camera x (m, right)")
    axis.set_ylabel("camera y (m, down)")
    axis.set_zlabel("camera z (m, forward)")
    axis.set_title(
        f"{candidate.backend}: score={candidate.score:.3f}, "
        f"width={candidate.width_m * 1000.0:.1f} mm"
    )
    axis.view_init(elev=24, azim=-64)
    axis.legend(loc="upper left", fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def save_point_cloud_ply(grasp_input: GraspInput, output_path: Path) -> Path:
    """Write the affordance-filtered RGB point cloud as an ASCII PLY file."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.rint(grasp_input.colors_rgb * 255.0).astype(np.uint8)
    with output_path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(grasp_input.points_xyz_m)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(grasp_input.points_xyz_m, rgb):
            file.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return output_path
