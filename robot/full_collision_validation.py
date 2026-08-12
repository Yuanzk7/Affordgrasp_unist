"""Read-only xArm7 IK and full-link collision validation for a robot plan.

The validator queries the controller for IK solutions but never enables motion or
sends a pose command.  Official UFACTORY xArm7 and xArm Gripper collision meshes
are loaded in PyBullet together with a configured table box.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .transforms import (
    TransformError,
    load_json_object,
    pose_aa_to_transform,
    rotation_distance_rad,
    validate_transform,
)
from .xarm_grasp_execution import (
    RobotExecutionError,
    _check_robot_status,
    _connect,
    _load_config,
    _number,
    _plan_ready,
    _ranked_plan_candidates,
    _rotation_to_rpy,
    _sha256_file,
)


class CollisionValidationError(RuntimeError):
    """Raised when collision validation cannot be completed."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any, label: str, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollisionValidationError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        raise CollisionValidationError(
            f"{label} must be finite"
            + (f" and >= {minimum}" if minimum is not None else "")
        )
    return result


def _collision_config(config_path: Path) -> Dict[str, Any]:
    raw = dict(load_json_object(config_path))
    value = raw.get("collision_environment")
    if not isinstance(value, dict):
        raise CollisionValidationError(
            "robot config collision_environment must be an object"
        )
    center = np.asarray(value.get("table_center_xy_m"), dtype=np.float64)
    size = np.asarray(value.get("table_size_xy_m"), dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise CollisionValidationError("table_center_xy_m must contain two values")
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise CollisionValidationError("table_size_xy_m must contain positive values")
    ignored = value.get("ignored_table_links", ["link_base"])
    if not isinstance(ignored, list) or not all(isinstance(x, str) for x in ignored):
        raise CollisionValidationError("ignored_table_links must be a string list")
    return {
        "table_top_z_m": _finite(value.get("table_top_z_m"), "table_top_z_m"),
        "table_center_xy_m": center,
        "table_size_xy_m": size,
        "table_thickness_m": _finite(
            value.get("table_thickness_m"), "table_thickness_m", 0.001
        ),
        "required_table_clearance_m": _finite(
            value.get("required_table_clearance_m"),
            "required_table_clearance_m",
            0.0,
        ),
        "self_collision_clearance_m": _finite(
            value.get("self_collision_clearance_m"),
            "self_collision_clearance_m",
            0.0,
        ),
        "nominal_model_uncertainty_m": _finite(
            value.get("nominal_model_uncertainty_m"),
            "nominal_model_uncertainty_m",
            0.0,
        ),
        "maximum_model_error_m": _finite(
            value.get("maximum_model_error_m"),
            "maximum_model_error_m",
            0.0,
        ),
        "linear_sample_step_m": _finite(
            value.get("linear_sample_step_m"), "linear_sample_step_m", 0.001
        ),
        "angular_sample_step_deg": _finite(
            value.get("angular_sample_step_deg"),
            "angular_sample_step_deg",
            0.1,
        ),
        "geometry_verified": value.get("geometry_verified") is True,
        "ignored_table_links": set(ignored),
    }


def _rpy_to_rotation(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _interpolate(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    relative = first[:3, :3].T @ second[:3, :3]
    rotation_vector = cv2.Rodrigues(relative)[0].reshape(3)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = first[:3, :3] @ cv2.Rodrigues(alpha * rotation_vector)[0]
    result[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    return result


def _path_targets(
    start: np.ndarray,
    waypoints: Mapping[str, Any],
    linear_step: float,
    angular_step_deg: float,
    start_name: str = "current",
) -> Sequence[Dict[str, Any]]:
    named = [(start_name, start)]
    for name in ("pregrasp", "grasp", "retreat", "lift"):
        named.append((name, validate_transform(np.asarray(waypoints[name]), name)))
    samples = [
        {"segment": start_name, "alpha": 0.0, "target": named[0][1]}
    ]
    angular_step = math.radians(angular_step_deg)
    for (start_name, start), (end_name, end) in zip(named, named[1:]):
        distance = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
        angle = rotation_distance_rad(start[:3, :3], end[:3, :3])
        count = max(
            1,
            int(math.ceil(distance / linear_step)),
            int(math.ceil(angle / angular_step)),
        )
        for index in range(1, count + 1):
            alpha = index / count
            samples.append(
                {
                    "segment": f"{start_name}->{end_name}",
                    "alpha": alpha,
                    "target": _interpolate(start, end, alpha),
                }
            )
    return samples


def _controller_fk(
    arm: Any,
    angles: np.ndarray,
    label: str,
) -> np.ndarray:
    fk_code, fk_pose = arm.get_forward_kinematics(
        angles.tolist(), input_is_radian=True, return_is_radian=True
    )
    if fk_code != 0 or len(fk_pose) < 6:
        raise CollisionValidationError(
            f"controller FK failed at {label}, code={fk_code}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_to_rotation(fk_pose[3:6])
    transform[:3, 3] = np.asarray(fk_pose[:3], dtype=np.float64) / 1000.0
    return transform


def _ready_tcp_targets(
    current: np.ndarray,
    ready: np.ndarray,
    linear_step: float,
    angular_step_deg: float,
) -> Sequence[Dict[str, Any]]:
    current = validate_transform(current, "current TCP")
    ready = validate_transform(ready, "ready TCP")
    distance = float(np.linalg.norm(ready[:3, 3] - current[:3, 3]))
    angle = rotation_distance_rad(current[:3, :3], ready[:3, :3])
    count = max(
        1,
        int(math.ceil(distance / linear_step)),
        int(math.ceil(angle / math.radians(angular_step_deg))),
    )
    samples = [{"segment": "current", "alpha": 0.0, "target": current}]
    for index in range(1, count + 1):
        alpha = index / count
        samples.append(
            {
                "segment": "current->ready",
                "alpha": alpha,
                "target": _interpolate(current, ready, alpha),
            }
        )
    return samples


def _query_ik_path(
    arm: Any,
    targets: Sequence[Dict[str, Any]],
    maximum_joint_step_rad: float,
    initial_angles: Optional[Sequence[float]] = None,
) -> Sequence[Dict[str, Any]]:
    if initial_angles is None:
        code, current_angles = arm.get_servo_angle(is_radian=True)
        if code != 0 or len(current_angles) < 7:
            raise CollisionValidationError(
                f"could not read current xArm joint angles, code={code}"
            )
        previous = np.asarray(current_angles[:7], dtype=np.float64)
    else:
        previous = np.asarray(initial_angles, dtype=np.float64)
        if previous.shape != (7,) or not np.all(np.isfinite(previous)):
            raise CollisionValidationError("initial IK joint state is invalid")
    results = []
    for index, sample in enumerate(targets):
        target = sample["target"]
        if index == 0:
            angles = previous.copy()
        elif "planned_joints" in sample:
            angles = np.asarray(sample["planned_joints"], dtype=np.float64)
        else:
            rpy = _rotation_to_rpy(target[:3, :3])
            pose = [*(target[:3, 3] * 1000.0).tolist(), *rpy.tolist()]
            code, values = arm.get_inverse_kinematics(
                pose,
                input_is_radian=True,
                return_is_radian=True,
                limited=True,
                ref_angles=previous.tolist(),
            )
            if code != 0 or len(values) < 7:
                raise CollisionValidationError(
                    f"xArm IK failed at {sample['segment']} alpha={sample['alpha']:.3f}, "
                    f"code={code}"
                )
            angles = np.asarray(values[:7], dtype=np.float64)
        if index > 0:
            jump = np.arctan2(np.sin(angles - previous), np.cos(angles - previous))
            if float(np.max(np.abs(jump))) > maximum_joint_step_rad:
                joint = int(np.argmax(np.abs(jump)))
                raise CollisionValidationError(
                    f"joint discontinuity at {sample['segment']} "
                    f"alpha={sample['alpha']:.3f}, joint={joint + 1}, "
                    f"step={np.rad2deg(abs(jump[joint])):.1f} deg"
                )
            limit_code, limited = arm.is_joint_limit(
                angles.tolist(), is_radian=True
            )
            if limit_code != 0 or limited:
                raise CollisionValidationError(
                    f"joint limit at {sample['segment']} alpha={sample['alpha']:.3f}"
                )

        controller_fk = _controller_fk(arm, angles, sample["segment"])
        results.append(
            {
                **sample,
                "joints": angles,
                "controller_fk_translation_error_m": float(
                    np.linalg.norm(controller_fk[:3, 3] - target[:3, 3])
                ),
                "controller_fk_rotation_error_deg": math.degrees(
                    rotation_distance_rad(controller_fk[:3, :3], target[:3, :3])
                ),
            }
        )
        previous = angles
    return results


def _description_paths(root: Path) -> Tuple[Path, Path, Path]:
    root = root.expanduser().resolve()
    description = root / "xarm_description"
    if not description.is_dir() and root.name == "xarm_description":
        description = root
        root = root.parent
    controller = root / "xarm_controller"
    srdf = root / "xarm_moveit_config" / "srdf" / "_xarm7_macro.srdf.xacro"
    required = [
        description / "urdf" / "xarm_device.urdf.xacro",
        description / "meshes" / "xarm7",
        description / "meshes" / "gripper" / "xarm",
        controller,
        srdf,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise CollisionValidationError(
            "official xarm_ros2 collision model is incomplete: " + ", ".join(missing)
        )
    return description, controller, srdf


def _generate_urdf(
    xarm_root: Path,
    output: Path,
    model1300: bool,
    gripper_version: str,
) -> Tuple[Path, Path]:
    try:
        import xacro
    except ImportError as exc:
        raise CollisionValidationError(
            "xacro is missing; install it in the AnyGrasp environment"
        ) from exc
    description, controller, srdf = _description_paths(xarm_root)
    with tempfile.TemporaryDirectory(prefix="affordgrasp_xacro_") as temporary:
        temporary_description = Path(temporary) / "xarm_description"
        shutil.copytree(description / "urdf", temporary_description / "urdf")
        shutil.copytree(description / "config", temporary_description / "config")
        for path in (temporary_description / "urdf").rglob("*.xacro"):
            text = path.read_text(encoding="utf-8")
            text = text.replace("$(find xarm_description)", str(temporary_description))
            text = text.replace("$(find xarm_controller)", str(controller))
            path.write_text(text, encoding="utf-8")
        document = xacro.process_file(
            str(temporary_description / "urdf" / "xarm_device.urdf.xacro"),
            mappings={
                "dof": "7",
                "robot_type": "xarm",
                "add_gripper": "true",
                "model1300": "true" if model1300 else "false",
                "limited": "true",
                "mesh_suffix": "stl",
                "gripper_version": gripper_version,
            },
        )
        xml = document.toprettyxml(indent="  ")
    xml = xml.replace("package://xarm_description", str(description))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml, encoding="utf-8")
    return output, srdf


def _disabled_collision_pairs(srdf_path: Path) -> set[frozenset[str]]:
    text = srdf_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'<disable_collisions\s+link1="\$\{prefix\}([^\"]+)"\s+'
        r'link2="\$\{prefix\}([^\"]+)"'
    )
    return {frozenset(pair) for pair in pattern.findall(text)}


def _minimum_table_distance(
    bullet: Any,
    robot: int,
    table: int,
    link_names: Mapping[int, str],
    ignored: set[str],
    only: Optional[set[str]] = None,
) -> Tuple[float, str]:
    closest = (float("inf"), "")
    for point in bullet.getClosestPoints(robot, table, distance=1.0):
        link = link_names.get(int(point[3]), f"link_index_{point[3]}")
        if link in ignored or (only is not None and link not in only):
            continue
        distance = float(point[8])
        if distance < closest[0]:
            closest = (distance, link)
    return closest


def _self_clearances(
    bullet: Any,
    robot: int,
    link_names: Mapping[int, str],
    disabled: set[frozenset[str]],
    query_distance: float,
) -> Sequence[Tuple[float, str, str]]:
    found = []
    for point in bullet.getClosestPoints(robot, robot, distance=query_distance):
        first_index, second_index = int(point[3]), int(point[4])
        if first_index >= second_index:
            continue
        first = link_names.get(first_index, f"link_index_{first_index}")
        second = link_names.get(second_index, f"link_index_{second_index}")
        if frozenset((first, second)) in disabled:
            continue
        found.append((float(point[8]), first, second))
    return found


def _run_bullet(
    urdf_path: Path,
    srdf_path: Path,
    samples: Sequence[Dict[str, Any]],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    try:
        import pybullet as bullet
    except ImportError as exc:
        raise CollisionValidationError(
            "pybullet is missing; install it in the AnyGrasp environment"
        ) from exc
    client = bullet.connect(bullet.DIRECT)
    if client < 0:
        raise CollisionValidationError("could not start PyBullet in DIRECT mode")
    try:
        robot = bullet.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=bullet.URDF_USE_SELF_COLLISION,
        )
        half_xy = settings["table_size_xy_m"] / 2.0
        thickness = settings["table_thickness_m"]
        table_shape = bullet.createCollisionShape(
            bullet.GEOM_BOX,
            halfExtents=[float(half_xy[0]), float(half_xy[1]), thickness / 2.0],
        )
        table = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_shape,
            basePosition=[
                *settings["table_center_xy_m"].tolist(),
                settings["table_top_z_m"] - thickness / 2.0,
            ],
        )
        joint_indices: Dict[str, int] = {}
        link_names: Dict[int, str] = {-1: "world"}
        for index in range(bullet.getNumJoints(robot)):
            info = bullet.getJointInfo(robot, index)
            joint_indices[info[1].decode("utf-8")] = index
            link_names[index] = info[12].decode("utf-8")
        arm_indices = [joint_indices[f"joint{i}"] for i in range(1, 8)]
        gripper_joint_names = [
            "drive_joint",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        ]
        gripper_indices = [joint_indices[name] for name in gripper_joint_names]
        gripper_links = {
            "xarm_gripper_base_link",
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        }
        tcp_index = next(index for index, name in link_names.items() if name == "link_tcp")
        disabled = _disabled_collision_pairs(srdf_path)
        model_errors = []
        records = []
        minimum_table = (float("inf"), "", "", "", "")
        minimum_gripper = (float("inf"), "", "", "", "")
        minimum_self = (float("inf"), "", "", "", "")
        collisions = []

        for sample_index, sample in enumerate(samples):
            for index, angle in zip(arm_indices, sample["joints"]):
                bullet.resetJointState(robot, index, float(angle))
            target = sample["target"]
            tcp_state = bullet.getLinkState(
                robot, tcp_index, computeForwardKinematics=True
            )
            model_error = float(
                np.linalg.norm(np.asarray(tcp_state[4]) - target[:3, 3])
            )
            model_errors.append(model_error)
            uncertainty = max(
                settings["nominal_model_uncertainty_m"], model_error
            )
            query_self = settings["self_collision_clearance_m"] + uncertainty
            for gripper_state, angle in (("open", 0.0), ("closed", 0.85)):
                for index in gripper_indices:
                    bullet.resetJointState(robot, index, angle)
                bullet.performCollisionDetection()
                table_distance, table_link = _minimum_table_distance(
                    bullet,
                    robot,
                    table,
                    link_names,
                    settings["ignored_table_links"],
                )
                gripper_distance, gripper_link = _minimum_table_distance(
                    bullet,
                    robot,
                    table,
                    link_names,
                    settings["ignored_table_links"],
                    gripper_links,
                )
                effective_table = table_distance - uncertainty
                effective_gripper = gripper_distance - uncertainty
                identity = (
                    sample["segment"],
                    f"{sample['alpha']:.3f}",
                    gripper_state,
                )
                if effective_table < minimum_table[0]:
                    minimum_table = (effective_table, table_link, *identity)
                if effective_gripper < minimum_gripper[0]:
                    minimum_gripper = (
                        effective_gripper,
                        gripper_link,
                        *identity,
                    )
                for distance, first, second in _self_clearances(
                    bullet,
                    robot,
                    link_names,
                    disabled,
                    query_self,
                ):
                    effective_self = distance - uncertainty
                    if effective_self < minimum_self[0]:
                        minimum_self = (
                            effective_self,
                            f"{first} <-> {second}",
                            *identity,
                        )
                    if effective_self < settings["self_collision_clearance_m"]:
                        collisions.append(
                            {
                                "type": "self",
                                "links": [first, second],
                                "effective_clearance_m": effective_self,
                                "sample_index": sample_index,
                                "segment": sample["segment"],
                                "alpha": sample["alpha"],
                                "gripper_state": gripper_state,
                            }
                        )
                if effective_table < settings["required_table_clearance_m"]:
                    collisions.append(
                        {
                            "type": "table",
                            "link": table_link,
                            "effective_clearance_m": effective_table,
                            "sample_index": sample_index,
                            "segment": sample["segment"],
                            "alpha": sample["alpha"],
                            "gripper_state": gripper_state,
                        }
                    )
            records.append(
                {
                    "index": sample_index,
                    "segment": sample["segment"],
                    "alpha": sample["alpha"],
                    "target_xyz_m": target[:3, 3].tolist(),
                    "joints_rad": sample["joints"].tolist(),
                    "controller_fk_translation_error_m": sample[
                        "controller_fk_translation_error_m"
                    ],
                    "controller_fk_rotation_error_deg": sample[
                        "controller_fk_rotation_error_deg"
                    ],
                    "nominal_urdf_tcp_error_m": model_error,
                }
            )
        maximum_model_error = max(model_errors, default=float("inf"))
        model_ok = maximum_model_error <= settings["maximum_model_error_m"]
        modeled_passed = not collisions and model_ok
        return {
            "modeled_collision_check_passed": modeled_passed,
            "sample_count": len(records),
            "maximum_nominal_urdf_tcp_error_m": maximum_model_error,
            "maximum_allowed_model_error_m": settings["maximum_model_error_m"],
            "minimum_full_link_table_clearance_m": minimum_table[0],
            "minimum_full_link_table_clearance_context": {
                "link": minimum_table[1],
                "segment": minimum_table[2],
                "alpha": minimum_table[3],
                "gripper_state": minimum_table[4],
            },
            "minimum_gripper_table_clearance_m": minimum_gripper[0],
            "minimum_gripper_table_clearance_context": {
                "link": minimum_gripper[1],
                "segment": minimum_gripper[2],
                "alpha": minimum_gripper[3],
                "gripper_state": minimum_gripper[4],
            },
            "minimum_self_clearance_m": (
                None if math.isinf(minimum_self[0]) else minimum_self[0]
            ),
            "minimum_self_clearance_context": (
                None
                if math.isinf(minimum_self[0])
                else {
                    "links": minimum_self[1],
                    "segment": minimum_self[2],
                    "alpha": minimum_self[3],
                    "gripper_state": minimum_self[4],
                }
            ),
            "collision_count": len(collisions),
            "collisions": collisions[:100],
            "samples": records,
        }
    finally:
        bullet.disconnect()


def validate_collision(
    plan_path: Path,
    config_path: Path,
    xarm_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    source_plan_sha256 = _sha256_file(plan_path)
    source_robot_config_sha256 = _sha256_file(config_path)
    plan = load_json_object(plan_path)
    plan_sources = plan.get("source_sha256")
    if not isinstance(plan_sources, dict):
        raise CollisionValidationError(
            "robot plan has no source hashes; rebuild robot-plan"
        )
    if plan_sources.get("robot_config") != source_robot_config_sha256:
        raise CollisionValidationError(
            "robot config changed after plan generation; rebuild robot-plan"
        )
    candidates = _ranked_plan_candidates(plan)
    config = _load_config(config_path)
    ready = _plan_ready(plan, config)
    settings = _collision_config(config_path)
    urdf_path, srdf_path = _generate_urdf(
        xarm_root,
        output_path.with_name("xarm7_gripper_collision.urdf"),
        bool(dict(load_json_object(config_path)).get("xarm_model1300", True)),
        str(dict(load_json_object(config_path)).get("xarm_gripper_version", "G1")),
    )
    evaluated_candidate_count = 0
    collision_tested_candidate_count = 0
    selected: Optional[Dict[str, Any]] = None
    bullet_result: Optional[Dict[str, Any]] = None
    arm = _connect(config)
    try:
        status = _check_robot_status(arm, config)
        current = pose_aa_to_transform(status["tcp_pose_mm_rad"])
        ready_targets = _ready_tcp_targets(
            current,
            np.asarray(ready["transform"], dtype=np.float64),
            settings["linear_sample_step_m"],
            settings["angular_sample_step_deg"],
        )
        ready_transform = ready_targets[-1]["target"]
        ready_sample_index = len(ready_targets) - 1
        ready_samples = _query_ik_path(
            arm,
            ready_targets,
            _number(config, "maximum_joint_step_rad"),
        )
        resolved_ready_joints = np.asarray(
            ready_samples[ready_sample_index]["joints"], dtype=np.float64
        )
        for candidate in candidates:
            evaluated_candidate_count += 1
            task_targets = _path_targets(
                ready_transform,
                candidate["waypoints"],
                settings["linear_sample_step_m"],
                settings["angular_sample_step_deg"],
                start_name="ready",
            )
            try:
                task_samples = _query_ik_path(
                    arm,
                    task_targets,
                    _number(config, "maximum_joint_step_rad"),
                    initial_angles=resolved_ready_joints,
                )
            except CollisionValidationError:
                continue
            samples = [*ready_samples, *task_samples[1:]]
            collision_tested_candidate_count += 1
            candidate_result = _run_bullet(
                urdf_path, srdf_path, samples, settings
            )
            if candidate_result["modeled_collision_check_passed"] is True:
                selected = dict(candidate)
                bullet_result = candidate_result
                break
    finally:
        arm.disconnect()

    initial_distance = float(
        np.linalg.norm(ready_transform[:3, 3] - current[:3, 3])
    )
    initial_distance_ok = initial_distance <= _number(config, "maximum_initial_move_m")
    modeled_passed = selected is not None and bullet_result is not None
    if selected is None or bullet_result is None:
        tabletop_execution_required: Optional[bool] = None
        tabletop_execution_verified = False
        bullet_result = {
            "modeled_collision_check_passed": False,
            "sample_count": 0,
            "maximum_nominal_urdf_tcp_error_m": None,
            "maximum_allowed_model_error_m": settings["maximum_model_error_m"],
            "minimum_full_link_table_clearance_m": None,
            "minimum_full_link_table_clearance_context": None,
            "minimum_gripper_table_clearance_m": None,
            "minimum_gripper_table_clearance_context": None,
            "minimum_self_clearance_m": None,
            "minimum_self_clearance_context": None,
            "collision_count": None,
            "collisions": [],
            "samples": [],
        }
    else:
        grasp = validate_transform(
            np.asarray(selected["waypoints"]["grasp"]), "grasp"
        )
        tabletop_execution_required = bool(
            float(grasp[2, 3]) < _number(config, "minimum_transit_z_m")
        )
        tabletop_execution_verified = bool(
            not tabletop_execution_required
            or config.get("tabletop_execution_verified") is True
        )
    safe_for_execution = bool(
        modeled_passed
        and initial_distance_ok
        and settings["geometry_verified"]
        and config.get("tool_alignment_verified") is True
        and tabletop_execution_verified
    )
    if _sha256_file(plan_path) != source_plan_sha256:
        raise CollisionValidationError(
            "robot plan changed while collision validation was running"
        )
    if _sha256_file(config_path) != source_robot_config_sha256:
        raise CollisionValidationError(
            "robot config changed while collision validation was running"
        )
    start_joints = np.asarray(
        ready_samples[0]["joints"], dtype=np.float64
    ).tolist()
    result = {
        "schema_version": 3,
        "validation_type": "xarm7_full_link_table_collision",
        "validation_status": "completed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "read_only_robot_query": True,
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": source_plan_sha256,
        "source_robot_config": str(config_path.resolve()),
        "source_robot_config_sha256": source_robot_config_sha256,
        "official_model_root": str(xarm_root.resolve()),
        "robot_status": status,
        "robot_start_joints_rad": start_joints,
        "ready": plan["ready"],
        "selection_method": plan["selection_method"],
        "candidate_count": len(candidates),
        "evaluated_candidate_count": evaluated_candidate_count,
        "collision_tested_candidate_count": collision_tested_candidate_count,
        "selected": selected,
        "selected_candidate_index": (
            None if selected is None else selected["candidate_index"]
        ),
        "resolved_ready_joint_angles_rad": resolved_ready_joints.tolist(),
        "resolved_ready_joint_angles_deg": np.rad2deg(
            resolved_ready_joints
        ).tolist(),
        "initial_tcp_to_ready_distance_m": initial_distance,
        "maximum_initial_move_m": _number(config, "maximum_initial_move_m"),
        "initial_distance_ok": initial_distance_ok,
        "tabletop_execution_required": tabletop_execution_required,
        "tabletop_execution_verified": tabletop_execution_verified,
        "environment": {
            "table_top_z_m": settings["table_top_z_m"],
            "table_center_xy_m": settings["table_center_xy_m"].tolist(),
            "table_size_xy_m": settings["table_size_xy_m"].tolist(),
            "table_thickness_m": settings["table_thickness_m"],
            "required_table_clearance_m": settings[
                "required_table_clearance_m"
            ],
            "self_collision_clearance_m": settings[
                "self_collision_clearance_m"
            ],
            "geometry_verified": settings["geometry_verified"],
        },
        **bullet_result,
        "safe_for_execution": safe_for_execution,
        "blocking_reasons": [
            reason
            for condition, reason in (
                (
                    not modeled_passed,
                    "no score-ranked candidate passed full collision validation",
                ),
                (
                    not initial_distance_ok,
                    "current TCP is farther from ready pose than maximum_initial_move_m",
                ),
                (
                    not settings["geometry_verified"],
                    "physical table/support geometry has not been measured and verified",
                ),
                (
                    config.get("tool_alignment_verified") is not True,
                    "tool alignment has not been physically verified",
                ),
                (
                    selected is not None and not tabletop_execution_verified,
                    "tabletop gripper clearance has not been physically verified",
                ),
            )
            if condition
        ],
        "limitations": [
            "The official nominal UFACTORY URDF is used; the measured TCP mismatch is subtracted from clearance.",
            "Only the configured table box and robot self-collision are checked.",
            "Transparent platforms, target-object contact, cables, camera stands, and people are not modeled.",
            "No robot motion command is sent by this validator.",
        ],
    }
    _write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--robot-config", type=Path, default=Path("robot_config.json")
    )
    parser.add_argument(
        "--xarm-ros2-root",
        type=Path,
        default=Path(os.environ.get("AFFORDGRASP_XARM_ROS2_ROOT", "xarm_ros2")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/robot_collision_validation.json"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_path = arguments.output.expanduser().resolve()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        # 먼저 이전 승인 결과를 무효화한다. 새 검사가 중간에 실패해도 과거의
        # safe_for_execution=true 파일이 남아 실제 실행에 재사용되지 않는다.
        _write_json(
            output_path,
            {
                "schema_version": 3,
                "validation_type": "xarm7_full_link_table_collision",
                "validation_status": "running",
                "validation_started_at": started_at,
                "safe_for_execution": False,
                "blocking_reasons": ["collision validation is still running"],
            },
        )
        result = validate_collision(
            arguments.plan.expanduser().resolve(),
            arguments.robot_config.expanduser().resolve(),
            arguments.xarm_ros2_root.expanduser().resolve(),
            output_path,
        )
        summary = {
            "output": str(output_path),
            "selection_method": result["selection_method"],
            "candidate_count": result["candidate_count"],
            "evaluated_candidate_count": result["evaluated_candidate_count"],
            "selected_candidate_index": result["selected_candidate_index"],
            "modeled_collision_check_passed": result[
                "modeled_collision_check_passed"
            ],
            "safe_for_execution": result["safe_for_execution"],
            "minimum_gripper_table_clearance_m": result[
                "minimum_gripper_table_clearance_m"
            ],
            "minimum_full_link_table_clearance_m": result[
                "minimum_full_link_table_clearance_m"
            ],
            "collision_count": result["collision_count"],
            "blocking_reasons": result["blocking_reasons"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        # A completed collision analysis is a successful command even when a
        # separate physical prerequisite keeps real execution blocked.  A
        # modeled collision itself remains a non-zero validation result.
        return 0 if result["modeled_collision_check_passed"] else 2
    except (
        CollisionValidationError,
        RobotExecutionError,
        TransformError,
        OSError,
        ValueError,
    ) as exc:
        try:
            _write_json(
                output_path,
                {
                    "schema_version": 3,
                    "validation_type": "xarm7_full_link_table_collision",
                    "validation_status": "failed",
                    "validation_started_at": started_at,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "safe_for_execution": False,
                    "blocking_reasons": [str(exc)],
                },
            )
        except OSError:
            pass
        print(f"Collision validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
