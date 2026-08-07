"""Plan or explicitly execute an AnyGrasp result on an xArm7."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .transforms import (
    TransformError,
    load_base_to_camera,
    load_json_object,
    make_transform,
    rotation_distance_rad,
    transform_to_pose_aa,
    validate_rotation,
    validate_transform,
)
from .xarm_connection import connect_xarm, read_tcp_offset


class RobotExecutionError(RuntimeError):
    """Raised before or during a guarded real-robot action."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _number(payload: Mapping[str, Any], name: str, minimum: float = 0.0) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobotExecutionError(f"robot config {name!r} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < minimum:
        raise RobotExecutionError(
            f"robot config {name!r} must be finite and >= {minimum}"
        )
    return value


def _load_config(path: Path) -> Dict[str, Any]:
    payload = dict(load_json_object(path))
    if payload.get("robot_model") != "xArm7":
        raise RobotExecutionError("robot_model must be xArm7")
    ip = payload.get("robot_ip")
    if not isinstance(ip, str) or not ip.strip():
        raise RobotExecutionError("robot_ip is missing")
    workspace = np.asarray(payload.get("workspace_m"), dtype=np.float64)
    if workspace.shape != (3, 2) or np.any(workspace[:, 0] >= workspace[:, 1]):
        raise RobotExecutionError("workspace_m must be [[xmin,xmax], ...]")
    payload["workspace_m"] = workspace
    payload["R_anygrasp_to_tcp"] = validate_rotation(
        np.asarray(payload.get("R_anygrasp_to_tcp"), dtype=np.float64),
        "R_anygrasp_to_tcp",
    )
    tcp_offset = np.asarray(payload.get("required_tcp_offset_mm"), dtype=np.float64)
    if tcp_offset.shape != (6,) or not np.all(np.isfinite(tcp_offset)):
        raise RobotExecutionError("required_tcp_offset_mm must contain six values")
    payload["required_tcp_offset_mm"] = tcp_offset
    return payload


def _candidate_records(grasp_payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    candidates = grasp_payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        return [candidate for candidate in candidates if isinstance(candidate, dict)]
    selected = grasp_payload.get("selected_grasp")
    if isinstance(selected, dict):
        return [selected]
    raise RobotExecutionError("grasp result contains no candidates")


def _point_in_workspace(point: np.ndarray, workspace: np.ndarray) -> bool:
    return bool(np.all(point >= workspace[:, 0]) and np.all(point <= workspace[:, 1]))


def _candidate_plan(
    candidate: Mapping[str, Any],
    base_to_camera: np.ndarray,
    config: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Sequence[str]]:
    rejected = []
    try:
        rotation_camera_grasp = validate_rotation(
            np.asarray(candidate.get("R"), dtype=np.float64),
            "AnyGrasp candidate R",
        )
        center_camera = np.asarray(candidate.get("t"), dtype=np.float64).reshape(3)
        width = float(candidate.get("w"))
        score = float(candidate.get("score"))
    except (TypeError, ValueError, TransformError):
        return None, ["malformed candidate"]
    if not np.all(np.isfinite(center_camera)) or not np.isfinite(width + score):
        return None, ["candidate contains non-finite values"]
    if score < _number(config, "minimum_anygrasp_score"):
        rejected.append("score below configured minimum")
    if width <= 0.0 or width > _number(config, "gripper_max_opening_m"):
        rejected.append("required width exceeds xArm Gripper opening")
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if metadata.get("collision_detection") is not True:
        rejected.append("AnyGrasp collision detection was not enabled")
    if metadata.get("region_steering") is not True:
        rejected.append("AnyGrasp region steering was not enabled")

    tip_value = candidate.get("gripper_tip_xyz_m")
    if tip_value is None:
        insertion_depth = float(metadata.get("insertion_depth_m", 0.0))
        tip_camera = center_camera + insertion_depth * rotation_camera_grasp[:, 0]
    else:
        try:
            tip_camera = np.asarray(tip_value, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return None, ["malformed gripper tip"]

    rotation_base_camera = base_to_camera[:3, :3]
    rotation_base_grasp = rotation_base_camera @ rotation_camera_grasp
    approach_base = rotation_base_grasp[:, 0]
    maximum_z = float(config.get("maximum_approach_z", -0.20))
    if float(approach_base[2]) > maximum_z:
        rejected.append(
            "approach is not sufficiently downward in the robot base frame"
        )

    tip_base = (
        rotation_base_camera @ tip_camera + base_to_camera[:3, 3]
    )
    rotation_base_tcp = rotation_base_grasp @ config["R_anygrasp_to_tcp"]
    target = make_transform(rotation_base_tcp, tip_base)
    pregrasp = target.copy()
    pregrasp[:3, 3] -= approach_base * _number(config, "pregrasp_offset_m")
    retreat = target.copy()
    retreat[:3, 3] -= approach_base * _number(config, "retreat_offset_m")
    lift = retreat.copy()
    lift[2, 3] += _number(config, "lift_offset_m")
    workspace = config["workspace_m"]
    for name, pose in (
        ("pregrasp", pregrasp),
        ("grasp", target),
        ("retreat", retreat),
        ("lift", lift),
    ):
        if not _point_in_workspace(pose[:3, 3], workspace):
            rejected.append(f"{name} lies outside workspace")

    if rejected:
        return None, rejected
    objective = candidate.get("selection_objective", score)
    try:
        objective = float(objective)
    except (TypeError, ValueError):
        objective = score
    return {
        "candidate_index": int(candidate.get("index", 0)),
        "anygrasp_score": score,
        "selection_objective": objective,
        "required_width_m": width,
        "approach_vector_base": approach_base.tolist(),
        "waypoints": {
            "pregrasp": pregrasp.tolist(),
            "grasp": target.tolist(),
            "retreat": retreat.tolist(),
            "lift": lift.tolist(),
        },
    }, []


def build_plan(
    grasp_path: Path,
    calibration_path: Path,
    config_path: Path,
) -> Dict[str, Any]:
    grasp = load_json_object(grasp_path)
    if grasp.get("backend") != "anygrasp":
        raise RobotExecutionError("only a real AnyGrasp result can drive the robot")
    calibration = load_json_object(calibration_path)
    base_to_camera = load_base_to_camera(calibration_path)
    config = _load_config(config_path)
    calibration_robot = calibration.get("robot")
    if not isinstance(calibration_robot, dict):
        raise RobotExecutionError("calibration has no robot metadata")
    if calibration_robot.get("ip") != config["robot_ip"]:
        raise RobotExecutionError("calibration robot IP does not match robot config")
    calibration_tcp_offset = np.asarray(
        calibration_robot.get("tcp_offset_mm_rad"), dtype=np.float64
    )
    if calibration_tcp_offset.shape != (6,) or not np.allclose(
        calibration_tcp_offset,
        config["required_tcp_offset_mm"],
        atol=_number(config, "tcp_offset_tolerance_mm"),
    ):
        raise RobotExecutionError(
            "calibration TCP offset does not match robot config"
        )
    calibration_camera = calibration.get("camera")
    grasp_inputs = grasp.get("inputs")
    if not isinstance(calibration_camera, dict) or not isinstance(grasp_inputs, dict):
        raise RobotExecutionError("camera identity metadata is missing")
    calibration_serial = str(calibration_camera.get("serial_number", ""))
    grasp_serial = str(grasp_inputs.get("camera_serial_number", ""))
    if not calibration_serial or calibration_serial != grasp_serial:
        raise RobotExecutionError(
            "grasp D435 serial does not match the calibrated D435"
        )
    accepted = []
    rejected = []
    for ordinal, candidate in enumerate(_candidate_records(grasp)):
        plan, reasons = _candidate_plan(candidate, base_to_camera, config)
        if plan is None:
            rejected.append({"candidate": ordinal, "reasons": list(reasons)})
        else:
            accepted.append(plan)
    if not accepted:
        raise RobotExecutionError(
            "no AnyGrasp candidate passed robot workspace, approach, and gripper checks"
        )
    accepted.sort(
        key=lambda item: (item["selection_objective"], item["anygrasp_score"]),
        reverse=True,
    )
    selected = accepted[0]
    return {
        "schema_version": 1,
        "plan_type": "xarm7_anygrasp_pick",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_ip": config["robot_ip"],
        "coordinate_frame": "xArm base",
        "source_grasp_result": str(grasp_path.resolve()),
        "source_calibration": str(calibration_path.resolve()),
        "source_robot_config": str(config_path.resolve()),
        "tool_alignment_verified": config.get("tool_alignment_verified") is True,
        "selected": selected,
        "accepted_candidate_count": len(accepted),
        "rejected": rejected,
        "workspace_m": config["workspace_m"].tolist(),
        "limitations": [
            "AnyGrasp collision checks the local gripper pose, not the full xArm trajectory.",
            "Direct Cartesian segments require a physically cleared robot cell.",
            "Full execution is blocked until tool alignment is visually verified.",
        ],
    }


def _connect(config: Mapping[str, Any]) -> Any:
    return connect_xarm(config["robot_ip"], RobotExecutionError)


def _check_robot_status(arm: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    code, pose = arm.get_position_aa(is_radian=True)
    if code != 0:
        raise RobotExecutionError(f"could not read xArm TCP pose, code={code}")
    error_code = int(getattr(arm, "error_code", 0))
    warn_code = int(getattr(arm, "warn_code", 0))
    if error_code != 0:
        raise RobotExecutionError(
            f"xArm error_code={error_code}; inspect and clear it in UFACTORY Studio"
        )
    required_offset = config["required_tcp_offset_mm"]
    current_offset = read_tcp_offset(
        arm,
        RobotExecutionError,
        expected=required_offset,
        tolerance=_number(config, "tcp_offset_tolerance_mm"),
    )
    offset_ok = current_offset.shape == (6,) and np.allclose(
        current_offset,
        required_offset,
        atol=_number(config, "tcp_offset_tolerance_mm"),
    )
    return {
        "connected": bool(arm.connected),
        "state": int(getattr(arm, "state", -1)),
        "mode": int(getattr(arm, "mode", -1)),
        "error_code": error_code,
        "warn_code": warn_code,
        "tcp_pose_mm_rad": list(map(float, pose[:6])),
        "tcp_offset_mm_rad": current_offset.tolist(),
        "required_tcp_offset_mm_rad": required_offset.tolist(),
        "tcp_offset_ok": bool(offset_ok),
    }


def _rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    # xArm uses roll/pitch/yaw with R = Rz(yaw) Ry(pitch) Rx(roll).
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _check_ik(
    arm: Any,
    waypoints: Mapping[str, Any],
    maximum_joint_step_rad: float,
) -> Dict[str, Any]:
    results = {}
    reference = None
    code, current_angles = arm.get_servo_angle(is_radian=True)
    if code == 0:
        reference = current_angles
    for name in ("pregrasp", "grasp", "retreat", "lift"):
        transform = validate_transform(np.asarray(waypoints[name], dtype=np.float64))
        rpy = _rotation_to_rpy(transform[:3, :3])
        pose = [*(transform[:3, 3] * 1000.0).tolist(), *rpy.tolist()]
        code, angles = arm.get_inverse_kinematics(
            pose,
            input_is_radian=True,
            return_is_radian=True,
            limited=True,
            ref_angles=reference,
        )
        if code != 0 or not angles:
            raise RobotExecutionError(f"xArm IK failed for {name}, code={code}")
        if reference is not None:
            previous = np.asarray(reference, dtype=np.float64)
            current = np.asarray(angles, dtype=np.float64)
            wrapped_delta = np.arctan2(
                np.sin(current - previous), np.cos(current - previous)
            )
            largest_step = float(np.max(np.abs(wrapped_delta)))
            if largest_step > maximum_joint_step_rad:
                raise RobotExecutionError(
                    f"xArm IK for {name} requires a joint jump of "
                    f"{np.rad2deg(largest_step):.1f} deg"
                )
        results[name] = list(map(float, angles))
        reference = angles
    return results


def _check_code(arm: Any, code: int, operation: str) -> None:
    if code != 0 or int(getattr(arm, "error_code", 0)) != 0:
        raise RobotExecutionError(
            f"{operation} failed: code={code}, error={arm.error_code}, "
            f"warn={arm.warn_code}, state={arm.state}"
        )


def _move(arm: Any, transform: np.ndarray, speed: float, acceleration: float, label: str) -> None:
    code = arm.set_position_aa(
        transform_to_pose_aa(transform),
        speed=speed,
        mvacc=acceleration,
        is_radian=True,
        wait=True,
        timeout=60,
        radius=-1,
    )
    _check_code(arm, int(code), f"move to {label}")
    read_code, reached_pose = arm.get_position_aa(is_radian=True)
    _check_code(arm, int(read_code), f"read pose after {label}")
    reached = make_transform(
        cv2.Rodrigues(np.asarray(reached_pose[3:6], dtype=np.float64))[0],
        np.asarray(reached_pose[:3], dtype=np.float64) / 1000.0,
    )
    translation_error = float(np.linalg.norm(reached[:3, 3] - transform[:3, 3]))
    rotation_error = rotation_distance_rad(reached[:3, :3], transform[:3, :3])
    if translation_error > 0.005 or rotation_error > np.deg2rad(3.0):
        raise RobotExecutionError(
            f"{label} tracking error is too high: "
            f"{translation_error * 1000:.1f} mm, {np.rad2deg(rotation_error):.1f} deg"
        )


def execute_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    mode: str,
) -> Dict[str, Any]:
    arm = _connect(config)
    motion_started = False
    try:
        status = _check_robot_status(arm, config)
        if mode == "connect":
            return {"mode": mode, "status": status}
        if not status["tcp_offset_ok"]:
            raise RobotExecutionError(
                "xArm TCP offset does not match the configured xArm Gripper offset; "
                f"current={status['tcp_offset_mm_rad']}, "
                f"required={status['required_tcp_offset_mm_rad']}"
            )
        if mode == "full" and config.get("tool_alignment_verified") is not True:
            raise RobotExecutionError(
                "full grasp is blocked until the pregrasp orientation is visually "
                "checked and tool_alignment_verified is set to true"
            )
        waypoints = plan["selected"]["waypoints"]
        pregrasp = validate_transform(np.asarray(waypoints["pregrasp"]))
        current_position = np.asarray(status["tcp_pose_mm_rad"][:3]) / 1000.0
        initial_distance = float(
            np.linalg.norm(pregrasp[:3, 3] - current_position)
        )
        if initial_distance > _number(config, "maximum_initial_move_m"):
            raise RobotExecutionError(
                "current TCP is too far from pregrasp for a direct Cartesian move: "
                f"{initial_distance:.3f} m"
            )
        ik = _check_ik(
            arm,
            waypoints,
            _number(config, "maximum_joint_step_rad"),
        )

        _check_code(
            arm,
            int(arm.set_collision_sensitivity(5)),
            "collision sensitivity",
        )
        self_collision_code = arm.set_self_collision_detection(True)
        _check_code(arm, int(self_collision_code), "self-collision detection")
        _check_code(arm, int(arm.motion_enable(True)), "motion enable")
        _check_code(arm, int(arm.set_mode(0)), "position mode")
        _check_code(arm, int(arm.set_state(0)), "ready state")
        motion_started = True

        gripper_speed = int(_number(config, "gripper_speed_units"))
        open_units = int(_number(config, "gripper_open_units"))
        code = arm.set_gripper_enable(True)
        _check_code(arm, int(code), "gripper enable")
        _check_code(arm, int(arm.set_gripper_mode(0)), "gripper mode")
        _check_code(
            arm,
            int(arm.set_gripper_position(open_units, wait=True, speed=gripper_speed)),
            "open gripper",
        )
        speed = _number(config, "travel_speed_mm_s")
        acceleration = _number(config, "travel_acceleration_mm_s2")
        _move(arm, pregrasp, speed, acceleration, "pregrasp")
        if mode == "pregrasp":
            return {"mode": mode, "status_before": status, "ik": ik}

        grasp = validate_transform(np.asarray(waypoints["grasp"]))
        _move(
            arm,
            grasp,
            _number(config, "grasp_speed_mm_s"),
            acceleration,
            "grasp",
        )
        close_units = int(_number(config, "gripper_close_units"))
        _check_code(
            arm,
            int(arm.set_gripper_position(close_units, wait=True, speed=gripper_speed)),
            "close gripper",
        )
        grip_code, grip_position = arm.get_gripper_position()
        _check_code(arm, int(grip_code), "read gripper position")
        retreat = validate_transform(np.asarray(waypoints["retreat"]))
        lift = validate_transform(np.asarray(waypoints["lift"]))
        _move(arm, retreat, speed, acceleration, "retreat")
        _move(arm, lift, speed, acceleration, "lift")
        return {
            "mode": mode,
            "status_before": status,
            "ik": ik,
            "gripper_position_after_close": float(grip_position),
        }
    except Exception:
        if motion_started:
            try:
                arm.emergency_stop()
            except Exception:
                pass
        raise
    finally:
        arm.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-result", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=Path("calibration/eye_to_hand.json")
    )
    parser.add_argument(
        "--robot-config", type=Path, default=Path("robot_config.json")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/robot_plan.json"))
    parser.add_argument(
        "--mode", choices=("plan", "connect", "pregrasp", "full"), default="plan"
    )
    parser.add_argument("--confirm", default="")
    parser.add_argument("--acknowledge-cleared-workspace", action="store_true")
    parser.add_argument("--acknowledge-estop-ready", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config_path = arguments.robot_config.expanduser().resolve()
        config = _load_config(config_path)
        plan = build_plan(
            arguments.grasp_result.expanduser().resolve(),
            arguments.calibration.expanduser().resolve(),
            config_path,
        )
        output_path = arguments.output.expanduser().resolve()
        _write_json(output_path, plan)
        if arguments.mode == "plan":
            print(
                json.dumps(
                    {"plan": str(output_path), "selected": plan["selected"]},
                    indent=2,
                )
            )
            return 0
        if arguments.mode in {"pregrasp", "full"}:
            expected = "MOVE_XARM7_" + config["robot_ip"].replace(".", "_")
            if arguments.confirm != expected:
                raise RobotExecutionError(f"--confirm must be exactly {expected}")
            if not arguments.acknowledge_cleared_workspace:
                raise RobotExecutionError("--acknowledge-cleared-workspace is required")
            if not arguments.acknowledge_estop_ready:
                raise RobotExecutionError("--acknowledge-estop-ready is required")
        result = execute_plan(plan, config, arguments.mode)
        execution_path = output_path.with_name(output_path.stem + "_execution.json")
        _write_json(execution_path, {"plan": plan, "execution": result})
        print(json.dumps({"execution": str(execution_path), **result}, indent=2))
    except (RobotExecutionError, TransformError, OSError, ValueError) as exc:
        print(f"Robot execution error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
