#!/usr/bin/env python3
"""Run AffordGrasp with the ThinkGrasp PyBullet environment.

The simulator replaces the two hardware-specific ends of AffordGrasp:

* ThinkGrasp ``render_camera`` replaces the RealSense capture.
* ThinkGrasp ``Environment.step`` replaces xArm execution.

By default the script only creates the scene and writes an AffordGrasp-compatible
RGB-D capture.  Use ``--run-pipeline`` to run ICAR, VLPart, affordance masking,
and AnyGrasp.  Add ``--execute`` to execute the selected grasp in PyBullet.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
THINKGRASP_ROOT = PROJECT_ROOT / "third_party" / "ThinkGrasp"
CAPTURE_ROOT = PROJECT_ROOT / "captures" / "icar_d435"
RUN_ROOT = PROJECT_ROOT / "runs"
DEPTH_SCALE_M_PER_UNIT = 0.001
XARM_ROOT = Path(
    os.environ.get("AFFORDGRASP_XARM_ROS2_ROOT", PROJECT_ROOT / "xarm_ros2")
)
XARM_URDF_CACHE = RUN_ROOT / "_simulation" / "xarm7_gripper.urdf"
XARM_ANYGRASP_TO_TCP = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


class SimulationError(RuntimeError):
    """Raised when simulation capture or execution cannot finish."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ThinkGrasp PyBullet 환경에서 AffordGrasp를 실행합니다."
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=Path("affordgrasp_cases/ball_visible.txt"),
        help=(
            "ThinkGrasp 기준 case 파일 "
            "(기본: affordgrasp_cases/ball_visible.txt)"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="sim_case00",
        help="captures/와 runs/에 사용할 실행 prefix",
    )
    parser.add_argument(
        "--instruction",
        help="AffordGrasp 작업 지시; 생략하면 case 파일의 첫 줄 사용",
    )
    parser.add_argument(
        "--xarm-root",
        type=Path,
        default=XARM_ROOT,
        help="공식 xarm_ros2 저장소 경로",
    )
    parser.add_argument(
        "--xarm-urdf",
        type=Path,
        help="미리 생성한 xArm7+Gripper URDF; 생략하면 자동 생성",
    )
    parser.add_argument("--camera-index", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=0.5,
        help="GUI 로봇 동작 재생 배속 (기본: 0.5, 1.0=실시간)",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="캡처 후 ICAR, localization, mask, grasp 단계를 실행",
    )
    parser.add_argument(
        "--icar-python",
        type=Path,
        help=(
            "ICAR에 사용할 Python 실행 파일. 생략하면 "
            "AFFORDGRASP_ICAR_PYTHON 또는 Conda base Python 사용"
        ),
    )
    parser.add_argument(
        "--grasp-result",
        type=Path,
        help="기존 grasp_pose_result.json을 시각화하거나 실행",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="선택된 grasp를 PyBullet xArm7에서 실행",
    )
    parser.add_argument(
        "--max-approach-angle",
        type=float,
        help="시뮬레이션 하향 접근 최대 각도(deg, 기본: 20)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="PyBullet GUI 없이 실행",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="완료 후 GUI 유지 루프를 실행하지 않음",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.prefix or not arguments.prefix[0].isalnum():
        raise SimulationError("prefix는 영문자 또는 숫자로 시작해야 합니다")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(character not in allowed for character in arguments.prefix):
        raise SimulationError("prefix에는 영문자, 숫자, '.', '_', '-'만 사용할 수 있습니다")
    if arguments.execute and not (
        arguments.run_pipeline or arguments.grasp_result is not None
    ):
        raise SimulationError(
            "--execute에는 --run-pipeline 또는 --grasp-result가 필요합니다"
        )
    if arguments.run_pipeline and arguments.grasp_result is not None:
        raise SimulationError("--run-pipeline과 --grasp-result는 함께 사용할 수 없습니다")
    if arguments.max_approach_angle is not None and not (
        0.0 < arguments.max_approach_angle <= 90.0
    ):
        raise SimulationError("--max-approach-angle은 (0, 90] 범위여야 합니다")
    if not np.isfinite(arguments.simulation_speed) or not (
        0.05 <= arguments.simulation_speed <= 10.0
    ):
        raise SimulationError("--simulation-speed는 0.05~10 범위여야 합니다")
    if arguments.no_gui:
        arguments.no_hold = True


def _resolve_case(case_argument: Path) -> Path:
    case_path = case_argument.expanduser()
    if not case_path.is_absolute():
        case_path = THINKGRASP_ROOT / case_path
    case_path = case_path.resolve()
    if not case_path.is_file():
        raise SimulationError(f"case 파일이 없습니다: {case_path}")
    return case_path


def _resolve_icar_python(argument: Optional[Path]) -> Path:
    configured = os.environ.get("AFFORDGRASP_ICAR_PYTHON", "").strip()
    candidates = []
    if argument is not None:
        candidates.append(argument.expanduser())
    elif configured:
        candidates.append(Path(configured).expanduser())
    else:
        conda_executable = os.environ.get("CONDA_EXE", "").strip()
        if conda_executable:
            candidates.append(Path(conda_executable).expanduser().parent / "python")

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise SimulationError(
        "ICAR Python을 찾지 못했습니다. 예: "
        "--icar-python /home/unist/anaconda3/bin/python"
    )


def _configured_anygrasp_python() -> Path:
    config_path = PROJECT_ROOT / "config.env"
    shell = r'''
set -euo pipefail
if [[ -f "$1" ]]; then
  source "$1"
fi
if [[ -z "${AFFORDGRASP_ANYGRASP_ENV:-}" ]]; then
  exit 2
fi
printf '%s/bin/python' "$AFFORDGRASP_ANYGRASP_ENV"
'''
    completed = subprocess.run(
        ["bash", "-c", shell, "affordgrasp-xacro", str(config_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SimulationError(
            "config.env의 AFFORDGRASP_ANYGRASP_ENV에서 xacro Python을 "
            "찾을 수 없습니다"
        )
    python_path = Path(completed.stdout.strip()).expanduser().resolve()
    if not python_path.is_file():
        raise SimulationError(f"AnyGrasp Python이 없습니다: {python_path}")
    return python_path


def _prepare_xarm_urdf(arguments: argparse.Namespace) -> Path:
    if arguments.xarm_urdf is not None:
        urdf = arguments.xarm_urdf.expanduser().resolve()
        if not urdf.is_file():
            raise SimulationError(f"xArm7 URDF가 없습니다: {urdf}")
        return urdf

    xarm_root = arguments.xarm_root.expanduser().resolve()
    if not (xarm_root / "xarm_description").is_dir():
        raise SimulationError(f"공식 xarm_ros2 모델이 없습니다: {xarm_root}")
    model1300 = True
    gripper_version = "G1"
    robot_config = PROJECT_ROOT / "robot_config.json"
    if robot_config.is_file():
        try:
            payload = json.loads(robot_config.read_text(encoding="utf-8"))
            model1300 = bool(payload.get("xarm_model1300", True))
            gripper_version = str(payload.get("xarm_gripper_version", "G1"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SimulationError(f"robot_config.json을 읽을 수 없습니다: {exc}") from exc

    generator_python = _configured_anygrasp_python()
    XARM_URDF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(generator_python),
        "-m",
        "affordgrasp_icar.simulation.generate_xarm7_urdf",
        "--xarm-root",
        str(xarm_root),
        "--output",
        str(XARM_URDF_CACHE),
        "--gripper-version",
        gripper_version,
    ]
    if model1300:
        command.append("--model1300")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT.parent)
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SimulationError(f"xArm7 URDF 생성기를 실행할 수 없습니다: {exc}") from exc
    if completed.returncode != 0 or not XARM_URDF_CACHE.is_file():
        details = (completed.stderr or completed.stdout).strip()
        raise SimulationError(f"xArm7 URDF 자동 생성 실패: {details}")
    print(f"xArm7 URDF: {XARM_URDF_CACHE}")
    return XARM_URDF_CACHE


def _xarm_additional_grasp_depth() -> float:
    """Use the same grasp-depth correction as physical xArm execution."""
    depth = 0.01
    config_path = PROJECT_ROOT / "robot_config.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            depth = float(payload.get("additional_grasp_depth_m", depth))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SimulationError(
                f"robot_config.json의 additional_grasp_depth_m가 잘못됐습니다: {exc}"
            ) from exc
    if not np.isfinite(depth) or not 0.0 <= depth <= 0.05:
        raise SimulationError(
            "additional_grasp_depth_m는 0~0.05 m 범위여야 합니다"
        )
    return depth


def _load_thinkgrasp(
    xarm_urdf: Path, simulation_speed: float
) -> Tuple[Any, Any]:
    if not THINKGRASP_ROOT.is_dir():
        raise SimulationError(f"ThinkGrasp 폴더가 없습니다: {THINKGRASP_ROOT}")
    sys.path.insert(0, str(THINKGRASP_ROOT))
    try:
        import pybullet as bullet
    except ImportError as exc:
        raise SimulationError(
            "ThinkGrasp/pybullet을 불러오지 못했습니다. thinkgrasp Conda 환경에서 "
            "실행하세요"
        ) from exc
    from simulation.xarm7_environment import XArm7Environment

    def create_xarm7(gui: bool = True) -> Any:
        return XArm7Environment(
            xarm_urdf,
            gui=gui,
            simulation_speed=simulation_speed,
        )

    return bullet, create_xarm7


def _camera_metadata(
    camera: Dict[str, Any],
    case_path: Path,
    seed: int,
) -> Dict[str, Any]:
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    height, width = (int(value) for value in camera["image_size"])
    return {
        "camera_name": "ThinkGrasp PyBullet RealSenseD435",
        "serial_number": "PYBULLET",
        "firmware_version": "simulation",
        "coordinate_system": "depth aligned to RGB/color pixels",
        "depth_format": "uint16 PNG (millimetres)",
        "depth_scale_meters_per_unit": DEPTH_SCALE_M_PER_UNIT,
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
            "distortion_model": "none",
            "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "simulation": {
            "case": str(case_path),
            "seed": seed,
            "world_from_camera": {
                "translation_xyz_m": [float(value) for value in camera["position"]],
                "quaternion_xyzw": [float(value) for value in camera["rotation"]],
            },
        },
    }


def _write_capture(
    env: Any,
    camera: Dict[str, Any],
    prefix: str,
    case_path: Path,
    seed: int,
) -> Dict[str, Path]:
    color, depth_m, segmentation = env.render_camera(camera)
    color = np.asarray(color, dtype=np.uint8)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] != 3:
        raise SimulationError(f"잘못된 RGB shape: {color.shape}")
    if depth_m.shape != color.shape[:2]:
        raise SimulationError(
            f"RGB/depth 크기가 다릅니다: {color.shape[:2]} vs {depth_m.shape}"
        )
    if not np.all(np.isfinite(depth_m)):
        raise SimulationError("시뮬레이션 depth에 non-finite 값이 있습니다")

    CAPTURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb": CAPTURE_ROOT / f"{prefix}_rgb.png",
        "depth_raw": CAPTURE_ROOT / f"{prefix}_depth_raw.png",
        "depth_filtered": CAPTURE_ROOT / f"{prefix}_depth_filtered.png",
        "depth_preview": CAPTURE_ROOT / f"{prefix}_depth_preview.png",
        "segmentation": CAPTURE_ROOT / f"{prefix}_segmentation.png",
        "camera": CAPTURE_ROOT / f"{prefix}_camera.json",
    }

    depth_units = np.clip(
        np.rint(depth_m / DEPTH_SCALE_M_PER_UNIT),
        0,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    valid = depth_m > 0
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth_m[valid], (2.0, 98.0))
        if far > near:
            normalized = 1.0 - np.clip((depth_m - near) / (far - near), 0.0, 1.0)
            preview[valid] = np.rint(normalized[valid] * 255.0).astype(np.uint8)

    segmentation_u16 = np.asarray(segmentation, dtype=np.uint16)
    Image.fromarray(color, mode="RGB").save(paths["rgb"])
    Image.fromarray(depth_units).save(paths["depth_raw"])
    Image.fromarray(depth_units).save(paths["depth_filtered"])
    Image.fromarray(preview, mode="L").save(paths["depth_preview"])
    Image.fromarray(segmentation_u16).save(paths["segmentation"])
    paths["camera"].write_text(
        json.dumps(
            _camera_metadata(camera, case_path, seed),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _run_icar(
    prefix: str,
    instruction: str,
    icar_python: Path,
) -> None:
    rgb_path = CAPTURE_ROOT / f"{prefix}_rgb.png"
    config_path = PROJECT_ROOT / "config.env"
    shell = r'''
set -euo pipefail
config_path=$1
icar_python=$2
rgb_path=$3
instruction=$4
if [[ -f "$config_path" ]]; then
  source "$config_path"
fi
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  "$icar_python" -m affordgrasp_icar \
  --image "$rgb_path" \
  --instruction "$instruction"
'''
    try:
        completed = subprocess.run(
            [
                "bash",
                "-c",
                shell,
                "affordgrasp-sim-icar",
                str(config_path),
                str(icar_python),
                str(rgb_path),
                instruction,
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except OSError as exc:
        raise SimulationError(f"ICAR Python 실행 실패: {exc}") from exc
    if completed.returncode == 3:
        raise SimulationError(
            "ICAR가 visible하고 안전한 grasp 대상을 찾지 못했습니다. "
            "대상이 카메라에 보이는지 확인하거나 더 구체적인 --instruction을 사용하세요"
        )
    if completed.returncode != 0:
        raise SimulationError(
            f"AffordGrasp icar 단계가 실패했습니다 (exit={completed.returncode})"
        )


def _run_affordgrasp_pipeline(
    prefix: str,
    instruction: str,
    icar_python: Path,
) -> Path:
    pipeline = PROJECT_ROOT / "run_affordgrasp_pipeline.sh"
    if not pipeline.is_file():
        raise SimulationError(f"AffordGrasp pipeline 스크립트가 없습니다: {pipeline}")

    print("\n[AffordGrasp] stage=icar", flush=True)
    _run_icar(prefix, instruction, icar_python)

    commands = (
        ("localization", prefix),
        ("mask", prefix),
        ("grasp", prefix),
    )
    for stage in commands:
        command = ["bash", str(pipeline), "--stage", *stage]
        print(f"\n[AffordGrasp] stage={stage[0]}", flush=True)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            raise SimulationError(
                f"AffordGrasp {stage[0]} 단계가 실패했습니다 (exit={exc.returncode})"
            ) from exc

    result_path = RUN_ROOT / prefix / "grasp" / "grasp_pose_result.json"
    if not result_path.is_file():
        raise SimulationError(f"grasp 결과가 생성되지 않았습니다: {result_path}")
    return result_path


def _read_grasp_result(result_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationError(f"grasp 결과를 읽을 수 없습니다: {result_path}") from exc
    if not isinstance(payload.get("selected_grasp"), dict):
        raise SimulationError("grasp 결과에 selected_grasp가 없습니다")
    return payload


def _select_simulation_grasp(
    payload: Dict[str, Any],
    camera: Dict[str, Any],
    bullet: Any,
    env: Any,
    max_approach_angle_deg: float,
    require_reachable: bool,
) -> Dict[str, Any]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        candidates = [payload["selected_grasp"]]

    rotation_world_camera = np.asarray(
        bullet.getMatrixFromQuaternion(camera["rotation"]), dtype=np.float64
    ).reshape(3, 3)
    maximum_z = -float(np.cos(np.deg2rad(max_approach_angle_deg)))
    feasible = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            rotation_camera_grasp = np.asarray(candidate["R"], dtype=np.float64)
            approach_world = rotation_world_camera @ rotation_camera_grasp[:, 0]
            score = float(candidate["score"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if rotation_camera_grasp.shape == (3, 3) and approach_world[2] <= maximum_z:
            feasible.append((score, candidate, float(approach_world[2])))

    if not feasible:
        message = (
            f"하향 접근 각도 {max_approach_angle_deg:.1f}도 이내인 grasp 후보가 "
            "없습니다"
        )
        if require_reachable:
            raise SimulationError(message + "; xArm7을 움직이지 않습니다")
        print(f"경고: {message}. AffordGrasp 기본 후보를 표시합니다.", file=sys.stderr)
        return payload["selected_grasp"]

    if require_reachable:
        reachable = []
        for score, candidate, approach_z in feasible:
            try:
                action = _to_xarm_action(candidate, camera, bullet)
                is_reachable, failed_waypoint, error = env.check_grasp_reachability(
                    action
                )
            except (KeyError, TypeError, ValueError, SimulationError) as exc:
                print(
                    "xArm7 reachability: "
                    f"candidate={candidate.get('index', '?')}, invalid={exc}",
                    file=sys.stderr,
                )
                continue
            print(
                "xArm7 reachability: "
                f"candidate={candidate.get('index', '?')}, "
                f"reachable={is_reachable}, waypoint={failed_waypoint}, "
                f"position_error={error:.4f} m"
            )
            if is_reachable:
                reachable.append((score, candidate, approach_z))
        if not reachable:
            raise SimulationError(
                "접근 각도 조건을 통과한 후보 중 xArm7이 pregrasp부터 grasp까지 "
                "도달할 수 있는 후보가 없습니다; xArm7을 움직이지 않습니다"
            )
        feasible = reachable

    _, selected, approach_z = max(feasible, key=lambda item: item[0])
    print(
        "simulation candidate: "
        f"index={selected.get('index', '?')}, score={selected['score']:.4f}, "
        f"world approach z={approach_z:.4f}"
    )
    return selected


def _visible_target_pixels(segmentation_path: Path, target_ids: Sequence[int]) -> int:
    try:
        segmentation = np.asarray(Image.open(segmentation_path))
    except OSError as exc:
        raise SimulationError(
            f"segmentation 결과를 읽을 수 없습니다: {segmentation_path}"
        ) from exc
    if not target_ids:
        return 0
    return int(np.count_nonzero(np.isin(segmentation, np.asarray(target_ids))))


def _to_xarm_action(
    selected: Dict[str, Any],
    camera: Dict[str, Any],
    bullet: Any,
) -> np.ndarray:
    try:
        rotation_camera_grasp = np.asarray(selected["R"], dtype=np.float64)
        point_camera = np.asarray(selected["gripper_tip_xyz_m"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise SimulationError("selected_grasp의 R 또는 gripper_tip_xyz_m가 잘못됐습니다") from exc
    if rotation_camera_grasp.shape != (3, 3) or point_camera.shape != (3,):
        raise SimulationError("selected grasp pose shape이 잘못됐습니다")

    rotation_world_camera = np.asarray(
        bullet.getMatrixFromQuaternion(camera["rotation"]), dtype=np.float64
    ).reshape(3, 3)
    translation_world_camera = np.asarray(camera["position"], dtype=np.float64)
    point_world = rotation_world_camera @ point_camera + translation_world_camera
    rotation_world_grasp = rotation_world_camera @ rotation_camera_grasp

    # Same AnyGrasp-to-TCP convention used by robot_config.json for the
    # physical xArm Gripper. TCP +Z is the grasp approach direction.
    point_world = (
        point_world
        + _xarm_additional_grasp_depth() * rotation_world_grasp[:, 0]
    )
    rotation_world_tip = rotation_world_grasp @ XARM_ANYGRASP_TO_TCP
    from scipy.spatial.transform import Rotation

    quaternion_xyzw = Rotation.from_matrix(rotation_world_tip).as_quat()
    return np.concatenate((point_world, quaternion_xyzw)).astype(np.float64)


def _draw_grasp_axes(action: np.ndarray, bullet: Any, client_id: int) -> None:
    position = action[:3]
    rotation = np.asarray(
        bullet.getMatrixFromQuaternion(action[3:]), dtype=np.float64
    ).reshape(3, 3)
    for axis, color in zip(rotation.T, ((1, 0, 0), (0, 1, 0), (0, 0, 1))):
        bullet.addUserDebugLine(
            position,
            position + 0.08 * axis,
            lineColorRGB=color,
            lineWidth=3,
            lifeTime=0,
            physicsClientId=client_id,
        )


def _hold_gui(env: Any, bullet: Any) -> None:
    print("PyBullet 창을 유지합니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        while bullet.isConnected(env._client_id):
            bullet.stepSimulation(physicsClientId=env._client_id)
            time.sleep(env.time_step)
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        _validate_arguments(arguments)
        case_path = _resolve_case(arguments.case)
        requested_grasp_result = None
        if arguments.grasp_result is not None:
            requested_grasp_result = arguments.grasp_result.expanduser().resolve()
        icar_python = None
        if arguments.run_pipeline:
            icar_python = _resolve_icar_python(arguments.icar_python)
        xarm_urdf = _prepare_xarm_urdf(arguments)
        bullet, environment_class = _load_thinkgrasp(
            xarm_urdf, arguments.simulation_speed
        )

        # ThinkGrasp uses relative asset paths internally.
        os.chdir(THINKGRASP_ROOT)
        try:
            env = environment_class(gui=not arguments.no_gui)
            env.seed(arguments.seed)
            env.reset()
            scene_ok, case_instruction = env.add_object_push_from_file(str(case_path))
        except Exception as exc:
            raise SimulationError(
                f"ThinkGrasp 장면을 불러오지 못했습니다: {exc}"
            ) from exc
        if not scene_ok:
            raise SimulationError("장면 초기화 중 일부 물체가 안정화되지 않았습니다")
        instruction = arguments.instruction or case_instruction
        camera = env.agent_cams[arguments.camera_index]
        print("simulation robot: xarm7")

        capture_paths = _write_capture(
            env=env,
            camera=camera,
            prefix=arguments.prefix,
            case_path=case_path,
            seed=arguments.seed,
        )
        print("\n시뮬레이션 RGB-D 저장 완료")
        print(json.dumps({key: str(value) for key, value in capture_paths.items()}, indent=2))
        print(f"instruction: {instruction}")
        target_pixels = _visible_target_pixels(
            capture_paths["segmentation"],
            getattr(env, "target_obj_ids", ()),
        )
        print(f"ground-truth target visible pixels: {target_pixels}")
        if arguments.run_pipeline and instruction == case_instruction:
            if target_pixels == 0:
                raise SimulationError(
                    "case의 정답 target이 선택한 카메라에 완전히 가려져 있습니다. "
                    "다른 --camera-index 또는 더 단순한 --case를 사용하세요"
                )
            if target_pixels < 200:
                print(
                    "경고: target이 200픽셀 미만으로 작거나 많이 가려져 있어 "
                    "ICAR가 none을 반환할 수 있습니다.",
                    file=sys.stderr,
                )

        result_path: Optional[Path] = None
        if arguments.run_pipeline:
            assert icar_python is not None
            print(f"ICAR Python: {icar_python}")
            result_path = _run_affordgrasp_pipeline(
                arguments.prefix,
                instruction,
                icar_python,
            )
        elif requested_grasp_result is not None:
            result_path = requested_grasp_result

        if result_path is not None:
            grasp_payload = _read_grasp_result(result_path)
            maximum_approach_angle = arguments.max_approach_angle
            if maximum_approach_angle is None:
                maximum_approach_angle = 20.0
            selected = _select_simulation_grasp(
                grasp_payload,
                camera,
                bullet,
                env,
                maximum_approach_angle,
                require_reachable=arguments.execute,
            )
            action = _to_xarm_action(selected, camera, bullet)
            print(f"grasp result: {result_path}")
            print(
                "xarm7 action [x, y, z, qx, qy, qz, qw]: "
                f"{action.tolist()}"
            )
            if not arguments.no_gui:
                _draw_grasp_axes(action, bullet, env._client_id)
            if arguments.execute:
                reward, done = env.step(action)
                print(f"simulation reward={reward}, success={done}")

        if not arguments.no_hold:
            _hold_gui(env, bullet)
        return 0
    except SimulationError as exc:
        print(f"AffordGrasp Simulation 오류: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n시뮬레이션을 종료합니다.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
