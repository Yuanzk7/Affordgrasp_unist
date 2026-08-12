#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
사용법:
  인식 실행:   ./run_affordgrasp_pipeline.sh "사용자 작업 지시" 촬영_prefix
  촬영→로봇:  ./run_affordgrasp_pipeline.sh --execute \
                 --confirm MOVE_XARM7_IP_WITH_UNDERSCORES \
                 --acknowledge-cleared-workspace \
                 --acknowledge-estop-ready \
                 "사용자 작업 지시" 촬영_prefix
  단계 재실행: ./run_affordgrasp_pipeline.sh --stage icar "사용자 작업 지시" 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage localization 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage mask 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage grasp 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage robot-plan 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage robot-collision 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage robot-execute [승인 옵션] 촬영_prefix

--stage 재실행은 새로 촬영하지 않고 기존 captures/ 촬영본과
runs/<prefix>/json 결과를 사용해 해당 단계만 다시 실행한다.
--execute는 촬영부터 full-link 충돌 검증까지 새로 수행하고, 검증을 통과한
동일 plan을 xArm7에서 집기·후퇴·리프트까지 실행한다.
설정은 config.env에서 읽는다.
EOF
  exit 64
}

PIPELINE_STAGE=all
PIPELINE_STAGE_EXPLICIT=false
PIPELINE_EXECUTE=false
PIPELINE_CONFIRM=""
PIPELINE_ACK_WORKSPACE=false
PIPELINE_ACK_ESTOP=false
declare -a PIPELINE_POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --stage)
      [[ $# -ge 2 ]] || usage
      PIPELINE_STAGE=$2
      PIPELINE_STAGE_EXPLICIT=true
      shift 2
      ;;
    --execute)
      PIPELINE_EXECUTE=true
      shift
      ;;
    --confirm)
      [[ $# -ge 2 ]] || usage
      PIPELINE_CONFIRM=$2
      shift 2
      ;;
    --acknowledge-cleared-workspace)
      PIPELINE_ACK_WORKSPACE=true
      shift
      ;;
    --acknowledge-estop-ready)
      PIPELINE_ACK_ESTOP=true
      shift
      ;;
    -h | --help)
      usage
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PIPELINE_POSITIONAL+=("$1")
        shift
      done
      ;;
    -*)
      echo "알 수 없는 옵션: $1" >&2
      usage
      ;;
    *)
      PIPELINE_POSITIONAL+=("$1")
      shift
      ;;
  esac
done

set -- "${PIPELINE_POSITIONAL[@]}"

if $PIPELINE_EXECUTE && $PIPELINE_STAGE_EXPLICIT; then
  echo "--execute와 --stage는 함께 사용할 수 없습니다." >&2
  usage
fi
case $PIPELINE_STAGE in
  all | icar)
    [[ $# -eq 2 ]] || usage
    PIPELINE_INSTRUCTION=$1
    PIPELINE_PREFIX=$2
    ;;
  localization | mask | grasp | robot-plan | robot-collision | robot-execute)
    [[ $# -eq 1 ]] || usage
    PIPELINE_PREFIX=$1
    ;;
  *) usage ;;
esac

if [[ $PIPELINE_STAGE != robot-execute ]] && ! $PIPELINE_EXECUTE; then
  if [[ -n $PIPELINE_CONFIRM ]] || $PIPELINE_ACK_WORKSPACE || $PIPELINE_ACK_ESTOP; then
    echo "로봇 승인 옵션은 --execute 또는 --stage robot-execute에서만 사용합니다." >&2
    usage
  fi
fi

if [[ ! "$PIPELINE_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "촬영 prefix는 영문자/숫자로 시작하고 영문자, 숫자, '.', '_', '-'만 사용할 수 있습니다." >&2
  exit 64
fi

PIPELINE_PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PIPELINE_PROJECT_DIR"

# config.env가 있으면 모든 AFFORDGRASP_* 설정을 그 파일에서 읽는다.
if [[ -f "$PIPELINE_PROJECT_DIR/config.env" ]]; then
  # shellcheck source=config.env
  source "$PIPELINE_PROJECT_DIR/config.env"
  echo "설정 파일 적용: $PIPELINE_PROJECT_DIR/config.env"
fi

PIPELINE_PROVIDER=${AFFORDGRASP_PROVIDER:-gemini}
PIPELINE_MIN_CONFIDENCE=${AFFORDGRASP_MIN_CONFIDENCE:-0.70}
PIPELINE_OBJECT_THRESHOLD=${AFFORDGRASP_OBJECT_THRESHOLD:-0.01}
PIPELINE_OBJECT_TOP_K=${AFFORDGRASP_OBJECT_TOP_K:-10}
PIPELINE_SELECTION_METHOD=${AFFORDGRASP_SELECTION_METHOD:-gemini-top-k}
PIPELINE_SELECTOR_MODEL=${AFFORDGRASP_SELECTOR_MODEL:-gemini-3.5-flash-lite}
PIPELINE_SELECTOR_CONFIDENCE=${AFFORDGRASP_SELECTOR_CONFIDENCE:-0.55}
PIPELINE_SELECTOR_TIMEOUT=${AFFORDGRASP_SELECTOR_TIMEOUT:-60}
PIPELINE_PART_THRESHOLD=${AFFORDGRASP_PART_THRESHOLD:-0.01}
PIPELINE_DEVICE=${AFFORDGRASP_DEVICE:-cuda}
PIPELINE_VLPART_ROOT=${AFFORDGRASP_VLPART_ROOT:-$PIPELINE_PROJECT_DIR/VLPart}
PIPELINE_VLPART_WEIGHTS=${AFFORDGRASP_VLPART_WEIGHTS:-$PIPELINE_VLPART_ROOT/models/r50_pascalpart.pth}
PIPELINE_VLPART_ENV=${AFFORDGRASP_VLPART_ENV:-$PIPELINE_VLPART_ROOT/.conda}
PIPELINE_GRASP_DEPTH_SOURCE=${AFFORDGRASP_GRASP_DEPTH_SOURCE:-filtered}
PIPELINE_MAX_GRIPPER_WIDTH=${AFFORDGRASP_MAX_GRIPPER_WIDTH:-0.10}
PIPELINE_GRIPPER_HEIGHT=${AFFORDGRASP_GRIPPER_HEIGHT:-0.03}
PIPELINE_ANYGRASP_SDK=${AFFORDGRASP_ANYGRASP_SDK:-}
PIPELINE_ANYGRASP_CHECKPOINT=${AFFORDGRASP_ANYGRASP_CHECKPOINT:-}
PIPELINE_ANYGRASP_ENV=${AFFORDGRASP_ANYGRASP_ENV:-}
PIPELINE_OMP_NUM_THREADS=${AFFORDGRASP_OMP_NUM_THREADS:-16}
PIPELINE_ROBOT_IP=${AFFORDGRASP_ROBOT_IP:-192.168.1.216}
PIPELINE_ROBOT_CONFIG=${AFFORDGRASP_ROBOT_CONFIG:-$PIPELINE_PROJECT_DIR/robot_config.json}
PIPELINE_EYE_TO_HAND_CALIBRATION=${AFFORDGRASP_EYE_TO_HAND_CALIBRATION:-$PIPELINE_PROJECT_DIR/calibration/eye_to_hand.json}
PIPELINE_XARM_ROS2_ROOT=${AFFORDGRASP_XARM_ROS2_ROOT:-$PIPELINE_PROJECT_DIR/xarm_ros2}
PIPELINE_COLLISION_MAX_AGE=${AFFORDGRASP_COLLISION_VALIDATION_MAX_AGE_SECONDS:-300}

PIPELINE_CAPTURE_DIR=${AFFORDGRASP_CAPTURE_DIR:-$PIPELINE_PROJECT_DIR/captures/icar_d435}
PIPELINE_RUN_ROOT=${AFFORDGRASP_RUN_ROOT:-$PIPELINE_PROJECT_DIR/runs}
PIPELINE_RUN_DIR=$PIPELINE_RUN_ROOT/$PIPELINE_PREFIX
PIPELINE_JSON_DIR=$PIPELINE_RUN_DIR/json
PIPELINE_LOCALIZATION_DIR=$PIPELINE_RUN_DIR/object_localization
PIPELINE_MASK_DIR=$PIPELINE_RUN_DIR/affordance_mask
PIPELINE_GRASP_DIR=$PIPELINE_RUN_DIR/grasp
PIPELINE_ROBOT_DIR=$PIPELINE_RUN_DIR/robot
PIPELINE_RGB_IMAGE=$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_rgb.png
PIPELINE_DEPTH_IMAGE=$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_depth_${PIPELINE_GRASP_DEPTH_SOURCE}.png
PIPELINE_CAMERA_INFO=$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_camera.json
PIPELINE_AFFORDANCE_MASK=$PIPELINE_MASK_DIR/affordance_mask.png

if $PIPELINE_EXECUTE || [[ $PIPELINE_STAGE == robot-execute ]]; then
  PIPELINE_EXPECTED_CONFIRM="MOVE_XARM7_${PIPELINE_ROBOT_IP//./_}"
  if [[ $PIPELINE_CONFIRM != "$PIPELINE_EXPECTED_CONFIRM" ]]; then
    echo "--confirm은 정확히 $PIPELINE_EXPECTED_CONFIRM 이어야 합니다." >&2
    exit 64
  fi
  if ! $PIPELINE_ACK_WORKSPACE; then
    echo "--acknowledge-cleared-workspace가 필요합니다." >&2
    exit 64
  fi
  if ! $PIPELINE_ACK_ESTOP; then
    echo "--acknowledge-estop-ready가 필요합니다." >&2
    exit 64
  fi
fi

case $PIPELINE_GRASP_DEPTH_SOURCE in
  raw | filtered) ;;
  *)
    echo "AFFORDGRASP_GRASP_DEPTH_SOURCE는 raw 또는 filtered여야 합니다." >&2
    exit 64
    ;;
esac

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "$2: $1" >&2
    exit 2
  fi
}

require_vlpart() {
  if [[ ! -d "$PIPELINE_VLPART_ROOT" ]]; then
    echo "VLPart 폴더가 없습니다: $PIPELINE_VLPART_ROOT" >&2
    exit 2
  fi
  require_file "$PIPELINE_VLPART_WEIGHTS" "VLPart checkpoint가 없습니다"
  if [[ ! -d "$PIPELINE_VLPART_ENV" ]]; then
    echo "VLPart Conda 환경이 없습니다: $PIPELINE_VLPART_ENV" >&2
    exit 2
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda 명령을 찾을 수 없습니다." >&2
    exit 2
  fi
}

require_robot_prerequisites() {
  require_file "$PIPELINE_EYE_TO_HAND_CALIBRATION" \
    "통과한 eye-to-hand 캘리브레이션이 없습니다"
  require_file "$PIPELINE_ROBOT_CONFIG" \
    "로봇 설정 파일이 없습니다"
  if [[ ! -d $PIPELINE_XARM_ROS2_ROOT/xarm_description ]]; then
    echo "공식 xarm_ros2 모델이 없습니다: $PIPELINE_XARM_ROS2_ROOT" >&2
    exit 2
  fi
  if [[ -z $PIPELINE_ANYGRASP_ENV || ! -d $PIPELINE_ANYGRASP_ENV ]]; then
    echo "로봇 단계에 사용할 Conda 환경이 없습니다: $PIPELINE_ANYGRASP_ENV" >&2
    exit 2
  fi
}

run_capture() {
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  python -m affordgrasp_icar \
    --camera \
    --capture-only \
    --capture-dir "$PIPELINE_CAPTURE_DIR" \
    --capture-prefix "$PIPELINE_PREFIX" \
    --json-dir "$PIPELINE_JSON_DIR"
}

run_icar() {
  require_file "$PIPELINE_RGB_IMAGE" "촬영된 RGB 이미지가 없습니다"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  python -m affordgrasp_icar \
    --provider "$PIPELINE_PROVIDER" \
    --image "$PIPELINE_RGB_IMAGE" \
    --instruction "$PIPELINE_INSTRUCTION" \
    --min-confidence "$PIPELINE_MIN_CONFIDENCE" \
    --json-dir "$PIPELINE_JSON_DIR"
}

run_localization() {
  require_vlpart
  require_file "$PIPELINE_RGB_IMAGE" "촬영된 RGB 이미지가 없습니다"
  require_file "$PIPELINE_JSON_DIR/grounding_request.json" \
    "ICAR grounding request가 없습니다"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  conda run --prefix "$PIPELINE_VLPART_ENV" \
  python -m affordgrasp_icar.grounding.object_localization_cli \
    --image "$PIPELINE_RGB_IMAGE" \
    --request "$PIPELINE_JSON_DIR/grounding_request.json" \
    --output-dir "$PIPELINE_LOCALIZATION_DIR" \
    --json-dir "$PIPELINE_JSON_DIR" \
    --vlpart-root "$PIPELINE_VLPART_ROOT" \
    --weights "$PIPELINE_VLPART_WEIGHTS" \
    --confidence-threshold "$PIPELINE_OBJECT_THRESHOLD" \
    --top-k "$PIPELINE_OBJECT_TOP_K" \
    --selection-method "$PIPELINE_SELECTION_METHOD" \
    --selector-model "$PIPELINE_SELECTOR_MODEL" \
    --selector-confidence-threshold "$PIPELINE_SELECTOR_CONFIDENCE" \
    --selector-timeout "$PIPELINE_SELECTOR_TIMEOUT" \
    --device "$PIPELINE_DEVICE"
}

run_mask() {
  require_vlpart
  require_file "$PIPELINE_JSON_DIR/object_localization.json" \
    "Object Localization 결과가 없습니다"
  require_file "$PIPELINE_JSON_DIR/grounding_request.json" \
    "ICAR grounding request가 없습니다"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  conda run --prefix "$PIPELINE_VLPART_ENV" \
  python -m affordgrasp_icar.grounding.affordance_mask_cli \
    --localization "$PIPELINE_JSON_DIR/object_localization.json" \
    --request "$PIPELINE_JSON_DIR/grounding_request.json" \
    --output-dir "$PIPELINE_MASK_DIR" \
    --json-dir "$PIPELINE_JSON_DIR" \
    --vlpart-root "$PIPELINE_VLPART_ROOT" \
    --weights "$PIPELINE_VLPART_WEIGHTS" \
    --confidence-threshold "$PIPELINE_PART_THRESHOLD" \
    --input-size 224 \
    --device "$PIPELINE_DEVICE"
}

run_grasp() {
  require_file "$PIPELINE_RGB_IMAGE" "촬영된 RGB 이미지가 없습니다"
  require_file "$PIPELINE_DEPTH_IMAGE" "촬영된 depth 이미지가 없습니다"
  require_file "$PIPELINE_CAMERA_INFO" "카메라 내부 파라미터 JSON이 없습니다"
  require_file "$PIPELINE_AFFORDANCE_MASK" "Affordance mask가 없습니다"

  if [[ -n $PIPELINE_ANYGRASP_SDK && ! -d $PIPELINE_ANYGRASP_SDK ]]; then
    echo "AnyGrasp SDK 폴더가 없습니다: $PIPELINE_ANYGRASP_SDK" >&2
    exit 2
  fi
  if [[ -n $PIPELINE_ANYGRASP_CHECKPOINT ]]; then
    require_file "$PIPELINE_ANYGRASP_CHECKPOINT" \
      "AnyGrasp checkpoint가 없습니다"
  fi

  local grasp_mpl_cache=$PIPELINE_GRASP_DIR/.matplotlib
  local grasp_xdg_cache=$PIPELINE_GRASP_DIR/.cache
  mkdir -p "$grasp_mpl_cache" "$grasp_xdg_cache"

  local -a grasp_command=(
    python -m affordgrasp_icar.grasp.grasp_pose_generation
    --rgb "$PIPELINE_RGB_IMAGE"
    --depth "$PIPELINE_DEPTH_IMAGE"
    --camera "$PIPELINE_CAMERA_INFO"
    --mask "$PIPELINE_AFFORDANCE_MASK"
    --output-dir "$PIPELINE_GRASP_DIR"
    --max-gripper-width "$PIPELINE_MAX_GRIPPER_WIDTH"
    --gripper-height "$PIPELINE_GRIPPER_HEIGHT"
  )

  if [[ -z $PIPELINE_ANYGRASP_SDK && -z $PIPELINE_ANYGRASP_CHECKPOINT ]]; then
    echo "AnyGrasp에는 AFFORDGRASP_ANYGRASP_SDK 또는 AFFORDGRASP_ANYGRASP_CHECKPOINT가 필요합니다." >&2
    exit 2
  fi
  if [[ -n $PIPELINE_ANYGRASP_SDK ]]; then
    grasp_command+=(--anygrasp-sdk "$PIPELINE_ANYGRASP_SDK")
  fi
  if [[ -n $PIPELINE_ANYGRASP_CHECKPOINT ]]; then
    grasp_command+=(--checkpoint "$PIPELINE_ANYGRASP_CHECKPOINT")
  fi

  if [[ -n $PIPELINE_ANYGRASP_ENV ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "conda 명령을 찾을 수 없습니다." >&2
      exit 2
    fi
    if [[ ! -d $PIPELINE_ANYGRASP_ENV ]]; then
      echo "Grasp Conda 환경이 없습니다: $PIPELINE_ANYGRASP_ENV" >&2
      exit 2
    fi
    OMP_NUM_THREADS="$PIPELINE_OMP_NUM_THREADS" \
    MPLCONFIGDIR="$grasp_mpl_cache" \
    XDG_CACHE_HOME="$grasp_xdg_cache" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
    conda run --prefix "$PIPELINE_ANYGRASP_ENV" "${grasp_command[@]}"
  else
    OMP_NUM_THREADS="$PIPELINE_OMP_NUM_THREADS" \
    MPLCONFIGDIR="$grasp_mpl_cache" \
    XDG_CACHE_HOME="$grasp_xdg_cache" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. "${grasp_command[@]}"
  fi
}

run_robot_plan() {
  require_file "$PIPELINE_GRASP_DIR/grasp_pose_result.json" \
    "AnyGrasp 결과가 없습니다"
  require_file "$PIPELINE_EYE_TO_HAND_CALIBRATION" \
    "통과한 eye-to-hand 캘리브레이션이 없습니다"
  require_file "$PIPELINE_ROBOT_CONFIG" \
    "로봇 설정 파일이 없습니다"
  if [[ -z $PIPELINE_ANYGRASP_ENV || ! -d $PIPELINE_ANYGRASP_ENV ]]; then
    echo "Robot plan에 사용할 Conda 환경이 없습니다: $PIPELINE_ANYGRASP_ENV" >&2
    exit 2
  fi
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  conda run --prefix "$PIPELINE_ANYGRASP_ENV" \
  python -m affordgrasp_icar.robot.xarm_grasp_execution \
    --grasp-result "$PIPELINE_GRASP_DIR/grasp_pose_result.json" \
    --calibration "$PIPELINE_EYE_TO_HAND_CALIBRATION" \
    --robot-config "$PIPELINE_ROBOT_CONFIG" \
    --output "$PIPELINE_ROBOT_DIR/robot_plan.json"
}

run_robot_collision() {
  require_file "$PIPELINE_ROBOT_DIR/robot_plan.json" \
    "xArm7 robot plan이 없습니다"
  require_file "$PIPELINE_ROBOT_CONFIG" \
    "로봇 설정 파일이 없습니다"
  if [[ ! -d $PIPELINE_XARM_ROS2_ROOT/xarm_description ]]; then
    echo "공식 xarm_ros2 모델이 없습니다: $PIPELINE_XARM_ROS2_ROOT" >&2
    exit 2
  fi
  if [[ -z $PIPELINE_ANYGRASP_ENV || ! -d $PIPELINE_ANYGRASP_ENV ]]; then
    echo "충돌 검증에 사용할 Conda 환경이 없습니다: $PIPELINE_ANYGRASP_ENV" >&2
    exit 2
  fi
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  conda run --prefix "$PIPELINE_ANYGRASP_ENV" \
  python -m affordgrasp_icar.robot.full_collision_validation \
    --plan "$PIPELINE_ROBOT_DIR/robot_plan.json" \
    --robot-config "$PIPELINE_ROBOT_CONFIG" \
    --xarm-ros2-root "$PIPELINE_XARM_ROS2_ROOT" \
    --output "$PIPELINE_ROBOT_DIR/collision_validation.json"
}

run_robot_execute() {
  require_file "$PIPELINE_ROBOT_DIR/robot_plan.json" \
    "실행할 xArm7 robot plan이 없습니다"
  require_file "$PIPELINE_ROBOT_DIR/collision_validation.json" \
    "최신 전체 링크 충돌 검증 결과가 없습니다"
  require_file "$PIPELINE_ROBOT_CONFIG" \
    "로봇 설정 파일이 없습니다"
  if [[ -z $PIPELINE_ANYGRASP_ENV || ! -d $PIPELINE_ANYGRASP_ENV ]]; then
    echo "Robot 실행에 사용할 Conda 환경이 없습니다: $PIPELINE_ANYGRASP_ENV" >&2
    exit 2
  fi

  local -a robot_command=(
    python -m affordgrasp_icar.robot.xarm_grasp_execution
    --plan "$PIPELINE_ROBOT_DIR/robot_plan.json"
    --collision-validation "$PIPELINE_ROBOT_DIR/collision_validation.json"
    --robot-config "$PIPELINE_ROBOT_CONFIG"
    --output "$PIPELINE_ROBOT_DIR/robot_plan.json"
    --execute
    --maximum-validation-age-seconds "$PIPELINE_COLLISION_MAX_AGE"
    --confirm "$PIPELINE_CONFIRM"
  )
  if $PIPELINE_ACK_WORKSPACE; then
    robot_command+=(--acknowledge-cleared-workspace)
  fi
  if $PIPELINE_ACK_ESTOP; then
    robot_command+=(--acknowledge-estop-ready)
  fi

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
  conda run --prefix "$PIPELINE_ANYGRASP_ENV" "${robot_command[@]}"
}

mkdir -p \
  "$PIPELINE_CAPTURE_DIR" \
  "$PIPELINE_JSON_DIR" \
  "$PIPELINE_LOCALIZATION_DIR" \
  "$PIPELINE_MASK_DIR" \
  "$PIPELINE_GRASP_DIR" \
  "$PIPELINE_ROBOT_DIR"

case $PIPELINE_STAGE in
  all)
    require_vlpart
    if $PIPELINE_EXECUTE; then
      require_robot_prerequisites
    fi
    run_capture
    run_icar
    run_localization
    run_mask
    run_grasp
    if $PIPELINE_EXECUTE; then
      run_robot_plan
      run_robot_collision
      run_robot_execute
    fi
    printf 'AffordGrasp 전체 실행 완료\n'
    printf 'RGB: %s\n' "$PIPELINE_RGB_IMAGE"
    printf 'Raw depth: %s\n' "$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_depth_raw.png"
    printf 'Filtered depth: %s\n' "$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_depth_filtered.png"
    printf 'Camera info: %s\n' "$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_camera.json"
    printf 'JSON: %s\n' "$PIPELINE_JSON_DIR"
    printf 'Top-k candidates: %s\n' "$PIPELINE_LOCALIZATION_DIR/top_k_candidates.png"
    printf 'Selected object: %s\n' "$PIPELINE_LOCALIZATION_DIR/selected_object_overlay.png"
    printf 'Masked object: %s\n' "$PIPELINE_LOCALIZATION_DIR/masked_object.png"
    printf 'Affordance mask: %s\n' "$PIPELINE_MASK_DIR/affordance_mask.png"
    printf 'Overlay: %s\n' "$PIPELINE_MASK_DIR/affordance_overlay.png"
    printf 'Grasp backend: anygrasp\n'
    printf 'Grasp result: %s\n' "$PIPELINE_GRASP_DIR/grasp_pose_result.json"
    printf 'Grasp 3D: %s\n' "$PIPELINE_GRASP_DIR/grasp_pose_3d.png"
    printf 'Scene point cloud: %s\n' "$PIPELINE_GRASP_DIR/scene_point_cloud.ply"
    printf 'Affordance point cloud: %s\n' "$PIPELINE_GRASP_DIR/affordance_point_cloud.ply"
    if $PIPELINE_EXECUTE; then
      printf 'Robot plan: %s\n' "$PIPELINE_ROBOT_DIR/robot_plan.json"
      printf 'Collision validation: %s\n' "$PIPELINE_ROBOT_DIR/collision_validation.json"
      printf 'Robot execution: %s\n' "$PIPELINE_ROBOT_DIR/robot_plan_execution.json"
    fi
    ;;
  icar)
    run_icar
    printf 'ICAR 재실행 완료: %s\n' "$PIPELINE_JSON_DIR"
    ;;
  localization)
    run_localization
    printf 'Object Localization 재실행 완료\n'
    printf 'Top-k candidates: %s\n' "$PIPELINE_LOCALIZATION_DIR/top_k_candidates.png"
    printf 'Selected object: %s\n' "$PIPELINE_LOCALIZATION_DIR/selected_object_overlay.png"
    printf 'Masked object: %s\n' "$PIPELINE_LOCALIZATION_DIR/masked_object.png"
    ;;
  mask)
    run_mask
    printf 'Affordance Mask 재실행 완료\n'
    printf 'Affordance mask: %s\n' "$PIPELINE_MASK_DIR/affordance_mask.png"
    printf 'Overlay: %s\n' "$PIPELINE_MASK_DIR/affordance_overlay.png"
    ;;
  grasp)
    run_grasp
    printf 'Grasp Pose Generation 재실행 완료\n'
    printf 'Backend: anygrasp\n'
    printf 'Result: %s\n' "$PIPELINE_GRASP_DIR/grasp_pose_result.json"
    printf '3D visualization: %s\n' "$PIPELINE_GRASP_DIR/grasp_pose_3d.png"
    printf 'Scene point cloud: %s\n' "$PIPELINE_GRASP_DIR/scene_point_cloud.ply"
    printf 'Affordance point cloud: %s\n' "$PIPELINE_GRASP_DIR/affordance_point_cloud.ply"
    ;;
  robot-plan)
    run_robot_plan
    printf 'xArm7 robot plan 생성 완료\n'
    printf 'Plan: %s\n' "$PIPELINE_ROBOT_DIR/robot_plan.json"
    printf '이 단계는 로봇에 연결하거나 이동 명령을 보내지 않습니다.\n'
    ;;
  robot-collision)
    run_robot_collision
    printf 'xArm7 전체 링크·테이블 충돌 검증 완료\n'
    printf 'Result: %s\n' "$PIPELINE_ROBOT_DIR/collision_validation.json"
    printf '이 단계는 읽기 전용 IK만 조회하며 로봇을 움직이지 않습니다.\n'
    ;;
  robot-execute)
    run_robot_execute
    printf '검증된 xArm7 plan 실행 완료\n'
    printf 'Execution: %s\n' "$PIPELINE_ROBOT_DIR/robot_plan_execution.json"
    ;;
esac
