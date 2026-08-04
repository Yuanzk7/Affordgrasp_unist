#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
사용법:
  전체 실행:   ./run_affordgrasp_pipeline.sh "사용자 작업 지시" 촬영_prefix
  단계 재실행: ./run_affordgrasp_pipeline.sh --stage icar "사용자 작업 지시" 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage localization 촬영_prefix
               ./run_affordgrasp_pipeline.sh --stage mask 촬영_prefix

--stage 재실행은 새로 촬영하지 않고 기존 captures/ 촬영본과
runs/<prefix>/json 결과를 사용해 해당 단계만 다시 실행한다.
설정은 config.env에서 읽는다.
EOF
  exit 64
}

PIPELINE_STAGE=all
if [[ ${1:-} == "--stage" ]]; then
  [[ $# -ge 2 ]] || usage
  PIPELINE_STAGE=$2
  shift 2
fi

case $PIPELINE_STAGE in
  all | icar)
    [[ $# -eq 2 ]] || usage
    PIPELINE_INSTRUCTION=$1
    PIPELINE_PREFIX=$2
    ;;
  localization | mask)
    [[ $# -eq 1 ]] || usage
    PIPELINE_PREFIX=$1
    ;;
  *) usage ;;
esac

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
PIPELINE_SELECTOR_MODEL=${AFFORDGRASP_SELECTOR_MODEL:-gemini-3.6-flash}
PIPELINE_SELECTOR_CONFIDENCE=${AFFORDGRASP_SELECTOR_CONFIDENCE:-0.55}
PIPELINE_SELECTOR_TIMEOUT=${AFFORDGRASP_SELECTOR_TIMEOUT:-60}
PIPELINE_PART_THRESHOLD=${AFFORDGRASP_PART_THRESHOLD:-0.01}
PIPELINE_DEVICE=${AFFORDGRASP_DEVICE:-cpu}
PIPELINE_VLPART_ROOT=${AFFORDGRASP_VLPART_ROOT:-$PIPELINE_PROJECT_DIR/VLPart}
PIPELINE_VLPART_WEIGHTS=${AFFORDGRASP_VLPART_WEIGHTS:-$PIPELINE_VLPART_ROOT/models/r50_pascalpart.pth}
PIPELINE_VLPART_ENV=${AFFORDGRASP_VLPART_ENV:-$PIPELINE_VLPART_ROOT/.conda}

PIPELINE_CAPTURE_DIR=$PIPELINE_PROJECT_DIR/captures/icar_d435
PIPELINE_RUN_DIR=$PIPELINE_PROJECT_DIR/runs/$PIPELINE_PREFIX
PIPELINE_JSON_DIR=$PIPELINE_RUN_DIR/json
PIPELINE_LOCALIZATION_DIR=$PIPELINE_RUN_DIR/object_localization
PIPELINE_MASK_DIR=$PIPELINE_RUN_DIR/affordance_mask
PIPELINE_RGB_IMAGE=$PIPELINE_CAPTURE_DIR/${PIPELINE_PREFIX}_rgb.png

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

mkdir -p \
  "$PIPELINE_CAPTURE_DIR" \
  "$PIPELINE_JSON_DIR" \
  "$PIPELINE_LOCALIZATION_DIR" \
  "$PIPELINE_MASK_DIR"

case $PIPELINE_STAGE in
  all)
    require_vlpart
    run_capture
    run_icar
    run_localization
    run_mask
    printf 'AffordGrasp 실행 완료\n'
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
esac
