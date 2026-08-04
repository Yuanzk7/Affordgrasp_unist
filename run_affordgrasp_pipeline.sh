#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "사용법: $0 \"사용자 작업 지시\" 촬영_prefix" >&2
  exit 64
fi

PIPELINE_INSTRUCTION=$1
PIPELINE_PREFIX=$2

if [[ ! "$PIPELINE_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "촬영 prefix는 영문자/숫자로 시작하고 영문자, 숫자, '.', '_', '-'만 사용할 수 있습니다." >&2
  exit 64
fi

PIPELINE_PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PIPELINE_PROJECT_DIR"

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

if [[ ! -d "$PIPELINE_VLPART_ROOT" ]]; then
  echo "VLPart 폴더가 없습니다: $PIPELINE_VLPART_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PIPELINE_VLPART_WEIGHTS" ]]; then
  echo "VLPart checkpoint가 없습니다: $PIPELINE_VLPART_WEIGHTS" >&2
  exit 2
fi
if [[ ! -d "$PIPELINE_VLPART_ENV" ]]; then
  echo "VLPart Conda 환경이 없습니다: $PIPELINE_VLPART_ENV" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda 명령을 찾을 수 없습니다." >&2
  exit 2
fi

mkdir -p \
  "$PIPELINE_CAPTURE_DIR" \
  "$PIPELINE_JSON_DIR" \
  "$PIPELINE_LOCALIZATION_DIR" \
  "$PIPELINE_MASK_DIR"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
python -m affordgrasp_icar \
  --camera \
  --capture-only \
  --capture-dir "$PIPELINE_CAPTURE_DIR" \
  --capture-prefix "$PIPELINE_PREFIX" \
  --json-dir "$PIPELINE_JSON_DIR"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
python -m affordgrasp_icar \
  --provider "$PIPELINE_PROVIDER" \
  --image "$PIPELINE_RGB_IMAGE" \
  --instruction "$PIPELINE_INSTRUCTION" \
  --min-confidence "$PIPELINE_MIN_CONFIDENCE" \
  --json-dir "$PIPELINE_JSON_DIR"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix "$PIPELINE_VLPART_ENV" \
python -m affordgrasp_icar.object_localization_cli \
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

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix "$PIPELINE_VLPART_ENV" \
python -m affordgrasp_icar.affordance_mask_cli \
  --localization "$PIPELINE_JSON_DIR/object_localization.json" \
  --request "$PIPELINE_JSON_DIR/grounding_request.json" \
  --output-dir "$PIPELINE_MASK_DIR" \
  --json-dir "$PIPELINE_JSON_DIR" \
  --vlpart-root "$PIPELINE_VLPART_ROOT" \
  --weights "$PIPELINE_VLPART_WEIGHTS" \
  --confidence-threshold "$PIPELINE_PART_THRESHOLD" \
  --input-size 224 \
  --device "$PIPELINE_DEVICE"

printf 'AffordGrasp 실행 완료\n'
printf 'RGB: %s\n' "$PIPELINE_RGB_IMAGE"
printf 'JSON: %s\n' "$PIPELINE_JSON_DIR"
printf 'Top-k candidates: %s\n' "$PIPELINE_LOCALIZATION_DIR/top_k_candidates.png"
printf 'Selected object: %s\n' "$PIPELINE_LOCALIZATION_DIR/selected_object_overlay.png"
printf 'Masked object: %s\n' "$PIPELINE_LOCALIZATION_DIR/masked_object.png"
printf 'Affordance mask: %s\n' "$PIPELINE_MASK_DIR/affordance_mask.png"
printf 'Overlay: %s\n' "$PIPELINE_MASK_DIR/affordance_overlay.png"
