# AffordGrasp 실제 실행 가이드

## 1. 실행 결과

이 프로젝트는 RealSense D435로 새 RGB-D 장면을 촬영하고 다음 단계를 실제로
실행한다.

```text
D435 촬영
  → ICAR JSON 생성
  → VLPart top-k 후보 생성
  → Gemini가 ICAR 대상과 일치하는 후보 재선택
  → Object Localization 및 masked_object.png 생성
  → Affordance Mask 및 overlay 생성
```

Depth point cloud, AnyGrasp, 로봇 grasp 실행은 아직 포함하지 않는다.

## 2. 사전 준비

Gemini를 사용할 경우 API key를 설정한다.

```bash
export GEMINI_API_KEY="본인의_Gemini_API_KEY"
export AFFORDGRASP_PROVIDER=gemini
```

OpenAI를 사용할 경우:

```bash
export OPENAI_API_KEY="본인의_OpenAI_API_KEY"
export AFFORDGRASP_PROVIDER=openai
```

D435의 원본 metric depth를 저장하기 위해 실행하는 Python 환경에
`pyrealsense2`가 필요하다.

```bash
python -m pip install pyrealsense2
```

VLPart runtime은 다음 위치에 있어야 한다.

```text
./VLPart/.conda
./VLPart/configs/pascal_part/r50_pascalpart.yaml
./VLPart/models/r50_pascalpart.pth
```

## 3. 전체 파이프라인 실행

프로젝트 폴더에서 작업 지시와 촬영 prefix를 전달한다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar

./run_affordgrasp_pipeline.sh \
  "I need to pick up the pliers safely." \
  01_pliers
```

스크립트는 실행할 때마다 D435로 새 장면을 촬영한다. 동일한 prefix를 다시
사용하면 해당 촬영 이미지와 실행 결과를 덮어쓸 수 있으므로 장면마다 다른
prefix를 사용한다.

## 4. 생성 파일

위 명령은 다음 파일을 실제로 생성한다.

```text
captures/icar_d435/
├── 01_pliers_rgb.png
├── 01_pliers_depth_raw.png
├── 01_pliers_depth_filtered.png
├── 01_pliers_depth_preview.png
└── 01_pliers_camera.json

runs/01_pliers/
├── json/
│   ├── capture_result.json
│   ├── icar_result.json
│   ├── grounding_request.json
│   ├── object_localization.json
│   └── affordance_mask.json
├── object_localization/
│   ├── top_k_candidates.png
│   ├── selected_object_overlay.png
│   └── masked_object.png
└── affordance_mask/
    ├── affordance_mask.png
    └── affordance_overlay.png
```

JSON은 장면별 `json/` 폴더 한 곳에 모이고, 이미지는 각 처리 단계 폴더에
저장된다.

- `depth_raw.png`: RGB 픽셀 좌표에 정렬된 무손실 `uint16` Z16 depth
- `depth_filtered.png`: spatial/temporal 및 작은 틈 보간 후처리 depth
- `depth_preview.png`: 사람이 확인하는 컬러 이미지이며 3D 계산에 사용하지 않음
- `camera.json`: depth scale, `fx`, `fy`, `cx`, `cy`, 유효·보간 depth 비율

미터 단위 거리는 `depth_raw_value * depth_scale_meters_per_unit`로 계산한다.
Point cloud와 AnyGrasp에는 preview가 아니라 raw 또는 filtered depth와
`camera.json`을 사용해야 한다.

## 5. 결과 확인

다음 파일을 직접 열어 실제 물체와 결과가 일치하는지 확인한다.

1. `01_pliers_rgb.png`: 대상 물체 전체와 잡을 part가 보이는지 확인
2. `01_pliers_depth_preview.png`: 대상 영역의 검은 depth hole을 확인
3. `01_pliers_camera.json`: `filtered_valid_ratio`와 intrinsics를 확인
4. `top_k_candidates.png`: 대상 물체가 번호 후보 중 하나에 포함됐는지 확인
5. `selected_object_overlay.png`: Gemini가 올바른 번호를 골랐는지 확인
6. `masked_object.png`: 선택된 bounding box 내부만 남았는지 확인
7. `affordance_overlay.png`: cyan 영역이 ICAR가 선택한 part인지 확인
8. `grounding_request.json`: object, part, affordance가 실제 장면과 일치하는지 확인

`object_localization.json`에는 전체 후보의 번호, bbox, VLPart 점수와 Gemini가
선택한 번호, confidence, 선택 이유가 함께 기록된다.

JSON 내용은 다음처럼 확인할 수 있다.

```bash
python -m json.tool ./runs/01_pliers/json/grounding_request.json
python -m json.tool ./runs/01_pliers/json/object_localization.json
python -m json.tool ./runs/01_pliers/json/affordance_mask.json
```

## 6. 실행 설정 변경

기본값:

```text
ICAR 최소 confidence:       0.70
Object 후보 최소 점수:      0.01
Object 후보 수:             10
Object 선택 방식:           gemini-top-k
Gemini 선택 confidence:     0.55
Affordance Mask 점수:       0.01
VLPart 장치:                cpu
```

환경변수로 변경할 수 있다.

```bash
export AFFORDGRASP_MIN_CONFIDENCE=0.60
export AFFORDGRASP_OBJECT_THRESHOLD=0.01
export AFFORDGRASP_OBJECT_TOP_K=15
export AFFORDGRASP_SELECTOR_CONFIDENCE=0.60
export AFFORDGRASP_PART_THRESHOLD=0.01
export AFFORDGRASP_DEVICE=cuda
```

대상이 `top_k_candidates.png`에 아예 없다면 `AFFORDGRASP_OBJECT_TOP_K`를
15~20으로 올린다. 후보 threshold를 지나치게 낮추면 배경 후보가 많아지므로
기본값 0.01을 먼저 사용한다. API 없이 기존 방식으로 실행해야 할 때만
`AFFORDGRASP_SELECTION_METHOD=vlpart-score`를 사용한다. 이 방식은 최고 VLPart
점수 하나를 선택하므로 다물체 장면에서는 권장하지 않는다.

## 7. 단계별 수동 실행

전체 스크립트가 특정 단계에서 중단된 경우, 생성된 prefix 경로를 사용해 그
단계부터 다시 실행할 수 있다.

### Object Localization 재실행

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix ./VLPart/.conda \
python -m affordgrasp_icar.object_localization_cli \
  --image ./captures/icar_d435/01_pliers_rgb.png \
  --request ./runs/01_pliers/json/grounding_request.json \
  --output-dir ./runs/01_pliers/object_localization \
  --json-dir ./runs/01_pliers/json \
  --vlpart-root ./VLPart \
  --weights ./VLPart/models/r50_pascalpart.pth \
  --confidence-threshold 0.01 \
  --top-k 10 \
  --selection-method gemini-top-k \
  --selector-model gemini-3.6-flash \
  --selector-confidence-threshold 0.55 \
  --device cpu
```

### Affordance Mask 재실행

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix ./VLPart/.conda \
python -m affordgrasp_icar.affordance_mask_cli \
  --localization ./runs/01_pliers/json/object_localization.json \
  --request ./runs/01_pliers/json/grounding_request.json \
  --output-dir ./runs/01_pliers/affordance_mask \
  --json-dir ./runs/01_pliers/json \
  --vlpart-root ./VLPart \
  --weights ./VLPart/models/r50_pascalpart.pth \
  --confidence-threshold 0.01 \
  --input-size 224 \
  --device cpu
```

## 8. 오류 처리

- `grounding_request.json`이 없으면 ICAR가 안전 게이트를 통과했는지 확인한다.
- `found no object`이면 object 이름, 물체 크기, 가림 상태를 확인한다.
- `Gemini found no ... among top-k`이면 후보 이미지에 대상이 있는지 확인하고
  `--top-k`를 15~20으로 올린다.
- `Gemini top-k confidence ... below`이면 선택이 모호한 상태이므로 후보 이미지를
  확인하고 대상이 더 선명하게 보이도록 다시 촬영한다.
- `found no part`이면 `part_name`이 이미지에서 실제로 보이는지 확인한다.
- mask가 다른 물체에 생성되면 Object Localization 결과부터 다시 촬영·실행한다.
- `num_batches_tracked` 메시지는 checkpoint 호환 경고이며 추론이 완료됐다면 치명적인 오류가 아니다.

## 9. 참고

- [AffordGrasp 논문](https://arxiv.org/abs/2503.00778)
- [AffordGrasp Visual Affordance Grounding](https://arxiv.org/html/2503.00778v1#S3.SS2)
- [VLPart 공식 저장소](https://github.com/facebookresearch/VLPart)
