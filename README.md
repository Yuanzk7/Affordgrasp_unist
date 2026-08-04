# AffordGrasp 파이프라인

[AffordGrasp](https://arxiv.org/abs/2503.00778) 구현. RealSense D435로 RGB-D
장면을 촬영하고 다음을 실행한다.

```text
D435 촬영 → ICAR (VLM 추론) → Object Localization (VLPart + Gemini top-k 재선택)
→ Affordance Mask
```

AnyGrasp grasp 실행은 아직 메인 파이프라인에 없다 (`grasp/`는 번들 샘플 데모).

## 코드 구조

```text
affordgrasp_icar/
├── camera.py       공용: D435 RGB-D 촬영
├── icar/           1단계 In-Context Affordance Reasoning
├── grounding/      2단계 Visual Affordance Grounding
└── grasp/          3단계 Grasp Pose Generation
```

## 준비

```bash
export GEMINI_API_KEY="..."   # 또는 OPENAI_API_KEY (provider는 config.env에서 선택)
python -m pip install pyrealsense2
```

VLPart runtime 위치:

```text
./VLPart/.conda
./VLPart/configs/pascal_part/r50_pascalpart.yaml
./VLPart/models/r50_pascalpart.pth
```

## 실행

```bash
./run_affordgrasp_pipeline.sh "I need to pick up the pliers safely." 01_pliers
```

실행마다 새 장면을 촬영한다. 같은 prefix는 기존 결과를 덮어쓴다.

단계별 재실행 (기존 촬영본·JSON 사용, 경로는 prefix에서 자동 생성):

```bash
./run_affordgrasp_pipeline.sh --stage icar "새 지시" 01_pliers
./run_affordgrasp_pipeline.sh --stage localization 01_pliers
./run_affordgrasp_pipeline.sh --stage mask 01_pliers
```

## 설정

모든 실행 설정은 `config.env` 한 파일에서 관리한다 (항목 설명은 파일 주석
참조). API 키는 이 파일에 넣지 말고 셸 환경변수나
`affordgrasp_icar/local_config.py`에 둔다.

- 대상이 `top_k_candidates.png`에 없으면 `AFFORDGRASP_OBJECT_TOP_K`를 15~20으로 올린다.
- API 없이 실행할 때만 `AFFORDGRASP_SELECTION_METHOD=vlpart-score` (다물체 장면 비권장).

## 출력

```text
captures/icar_d435/<prefix>_{rgb,depth_raw,depth_filtered,depth_preview}.png, <prefix>_camera.json
runs/<prefix>/
├── json/                  capture·icar·grounding·localization·mask 결과 JSON
├── object_localization/   top_k_candidates.png, selected_object_overlay.png, masked_object.png
└── affordance_mask/       affordance_mask.png, affordance_overlay.png
```

- 3D 계산에는 preview가 아닌 `depth_raw`/`depth_filtered`(`uint16`)와 `camera.json`을 사용한다.
  미터 거리 = `depth_value * depth_scale_meters_per_unit`
- 결과 검증: `top_k_candidates.png`에 대상 포함 여부 → `selected_object_overlay.png`
  선택 번호 → `affordance_overlay.png` cyan 영역이 의도한 part인지 순으로 확인한다.

## 오류 처리

- `grounding_request.json` 없음: ICAR 안전 게이트(confidence < 0.70) 차단 여부 확인
- `found no object`: object 이름·물체 크기·가림 상태 확인
- `Gemini found no ... among top-k`: 후보 이미지 확인 후 `AFFORDGRASP_OBJECT_TOP_K` 상향
- `Gemini top-k confidence ... below`: 대상이 선명하게 보이도록 재촬영
- `found no part`: `part_name`이 이미지에서 실제로 보이는지 확인
- `num_batches_tracked` 경고: checkpoint 호환 경고로 추론이 완료됐다면 무시 가능
