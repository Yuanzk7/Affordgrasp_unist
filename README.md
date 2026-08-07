# AffordGrasp 파이프라인

[AffordGrasp](https://arxiv.org/abs/2503.00778) 구현. RealSense D435로 RGB-D
장면을 촬영하고 다음을 실행한다.

```text
D435 촬영 → ICAR (VLM 추론) → Object Localization (VLPart + Gemini top-k 재선택)
→ Affordance Mask → AnyGrasp (region steering + collision detection)
```

`run_affordgrasp_pipeline.sh`는 촬영부터 Grasp Pose Generation과 3D 출력까지
순서대로 실행한다. `grasp/`는 전체 장면 포인트를 충돌 검사에 유지하고,
affordance mask를 AnyGrasp의 `region_steering`으로 전달한다. PCA baseline은
입력·시각화 진단용 fallback으로만 남겨 둔다.

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
./run_affordgrasp_pipeline.sh --stage grasp 01_pliers
```

기본 grasp backend는 `anygrasp`다. SDK·라이선스·checkpoint·전용 Conda 환경은
`config.env`에서 지정한다.

## Grasp 공통 인터페이스와 3D 시각화

AnyGrasp 라이선스 없이 번들 D435 샘플로 입력·pose·3D 출력을 검증한다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
python -m affordgrasp_icar.grasp.grasp_pose_generation \
  --backend baseline \
  --sample-dir examples/d435_sample \
  --output-dir runs/grasp_baseline
```

출력은 `grasp_pose_result.json`, `grasp_pose_3d.png`, 전체 충돌 문맥을 담은
`scene_point_cloud.ply`, `affordance_point_cloud.ply`, mask와 overlay다.
Baseline pose는 연결과 시각화 검증 전용이며 실제 로봇에 실행하지 않는다.

- 검은 선: parallel-jaw gripper
- 빨간 축: 접근 방향
- 초록 축: jaw closing 방향
- 파란 축: 오른손 좌표계를 완성하는 gripper 축

실제 파이프라인 산출물로 실행할 때는 네 입력을 명시한다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
python -m affordgrasp_icar.grasp.grasp_pose_generation \
  --backend anygrasp \
  --anygrasp-sdk /path/to/anygrasp_sdk \
  --checkpoint /path/to/checkpoint_detection.tar \
  --rgb captures/icar_d435/<prefix>_rgb.png \
  --depth captures/icar_d435/<prefix>_depth_filtered.png \
  --camera captures/icar_d435/<prefix>_camera.json \
  --mask runs/<prefix>/affordance_mask/affordance_mask.png \
  --output-dir runs/<prefix>/grasp
```

AnyGrasp는 RGB 색 대신 전체 scene point cloud를 추론에 사용하고, mask와 같은
길이의 3D region을 생성해 grasp 위치를 제한한다. 주변 물체 포인트는 삭제하지
않으므로 collision detection에서 그대로 사용된다.

전체 파이프라인에서는 `config.env`의 실제 경로를 사용한다.

```bash
export AFFORDGRASP_GRASP_BACKEND=anygrasp
export AFFORDGRASP_ANYGRASP_SDK=/path/to/anygrasp_sdk
export AFFORDGRASP_ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
export AFFORDGRASP_ANYGRASP_ENV=/path/to/anygrasp/conda-env
```

## 설정

모든 실행 설정은 `config.env` 한 파일에서 관리한다 (항목 설명은 파일 주석
참조). API 키는 이 파일에 넣지 말고 셸 환경변수나
`affordgrasp_icar/local_config.py`에 둔다.

- 대상이 `top_k_candidates.png`에 없으면 `AFFORDGRASP_OBJECT_TOP_K`를 15~20으로 올린다.
- API 없이 실행할 때만 `AFFORDGRASP_SELECTION_METHOD=vlpart-score` (다물체 장면 비권장).
- Object Localization은 `toilet paper`·`paper towel` 계열에 built-in alias
  ensemble을 자동 적용하고, 동일 박스를 병합한 뒤 Gemini top-k로 넘긴다.
- 다른 객체의 별칭은 Object Localization CLI에
  `--object-alias "별칭"`을 여러 번 지정해 추가할 수 있다.

## 출력

```text
captures/icar_d435/<prefix>_{rgb,depth_raw,depth_filtered,depth_preview}.png, <prefix>_camera.json
runs/<prefix>/
├── json/                  capture·icar·grounding·localization·mask 결과 JSON
├── object_localization/   top_k_candidates.png, selected_object_overlay.png, masked_object.png
├── affordance_mask/       affordance_mask.png, affordance_overlay.png
└── grasp/                 grasp_pose_result.json, grasp_pose_3d.png,
                          scene_point_cloud.ply, affordance_point_cloud.ply
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
- `Affordance mask가 없습니다`: `--stage mask <prefix>`를 먼저 실행
- AnyGrasp checkpoint/license 오류: `config.env`의 SDK·checkpoint·Conda 환경 확인
- `num_batches_tracked` 경고: checkpoint 호환 경고로 추론이 완료됐다면 무시 가능
