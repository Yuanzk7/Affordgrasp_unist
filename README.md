# AffordGrasp 파이프라인

RealSense D435 영상에서 작업 대상과 잡을 부분을 찾고 AnyGrasp로 grasp pose를
생성하는 파이프라인이다.

```text
D435 촬영 → ICAR → VLPart 객체 검출 → Affordance mask → AnyGrasp → 3D 시각화
```

## 1. 준비

프로젝트 폴더와 Conda 환경을 활성화한다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar
conda activate anygrasp
```

Gemini를 사용할 경우 API 키를 설정한다.

```bash
export GEMINI_API_KEY="발급받은_API_KEY"
```

주요 로컬 경로는 [config.env](config.env)에서 관리한다.

- VLPart 환경 및 checkpoint
- AnyGrasp SDK, license, checkpoint 및 Conda 환경
- D435 depth 종류와 gripper 최대 폭
- Gemini top-k 및 confidence 설정

API 키는 `config.env`에 저장하지 않는다.

## 2. 전체 실행

```bash
./run_affordgrasp_pipeline.sh "I need to pick up the pliers safely." 01_pliers
```

- 첫 번째 인자: 로봇에게 수행시키려는 작업
- 두 번째 인자: 촬영 및 결과 폴더에 사용할 prefix
- 같은 prefix를 다시 사용하면 기존 결과를 덮어쓸 수 있다.

전체 실행은 촬영부터 AnyGrasp와 카메라 좌표계 시뮬레이션까지 수행한다.

## 3. 단계별 재실행

기존 촬영본을 사용하므로 D435로 다시 촬영하지 않는다.

```bash
./run_affordgrasp_pipeline.sh --stage icar "새 작업 지시" 01_pliers
./run_affordgrasp_pipeline.sh --stage localization 01_pliers
./run_affordgrasp_pipeline.sh --stage mask 01_pliers
./run_affordgrasp_pipeline.sh --stage grasp 01_pliers
./run_affordgrasp_pipeline.sh --stage camera-sim 01_pliers
./run_affordgrasp_pipeline.sh --stage robot-plan 01_pliers
```

| 단계 | 기능 |
|---|---|
| `icar` | 작업 지시에서 object, part, affordance 추론 |
| `localization` | VLPart 후보 중 작업 대상 선택 |
| `mask` | 잡아야 할 part mask 생성 |
| `grasp` | RGB-D와 mask로 AnyGrasp pose 생성 |
| `camera-sim` | 카메라 좌표계에서 grasp 경로 시각화 |
| `robot-plan` | 캘리브레이션을 이용해 xArm base 경로 생성 |

단계를 재실행할 때는 앞 단계의 결과가 먼저 존재해야 한다.

## 4. 결과 확인

```text
captures/icar_d435/
└── <prefix>_rgb.png, depth_raw.png, depth_filtered.png, camera.json

runs/<prefix>/
├── json/                  ICAR 및 각 단계 JSON
├── object_localization/   객체 후보와 선택 결과
├── affordance_mask/       part mask와 overlay
├── grasp/                 grasp pose와 point cloud
└── camera_simulation/     경로 JSON, 3D PNG, GIF
```

다음 순서로 결과를 확인한다.

1. `top_k_candidates.png`: 대상 물체가 후보에 포함됐는가?
2. `selected_object_overlay.png`: 올바른 물체가 선택됐는가?
3. `affordance_overlay.png`: cyan 영역이 잡을 부분과 일치하는가?
4. `grasp_pose_3d.png`: gripper 위치와 접근 방향이 적절한가?
5. `camera_trajectory.gif`: 접근·grasp·후퇴 경로가 적절한가?

3D 계산에는 preview가 아닌 `uint16` raw/filtered depth와 `camera.json`을 사용한다.

## 5. 카메라 좌표계 시뮬레이션

ArUco와 eye-to-hand 캘리브레이션 없이 실행할 수 있다.

```bash
./run_affordgrasp_pipeline.sh --stage camera-sim 01_pliers
```

```text
pregrasp → grasp 및 jaw closing → retreat → lift
```

출력 파일:

- `camera_trajectory.json`: waypoint 위치, 회전, jaw 폭
- `camera_trajectory_3d.png`: point cloud 위에 표시한 전체 경로
- `camera_trajectory.gif`: gripper 이동 애니메이션

카메라 좌표는 `+x=영상 오른쪽`, `+y=영상 아래`, `+z=카메라 전방`이다. 기본
lift 방향은 영상 위쪽인 camera `-Y`다.

이 단계는 xArm에 연결하지 않으며 관절 IK도 계산하지 않는다. 실제 로봇으로
실행하려면 eye-to-hand 캘리브레이션이 필요하다.

## 6. AnyGrasp 설정

[config.env](config.env)의 다음 항목을 실제 설치 경로에 맞춘다.

```bash
export AFFORDGRASP_ANYGRASP_SDK=/path/to/anygrasp_sdk
export AFFORDGRASP_ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
export AFFORDGRASP_ANYGRASP_ENV=/path/to/anygrasp-env
export AFFORDGRASP_MAX_GRIPPER_WIDTH=0.085
```

`AFFORDGRASP_MAX_GRIPPER_WIDTH=0.085`는 85mm보다 넓은 grasp를 실제 xArm
Gripper로 실행하지 않도록 제한한다.

## 7. Eye-to-hand 캘리브레이션

카메라 좌표를 실제 xArm base 좌표로 변환할 때만 필요하다. D435와 로봇을
움직이지 않는다면 한 번 계산한 결과를 계속 사용할 수 있다.

TCP offset 확인:

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  status --robot-ip 192.168.1.216
```

마커 생성:

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  generate-marker \
  --output calibration/aruco_4x4_50_id0.png
```

검은 마커 한 변이 60mm가 되도록 출력하고 gripper의 움직이지 않는 몸체에
고정한다. 로봇 자세를 충분히 다르게 바꾸면서 다음 촬영을 최소 12회 반복한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  capture \
  --robot-ip 192.168.1.216 \
  --marker-id 0 \
  --marker-length-m 0.060
```

행렬 계산:

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration solve
```

결과는 `calibration/eye_to_hand.json`에 저장된다.

## 8. 실제 xArm7 실행

먼저 로봇을 움직이지 않는 offline plan을 생성한다.

```bash
./run_affordgrasp_pipeline.sh --stage robot-plan 01_pliers
```

그다음 읽기 전용 연결만 확인한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.xarm_grasp_execution \
  --grasp-result runs/01_pliers/grasp/grasp_pose_result.json \
  --mode connect
```

실제 이동은 캘리브레이션, TCP offset, 작업공간, 비상정지 준비와 gripper 방향을
모두 검증한 뒤에만 수행한다. AnyGrasp 충돌 검사는 gripper 주변만 검사하므로
xArm 전체 링크의 안전한 경로를 보장하지 않는다.

## 9. 자주 발생하는 오류

| 오류 | 확인할 내용 |
|---|---|
| `grounding_request.json` 없음 | ICAR confidence가 기준보다 낮지 않은지 확인 |
| `found no object` | 물체 이름, 크기, 가림 및 촬영 구도 확인 |
| `Gemini found no ... among top-k` | 후보 이미지 확인 후 top-k 증가 |
| `found no part` | 요청한 part가 RGB 영상에 보이는지 확인 |
| `Affordance mask가 없습니다` | `--stage mask`를 먼저 실행 |
| AnyGrasp license/checkpoint 오류 | SDK, license, checkpoint 경로 확인 |
| `candidate width ... exceeds ...` | grasp 폭이 실제 gripper 최대 폭보다 큼 |
| `ArUco marker was not detected` | 마커 전체가 D435 RGB 영상에 보이는지 확인 |

## 코드 구조

```text
affordgrasp_icar/
├── camera.py      D435 RGB-D 촬영
├── icar/          작업 및 affordance 추론
├── grounding/     객체 검출과 affordance mask
├── grasp/         AnyGrasp와 카메라 좌표 시각화
└── robot/         캘리브레이션, robot plan, guarded execution
```
