# AffordGrasp 파이프라인

RealSense D435 영상에서 작업 대상과 잡을 부분을 찾고, AnyGrasp grasp pose를
xArm7 집기 동작으로 연결하는 프로젝트다.

```text
D435 → ICAR → VLPart → Affordance mask → AnyGrasp
     → xArm7 plan → IK·충돌 검증 → xArm Gripper
```

## 준비

```bash
cd /home/unist/Test_hand/affordgrasp_icar
conda activate anygrasp
export GEMINI_API_KEY="발급받은_API_KEY"
```

[`config.env.example`](config.env.example)을 `config.env`로 복사한 뒤 로컬
설치 경로와 로봇 주소를 설정한다. 로봇과 안전 설정은
[robot_config.json](robot_config.json)에서 관리한다. API 키는 파일에 저장하지
않는다.

```bash
cp config.env.example config.env
```

필수 구성:

- Intel RealSense D435
- VLPart 모델과 환경
- AnyGrasp SDK, 라이선스, checkpoint와 환경
- xArm7, xArm Gripper와 eye-to-hand 캘리브레이션

## 실행

인식과 카메라 좌표 시뮬레이션까지만 실행:

```bash
./run_affordgrasp_pipeline.sh \
  "I need to pick up the pliers safely." run01_pliers
```

촬영부터 실제 집기까지 실행:

```bash
./run_affordgrasp_pipeline.sh --execute \
  --robot-mode full \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "I need to pick up the pliers safely." run01_pliers
```

처음에는 그리퍼를 닫지 않는 `grasp-check` 모드로 높이와 방향을 확인한다.

```bash
./run_affordgrasp_pipeline.sh --execute \
  --robot-mode grasp-check \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "I need to pick up the pliers safely." check01_pliers
```

같은 prefix를 다시 사용하면 기존 결과가 바뀔 수 있다. 실제 실행 전에는 작업
영역을 비우고 비상정지를 준비한다.

## 단계별 실행

```bash
./run_affordgrasp_pipeline.sh --stage icar "새 작업 지시" run01_pliers
./run_affordgrasp_pipeline.sh --stage localization run01_pliers
./run_affordgrasp_pipeline.sh --stage mask run01_pliers
./run_affordgrasp_pipeline.sh --stage grasp run01_pliers
./run_affordgrasp_pipeline.sh --stage camera-sim run01_pliers
./run_affordgrasp_pipeline.sh --stage robot-plan run01_pliers
./run_affordgrasp_pipeline.sh --stage robot-collision run01_pliers
```

| 단계 | 기능 |
|---|---|
| `icar` | 작업 지시에서 object, part, affordance 추론 |
| `localization` | VLPart 후보 중 대상 물체 선택 |
| `mask` | 잡을 부분의 mask 생성 |
| `grasp` | RGB-D와 mask로 AnyGrasp 후보 생성 |
| `camera-sim` | 카메라 좌표계 grasp 경로 시각화 |
| `robot-plan` | 후보를 xArm base 경로로 변환하고 점수순 정렬 |
| `robot-collision` | 점수순으로 IK와 전체 링크 충돌 검사 후 최종 후보 선택 |
| `robot-execute` | 검증된 최종 후보만 실제 실행 |

실제 실행만 다시 수행할 때:

```bash
./run_affordgrasp_pipeline.sh --stage robot-execute \
  --robot-mode full \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  run01_pliers
```

`robot-collision`은 로봇 상태와 IK를 읽지만 움직이지 않는다. 실제 실행은
`현재→ready→pregrasp→grasp→retreat→lift` 순서의 Cartesian 동작만 사용한다.

## 결과

```text
captures/icar_d435/
└── <prefix>_rgb.png, depth_raw.png, depth_filtered.png, camera.json

runs/<prefix>/
├── json/                  ICAR 및 단계별 JSON
├── object_localization/   객체 후보와 선택 결과
├── affordance_mask/       mask와 overlay
├── grasp/                 grasp pose와 point cloud
├── camera_simulation/     경로 JSON, PNG, GIF
└── robot/                 robot plan, 충돌 검증, 실행 기록
```

주요 확인 파일:

1. `selected_object_overlay.png`: 올바른 물체인지 확인
2. `affordance_overlay.png`: 잡을 영역이 맞는지 확인
3. `grasp_pose_3d.png`: gripper 자세 확인
4. `camera_trajectory.gif`: 접근·후퇴 경로 확인
5. `collision_validation.json`: `safe_for_execution` 확인

거리 계산에는 preview가 아니라 `uint16` raw/filtered depth를 사용한다.

## Eye-to-hand 캘리브레이션

D435 위치를 바꾸었거나 캘리브레이션 파일이 없을 때만 다시 수행한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  status --robot-ip 192.168.1.216

PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  generate-board --output calibration/charuco_4x5.png

PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  capture --robot-ip 192.168.1.216

PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration solve
```

보드는 100% 크기로 출력해 단단하게 고정하고, 카메라는 고정한 채 로봇 자세만
바꾸어 12장 이상 촬영한다. 결과는 `calibration/eye_to_hand.json`에 저장된다.

## 로봇 실행 주의사항

- xArm TCP offset은 xArm Gripper용 `[0, 0, 172, 0, 0, 0]`이어야 한다.
- ready 자세, 작업공간, 테이블과 충돌 기준은 `robot_config.json`에서 관리한다.
- 후보는 affordance 점수순으로 검사하며 충돌을 통과한 첫 후보만 실행한다.
- plan, 설정, 캘리브레이션 또는 로봇 시작 자세가 바뀌면 다시 검증한다.
- `safe_for_execution=false`이면 실행하지 않는다.
- 실제 환경의 물체, 케이블과 사람은 시뮬레이션에 포함되지 않는다.

## 자주 발생하는 오류

| 오류 | 확인할 내용 |
|---|---|
| `found no object/part` | 물체명, 가림, mask와 촬영 구도 확인 |
| Gemini 429/404 | API 할당량과 모델명 확인 |
| AnyGrasp license 오류 | SDK, license와 checkpoint 경로 확인 |
| `no AnyGrasp candidate passed...` | 물체 위치, grasp 폭, 접근 방향과 작업공간 확인 |
| `no score-ranked candidate passed...` | 생성 후보가 모두 IK 또는 충돌 검사에서 탈락 |
| `safe_for_execution=false` | `collision_validation.json` 확인 |
| Controller error C31 | 실제 충돌·걸림, payload와 충돌 민감도 확인 |
| calibration 오류 | 보드 인식, 카메라 고정과 촬영 자세 확인 |

## 코드 구조

```text
affordgrasp_icar/
├── camera.py      D435 RGB-D 촬영
├── icar/          작업과 affordance 추론
├── grounding/     객체 검출과 mask 생성
├── grasp/         AnyGrasp와 3D 시각화
└── robot/         캘리브레이션, 경로, 충돌 검증과 실행
```
