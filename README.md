# AffordGrasp 파이프라인

RealSense D435 영상에서 작업 대상과 잡을 부분을 찾고, AnyGrasp grasp pose를
xArm7의 실제 집기 동작까지 연결하는 파이프라인이다.

```text
D435 촬영 → ICAR → VLPart → Affordance mask → AnyGrasp
          → xArm7 plan → 실제 IK·전체 링크 충돌 검증 → xArm Gripper 집기
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

인식과 카메라 좌표 시뮬레이션까지만 실행:

```bash
./run_affordgrasp_pipeline.sh "I need to pick up the pliers safely." 01_pliers
```

- 첫 번째 인자: 로봇에게 수행시키려는 작업
- 두 번째 인자: 촬영 및 결과 폴더에 사용할 prefix
- 같은 prefix를 다시 사용하면 기존 결과를 덮어쓸 수 있다.

촬영부터 실제 집기까지 한 번에 실행:

```bash
./run_affordgrasp_pipeline.sh --execute \
  --robot-mode full \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "I need to pick up the pliers safely." 01_pliers
```

이 명령은 새로 촬영한 뒤 모든 인식 단계, robot plan과 충돌 검증을 차례로
수행한다. 검증된 **동일한 plan**만 실행하며 다음 중 하나라도 해당하면 로봇을
움직이지 않고 중단한다.

- 전체 링크·그리퍼·테이블 또는 self-collision 검사 실패
- IK 불연속, 관절 한계 또는 시작 위치 제한 위반
- 검증 뒤 plan이나 `robot_config.json` 변경
- 검증 후 300초 경과 또는 로봇 시작 자세 변경
- 캘리브레이션·TCP offset·물리 환경 승인 조건 불일치

처음 확인할 때는 그리퍼를 닫지 않는 모드를 권장한다.

```bash
./run_affordgrasp_pipeline.sh --execute \
  --robot-mode grasp-check \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "I need to pick up the pliers safely." check01_pliers
```

`grasp-check`는 pregrasp에서 grasp 높이까지 열린 그리퍼로 이동하고 멈춘다.
비상정지 장치를 손에 들고 사람·케이블·공구를 작업영역에서 치운 상태에서만
승인 옵션을 입력한다.

## 3. 단계별 재실행

기존 촬영본을 사용하므로 D435로 다시 촬영하지 않는다.

```bash
./run_affordgrasp_pipeline.sh --stage icar "새 작업 지시" 01_pliers
./run_affordgrasp_pipeline.sh --stage localization 01_pliers
./run_affordgrasp_pipeline.sh --stage mask 01_pliers
./run_affordgrasp_pipeline.sh --stage grasp 01_pliers
./run_affordgrasp_pipeline.sh --stage camera-sim 01_pliers
./run_affordgrasp_pipeline.sh --stage robot-plan 01_pliers
./run_affordgrasp_pipeline.sh --stage robot-collision 01_pliers
./run_affordgrasp_pipeline.sh --stage robot-execute \
  --robot-mode full \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  01_pliers
```

| 단계 | 기능 |
|---|---|
| `icar` | 작업 지시에서 object, part, affordance 추론 |
| `localization` | VLPart 후보 중 작업 대상 선택 |
| `mask` | 잡아야 할 part mask 생성 |
| `grasp` | RGB-D와 mask로 AnyGrasp pose 생성 |
| `camera-sim` | 카메라 좌표계에서 grasp 경로 시각화 |
| `robot-plan` | 캘리브레이션을 이용해 xArm base 경로 생성 |
| `robot-collision` | 실제 컨트롤러 IK와 공식 형상으로 전체 링크·테이블 충돌 검사 |
| `robot-execute` | 직전 검증과 정확히 일치하는 plan만 실제 로봇에서 실행 |

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
├── camera_simulation/     경로 JSON, 3D PNG, GIF
└── robot/                 base plan, 충돌 검증, 실행 기록
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

ChArUco 보드 생성:

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  generate-board \
  --output calibration/charuco_4x5.png \
  --squares-x 4 \
  --squares-y 5 \
  --square-length-m 0.030 \
  --marker-length-m 0.022
```

인쇄 설정에서 `실제 크기/100%`를 선택하고 `페이지에 맞춤`을 끈다. 패턴 크기는
120×150mm이고 흰 여백을 포함한 전체 크기는 140×170mm다. 출력 후 체스보드
사각형과 내부 ArUco 마커 한 변을 직접 측정한다. 정확히 30mm/22mm가 아니라면
아래 두 길이에 실측값을 입력한다. 보드는 휘지 않는 판에 붙이고 gripper의
움직이지 않는 몸체에 고정한다. 로봇 자세를 충분히 다르게 바꾸면서 촬영을 최소
12회, 가능하면 15~20회 반복한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  capture \
  --robot-ip 192.168.1.216 \
  --squares-x 4 \
  --squares-y 5 \
  --square-length-m 0.030 \
  --marker-length-m 0.022 \
  --min-charuco-corners 8
```

행렬 계산:

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration solve
```

결과는 `calibration/eye_to_hand.json`에 저장된다.
촬영 샘플은 기존 단일 ArUco 데이터와 분리된
`calibration/eye_to_hand_charuco_samples.json`에 저장된다.

## 8. 실제 xArm7 실행

현재 프로젝트의 표준 대기 TCP 자세는 xArm base 좌표계에서 다음과 같이 설정한다.

```json
"ready_tcp_pose_mm_deg": [300.0, 0.0, 250.0, 180.0, 0.0, 0.0],
"ready_tcp_speed_mm_s": 100.0,
"ready_tcp_acceleration_mm_s2": 150.0,
"ready_tcp_translation_tolerance_mm": 5.0,
"ready_tcp_rotation_tolerance_deg": 3.0
```

이는 `x=300mm, y=0mm, z=250mm, roll=180°, pitch=0°, yaw=0°`인 절대 TCP
자세다. SDK 연결은 기본적으로 radian 모드이므로 실행 코드는 반드시
`is_radian=False`를 명시한다. `현재→ready` 선형 Cartesian 경로도 전체 링크 충돌
검증을 통과해야 실제 `set_position()` 명령이 실행된다.

ready는 이 Cartesian 방식만 지원한다. 설정된 자세는 plan에 복사되고 설정 파일
해시로 고정된다. ready 값을 바꾸면 plan과 충돌 검증을 모두 다시 생성해야 한다.

먼저 로봇을 움직이지 않는 offline plan을 생성한다.

```bash
./run_affordgrasp_pipeline.sh --stage robot-plan 01_pliers
```

테이블 위 물체는 grasp 지점에 한해 base Z=-10mm까지 허용한다. pregrasp,
retreat, lift는 `robot_config.json`의 `minimum_transit_z_m`(기본 80mm) 이상이어야
한다. 이 구분 때문에 물체를 120mm 받침대 위에 올릴 필요는 없다. 낮은 tabletop
grasp의 실제 실행은 gripper와 테이블의 간격을 직접 검증하기 전까지 차단된다.

공식 xArm7/xArm Gripper 형상을 준비하고 전체 경로를 검사한다.

```bash
git clone --depth 1 --branch humble \
  https://github.com/xArm-Developer/xarm_ros2.git xarm_ros2

conda activate anygrasp
python -m pip install pybullet xacro

./run_affordgrasp_pipeline.sh --stage robot-collision 01_pliers
```

이 단계는 현재 관절각과 IK만 읽고 로봇을 움직이지 않는다. `현재→ready`와
`ready→pregrasp→grasp→retreat→lift`를 모두 10mm/5° 간격의 Cartesian 경로로
검사한다. 열린 그리퍼와 닫힌 그리퍼 형상을 모두 확인한다. 결과는
`runs/<prefix>/robot/collision_validation.json`에 저장된다.

`safe_for_execution=false`이면 실제 실행하지 않는다. 특히
`collision_environment.geometry_verified=false`는 테이블·받침대의 높이와 크기를
실측하지 않았다는 뜻이다. 투명 아크릴이나 유리는 D435 depth만으로 안전하게
측정할 수 없으므로 `robot_config.json`에 실측값을 입력한 뒤에만 `true`로 바꾼다.

물체 위에서 너무 일찍 닫히면 `robot_config.json`의
`additional_grasp_depth_m`으로 접근 방향의 추가 삽입 깊이를 조절한다. 이후
재조정은 5mm 단위로 하고, `grasp-check` 모드로 그리퍼를 닫지 않은 최종 높이를
먼저 확인한다.

그다음 읽기 전용 연결만 확인한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.xarm_grasp_execution \
  --robot-config robot_config.json \
  --mode connect
```

단계별 결과를 사용해 실제 실행만 다시 할 때는 직전에 `robot-collision`을
실행한 뒤 다음 명령을 사용한다.

```bash
./run_affordgrasp_pipeline.sh --stage robot-execute \
  --robot-mode full \
  --confirm MOVE_XARM7_192_168_1_216 \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  01_pliers
```

실제 순서는 `현재 관절→ready→그리퍼 열기→pregrasp→grasp→그리퍼 닫기
→retreat→lift`다.

실제 이동은 캘리브레이션, TCP offset, 작업공간, 비상정지 준비와 gripper 방향을
모두 검증한 뒤에만 수행한다. 충돌 결과가 오래됐거나 검증 뒤 로봇을 움직였다면
`robot-collision`부터 다시 실행해야 한다. AnyGrasp 자체 충돌 검사는 gripper
주변만 검사하므로 이 파이프라인은 별도로 xArm 전체 링크 경로를 검사한다.

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
| `no ArUco markers ... detected` | ChArUco 보드가 충분히 크고 선명하게 보이는지 확인 |
| `only ... ChArUco corners` | 보드의 가림·반사·흐림을 줄이고 더 많은 코너가 보이게 촬영 |

## 코드 구조

```text
affordgrasp_icar/
├── camera.py      D435 RGB-D 촬영
├── icar/          작업 및 affordance 추론
├── grounding/     객체 검출과 affordance mask
├── grasp/         AnyGrasp와 카메라 좌표 시각화
└── robot/         캘리브레이션, robot plan, guarded execution
```
