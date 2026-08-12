# AffordGrasp

AffordGrasp는 RGB-D 영상과 자연어 작업 지시로부터 대상 물체와 잡을 부위를
찾고, AnyGrasp 후보를 생성해 xArm7 집기 동작으로 연결하는 파이프라인이다.

```text
RGB-D 촬영 → ICAR 추론 → VLPart 객체·부위 분할 → AnyGrasp
           → 로봇 경로 생성 → IK·충돌 검증 → xArm7 실행
```

## 요구 사항

- Linux, Bash, Python, Conda
- Intel RealSense D435
- ICAR에 사용할 Gemini 또는 OpenAI API
- VLPart 소스, checkpoint, 실행 환경
- AnyGrasp SDK, license, checkpoint, 실행 환경
- 로봇 실행 시 xArm7, xArm Gripper, `xarm_ros2`, eye-to-hand 캘리브레이션

각 외부 프로젝트와 장치 드라이버를 먼저 설치한 뒤 경로를 `config.env`에
설정한다.

## 설정

저장소 루트에서 예시 파일을 로컬 설정으로 복사한다.

```bash
cp config.env.example config.env
cp robot_config.example.json robot_config.json
```

- [`config.env.example`](config.env.example): API provider, 모델, checkpoint, Conda 환경,
  로봇 주소 등 머신별 설정
- [`robot_config.example.json`](robot_config.example.json): 작업 공간, 관절·TCP 제한,
  테이블 기하, 이동 속도 등 로봇 안전 설정

`config.env`와 `robot_config.json`은 Git에 커밋하지 않는다. API 키도 설정
파일에 저장하지 말고 실행 셸의 환경변수로 제공한다.

```bash
export GEMINI_API_KEY="your-api-key"
# 또는
export OPENAI_API_KEY="your-api-key"
```

## 기본 실행

아래 명령은 새 RGB-D 프레임을 촬영하고 grasp pose까지 생성한다.
로봇은 연결하거나 움직이지 않는다.

```bash
./run_affordgrasp_pipeline.sh \
  "Pick up the mug by its handle." demo_mug
```

마지막 인자는 촬영과 결과를 구분할 prefix다. 영문자·숫자로 시작하고
영문자, 숫자, `.`, `_`, `-`만 사용할 수 있다.

### 단계 재실행

`--stage`는 새로 촬영하지 않고 같은 prefix의 기존 입력과 결과를 사용한다.

```bash
./run_affordgrasp_pipeline.sh --stage icar "Pick up the mug." demo_mug
./run_affordgrasp_pipeline.sh --stage localization demo_mug
./run_affordgrasp_pipeline.sh --stage mask demo_mug
./run_affordgrasp_pipeline.sh --stage grasp demo_mug
./run_affordgrasp_pipeline.sh --stage robot-plan demo_mug
./run_affordgrasp_pipeline.sh --stage robot-collision demo_mug
```

| 단계 | 기능 |
|---|---|
| `icar` | 작업 지시에서 대상, 부위, affordance 추론 |
| `localization` | VLPart 후보에서 대상 물체 선택 |
| `mask` | 잡을 부위의 mask 생성 |
| `grasp` | RGB-D와 mask로 AnyGrasp 후보 생성 |
| `robot-plan` | grasp 후보를 로봇 base 좌표계 경로로 변환 |
| `robot-collision` | IK와 전체 링크 충돌 검증 |
| `robot-execute` | 검증된 plan 실행 |

## 로봇 실행

로봇 실행 전에 캘리브레이션과 `robot_config.json`의 기하·제한값을 실제
설치와 맞게 검증해야 한다. 처음에는 그리퍼를 닫지 않는 `grasp-check`
모드로 자세와 높이를 확인한다.

```bash
source ./config.env
CONFIRM_TOKEN="MOVE_XARM7_${AFFORDGRASP_ROBOT_IP//./_}"

./run_affordgrasp_pipeline.sh --execute \
  --robot-mode grasp-check \
  --confirm "$CONFIRM_TOKEN" \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "Pick up the mug by its handle." demo_mug_check
```

`grasp-check`가 안전하게 통과한 후에만 `--robot-mode full`로 전체 집기를
실행한다. 실행 전에는 반드시 다음을 확인한다.

- 작업 공간에 사람, 케이블, 장애물이 없음
- 비상정지 장치를 즉시 사용할 수 있음
- TCP offset, payload, 속도, 작업 공간, 테이블 모델이 실제 설치와 일치함
- `collision_validation.json`의 `safe_for_execution`이 `true`임
- plan, 캘리브레이션, 로봇 설정, 시작 자세가 바뀌지 않음

충돌 검증은 모델에 포함된 로봇과 테이블만 고려한다. 실제 환경의 사람과
임시 장애물을 자동으로 보장하지 않는다.

## Eye-to-hand 캘리브레이션

카메라 위치가 바뀌었거나 유효한 캘리브레이션이 없으면 다시 수행한다.
상세한 하위 명령과 옵션은 도움말에서 확인할 수 있다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration --help
```

카메라는 고정하고 로봇 자세를 다양하게 바꾸어 샘플을 수집한다. 생성된 행렬은
다시 검증한 뒤 `AFFORDGRASP_EYE_TO_HAND_CALIBRATION`에서 지정한 경로에 저장한다.

## 결과

```text
captures/icar_d435/       RGB, raw/filtered depth, camera intrinsics
runs/<prefix>/json/       ICAR와 단계별 JSON
runs/<prefix>/object_localization/
runs/<prefix>/affordance_mask/
runs/<prefix>/grasp/      grasp pose와 point cloud
runs/<prefix>/robot/      plan, 충돌 검증, 실행 기록
```

실제 로봇 실행 전에는 최소한 다음 결과를 확인한다.

- `selected_object_overlay.png`: 올바른 물체가 선택되었는지
- `affordance_overlay.png`: 잡을 부위가 올바른지
- `grasp_pose_3d.png`: gripper 자세가 타당한지
- `collision_validation.json`: plan이 안전 검증을 통과했는지
