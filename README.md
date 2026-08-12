# AffordGrasp

RGB-D 영상과 자연어 지시에서 대상과 잡을 부위를 찾고, AnyGrasp 후보를
xArm7 집기 동작으로 연결하는 파이프라인이다.

```text
D435 → ICAR → VLPart → affordance mask → AnyGrasp
     → robot 동작 생성→ 충돌 검증 → 동작
```

## 1. 요구사항

**하드웨어**

- Intel RealSense D435
- VLPart·AnyGrasp를 실행할 NVIDIA GPU와 호환 CUDA 환경
- xArm7, xArm Gripper

**소프트웨어**

- Linux, Bash, Python 3.9 이상, Conda
- Intel librealsense와 `pyrealsense2`
- VLPart, Detectron2, CLIP, PyTorch
- AnyGrasp SDK, license, checkpoint, MinkowskiEngine
- 캘리브레이션 시 `xarm-python-sdk`
- 로봇 실행 시 `xarm-python-sdk`, `pybullet`, `xacro`, `xarm_ros2`

| 환경 | 사용 단계 |
|---|---|
| 현재 셸의 Python | D435 촬영, ICAR, 캘리브레이션 |
| `AFFORDGRASP_VLPART_ENV` | 물체 탐지와 affordance mask |
| `AFFORDGRASP_ANYGRASP_ENV` | grasp 생성, 충돌 검증, 로봇 실행 |

## 2. 설치 및 설정

저장소 루트에서 로컬 설정 파일을 만든다.

```bash
cp config.env.example config.env
cp robot_config.example.json robot_config.json
```

현재 Python 환경에 기본 패키지를 설치한다.

```bash
python -m pip install \
  numpy Pillow openai pyrealsense2 opencv-contrib-python xarm-python-sdk
```

VLPart와 AnyGrasp는 각 프로젝트의 설치 방법에 따라 PyTorch/CUDA 호환 버전으로
설치한다. AnyGrasp 환경에는 파이프라인용 패키지를 추가한다.

```bash
source ./config.env
conda run --prefix "$AFFORDGRASP_ANYGRASP_ENV" \
  python -m pip install matplotlib xarm-python-sdk pybullet xacro
```

`config.env`에서 VLPart, AnyGrasp, checkpoint, Conda 환경, 로봇 IP 경로를
설정한다. `xarm_ros2`의 기본 경로는 `./xarm_ros2`이며 다른 위치는
`AFFORDGRASP_XARM_ROS2_ROOT`로 지정한다.

API 키는 파일에 저장하지 않고 셸 환경변수로 제공한다.

```bash
export GEMINI_API_KEY="your-api-key"
# 또는
export OPENAI_API_KEY="your-api-key"
```

## 3. 캘리브레이션 및 로봇 설정

실제 로봇을 사용할 때만 진행한다.

1. D435를 움직이지 않도록 고정하고, 생성된 ChArUco 보드를 로봇 끝단에
   단단히 고정한다.
2. 로봇을 서로 다른 위치와 방향으로 옮긴 뒤 아래의 같은 명령을 반복한다.
3. 명령은 실행할 때마다 샘플 하나를 촬영한다. 12개 이상이 모이면 자동으로
   계산과 품질 검증을 수행해 `calibration/eye_to_hand.json`을 만든다.

```bash
source ./config.env
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration
```

첫 실행에서 `calibration/charuco_4x5.png`만 생성되면, 100% 크기로 인쇄해
고정한 뒤 같은 명령을 다시 실행한다. 인쇄된 한 칸의 실제 길이가 30 mm와
다르면 `--square-length-m`에 측정값을 미터 단위로 지정한다.

```bash
PYTHONPATH=.. python -m affordgrasp_icar.robot.eye_to_hand_calibration \
  --square-length-m 0.0298
```

캘리브레이션 중 로봇은 자동으로 움직이지 않는다. 각 촬영 전 로봇을 정지하고
보드 전체가 D435 영상에 보이게 한다.

캘리브레이션 후 `robot_config.json`의 작업 공간, TCP offset, ready 자세와
테이블 크기를 실제 설치에 맞춘다. 측정이 끝나면
`collision_environment.geometry_verified`를 `true`로 설정한다.

## 4. 실행

**인식과 grasp 생성만 실행 — 로봇은 움직이지 않음**

```bash
./run_affordgrasp_pipeline.sh \
  "Pick up the mug by its handle." demo_mug
```

**충돌 검증 후 실제 집기 실행**

```bash
source ./config.env
CONFIRM_TOKEN="MOVE_XARM7_${AFFORDGRASP_ROBOT_IP//./_}"

./run_affordgrasp_pipeline.sh --execute \
  --confirm "$CONFIRM_TOKEN" \
  --acknowledge-cleared-workspace \
  --acknowledge-estop-ready \
  "Pick up the mug by its handle." demo_mug
```

`--execute`는 별도 check 모드 없이 충돌 검증을 통과한 동일 plan을 바로
실행한다. 검증이 실패하면 로봇은 움직이지 않는다.

## 5. 전체 처리 순서

1. D435에서 RGB, raw/filtered depth와 카메라 내부 파라미터를 저장한다.
2. ICAR가 작업 지시에서 대상 물체, 잡을 부위와 affordance를 추론한다.
3. VLPart가 대상 물체를 찾고 잡을 부위의 mask를 생성한다.
4. AnyGrasp가 grasp 후보와 그리퍼 자세를 생성한다.
5. Eye-to-hand 행렬로 후보를 xArm base 좌표계 경로로 변환한다.
6. 현재 자세부터 lift까지 IK, 자체 충돌과 테이블 충돌을 검사한다.
7. `safe_for_execution=true`이면 다음 순서로 실제 로봇을 움직인다.

```text
ready → gripper open → pregrasp → grasp
      → gripper close → retreat → lift
```

실행 직전에는 검증 결과의 최신성, plan/config 해시와 로봇 시작 자세가
검증 당시와 일치하는지도 다시 확인한다.

## 6. 안전

- 비상정지 장치를 즉시 사용할 수 있는 상태에서 실행한다.
- 작업 공간에서 사람, 케이블과 임시 장애물을 제거한다.
- 캘리브레이션, TCP offset, 작업 공간과 테이블 모델을 실제 설치와 맞춘다.
- 같은 prefix를 재사용하면 기존 결과가 바뀔 수 있으므로 실행별로 구분한다.
- 충돌 검증은 설정된 로봇과 테이블만 검사한다. 사람, 케이블, 카메라
  스탠드와 대상 물체의 접촉은 보장하지 않는다.

## 7. 단계 재실행

새로 촬영하지 않고 같은 prefix의 기존 결과를 사용한다.

```bash
./run_affordgrasp_pipeline.sh --stage icar "Pick up the mug." demo_mug
./run_affordgrasp_pipeline.sh --stage localization demo_mug
./run_affordgrasp_pipeline.sh --stage mask demo_mug
./run_affordgrasp_pipeline.sh --stage grasp demo_mug
./run_affordgrasp_pipeline.sh --stage robot-plan demo_mug
./run_affordgrasp_pipeline.sh --stage robot-collision demo_mug
```

검증된 기존 plan을 실행할 때는 `--stage robot-execute`와 동일한 승인 옵션을
사용한다.

## 8. 결과

중요한 결과만 정리한 실제 실행 예시는 [`example/`](example/)에서 확인할 수
있다.

```text
captures/icar_d435/       RGB, depth, camera intrinsics
runs/<prefix>/json/       ICAR와 단계별 JSON
runs/<prefix>/object_localization/
runs/<prefix>/affordance_mask/
runs/<prefix>/grasp/      grasp pose와 point cloud
runs/<prefix>/robot/      plan, 충돌 검증, 실행 기록
```

실제 실행 전에는 다음 파일을 확인한다.

- `selected_object_overlay.png`
- `affordance_overlay.png`
- `grasp_pose_3d.png`
- `collision_validation.json`
