# VLPart Object Localization 가이드

## 1. 구현 범위

AffordGrasp Visual Affordance Grounding의 첫 단계만 구현한다.

```text
B_O = VLPart(I, O)

M_BO(i, j) = I(i, j)  if (i, j) is inside B_O
             0        otherwise
```

입력:

- D435 RGB 이미지 `I`
- ICAR가 선택한 물체 이름 `O`

출력:

- 물체 bounding box `B_O`
- bounding box 내부만 남긴 `masked_object.png`
- 점수와 좌표를 기록한 `object_localization.json`

Part mask, Depth, point cloud, AnyGrasp는 실행하지 않는다.

## 2. 현재 폴더의 VLPart 실행 환경

현재 필요한 저장소, checkpoint와 전용 Conda 환경은 다음 위치에 준비되어 있다.

```text
affordgrasp_icar/
└── VLPart/
    ├── .conda/
    ├── configs/pascal_part/r50_pascalpart.yaml
    └── models/r50_pascalpart.pth
```

`VLPart/`, checkpoint와 Conda 환경은 용량이 크므로 GitHub에는 업로드하지
않는다. 다른 PC에서 이 프로젝트를 clone한 경우
[VLPart 공식 저장소](https://github.com/facebookresearch/VLPart)의 설치 안내와
model zoo를 따라 위 구조로 별도 준비해야 한다.

Linux 경로는 대소문자를 구분한다. `VLPART`가 아니라 반드시 `VLPart`를
사용한다.

이전에 잘못된 환경변수를 설정했다면 다음처럼 수정한다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar

export VLPART_ROOT="$PWD/VLPart"
export VLPART_WEIGHTS="$PWD/VLPart/models/r50_pascalpart.pth"
```

경로를 인자로 직접 전달하면 환경변수를 설정하지 않아도 된다.

```bash
--vlpart-root ./VLPart \
--weights ./VLPart/models/r50_pascalpart.pth
```

다른 config를 사용할 때만 다음 변수를 추가한다.

```bash
export VLPART_CONFIG="/절대경로/config.yaml"
```

## 3. 물체 이름을 직접 지정해 실행

```bash
cd /home/unist/Test_hand/affordgrasp_icar

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix ./VLPart/.conda \
python -m affordgrasp_icar.object_localization_cli \
  --image ./captures/icar_d435/01_test_rgb.png \
  --object-name pliers \
  --output-dir ./runs/object_localization \
  --vlpart-root ./VLPart \
  --weights ./VLPart/models/r50_pascalpart.pth \
  --confidence-threshold 0.10 \
  --device cpu
```

현재 `01_test_rgb.png`는 `0.50`과 `0.20`에서는 검출되지 않았고,
`0.10` 이하에서 검출됐다. 낮은 threshold는 오검출 가능성도 높이므로 촬영할
때는 대상 물체를 화면 중앙에 크게 두고 다른 물체와 겹치지 않게 한다.

## 4. ICAR grounding 요청으로 실행

ICAR에서 생성한 JSON이 다음과 같다고 가정한다.

```json
{
  "task": "cut wire",
  "object_name": "pliers",
  "part_name": "handle",
  "affordance": "grasp",
  "confidence": 0.91
}
```

Object Localization은 이 중 `object_name`만 사용한다.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.. \
conda run --prefix ./VLPart/.conda \
python -m affordgrasp_icar.object_localization_cli \
  --image ./captures/icar_d435/01_test_rgb.png \
  --request ./grounding_request.json \
  --output-dir ./runs/object_localization \
  --vlpart-root ./VLPart \
  --weights ./VLPart/models/r50_pascalpart.pth \
  --confidence-threshold 0.10 \
  --device cpu
```

## 5. 출력

```text
runs/object_localization/
├── masked_object.png
└── object_localization.json
```

JSON 예시:

```json
{
  "object_name": "pliers",
  "bbox_xyxy": [468, 33, 635, 450],
  "score": 0.10814403742551804,
  "image_width": 640,
  "image_height": 480,
  "source_image": "/절대경로/01_test_rgb.png",
  "masked_image": "/절대경로/masked_object.png"
}
```

`bbox_xyxy`는 원본 RGB 좌표계의 `[x1, y1, x2, y2]`다. 여러 후보가 검출되면 confidence가 가장 높은 bounding box를 선택한다.

VLPart가 지정한 threshold 이상의 물체를 찾지 못하면 출력 mask를 생성하지 않고 종료 코드 `2`를 반환한다.

## 6. 테스트

VLPart 설치나 checkpoint 없이 수식과 파일 생성을 검사할 수 있다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.. \
  python -m unittest \
  affordgrasp_icar.tests.test_object_localization \
  -v
```

검사 범위:

- bounding box 밖의 픽셀이 모두 0인지 확인
- 원본 RGB 좌표와 pixel 값 보존
- 최고 점수 후보 선택
- VLPart `Instances`의 box와 score 변환
- threshold 미달 검출 차단
- masked PNG와 bounding box JSON 저장

## 7. 공식 참고

- [VLPart 공식 저장소](https://github.com/facebookresearch/VLPart)
- [VLPart 설치 안내](https://github.com/facebookresearch/VLPart/blob/main/INSTALL.md)
- [VLPart custom vocabulary 실행](https://github.com/facebookresearch/VLPart/blob/main/GETTING_STARTED.md)
- [VLPart checkpoint 및 config 목록](https://github.com/facebookresearch/VLPart/blob/main/MODEL_ZOO.md)
