# Intel RealSense D435 sample

이 폴더는 D435 촬영 코드의 출력 형식과 후속 처리 입력을 확인하기 위한 공개용 샘플입니다.

| 파일 | 형식과 용도 |
| --- | --- |
| `sample_rgb.png` | 640x480 RGB 이미지 |
| `sample_depth_raw.png` | RGB 픽셀에 정렬된 원본 `uint16` 깊이 이미지 |
| `sample_depth_filtered.png` | 공간·시간 필터를 적용한 `uint16` 깊이 이미지 |
| `sample_depth_preview.png` | 깊이를 사람이 확인할 수 있도록 색상으로 표현한 미리보기 |
| `sample_camera.json` | 깊이 단위, RGB 카메라 내부 파라미터와 상대 파일 경로 |

`sample_depth_raw.png`와 `sample_depth_filtered.png`는 일반 이미지 뷰어에서 검게 보일 수 있습니다. 거리 계산에는 미리보기가 아닌 16-bit 깊이 이미지를 원본 형식으로 읽어야 합니다.

```python
import json

import cv2
import numpy as np

with open("sample_camera.json", encoding="utf-8") as file:
    camera = json.load(file)

depth = cv2.imread(camera["files"]["depth_filtered"], cv2.IMREAD_UNCHANGED)
if depth is None or depth.dtype != np.uint16:
    raise RuntimeError("Expected a uint16 depth PNG")

depth_m = depth.astype(np.float32) * camera["depth_scale_meters_per_unit"]
```

공개 저장소에 안전하게 포함할 수 있도록 원본 촬영 JSON의 카메라 시리얼 번호와 로컬 절대경로는 제거했습니다.
