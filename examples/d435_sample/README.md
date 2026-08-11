# D435 샘플

D435 촬영 결과 형식을 보여주는 공개용 샘플이다.

| 파일 | 용도 |
|---|---|
| `sample_rgb.png` | 640×480 RGB |
| `sample_depth_raw.png` | RGB에 정렬된 원본 `uint16` depth |
| `sample_depth_filtered.png` | 필터를 적용한 `uint16` depth |
| `sample_depth_preview.png` | 사람이 확인하는 컬러 미리보기 |
| `sample_camera.json` | depth scale, 카메라 내부 파라미터와 파일 경로 |

16-bit depth는 일반 뷰어에서 검게 보일 수 있다. 거리 계산에는 preview가 아닌
raw 또는 filtered 파일을 `cv2.IMREAD_UNCHANGED`로 읽고,
`depth_scale_meters_per_unit`을 곱한다.

공개용 샘플에서는 카메라 시리얼 번호와 로컬 절대경로를 제거했다.
