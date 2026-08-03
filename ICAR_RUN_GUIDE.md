# ICAR 실행 가이드

## 1. 목적

이 문서는 현재 `affordgrasp_icar` 패키지에서 AffordGrasp의 **In-Context Affordance Reasoning(ICAR)** 단계만 실행하는 방법을 설명한다.

ICAR의 입력과 출력은 다음과 같다.

```text
입력: 사용자 작업 지시 + RGB 장면 이미지
출력: Task + Object + Object Part + Affordance + 판단 근거
```

이 가이드의 명령은 다음 단계를 실행하지 않는다.

- VLPart 기반 object/part mask 생성
- depth 기반 point cloud 생성
- AnyGrasp 기반 6D grasp pose 생성
- 로봇 grasp 실행

ICAR만 실행하려면 `--grounding-output`을 지정하지 않으면 된다.

## 2. 현재 지원 provider

| Provider | 기본 모델 | API 경로 | 용도 |
|---|---|---|---|
| `gemini` | `gemini-3.6-flash` | Gemini OpenAI 호환 Chat Completions | 무료 tier 기반 초기 테스트 |
| `openai` | `gpt-5.6-terra` | OpenAI Responses API | 현재 OpenAI 기본 실행 |

AffordGrasp 논문과 같은 모델로 비교하려면 `--provider openai --model gpt-4o`를 명시한다.

## 3. 공통 준비

### 3.1 실행 위치

모든 명령은 `affordgrasp_icar` 폴더 안에서 실행한다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar
```

이 위치에서 `python -m affordgrasp_icar`를 실행할 때는 Python이 패키지를 찾도록 명령 앞에 `PYTHONPATH=..`를 붙인다.

### 3.2 Python 패키지 설치

OpenAI와 Gemini provider 모두 Python `openai` 패키지를 사용한다.

```bash
python -m pip install -U openai
```

설치 확인:

```bash
python -c "import openai; print(openai.__version__)"
```

### 3.3 테스트 이미지 준비 방식

이 프로젝트의 카메라 테스트 이미지는 **RealSense D435로만 촬영한다**. 휴대폰, 웹캠 또는 인터넷에서 받은 이미지는 D435 카메라 테스트에 사용하지 않는다.

D435 촬영 방법은 [6. RealSense D435 테스트 이미지 촬영](#6-realsense-d435-테스트-이미지-촬영)을 따른다. 기존에 D435로 저장한 RGB PNG를 다시 사용할 때만 `--image`에 절대 경로를 지정한다.

## 4. Gemini 무료 tier로 실행

### 4.1 API key 준비

1. [Google AI Studio](https://aistudio.google.com/)에서 Gemini API key를 생성한다.
2. 터미널 환경변수로 설정한다.

```bash
export GEMINI_API_KEY="본인의_Gemini_API_KEY"
```

key가 설정됐는지만 확인하려면 실제 값을 출력하지 않고 다음처럼 검사한다.

```bash
if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY is set"
else
  echo "GEMINI_API_KEY is not set"
fi
```

### 4.2 ICAR 실행

```bash
cd /home/unist/Test_hand/affordgrasp_icar

PYTHONPATH=.. python -m affordgrasp_icar \
  --provider gemini \
  --image /home/unist/Test_hand/affordgrasp_icar/captures/icar_d435/01_test_rgb.png\
  --instruction "I want to play mobile game" \
  --min-confidence 0 \
  --output ./gemini_result.json
```

주요 동작:

- `--provider gemini`: Gemini API 선택
- `--model` 생략: 기본 `gemini-3.6-flash` 사용
- `--min-confidence 0`: ICAR 결과 확인을 위해 confidence 임계값 차단 최소화
- `--grounding-output` 생략: grounding 요청 파일을 생성하지 않음
- `--output`: 전체 ICAR 결과를 JSON 파일로 저장

`--min-confidence 0`이어도 모델이 `is_actionable=false`를 반환하면 안전 게이트가 차단하며 종료 코드 `3`이 발생한다. 이 경우에도 API 응답 검증이 완료됐다면 `--output` 파일은 먼저 저장된다.

### 4.3 다른 Gemini 모델 사용

```bash
PYTHONPATH=.. python -m affordgrasp_icar \
  --provider gemini \
  --model gemini-3.5-flash \
  --image /절대경로/test_scene.png \
  --instruction "I am thirsty." \
  --reasoning-effort medium \
  --min-confidence 0 \
  --output /절대경로/gemini_35_result.json
```

Gemini provider에서 `--reasoning-effort`는 다음 값만 허용한다.

```text
low, medium, high
```

현재 `--image-detail` 값은 Gemini 요청에 전달되지 않는다.

## 5. OpenAI로 실행

### 5.1 API key 설정

```bash
export OPENAI_API_KEY="본인의_OpenAI_API_KEY"
```

### 5.2 현재 기본 모델로 실행

```bash
cd /home/unist/Test_hand/affordgrasp_icar

PYTHONPATH=.. python -m affordgrasp_icar \
  --provider openai \
  --image /절대경로/test_scene.png \
  --instruction "I need to tighten screws; choose the right tool." \
  --image-detail high \
  --reasoning-effort medium \
  --min-confidence 0 \
  --output ./openai_result.json
```

`--model`을 생략하면 `gpt-5.6-terra`를 사용한다.

### 5.3 논문의 GPT-4o 조건으로 실행

```bash
PYTHONPATH=.. python -m affordgrasp_icar \
  --provider openai \
  --model gpt-4o \
  --image /절대경로/test_scene.png \
  --instruction "I need to tighten screws; choose the right tool." \
  --image-detail high \
  --min-confidence 0 \
  --output /절대경로/gpt4o_icar_result.json
```

GPT-4o 경로에서는 코드가 `reasoning_effort`를 API 요청에 추가하지 않는다.

## 6. RealSense D435 테스트 이미지 촬영

### 6.1 촬영 화면 확인

```bash
realsense-viewer
```

Color와 Depth 화면에서 물체가 잘 보이는지 확인한 뒤 Viewer를 완전히 종료한다.

### 6.2 RGB/Depth 촬영

```bash
cd /home/unist/Test_hand/affordgrasp_icar
mkdir -p ./captures/icar_d435

PYTHONPATH=.. python -m affordgrasp_icar \
  --camera \
  --capture-only \
  --capture-dir ./captures/icar_d435 \
  --capture-prefix 01_test
```

`--capture-prefix`만 변경하면 여러 장면을 구분해 촬영할 수 있다. 같은 접두어는 기존 파일을 덮어쓴다.

### 6.3 촬영 결과

```text
/home/unist/Test_hand/affordgrasp_icar/captures/icar_d435/01_test_rgb.png
/home/unist/Test_hand/affordgrasp_icar/captures/icar_d435/01_test_depth_preview.png
```

두 파일을 열어 RGB가 흐리거나 잘리지 않았는지, Depth에서 물체가 구분되는지만 확인한다. ICAR에는 `_rgb.png` 파일을 사용한다.

## 7. 주요 CLI 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--image PATH` | 저장된 RGB 장면 이미지 | `--camera`와 둘 중 하나 필수 |
| `--camera` | RealSense D435에서 현재 장면 캡처 | `false` |
| `--instruction TEXT` | 사용자 작업 지시 | 추론 시 필수 |
| `--provider` | `openai` 또는 `gemini` | `openai` |
| `--model` | provider가 지원하는 모델 | provider별 기본 모델 |
| `--image-detail` | `low`, `high`, `original`, `auto` | `high` |
| `--reasoning-effort` | 모델 추론 강도 | `medium` |
| `--min-confidence` | 안전 게이트 최소 confidence | `0.70` |
| `--output PATH` | 전체 ICAR 결과 저장 | 저장하지 않음 |
| `--grounding-output PATH` | 안전 게이트 통과 grounding 요청 저장 | 저장하지 않음 |
| `--capture-only` | 카메라 캡처 후 API 호출 없이 종료 | `false` |

환경변수로 기본 provider와 모델을 지정할 수도 있다.

```bash
export AFFORDGRASP_PROVIDER="gemini"
export AFFORDGRASP_MODEL="gemini-3.6-flash"
```

그 후 `--provider`, `--model`을 생략할 수 있다.

## 8. 출력 확인

성공한 ICAR 결과 예시는 다음과 같다.

```json
{
  "task_analysis": "The task is to tighten screws and requires applying torque.",
  "object_identification": "A screwdriver is visible and best fits the task.",
  "part_selection": "The handle should be grasped while preserving the tip.",
  "affordance_reasoning": "The handle supports a secure rotational grip.",
  "task": "tighten screws",
  "object": "screwdriver",
  "object_part": "handle",
  "affordance": "grasp",
  "is_actionable": true,
  "confidence": 0.92,
  "failure_reason": ""
}
```

저장된 JSON 확인:

```bash
python -m json.tool \
  ./gemini_result.json
```

핵심 판정 항목:

| 필드 | 확인할 내용 |
|---|---|
| `task` | 암시적인 사용자 의도가 올바르게 명시화됐는가 |
| `object` | 실제 이미지에 보이는 작업 관련 물체인가 |
| `object_part` | 작업 부위를 보존하는 안전한 파지 부위인가 |
| `affordance` | 짧은 영어 기본형 동사인가 |
| `is_actionable` | 하나의 안전한 목표가 명확한가 |
| `confidence` | `[0, 1]` 범위인가 |
| `failure_reason` | 실행 불가일 때 구체적인 이유가 있는가 |

## 9. 권장 테스트 시나리오

아래 모든 장면 이미지는 6장의 절차에 따라 D435로 촬영한다.

### 9.1 정상 선택

```text
장면: screwdriver, cup, sponge
지시: I need to tighten screws.
기대: screwdriver / handle / grasp
```

### 9.2 암시적 작업

```text
장면: mug, plate, spoon
지시: I am thirsty.
기대: mug / handle / grasp
```

### 9.3 작업 부위 보존

```text
장면: knife와 다른 주방 도구
지시: I want to cut fruit.
기대: knife / handle / grasp
주의: blade를 파지 부위로 선택하면 실패
```

### 9.4 적절한 물체 없음

```text
장면: cup와 sponge만 있음
지시: I need to tighten screws.
기대: is_actionable=false
```

### 9.5 모호한 장면

```text
장면: 비슷한 도구가 겹치거나 손잡이가 가려짐
기대: 낮은 confidence 또는 is_actionable=false
```

provider를 비교할 때는 동일한 이미지, 지시, 출력 schema, confidence 기준을 사용한다.

## 10. 종료 코드

| 종료 코드 | 의미 |
|---:|---|
| `0` | ICAR 성공 또는 카메라 캡처 전용 성공 |
| `2` | CLI 입력, provider 설정, API 호출 또는 JSON 검증 실패 |
| `3` | `is_actionable=false` 또는 confidence가 임계값 미만 |
| `4` | 처리된 RealSense D435 카메라 오류 |

직전 명령의 종료 코드 확인:

```bash
echo $?
```

종료 코드 `3`은 API 호출 자체의 실패를 의미하지 않는다. 전체 결과 JSON에서 `is_actionable`, `confidence`, `failure_reason`을 확인해야 한다.

## 11. 문제 해결

### `No module named 'openai'`

```bash
python -m pip install -U openai
```

현재 실행에 사용하는 `python`과 설치에 사용하는 `python -m pip`가 같은 환경인지 확인한다.

### `Gemini API key is missing`

```bash
export GEMINI_API_KEY="..."
```

같은 터미널에서 실행했는지 확인한다.

### `OpenAI API key is missing`

```bash
export OPENAI_API_KEY="..."
```

### `RGB image does not exist`

상대 경로 대신 절대 경로를 사용한다.

```bash
realpath /path/to/test_scene.png
```

### `unsupported image type`

이미지를 JPEG, PNG 또는 WebP로 변환한다.

### `rs-save-to-disk`를 찾을 수 없음

```bash
command -v rs-save-to-disk
```

아무것도 출력되지 않으면 공식 librealsense 설치 안내에 따라 `librealsense2-utils`를 설치한다.

### D435가 사용 중이거나 장치를 열 수 없음

`realsense-viewer`, RealSense ROS 노드, 다른 카메라 프로그램을 모두 종료하고 다시 실행한다. 한 번에 한 프로그램만 D435를 사용하게 한다.

### RGB 또는 depth 파일이 생성되지 않음

- Viewer에서 Color와 Depth 스트림이 둘 다 정상인지 확인한다.
- 촬영 중 D435나 물체를 움직이지 않는다.
- 현재 코드는 `rs-save-to-disk`가 만든 Color/Depth PNG 두 파일을 모두 기대하므로 한쪽만 없으면 카메라 오류로 처리한다.

### `Gemini reasoning_effort must be low, medium, or high`

Gemini에서는 다음처럼 실행한다.

```bash
--reasoning-effort medium
```

`none`, `xhigh`, `max`는 현재 Gemini provider 설정에서 허용하지 않는다.

### Gemini quota 또는 rate limit 오류

- Google AI Studio에서 사용량과 API key 상태를 확인한다.
- 잠시 후 다시 시도한다.
- 무료 tier 한도를 넘었다면 다음 quota 갱신을 기다리거나 유료 tier를 사용한다.

### 결과는 저장됐지만 종료 코드가 `3`

ICAR 응답은 정상적으로 생성됐지만 안전 게이트가 차단한 경우다.

```bash
python -m json.tool /절대경로/result.json
```

다음 값을 확인한다.

```text
is_actionable
confidence
failure_reason
```

## 12. API 호출 없는 mock 테스트

실제 API key나 네트워크 없이 provider별 요청 형식과 응답 파싱을 검사할 수 있다.

```bash
cd /home/unist/Test_hand/affordgrasp_icar

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=.. \
  python -m unittest discover \
  -s tests \
  -t .. \
  -v
```

현재 테스트 범위:

- OpenAI Responses API 요청 구조
- Gemini Chat Completions 요청 구조
- Gemini image data URL
- JSON Schema 전달
- provider별 기본 모델
- Gemini 호환 base URL
- provider별 API key 오류
- 기존 `OpenAIAffordanceReasoner` alias 호환성

mock 테스트는 실제 모델 접근 권한, API quota, 네트워크 또는 모델의 ICAR 정확도를 검증하지 않는다.

## 13. API key와 데이터 주의사항

- API key를 코드, 명령 기록, 문서 또는 결과 JSON에 넣지 않는다.
- 가능하면 `local_config.py`보다 환경변수를 사용한다.
- 현재 `.gitignore`는 `local_config.py`를 제외하지만 파일 복사와 화면 공유까지 막지는 않는다.
- Gemini 무료 tier에 전송한 콘텐츠는 Google 제품 개선에 사용될 수 있다.
- 로봇 카메라 장면에 사람, 화면, 문서 또는 민감한 정보가 보이면 외부 API 전송 전에 제거한다.
- 가격과 무료 quota는 변경될 수 있으므로 실행 전 공식 가격 페이지를 확인한다.
