"""CLI for the Object Localization stage of Visual Affordance Grounding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .object_localization import (
    DEFAULT_GEMINI_SELECTOR_MODEL,
    GeminiTopKObjectSelector,
    GeminiTopKSelectorConfig,
    ObjectLocalizationError,
    VLPartObjectDetector,
    config_from_environment,
    grounding_context_from_request,
    localize_object_file,
    object_name_from_grounding_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VLPart Object Localization: RGB에서 B_O를 찾고 "
            "bounding box 밖을 0으로 만든 M_BO를 저장합니다."
        )
    )
    parser.add_argument("--image", type=Path, required=True, help="입력 RGB 이미지")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--object-name", help="VLPart object query")
    query.add_argument(
        "--request",
        type=Path,
        help="ICAR가 생성한 grounding_request.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/object_localization"),
        help="masked image 저장 폴더",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        help="object_localization.json을 모아둘 공통 JSON 폴더",
    )
    parser.add_argument(
        "--vlpart-root",
        help="공식 VLPart 저장소 경로 (또는 VLPART_ROOT)",
    )
    parser.add_argument(
        "--config-file",
        help="VLPart YAML config 경로 (또는 VLPART_CONFIG)",
    )
    parser.add_argument(
        "--weights",
        help="VLPart checkpoint 경로 (또는 VLPART_WEIGHTS)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.01,
        help="top-k 후보에 포함할 최소 VLPart score (기본: 0.01)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Gemini가 재선택할 VLPart 후보 수 (기본: 10)",
    )
    parser.add_argument(
        "--selection-method",
        choices=("gemini-top-k", "vlpart-score"),
        default="gemini-top-k",
        help=(
            "후보 선택 방식 (기본: gemini-top-k). "
            "vlpart-score는 API 없이 기존 최고점 후보를 선택합니다."
        ),
    )
    parser.add_argument(
        "--selector-model",
        default=DEFAULT_GEMINI_SELECTOR_MODEL,
        help=(
            "top-k 이미지 재선택에 사용할 Gemini 모델 "
            f"(기본: {DEFAULT_GEMINI_SELECTOR_MODEL})"
        ),
    )
    parser.add_argument(
        "--selector-confidence-threshold",
        type=float,
        default=0.55,
        help="Gemini 후보 선택 최소 confidence (기본: 0.55)",
    )
    parser.add_argument(
        "--selector-timeout",
        type=float,
        default=60.0,
        help="Gemini 후보 선택 timeout 초 (기본: 60)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="VLPart 실행 장치 (기본: auto)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not 0.0 <= arguments.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if arguments.top_k <= 0:
        parser.error("--top-k must be positive")
    if not 0.0 <= arguments.selector_confidence_threshold <= 1.0:
        parser.error(
            "--selector-confidence-threshold must be between 0 and 1"
        )
    if arguments.selector_timeout <= 0:
        parser.error("--selector-timeout must be positive")

    try:
        object_name = arguments.object_name
        selection_context = None
        if arguments.request is not None:
            object_name = object_name_from_grounding_request(arguments.request)
            selection_context = grounding_context_from_request(arguments.request)

        config = config_from_environment(
            root=arguments.vlpart_root,
            weights=arguments.weights,
            config_file=arguments.config_file,
            confidence_threshold=arguments.confidence_threshold,
            device=arguments.device,
        )
        detector = VLPartObjectDetector(config)
        selector = None
        if arguments.selection_method == "gemini-top-k":
            selector = GeminiTopKObjectSelector(
                GeminiTopKSelectorConfig(
                    model=arguments.selector_model,
                    minimum_confidence=(
                        arguments.selector_confidence_threshold
                    ),
                    timeout_seconds=arguments.selector_timeout,
                )
            )
        result = localize_object_file(
            image_path=arguments.image,
            object_name=object_name,
            detector=detector,
            output_dir=arguments.output_dir,
            minimum_score=arguments.confidence_threshold,
            json_dir=arguments.json_dir,
            top_k=arguments.top_k,
            selector=selector,
            selection_context=selection_context,
        )
    except (ObjectLocalizationError, ValueError) as exc:
        print(f"Object Localization 오류: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
