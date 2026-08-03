"""CLI for the Object Localization stage of Visual Affordance Grounding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .object_localization import (
    ObjectLocalizationError,
    VLPartObjectDetector,
    config_from_environment,
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
        help="masked image와 결과 JSON 저장 폴더",
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
        default=0.5,
        help="최소 VLPart object score (기본: 0.5)",
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

    try:
        object_name = arguments.object_name
        if arguments.request is not None:
            object_name = object_name_from_grounding_request(arguments.request)

        config = config_from_environment(
            root=arguments.vlpart_root,
            weights=arguments.weights,
            config_file=arguments.config_file,
            confidence_threshold=arguments.confidence_threshold,
            device=arguments.device,
        )
        detector = VLPartObjectDetector(config)
        result = localize_object_file(
            image_path=arguments.image,
            object_name=object_name,
            detector=detector,
            output_dir=arguments.output_dir,
            minimum_score=arguments.confidence_threshold,
        )
    except (ObjectLocalizationError, ValueError) as exc:
        print(f"Object Localization 오류: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
