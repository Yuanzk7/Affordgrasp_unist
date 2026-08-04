"""CLI for AffordGrasp Affordance Mask Prediction (equation 4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .affordance_mask import (
    AffordanceMaskPredictionError,
    AffordanceTarget,
    VLPartAffordanceMaskPredictor,
    load_affordance_target,
    predict_affordance_mask_file,
)
from .object_localization import config_from_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VLPart Affordance Mask Prediction: M_BO에서 ICAR가 선택한 "
            "object part의 binary mask M_p*를 생성합니다."
        )
    )
    parser.add_argument(
        "--localization",
        type=Path,
        required=True,
        help="이전 단계의 object_localization.json",
    )
    parser.add_argument(
        "--request",
        type=Path,
        help="grounding request 또는 전체 ICAR 결과 JSON",
    )
    parser.add_argument("--object-name", help="직접 지정할 object name")
    parser.add_argument("--part-name", help="직접 지정할 optimal part p*")
    parser.add_argument("--affordance", help="직접 지정할 affordance a*")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/affordance_mask"),
        help="mask와 overlay 저장 폴더",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        help="affordance_mask.json을 모아둘 공통 JSON 폴더",
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
        default=0.10,
        help="최소 VLPart part score (기본: 0.10)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="논문 segmentation 입력 한 변 크기 (기본: 224, 0이면 원본 크기)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="VLPart 실행 장치 (기본: auto)",
    )
    return parser


def _target_from_arguments(arguments: argparse.Namespace) -> AffordanceTarget:
    if arguments.request is not None:
        if any(
            value is not None
            for value in (
                arguments.object_name,
                arguments.part_name,
                arguments.affordance,
            )
        ):
            raise AffordanceMaskPredictionError(
                "--request cannot be combined with direct target arguments"
            )
        return load_affordance_target(arguments.request)

    values = {
        "--object-name": arguments.object_name,
        "--part-name": arguments.part_name,
        "--affordance": arguments.affordance,
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise AffordanceMaskPredictionError(
            "use --request or provide all of: " + ", ".join(values)
        )
    return AffordanceTarget(
        object_name=arguments.object_name,
        part_name=arguments.part_name,
        affordance=arguments.affordance,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if not 0.0 <= arguments.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if arguments.input_size < 0:
        parser.error("--input-size must be 0 or a positive integer")

    try:
        target = _target_from_arguments(arguments)
        config = config_from_environment(
            root=arguments.vlpart_root,
            weights=arguments.weights,
            config_file=arguments.config_file,
            confidence_threshold=arguments.confidence_threshold,
            device=arguments.device,
        )
        predictor = VLPartAffordanceMaskPredictor(
            config,
            input_size=arguments.input_size or None,
        )
        result = predict_affordance_mask_file(
            localization_path=arguments.localization,
            target=target,
            predictor=predictor,
            output_dir=arguments.output_dir,
            minimum_score=arguments.confidence_threshold,
            json_dir=arguments.json_dir,
        )
    except (AffordanceMaskPredictionError, ValueError) as exc:
        print(f"Affordance Mask Prediction 오류: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
