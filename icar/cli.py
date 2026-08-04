"""Command-line entry point for ICAR inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..camera import CameraCaptureError, capture_realsense
from .reasoner import (
    AffordanceReasoner,
    AffordanceReasonerConfig,
    ReasoningError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AffordGrasp In-Context Affordance Reasoning: "
            "RGB 이미지와 사용자 지시에서 grasp 대상을 추론합니다."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="저장된 RGB 장면 이미지")
    source.add_argument(
        "--camera",
        action="store_true",
        help="연결된 Intel RealSense의 현재 장면을 캡처해 사용",
    )
    parser.add_argument("--instruction", help="사용자의 작업 지시")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures"),
        help="--camera로 캡처한 RGB/raw·filtered depth 저장 폴더",
    )
    parser.add_argument(
        "--capture-prefix",
        default="scene",
        help="캡처 파일명 prefix (기본: scene, 같은 이름은 덮어씀)",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="카메라 캡처만 검증하고 API 추론 없이 종료",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        default=os.environ.get("AFFORDGRASP_PROVIDER", "openai"),
        help="ICAR API provider (기본: openai, 무료 테스트는 gemini)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AFFORDGRASP_MODEL"),
        help=(
            "비전 및 Structured Outputs 지원 모델 "
            "(기본: openai=gpt-5.6-terra, gemini=gemini-3.6-flash)"
        ),
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default="high",
        help="ICAR는 기본적으로 part 인식을 위해 high를 사용합니다.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
        help="후속 grounding으로 전달할 최소 신뢰도 (기본 0.70)",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        help=(
            "모든 ICAR JSON을 모을 폴더 "
            "(icar_result.json, grounding_request.json)"
        ),
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")

    image_path = args.image
    if args.capture_only and not args.camera:
        parser.error("--capture-only requires --camera")
    if not args.capture_only and not args.instruction:
        parser.error("--instruction is required unless --capture-only is used")

    if args.camera:
        try:
            capture = capture_realsense(
                output_dir=args.capture_dir,
                prefix=args.capture_prefix,
            )
        except (CameraCaptureError, ValueError) as exc:
            print(f"카메라 오류: {exc}", file=sys.stderr)
            return 4
        image_path = str(capture.rgb_path)
        print(
            "RealSense 캡처 완료: "
            f"RGB={capture.rgb_path}, "
            f"raw depth={capture.depth_raw_path}, "
            f"filtered depth={capture.depth_filtered_path}, "
            f"camera info={capture.camera_info_path}, "
            f"depth valid={capture.raw_valid_ratio:.1%}"
            f"→{capture.filtered_valid_ratio:.1%}",
            file=sys.stderr,
        )
        if args.capture_only:
            capture_payload = capture.to_dict()
            print(json.dumps(capture_payload, ensure_ascii=False, indent=2))
            if args.json_dir is not None:
                _write_json(
                    args.json_dir / "capture_result.json",
                    capture_payload,
                )
            return 0

    try:
        config = AffordanceReasonerConfig(
            model=args.model,
            image_detail=args.image_detail,
            reasoning_effort=args.reasoning_effort,
            provider=args.provider,
        )
        result = AffordanceReasoner(config).reason(args.instruction, image_path)
    except (ReasoningError, ValueError) as exc:
        print(f"ICAR 오류: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.json_dir is not None:
        _write_json(args.json_dir / "icar_result.json", payload)

    if not result.actionable_at(args.min_confidence):
        reason = result.failure_reason or (
            f"confidence {result.confidence:.2f} < {args.min_confidence:.2f}"
        )
        print(
            f"안전 게이트 차단: {reason}. 로봇/grounding 명령을 생성하지 않습니다.",
            file=sys.stderr,
        )
        return 3

    if args.json_dir is not None:
        grounding = result.to_grounding_request(args.min_confidence).to_dict()
        _write_json(args.json_dir / "grounding_request.json", grounding)
    return 0
