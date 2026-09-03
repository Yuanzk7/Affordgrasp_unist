"""Command-line entry point for ICAR inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .models import UnsafeGroundingRequest
from .reasoner import (
    AffordanceReasoner,
    ReasoningError,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_DIR / "captures" / "icar_d435"
RUNS_DIR = PROJECT_DIR / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AffordGrasp In-Context Affordance Reasoning: "
            "장면을 촬영하거나 RGB 이미지에서 grasp 대상을 추론합니다."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="저장된 RGB 장면 이미지")
    source.add_argument(
        "--camera",
        metavar="PREFIX",
        help="RealSense 장면만 촬영하고 captures/icar_d435에 저장",
    )
    parser.add_argument("--instruction", help="사용자의 작업 지시")
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _result_dir(image_path: str) -> Path:
    prefix = Path(image_path).stem.removesuffix("_rgb")
    if not prefix:
        raise ValueError("image filename must contain a run prefix")
    return RUNS_DIR / prefix / "json"


def _run_camera(prefix: str) -> int:
    from ..camera import CameraCaptureError, capture_realsense

    try:
        capture = capture_realsense(output_dir=CAPTURE_DIR, prefix=prefix)
    except (CameraCaptureError, ValueError) as exc:
        print(f"카메라 오류: {exc}", file=sys.stderr)
        return 4

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
    print(json.dumps(capture.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _run_icar(image_path: str, instruction: str) -> int:
    try:
        result_dir = _result_dir(image_path)
        result = AffordanceReasoner().reason(instruction, image_path)
    except (ReasoningError, ValueError) as exc:
        print(f"ICAR 오류: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    _write_json(result_dir / "icar_result.json", payload)

    try:
        grounding = result.to_grounding_request()
    except UnsafeGroundingRequest as exc:
        print(
            f"ICAR 대상 없음: {exc}. grounding 요청을 생성하지 않습니다.",
            file=sys.stderr,
        )
        return 3

    _write_json(result_dir / "grounding_request.json", grounding.to_dict())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.camera is not None:
        if args.instruction:
            parser.error("--instruction cannot be used with standalone --camera")
        return _run_camera(args.camera)

    if not args.instruction:
        parser.error("--instruction is required with --image")
    return _run_icar(args.image, args.instruction)
