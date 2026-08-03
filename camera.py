"""Capture one RGB/depth-preview pair from an Intel RealSense camera."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union


class CameraCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class RealSenseCapture:
    rgb_path: Path
    depth_preview_path: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "rgb_path": str(self.rgb_path),
            "depth_preview_path": str(self.depth_preview_path),
        }


def capture_realsense(
    output_dir: Union[str, Path] = "captures",
    prefix: str = "scene",
) -> RealSenseCapture:
    """Capture current frames and return their saved paths."""

    executable = shutil.which("rs-save-to-disk")
    if executable is None:
        raise CameraCaptureError(
            "rs-save-to-disk가 없습니다. librealsense2-utils를 설치하세요."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"{prefix}_rgb.png"
    depth_path = output_dir / f"{prefix}_depth_preview.png"

    with tempfile.TemporaryDirectory() as temporary_dir:
        result = subprocess.run(
            [executable],
            cwd=temporary_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise CameraCaptureError(result.stderr.strip() or "카메라 캡처 실패")

        source = Path(temporary_dir)
        rgb_source = source / "rs-save-to-disk-output-Color.png"
        depth_source = source / "rs-save-to-disk-output-Depth.png"
        if not rgb_source.exists() or not depth_source.exists():
            raise CameraCaptureError("RGB 또는 Depth 프레임을 받지 못했습니다.")

        shutil.copy2(rgb_source, rgb_path)
        shutil.copy2(depth_source, depth_path)

    return RealSenseCapture(rgb_path, depth_path)
