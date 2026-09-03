"""Generate the official xArm7 + xArm Gripper URDF for PyBullet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ..robot.full_collision_validation import _generate_urdf


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xarm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gripper-version", default="G1")
    parser.add_argument("--model1300", action="store_true")
    arguments = parser.parse_args(argv)
    urdf, _ = _generate_urdf(
        arguments.xarm_root,
        arguments.output,
        arguments.model1300,
        arguments.gripper_version,
    )
    print(urdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
