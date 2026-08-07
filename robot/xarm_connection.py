"""Shared read-only connection/report helpers for the xArm SDK."""

from __future__ import annotations

import time
from typing import Any, Optional, Type

import numpy as np


def connect_xarm(
    robot_ip: str,
    error_type: Type[RuntimeError],
    report_timeout_s: float = 5.0,
) -> Any:
    """Connect and wait until the first complete rich controller report arrives."""

    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise error_type(
            "xArm SDK is missing; install it with "
            "'python -m pip install xarm-python-sdk'"
        ) from exc

    arm = XArmAPI(
        robot_ip,
        is_radian=True,
        do_not_open=False,
        enable_report=True,
        report_type="rich",
    )
    if not arm.connected:
        raise error_type(f"could not connect to xArm at {robot_ip}")

    # tcp_offset is a value cached from the asynchronous rich-report socket.
    # The SDK initializes the cache to zero, so reading it immediately after
    # connect can incorrectly report [0, 0, 0, 0, 0, 0].
    deadline = time.monotonic() + report_timeout_s
    internal = getattr(arm, "_arm", None)
    while time.monotonic() < deadline:
        if bool(getattr(internal, "_first_report_over", False)):
            return arm
        if not arm.connected:
            break
        time.sleep(0.05)

    arm.disconnect()
    raise error_type(
        f"xArm at {robot_ip} did not provide a complete rich status report "
        f"within {report_timeout_s:.1f} seconds"
    )


def read_tcp_offset(
    arm: Any,
    error_type: Type[RuntimeError],
    expected: Optional[np.ndarray] = None,
    tolerance: float = 0.0,
    timeout_s: float = 1.0,
) -> np.ndarray:
    """Read the cached offset, optionally allowing report updates to converge."""

    deadline = time.monotonic() + timeout_s
    last = np.empty(0, dtype=np.float64)
    while True:
        last = np.asarray(getattr(arm, "tcp_offset", []), dtype=np.float64)
        if last.shape == (6,) and np.all(np.isfinite(last)):
            if expected is None or np.allclose(last, expected, atol=tolerance):
                return last
        if time.monotonic() >= deadline or not arm.connected:
            break
        time.sleep(0.05)
    if last.shape != (6,) or not np.all(np.isfinite(last)):
        raise error_type("xArm rich report did not contain a valid TCP offset")
    return last
