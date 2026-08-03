"""Validated data contracts between affordance reasoning and grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


class InvalidReasoningResult(ValueError):
    """Raised when a model response does not satisfy the ICAR contract."""


class UnsafeGroundingRequest(RuntimeError):
    """Raised when a non-actionable result is passed to robot perception."""


_TEXT_FIELDS = (
    "task_analysis",
    "object_identification",
    "part_selection",
    "affordance_reasoning",
    "task",
    "object",
    "object_part",
    "affordance",
    "failure_reason",
)


def _require_text(payload: Dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise InvalidReasoningResult(f"{field!r} must be a string")
    value = value.strip()
    if field != "failure_reason" and not value:
        raise InvalidReasoningResult(f"{field!r} must not be empty")
    return value


@dataclass(frozen=True)
class GroundingRequest:
    """Minimal semantic target expected by visual affordance grounding."""

    task: str
    object_name: str
    part_name: str
    affordance: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AffordanceReasoningResult:
    """Paper-aligned ICAR result plus an execution safety gate."""

    task_analysis: str
    object_identification: str
    part_selection: str
    affordance_reasoning: str
    task: str
    object: str
    object_part: str
    affordance: str
    is_actionable: bool
    confidence: float
    failure_reason: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AffordanceReasoningResult":
        if not isinstance(payload, dict):
            raise InvalidReasoningResult("reasoning output must be a JSON object")

        missing = set(_TEXT_FIELDS + ("is_actionable", "confidence")) - set(payload)
        if missing:
            raise InvalidReasoningResult(
                "missing required fields: " + ", ".join(sorted(missing))
            )

        text = {field: _require_text(payload, field) for field in _TEXT_FIELDS}

        is_actionable = payload["is_actionable"]
        if not isinstance(is_actionable, bool):
            raise InvalidReasoningResult("'is_actionable' must be a boolean")

        confidence = payload["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise InvalidReasoningResult("'confidence' must be a number")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise InvalidReasoningResult("'confidence' must be between 0 and 1")

        if not is_actionable and not text["failure_reason"]:
            raise InvalidReasoningResult(
                "'failure_reason' is required when the result is not actionable"
            )

        return cls(
            **text,
            is_actionable=is_actionable,
            confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def actionable_at(self, minimum_confidence: float) -> bool:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        return self.is_actionable and self.confidence >= minimum_confidence

    def to_grounding_request(
        self, minimum_confidence: float = 0.70
    ) -> GroundingRequest:
        """Create a downstream request only after the safety gate passes."""

        if not self.actionable_at(minimum_confidence):
            reason = self.failure_reason or (
                f"confidence {self.confidence:.2f} is below "
                f"{minimum_confidence:.2f}"
            )
            raise UnsafeGroundingRequest(reason)

        return GroundingRequest(
            task=self.task,
            object_name=self.object,
            part_name=self.object_part,
            affordance=self.affordance,
            confidence=self.confidence,
        )
