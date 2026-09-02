"""Four-field data contracts between affordance reasoning and grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


class InvalidReasoningResult(ValueError):
    """Raised when a model response does not satisfy the ICAR contract."""


class UnsafeGroundingRequest(RuntimeError):
    """Raised when an unavailable ICAR label would reach grounding."""


_RESULT_FIELDS = ("task", "object", "object_part", "affordance")


def _require_text(payload: Dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise InvalidReasoningResult(f"{field!r} must be a string")
    value = value.strip()
    if not value:
        raise InvalidReasoningResult(f"{field!r} must not be empty")
    return value


@dataclass(frozen=True)
class GroundingRequest:
    """Minimal semantic target expected by visual affordance grounding."""

    task: str
    object_name: str
    part_name: str
    affordance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AffordanceReasoningResult:
    """Paper-aligned ICAR result: task, object, object part, affordance."""

    task: str
    object: str
    object_part: str
    affordance: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AffordanceReasoningResult":
        if not isinstance(payload, dict):
            raise InvalidReasoningResult("reasoning output must be a JSON object")

        expected = set(_RESULT_FIELDS)
        missing = expected - set(payload)
        if missing:
            raise InvalidReasoningResult(
                "missing required fields: " + ", ".join(sorted(missing))
            )
        unexpected = set(payload) - expected
        if unexpected:
            raise InvalidReasoningResult(
                "unexpected fields: " + ", ".join(sorted(unexpected))
            )

        text = {field: _require_text(payload, field) for field in _RESULT_FIELDS}
        return cls(**text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_grounding_request(self) -> GroundingRequest:
        """Create a downstream request when every canonical label is known."""

        unavailable = [
            field
            for field in _RESULT_FIELDS
            if getattr(self, field).casefold() == "none"
        ]
        if unavailable:
            raise UnsafeGroundingRequest(
                "unavailable ICAR fields: " + ", ".join(unavailable)
            )

        return GroundingRequest(
            task=self.task,
            object_name=self.object,
            part_name=self.object_part,
            affordance=self.affordance,
        )
