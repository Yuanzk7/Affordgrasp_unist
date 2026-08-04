"""AffordGrasp affordance-mask prediction on a localized object image."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import cv2
import numpy as np

from .object_localization import (
    ObjectLocalizationError,
    VLPartConfig,
    VLPartObjectDetector,
    clamp_bbox_xyxy,
)


class AffordanceMaskPredictionError(RuntimeError):
    """Raised when no usable part-level affordance mask can be produced."""


@dataclass(frozen=True)
class AffordanceTarget:
    """ICAR semantics consumed by AffordGrasp equation (4)."""

    object_name: str
    part_name: str
    affordance: str


@dataclass(frozen=True)
class AffordanceMaskCandidate:
    """One binary part-mask candidate returned by a segmentation backend."""

    mask: np.ndarray
    score: float
    query: str


class AffordanceMaskPredictor(Protocol):
    """Backend contract for part-level mask prediction."""

    def predict(
        self,
        masked_image_bgr: np.ndarray,
        target: AffordanceTarget,
    ) -> Sequence[AffordanceMaskCandidate]:
        """Return candidate masks in the input image coordinate system."""


@dataclass(frozen=True)
class AffordanceMaskResult:
    """Serializable output of the Affordance Mask Prediction stage."""

    object_name: str
    part_name: str
    affordance: str
    vlpart_query: str
    score: float
    object_bbox_xyxy: Tuple[int, int, int, int]
    mask_bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    mask_area_pixels: int
    image_width: int
    image_height: int
    object_localization: str
    masked_object_image: str
    affordance_mask: str
    affordance_overlay: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_name": self.object_name,
            "part_name": self.part_name,
            "affordance": self.affordance,
            "vlpart_query": self.vlpart_query,
            "score": self.score,
            "object_bbox_xyxy": list(self.object_bbox_xyxy),
            "mask_bbox_xyxy": list(self.mask_bbox_xyxy),
            "centroid_xy": list(self.centroid_xy),
            "mask_area_pixels": self.mask_area_pixels,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "object_localization": self.object_localization,
            "masked_object_image": self.masked_object_image,
            "affordance_mask": self.affordance_mask,
            "affordance_overlay": self.affordance_overlay,
        }


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AffordanceMaskPredictionError(
            f"{field} must be a non-empty string"
        )
    return value.strip()


def build_vlpart_part_query(target: AffordanceTarget) -> str:
    """Build one open-vocabulary label from ``O``, ``p*``, and ``a*``.

    VLPart documents object-part phrases such as ``fox head``. AffordGrasp
    equation (4) additionally conditions the prediction on the affordance, so
    the phrase is extended to ``<object> <part> for <affordance>``.
    """

    object_name = _require_text(target.object_name, "object_name")
    part_name = _require_text(target.part_name, "part_name")
    affordance = _require_text(target.affordance, "affordance")
    if object_name.casefold() == part_name.casefold():
        return f"{part_name} for {affordance}"
    return f"{object_name} {part_name} for {affordance}"


class VLPartAffordanceMaskPredictor:
    """Extract Detectron2 ``pred_masks`` from the existing VLPart runtime."""

    def __init__(
        self,
        config: VLPartConfig,
        input_size: Optional[int] = 224,
        detector: Optional[VLPartObjectDetector] = None,
    ) -> None:
        if input_size is not None and input_size <= 0:
            raise ValueError("input_size must be positive or None")
        self.config = config
        self.input_size = input_size
        self.detector = detector or VLPartObjectDetector(config)

    def predict(
        self,
        masked_image_bgr: np.ndarray,
        target: AffordanceTarget,
    ) -> Sequence[AffordanceMaskCandidate]:
        if not isinstance(masked_image_bgr, np.ndarray):
            raise AffordanceMaskPredictionError("masked image must be an array")
        if masked_image_bgr.ndim != 3 or masked_image_bgr.shape[2] != 3:
            raise AffordanceMaskPredictionError(
                "masked image must have shape (H, W, 3)"
            )

        height, width = masked_image_bgr.shape[:2]
        inference_image = masked_image_bgr
        if self.input_size is not None:
            inference_image = cv2.resize(
                masked_image_bgr,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

        query = build_vlpart_part_query(target)
        try:
            instances = self.detector._predict_instances(
                inference_image,
                query,
            )
        except ObjectLocalizationError as exc:
            raise AffordanceMaskPredictionError(str(exc)) from exc

        if instances is None or not instances.has("pred_masks"):
            return []

        masks = instances.pred_masks.detach().numpy()
        scores = instances.scores.detach().numpy()
        candidates: List[AffordanceMaskCandidate] = []
        for raw_mask, raw_score in zip(masks, scores):
            score = float(raw_score)
            if score < self.config.confidence_threshold:
                continue
            mask = np.asarray(raw_mask, dtype=np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_bool = mask.astype(bool)
            if not np.any(mask_bool):
                continue
            candidates.append(
                AffordanceMaskCandidate(
                    mask=mask_bool,
                    score=score,
                    query=query,
                )
            )
        return candidates


def load_affordance_target(
    request_path: Union[str, Path],
) -> AffordanceTarget:
    """Load either a grounding request or the full ICAR result JSON."""

    path = Path(request_path).expanduser().resolve()
    if not path.is_file():
        raise AffordanceMaskPredictionError(
            f"affordance request does not exist: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AffordanceMaskPredictionError(
            f"could not read affordance request JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AffordanceMaskPredictionError(
            "affordance request must be a JSON object"
        )

    return AffordanceTarget(
        object_name=_require_text(
            payload.get("object_name", payload.get("object")),
            "object_name",
        ),
        part_name=_require_text(
            payload.get("part_name", payload.get("object_part")),
            "part_name",
        ),
        affordance=_require_text(payload.get("affordance"), "affordance"),
    )


def _load_object_localization(
    localization_path: Union[str, Path],
) -> Tuple[Path, Dict[str, Any]]:
    path = Path(localization_path).expanduser().resolve()
    if not path.is_file():
        raise AffordanceMaskPredictionError(
            f"object localization JSON does not exist: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AffordanceMaskPredictionError(
            f"could not read object localization JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise AffordanceMaskPredictionError(
            "object localization result must be a JSON object"
        )
    return path, payload


def _mask_geometry(
    mask: np.ndarray,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float], int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise AffordanceMaskPredictionError(
            "predicted affordance mask is empty inside the object box"
        )
    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
    centroid = (float(xs.mean()), float(ys.mean()))
    return bbox, centroid, int(len(xs))


def predict_affordance_mask_file(
    localization_path: Union[str, Path],
    target: AffordanceTarget,
    predictor: AffordanceMaskPredictor,
    output_dir: Union[str, Path],
    minimum_score: float = 0.5,
    json_dir: Optional[Union[str, Path]] = None,
) -> AffordanceMaskResult:
    """Implement equation (4), constrain the result to B_O, and save outputs."""

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    localization, payload = _load_object_localization(localization_path)

    masked_value = payload.get("masked_image")
    if not isinstance(masked_value, str) or not masked_value.strip():
        raise AffordanceMaskPredictionError(
            "object localization JSON has no masked_image path"
        )
    masked_path = Path(masked_value).expanduser()
    if not masked_path.is_absolute():
        masked_path = localization.parent / masked_path
    masked_path = masked_path.resolve()
    if not masked_path.is_file():
        raise AffordanceMaskPredictionError(
            f"masked object image does not exist: {masked_path}"
        )

    image_bgr = cv2.imread(str(masked_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise AffordanceMaskPredictionError(
            f"OpenCV could not read masked object image: {masked_path}"
        )
    height, width = image_bgr.shape[:2]
    try:
        object_box = clamp_bbox_xyxy(
            payload.get("bbox_xyxy", ()),
            width,
            height,
        )
    except ObjectLocalizationError as exc:
        raise AffordanceMaskPredictionError(
            f"invalid object bounding box: {exc}"
        ) from exc

    candidates = [
        candidate
        for candidate in predictor.predict(image_bgr, target)
        if candidate.score >= minimum_score
    ]
    if not candidates:
        raise AffordanceMaskPredictionError(
            f"VLPart found no {target.part_name!r} part above score "
            f"{minimum_score:.2f}"
        )
    selected = max(candidates, key=lambda candidate: candidate.score)
    mask = np.asarray(selected.mask, dtype=bool)
    if mask.shape != (height, width):
        raise AffordanceMaskPredictionError(
            "predictor returned a mask with different image dimensions"
        )

    x1, y1, x2, y2 = object_box
    object_region = np.zeros((height, width), dtype=bool)
    object_region[y1:y2, x1:x2] = True
    mask &= object_region
    mask_box, centroid, area = _mask_geometry(mask)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / "affordance_mask.png"
    overlay_path = destination / "affordance_overlay.png"
    json_destination = (
        Path(json_dir).expanduser().resolve()
        if json_dir is not None
        else destination
    )
    json_destination.mkdir(parents=True, exist_ok=True)
    result_path = json_destination / "affordance_mask.json"

    binary_mask = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(mask_path), binary_mask):
        raise AffordanceMaskPredictionError(
            f"failed to write affordance mask: {mask_path}"
        )

    overlay = image_bgr.copy()
    cyan = np.zeros_like(image_bgr)
    cyan[:, :] = (255, 255, 0)
    blended = cv2.addWeighted(image_bgr, 0.45, cyan, 0.55, 0.0)
    overlay[mask] = blended[mask]
    if not cv2.imwrite(str(overlay_path), overlay):
        raise AffordanceMaskPredictionError(
            f"failed to write affordance overlay: {overlay_path}"
        )

    result = AffordanceMaskResult(
        object_name=_require_text(target.object_name, "object_name"),
        part_name=_require_text(target.part_name, "part_name"),
        affordance=_require_text(target.affordance, "affordance"),
        vlpart_query=selected.query,
        score=float(selected.score),
        object_bbox_xyxy=object_box,
        mask_bbox_xyxy=mask_box,
        centroid_xy=centroid,
        mask_area_pixels=area,
        image_width=width,
        image_height=height,
        object_localization=str(localization),
        masked_object_image=str(masked_path),
        affordance_mask=str(mask_path),
        affordance_overlay=str(overlay_path),
    )
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
