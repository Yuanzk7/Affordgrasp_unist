"""VLPart-based object localization and paper-aligned masked-image creation."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import cv2
import numpy as np


class ObjectLocalizationError(RuntimeError):
    """Raised when VLPart cannot produce a usable object bounding box."""


@dataclass(frozen=True)
class ObjectDetection:
    """One VLPart object candidate in XYXY image coordinates."""

    bbox_xyxy: Tuple[float, float, float, float]
    score: float


class ObjectDetector(Protocol):
    """Minimal backend contract used by the localization pipeline."""

    def predict(
        self,
        image_bgr: np.ndarray,
        object_name: str,
    ) -> Sequence[ObjectDetection]:
        """Return candidate boxes for one open-vocabulary object query."""


@dataclass(frozen=True)
class ObjectLocalizationResult:
    """Serializable result of AffordGrasp object localization."""

    object_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    image_width: int
    image_height: int
    source_image: str
    masked_image: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_name": self.object_name,
            "bbox_xyxy": list(self.bbox_xyxy),
            "score": self.score,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "source_image": self.source_image,
            "masked_image": self.masked_image,
        }


@dataclass(frozen=True)
class VLPartConfig:
    """Runtime paths and inference options for the official VLPart code."""

    root: Path
    weights: Path
    config_file: Optional[Path] = None
    confidence_threshold: float = 0.5
    device: str = "auto"

    def resolved_config_file(self) -> Path:
        if self.config_file is not None:
            return self.config_file.expanduser().resolve()
        return (
            self.root.expanduser().resolve()
            / "configs"
            / "pascal_part"
            / "r50_pascalpart.yaml"
        )


class VLPartObjectDetector:
    """Lazy adapter around the official VLPart Detectron2 predictor."""

    def __init__(self, config: VLPartConfig) -> None:
        if not 0.0 <= config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if config.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be one of: auto, cpu, cuda")
        self.config = config
        self._demos: Dict[str, Any] = {}

    def _validate_runtime_paths(self) -> Tuple[Path, Path, Path]:
        root = self.config.root.expanduser().resolve()
        weights = self.config.weights.expanduser().resolve()
        config_file = self.config.resolved_config_file()

        if not root.is_dir():
            raise ObjectLocalizationError(
                f"VLPart root directory does not exist: {root}"
            )
        if not (root / "vlpart").is_dir():
            raise ObjectLocalizationError(
                f"official VLPart package was not found under: {root}"
            )
        if not config_file.is_file():
            raise ObjectLocalizationError(
                f"VLPart config file does not exist: {config_file}"
            )
        if not weights.is_file():
            raise ObjectLocalizationError(
                f"VLPart weights file does not exist: {weights}"
            )
        return root, config_file, weights

    def _build_demo(self, object_name: str) -> Any:
        root, config_file, weights = self._validate_runtime_paths()
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        matplotlib_cache = root / "models" / "matplotlib_cache"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        try:
            import clip
            import torch
            from detectron2.config import get_cfg
            from demo.predictor import VisualizationDemo
            from vlpart.config import add_vlpart_config
        except ImportError as exc:
            raise ObjectLocalizationError(
                "VLPart runtime is incomplete. Install torchvision, Detectron2, "
                "and the official VLPart requirements in the active environment."
            ) from exc

        if not getattr(clip.load, "_affordgrasp_local_cache", False):
            original_clip_load = clip.load
            clip_cache = root / "models" / "clip"
            clip_cache.mkdir(parents=True, exist_ok=True)

            def load_clip_from_project_cache(
                name: str,
                device: Any = "cpu",
                jit: bool = False,
                download_root: Optional[str] = None,
            ) -> Any:
                return original_clip_load(
                    name,
                    device=device,
                    jit=jit,
                    download_root=download_root or str(clip_cache),
                )

            load_clip_from_project_cache._affordgrasp_local_cache = True
            clip.load = load_clip_from_project_cache

        torch_hub_cache = root / "models" / "torch_hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(torch_hub_cache))

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ObjectLocalizationError(
                "CUDA was requested, but torch.cuda.is_available() is false"
            )

        cfg = get_cfg()
        add_vlpart_config(cfg)
        cfg.merge_from_file(str(config_file))

        # Official VLPart configs contain paths relative to the VLPart repository.
        # Resolve them explicitly so this adapter can run from affordgrasp_icar.
        roi_box_head = cfg.MODEL.ROI_BOX_HEAD
        path_fields = (
            "ZEROSHOT_WEIGHT_PATH",
            "ZEROSHOT_WEIGHT_INFERENCE_PATH",
            "CAT_FREQ_PATH",
        )
        path_group_fields = (
            "ZEROSHOT_WEIGHT_PATH_GROUP",
            "CAT_FREQ_PATH_GROUP",
        )

        def resolve_vlpart_path(value: Any) -> Any:
            if not isinstance(value, str) or not value:
                return value
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return str(candidate)
            rooted = root / candidate
            return str(rooted) if rooted.exists() else value

        for field in path_fields:
            if hasattr(roi_box_head, field):
                setattr(
                    roi_box_head,
                    field,
                    resolve_vlpart_path(getattr(roi_box_head, field)),
                )
        for field in path_group_fields:
            if hasattr(roi_box_head, field):
                values = getattr(roi_box_head, field)
                setattr(
                    roi_box_head,
                    field,
                    [resolve_vlpart_path(value) for value in values],
                )

        cfg.MODEL.WEIGHTS = str(weights)
        cfg.MODEL.DEVICE = device
        threshold = self.config.confidence_threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
        cfg.freeze()

        arguments = SimpleNamespace(
            vocabulary="custom",
            custom_vocabulary=object_name,
        )
        return VisualizationDemo(cfg, arguments)

    def predict(
        self,
        image_bgr: np.ndarray,
        object_name: str,
    ) -> Sequence[ObjectDetection]:
        query = _require_object_name(object_name)
        if "," in query:
            raise ObjectLocalizationError(
                "object_name must contain exactly one VLPart query"
            )

        demo = self._demos.get(query)
        if demo is None:
            try:
                demo = self._build_demo(query)
            except ObjectLocalizationError:
                raise
            except Exception as exc:
                raise ObjectLocalizationError(
                    f"VLPart initialization failed for object query {query!r}: {exc}"
                ) from exc
            self._demos[query] = demo

        try:
            predictions = demo.predictor(image_bgr)
            instances = predictions.get("instances")
            if instances is None:
                return []
            instances = instances.to("cpu")
            boxes = instances.pred_boxes.tensor.detach().numpy()
            scores = instances.scores.detach().numpy()
        except Exception as exc:
            raise ObjectLocalizationError(
                f"VLPart inference failed for object query {query!r}: {exc}"
            ) from exc

        detections: List[ObjectDetection] = []
        for box, score in zip(boxes, scores):
            score_value = float(score)
            if score_value < self.config.confidence_threshold:
                continue
            detections.append(
                ObjectDetection(
                    bbox_xyxy=tuple(float(value) for value in box),
                    score=score_value,
                )
            )
        return detections


def _require_object_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectLocalizationError("object_name must be a non-empty string")
    return value.strip()


def clamp_bbox_xyxy(
    bbox_xyxy: Sequence[float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Clamp a floating-point XYXY box to a non-empty integer image region."""

    if len(bbox_xyxy) != 4:
        raise ObjectLocalizationError("bbox_xyxy must contain four values")
    if image_width <= 0 or image_height <= 0:
        raise ObjectLocalizationError("image dimensions must be positive")

    values = [float(value) for value in bbox_xyxy]
    if not all(math.isfinite(value) for value in values):
        raise ObjectLocalizationError("bbox_xyxy contains a non-finite value")

    x1 = max(0, min(image_width, math.floor(values[0])))
    y1 = max(0, min(image_height, math.floor(values[1])))
    x2 = max(0, min(image_width, math.ceil(values[2])))
    y2 = max(0, min(image_height, math.ceil(values[3])))
    if x2 <= x1 or y2 <= y1:
        raise ObjectLocalizationError(
            f"VLPart returned an empty bounding box: {(x1, y1, x2, y2)}"
        )
    return x1, y1, x2, y2


def create_masked_box_image(
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Implement M_BO: retain pixels inside B_O and zero everything outside."""

    if not isinstance(image_bgr, np.ndarray):
        raise ObjectLocalizationError("image must be a numpy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ObjectLocalizationError("image must have shape (H, W, 3)")

    image_height, image_width = image_bgr.shape[:2]
    box = clamp_bbox_xyxy(bbox_xyxy, image_width, image_height)
    x1, y1, x2, y2 = box
    masked = np.zeros_like(image_bgr)
    masked[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]
    return masked, box


def localize_object(
    image_bgr: np.ndarray,
    object_name: str,
    detector: ObjectDetector,
    minimum_score: float = 0.5,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """Select the highest-scoring VLPart box and create the M_BO image."""

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between 0 and 1")
    query = _require_object_name(object_name)
    detections = [
        detection
        for detection in detector.predict(image_bgr, query)
        if detection.score >= minimum_score
    ]
    if not detections:
        raise ObjectLocalizationError(
            f"VLPart found no {query!r} object above score {minimum_score:.2f}"
        )

    selected = max(detections, key=lambda detection: detection.score)
    masked, box = create_masked_box_image(image_bgr, selected.bbox_xyxy)
    return masked, box, selected.score


def localize_object_file(
    image_path: Union[str, Path],
    object_name: str,
    detector: ObjectDetector,
    output_dir: Union[str, Path],
    minimum_score: float = 0.5,
) -> ObjectLocalizationResult:
    """Run object localization and save M_BO plus a JSON bounding-box record."""

    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise ObjectLocalizationError(f"RGB image does not exist: {source}")

    image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ObjectLocalizationError(f"OpenCV could not read RGB image: {source}")

    masked, box, score = localize_object(
        image_bgr,
        object_name,
        detector,
        minimum_score=minimum_score,
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    masked_path = destination / "masked_object.png"
    result_path = destination / "object_localization.json"
    if not cv2.imwrite(str(masked_path), masked):
        raise ObjectLocalizationError(
            f"failed to write masked object image: {masked_path}"
        )

    height, width = image_bgr.shape[:2]
    result = ObjectLocalizationResult(
        object_name=_require_object_name(object_name),
        bbox_xyxy=box,
        score=float(score),
        image_width=width,
        image_height=height,
        source_image=str(source),
        masked_image=str(masked_path),
    )
    result_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def object_name_from_grounding_request(
    request_path: Union[str, Path],
) -> str:
    """Load only the ICAR object_name needed by this pipeline stage."""

    path = Path(request_path).expanduser().resolve()
    if not path.is_file():
        raise ObjectLocalizationError(f"grounding request does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ObjectLocalizationError(
            f"could not read grounding request JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ObjectLocalizationError("grounding request must be a JSON object")
    return _require_object_name(payload.get("object_name"))


def config_from_environment(
    root: Optional[str],
    weights: Optional[str],
    config_file: Optional[str],
    confidence_threshold: float,
    device: str,
) -> VLPartConfig:
    """Build configuration from CLI values with environment fallbacks."""

    resolved_root = root or os.environ.get("VLPART_ROOT")
    resolved_weights = weights or os.environ.get("VLPART_WEIGHTS")
    resolved_config = config_file or os.environ.get("VLPART_CONFIG")
    if not resolved_root:
        raise ObjectLocalizationError(
            "--vlpart-root or VLPART_ROOT is required"
        )
    if not resolved_weights:
        raise ObjectLocalizationError(
            "--weights or VLPART_WEIGHTS is required"
        )
    return VLPartConfig(
        root=Path(resolved_root),
        weights=Path(resolved_weights),
        config_file=Path(resolved_config) if resolved_config else None,
        confidence_threshold=confidence_threshold,
        device=device,
    )
