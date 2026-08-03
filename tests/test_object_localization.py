from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from affordgrasp_icar.object_localization import (
    ObjectDetection,
    ObjectLocalizationError,
    VLPartConfig,
    VLPartObjectDetector,
    create_masked_box_image,
    localize_object,
    localize_object_file,
    object_name_from_grounding_request,
)


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.queries = []

    def predict(self, image_bgr, object_name):
        self.queries.append(object_name)
        return self.detections


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def numpy(self):
        return self.values


class FakeInstances:
    def __init__(self):
        self.pred_boxes = SimpleNamespace(
            tensor=FakeTensor(
                [
                    [1.0, 2.0, 5.0, 6.0],
                    [0.0, 0.0, 2.0, 2.0],
                ]
            )
        )
        self.scores = FakeTensor([0.91, 0.32])

    def to(self, device):
        return self


class ObjectLocalizationTests(unittest.TestCase):
    def test_masked_image_matches_paper_box_equation(self):
        image = np.full((4, 6, 3), 17, dtype=np.uint8)

        masked, box = create_masked_box_image(
            image,
            (1.2, 0.8, 4.1, 3.2),
        )

        self.assertEqual(box, (1, 0, 5, 4))
        self.assertTrue(np.array_equal(masked[:, 1:5], image[:, 1:5]))
        self.assertTrue(np.all(masked[:, :1] == 0))
        self.assertTrue(np.all(masked[:, 5:] == 0))

    def test_highest_scoring_detection_is_selected(self):
        image = np.full((8, 8, 3), 100, dtype=np.uint8)
        detector = FakeDetector(
            [
                ObjectDetection((0, 0, 2, 2), 0.61),
                ObjectDetection((2, 1, 7, 6), 0.93),
            ]
        )

        masked, box, score = localize_object(
            image,
            "pliers",
            detector,
            minimum_score=0.5,
        )

        self.assertEqual(detector.queries, ["pliers"])
        self.assertEqual(box, (2, 1, 7, 6))
        self.assertAlmostEqual(score, 0.93)
        self.assertTrue(np.all(masked[1:6, 2:7] == 100))
        self.assertEqual(np.count_nonzero(masked), 5 * 5 * 3)

    def test_no_detection_above_threshold_is_rejected(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        detector = FakeDetector([ObjectDetection((0, 0, 5, 5), 0.49)])

        with self.assertRaisesRegex(ObjectLocalizationError, "found no"):
            localize_object(
                image,
                "cup",
                detector,
                minimum_score=0.5,
            )

    def test_vlpart_adapter_extracts_boxes_and_applies_threshold(self):
        detector = VLPartObjectDetector(
            VLPartConfig(
                root=Path("/unused/VLPart"),
                weights=Path("/unused/model.pth"),
                confidence_threshold=0.5,
            )
        )
        detector._demos["pliers"] = SimpleNamespace(
            predictor=lambda image: {"instances": FakeInstances()}
        )

        detections = detector.predict(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "pliers",
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox_xyxy, (1.0, 2.0, 5.0, 6.0))
        self.assertAlmostEqual(detections[0].score, 0.91)

    def test_file_pipeline_writes_mask_and_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "scene.png"
            image = np.full((6, 7, 3), 200, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            detector = FakeDetector(
                [ObjectDetection((1, 2, 6, 5), 0.88)]
            )

            result = localize_object_file(
                image_path,
                "screwdriver",
                detector,
                root / "output",
                minimum_score=0.5,
            )

            masked_path = root / "output" / "masked_object.png"
            json_path = root / "output" / "object_localization.json"
            self.assertTrue(masked_path.is_file())
            self.assertTrue(json_path.is_file())
            self.assertEqual(result.bbox_xyxy, (1, 2, 6, 5))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["object_name"], "screwdriver")
            self.assertEqual(payload["bbox_xyxy"], [1, 2, 6, 5])

    def test_object_name_can_be_loaded_from_grounding_request(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            request = Path(temporary_directory) / "grounding_request.json"
            request.write_text(
                json.dumps({"object_name": "pliers"}),
                encoding="utf-8",
            )

            self.assertEqual(
                object_name_from_grounding_request(request),
                "pliers",
            )


if __name__ == "__main__":
    unittest.main()
