from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from affordgrasp_icar.cli import build_parser
from affordgrasp_icar.reasoner import (
    DEFAULT_MODELS,
    GEMINI_OPENAI_BASE_URL,
    AffordanceReasoner,
    AffordanceReasonerConfig,
    ConfigurationError,
    OpenAIAffordanceReasoner,
)


VALID_RESULT = {
    "task_analysis": "The task is to tighten screws.",
    "object_identification": "A screwdriver is visible.",
    "part_selection": "The handle should be grasped.",
    "affordance_reasoning": "The handle supports a secure grip.",
    "task": "tighten screws",
    "object": "screwdriver",
    "object_part": "handle",
    "affordance": "grasp",
    "is_actionable": True,
    "confidence": 0.92,
    "failure_reason": "",
}


class ReasonerProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temporary_directory.name) / "scene.png"
        self.image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _fake_client(self) -> SimpleNamespace:
        openai_response = SimpleNamespace(output_text=json.dumps(VALID_RESULT))
        gemini_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(VALID_RESULT))
                )
            ]
        )
        return SimpleNamespace(
            responses=SimpleNamespace(create=Mock(return_value=openai_response)),
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=Mock(return_value=gemini_response)
                )
            ),
        )

    def test_provider_specific_default_models(self) -> None:
        self.assertEqual(
            AffordanceReasonerConfig().model,
            DEFAULT_MODELS["openai"],
        )
        self.assertEqual(
            AffordanceReasonerConfig(provider="gemini").model,
            DEFAULT_MODELS["gemini"],
        )
        self.assertIs(OpenAIAffordanceReasoner, AffordanceReasoner)

    def test_invalid_provider_and_gemini_effort_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider must be one of"):
            AffordanceReasonerConfig(provider="unknown")
        with self.assertRaisesRegex(ValueError, "Gemini reasoning_effort"):
            AffordanceReasonerConfig(
                provider="gemini",
                reasoning_effort="xhigh",
            )

    def test_openai_uses_responses_api(self) -> None:
        client = self._fake_client()
        reasoner = AffordanceReasoner(
            AffordanceReasonerConfig(provider="openai"),
            client=client,
        )

        result = reasoner.reason("Tighten the screws.", self.image_path)

        self.assertEqual(result.object, "screwdriver")
        client.responses.create.assert_called_once()
        client.chat.completions.create.assert_not_called()
        arguments = client.responses.create.call_args.kwargs
        self.assertEqual(arguments["model"], DEFAULT_MODELS["openai"])
        self.assertEqual(arguments["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual(arguments["reasoning"], {"effort": "medium"})
        self.assertFalse(arguments["store"])

    def test_gemini_uses_chat_completions_with_image_and_schema(self) -> None:
        client = self._fake_client()
        reasoner = AffordanceReasoner(
            AffordanceReasonerConfig(provider="gemini"),
            client=client,
        )

        result = reasoner.reason("Tighten the screws.", self.image_path)

        self.assertEqual(result.object_part, "handle")
        client.chat.completions.create.assert_called_once()
        client.responses.create.assert_not_called()
        arguments = client.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments["model"], DEFAULT_MODELS["gemini"])
        self.assertEqual(arguments["messages"][0]["role"], "system")
        image_content = arguments["messages"][1]["content"][1]
        self.assertEqual(image_content["type"], "image_url")
        self.assertTrue(
            image_content["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(arguments["response_format"]["type"], "json_schema")
        self.assertEqual(arguments["reasoning_effort"], "medium")

    def test_gemini_client_uses_compatibility_base_url(self) -> None:
        captured_arguments = {}
        fake_openai_module = ModuleType("openai")

        def create_client(**arguments):
            captured_arguments.update(arguments)
            return object()

        fake_openai_module.OpenAI = create_client
        reasoner = AffordanceReasoner(
            AffordanceReasonerConfig(provider="gemini")
        )

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            with patch.dict(sys.modules, {"openai": fake_openai_module}):
                reasoner._get_client()

        self.assertEqual(captured_arguments["api_key"], "test-key")
        self.assertEqual(
            captured_arguments["base_url"],
            GEMINI_OPENAI_BASE_URL,
        )

    def test_missing_gemini_key_has_provider_specific_error(self) -> None:
        reasoner = AffordanceReasoner(
            AffordanceReasonerConfig(provider="gemini")
        )
        empty_local_config = ModuleType("affordgrasp_icar.local_config")
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "affordgrasp_icar.local_config",
                empty_local_config,
                create=True,
            ):
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "GEMINI_API_KEY",
                ):
                    reasoner._get_client()

    def test_cli_accepts_gemini_without_explicit_model(self) -> None:
        parser = build_parser()
        arguments = parser.parse_args(
            [
                "--image",
                "scene.png",
                "--instruction",
                "Tighten the screws.",
                "--provider",
                "gemini",
            ]
        )

        config = AffordanceReasonerConfig(
            provider=arguments.provider,
            model=arguments.model,
        )
        self.assertEqual(config.model, DEFAULT_MODELS["gemini"])


if __name__ == "__main__":
    unittest.main()
