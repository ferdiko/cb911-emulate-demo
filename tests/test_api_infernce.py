import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api_infernce.run_together import parse_context
from api_infernce.together_classifier import (
    TOGETHER_API_KEY_ENV,
    TOGETHER_MODEL,
    TogetherQwenClassifier,
)
from screengrab_demo.models import Frame, SelectedFrame


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"category":"2FA","confidence":0.9,"evidence_frames":[1],"visible_text":[],"reason":"matched"}'
                    )
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class TogetherInferenceTests(unittest.TestCase):
    def test_classifier_uses_fixed_together_model(self):
        client = FakeClient()
        classifier = TogetherQwenClassifier(client=client)
        selected = [
            SelectedFrame(
                frame=Frame(number=1, name="frame1.png", data=b"abc", content_type="image/png"),
                reasons=("first_frame",),
            )
        ]

        result = classifier.classify(selected)

        self.assertEqual(result.category, "2FA")
        self.assertEqual(client.completions.kwargs["model"], TOGETHER_MODEL)
        self.assertEqual(client.completions.kwargs["temperature"], 0.0)
        self.assertEqual(client.completions.kwargs["messages"][0]["role"], "system")
        self.assertEqual(client.completions.kwargs["messages"][1]["role"], "user")
        self.assertEqual(client.completions.kwargs["response_format"], {"type": "json_object"})

    def test_classifier_requires_together_api_key_without_injected_client(self):
        with patch.dict("os.environ", {TOGETHER_API_KEY_ENV: ""}, clear=True):
            with self.assertRaises(RuntimeError):
                TogetherQwenClassifier()

    def test_parse_context(self):
        self.assertEqual(parse_context(["crm=ExampleCRM"]), {"crm": "ExampleCRM"})


if __name__ == "__main__":
    unittest.main()
