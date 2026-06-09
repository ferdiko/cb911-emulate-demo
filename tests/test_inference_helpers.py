import unittest

from inference.helpers import (
    InvalidClassificationResponse,
    build_multimodal_content,
    frame_to_data_url,
    parse_classification_response,
)
from screengrab_demo.models import Frame, SelectedFrame


class InferenceHelperTests(unittest.TestCase):
    def test_frame_to_data_url_uses_raw_frame_data(self):
        frame = Frame(number=1, name="frame1.png", data=b"abc", content_type="image/png")

        data_url = frame_to_data_url(frame)

        self.assertEqual(data_url, "data:image/png;base64,YWJj")

    def test_build_multimodal_content_includes_prompt_labels_and_images(self):
        selected = SelectedFrame(
            frame=Frame(number=7, name="frame7.png", data=b"abc", content_type="image/png"),
            reasons=("first_frame",),
        )

        content = build_multimodal_content([selected], context={"crm": "ExampleCRM"})

        self.assertEqual(content[0]["type"], "text")
        self.assertIn("ExampleCRM", content[0]["text"])
        self.assertEqual(content[1]["type"], "text")
        self.assertIn("Frame 7", content[1]["text"])
        self.assertEqual(content[2]["type"], "image_url")
        self.assertTrue(content[2]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_parse_classification_response_accepts_json_fences_and_aliases(self):
        result = parse_classification_response(
            """```json
            {
              "category": "ui changes",
              "confidence": 88,
              "evidence_frames": ["3", 4],
              "visible_text": ["Element not found"],
              "reason": "The target page changed."
            }
            ```"""
        )

        self.assertEqual(result.category, "ui_change")
        self.assertEqual(result.confidence, 0.88)
        self.assertEqual(result.evidence_frames, (3, 4))
        self.assertEqual(result.visible_text, ("Element not found",))

    def test_parse_error_includes_raw_response_preview(self):
        with self.assertRaises(InvalidClassificationResponse) as exc:
            parse_classification_response("The screenshots show a verification prompt.")

        self.assertIn("Raw response preview", str(exc.exception))
        self.assertIn("verification prompt", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
