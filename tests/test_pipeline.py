import unittest

from screengrab_demo.classifier import MockClassifier
from screengrab_demo.models import ClassificationResult, Frame
from screengrab_demo.pipeline import fallback_reason_for, run_pipeline


class PipelineTests(unittest.TestCase):
    def test_fallback_reason_for_low_confidence(self):
        result = ClassificationResult(category="other", confidence=0.2)

        reason = fallback_reason_for(result)

        self.assertIn("confidence", reason)

    def test_pipeline_runs_fallback_when_confidence_is_low(self):
        frames = [Frame(number=i + 1, name=f"frame{i + 1}.png") for i in range(20)]
        classifier = MockClassifier(
            forced_result=ClassificationResult(category="other", confidence=0.2)
        )

        result = run_pipeline(frames, classifier)

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.pass_name, "fallback")
        self.assertGreater(len(result.selected_frames), 12)

    def test_mock_classifier_uses_context_keywords(self):
        frames = [
            Frame(number=1, name="frame1.png"),
            Frame(number=2, name="frame2.png"),
            Frame(number=3, name="frame3.png"),
        ]

        result = run_pipeline(
            frames,
            MockClassifier(),
            context={"automation_error": "Enter verification code"},
        )

        self.assertEqual(result.classification.category, "2FA")
        self.assertIn(1, result.classification.evidence_frames)


if __name__ == "__main__":
    unittest.main()
