import unittest

from screengrab_demo import inference_config


class InferenceConfigTests(unittest.TestCase):
    def test_streamlit_backend_is_api(self):
        self.assertEqual(inference_config.INFERENCE_BACKEND, "api")
        self.assertEqual(inference_config.backend_label(), "Together API")
        self.assertIn("TOGETHER_API_KEY", inference_config.backend_requirement())


if __name__ == "__main__":
    unittest.main()
