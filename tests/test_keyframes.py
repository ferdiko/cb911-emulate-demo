import unittest

from screengrab_demo.keyframes import (
    INITIAL_KEYFRAME_LIMIT,
    evenly_spaced_indices,
    select_fallback_keyframes,
    select_keyframes,
)
from screengrab_demo.models import Frame


class KeyframeTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_edges(self):
        self.assertEqual(evenly_spaced_indices(10, 4), (0, 3, 6, 9))

    def test_selector_caps_results_and_keeps_edges(self):
        frames = [Frame(number=i + 1, name=f"frame{i + 1}.png") for i in range(40)]

        selected = select_keyframes(frames)

        self.assertLessEqual(len(selected), INITIAL_KEYFRAME_LIMIT)
        self.assertEqual(selected[0].frame.number, 1)
        self.assertEqual(selected[-1].frame.number, 40)

    def test_selector_includes_visual_transition(self):
        frames = [
            Frame(number=1, name="frame1.png", delta_to_previous=0.0),
            Frame(number=2, name="frame2.png", delta_to_previous=0.01),
            Frame(number=3, name="frame3.png", delta_to_previous=0.60),
            Frame(number=4, name="frame4.png", delta_to_previous=0.01),
        ]

        selected = select_keyframes(frames)
        numbers = [item.frame.number for item in selected]

        self.assertIn(3, numbers)

    def test_fallback_selector_expands_to_all_frames_when_under_limit(self):
        frames = [Frame(number=i + 1, name=f"frame{i + 1}.png") for i in range(20)]

        selected = select_fallback_keyframes(frames)

        self.assertEqual(len(selected), 20)


if __name__ == "__main__":
    unittest.main()
