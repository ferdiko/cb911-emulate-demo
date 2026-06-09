import unittest

from screengrab_demo.frame_io import natural_sort_key


class FrameIOTests(unittest.TestCase):
    def test_natural_sort_key_orders_numeric_fragments(self):
        names = ["frame10.png", "frame2.png", "frame1.png", "frame11.png"]

        ordered = sorted(names, key=natural_sort_key)

        self.assertEqual(ordered, ["frame1.png", "frame2.png", "frame10.png", "frame11.png"])


if __name__ == "__main__":
    unittest.main()
