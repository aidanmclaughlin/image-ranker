import tempfile
import unittest
from pathlib import Path
from PIL import Image

from image_ranker.ingest import InvalidImage, validate_image


class IngestTests(unittest.TestCase):
    def test_rejects_low_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jpg"; Image.new("RGB", (800, 800)).save(path)
            with self.assertRaises(InvalidImage): validate_image(path)

    def test_accepts_large_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.jpg"; Image.new("RGB", (2000, 1500)).save(path)
            self.assertEqual(validate_image(path), (2000, 1500, "jpg"))
