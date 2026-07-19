import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import image_ranker.config as config
from image_ranker.config import Settings


class SettingsTests(unittest.TestCase):
    def test_default_root_remains_the_checkout(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.load()

        self.assertEqual(settings.root, Path(config.__file__).resolve().parent.parent)
        self.assertEqual(settings.data, (settings.root / "data").resolve())

    def test_root_override_relocates_default_private_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Application Support" / "Lumen" / "runtime"
            with patch.dict(os.environ, {"IMAGE_RANKER_ROOT": str(root)}, clear=True):
                settings = Settings.load()

        self.assertEqual(settings.root, root.resolve())
        self.assertEqual(settings.data, root.resolve() / "data")
        self.assertEqual(settings.images, root.resolve() / "data" / "images")

    def test_explicit_data_directory_remains_independent_of_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "runtime"
            data = base / "private-data"
            with patch.dict(
                os.environ,
                {
                    "IMAGE_RANKER_ROOT": str(root),
                    "IMAGE_RANKER_DATA": str(data),
                },
                clear=True,
            ):
                settings = Settings.load()

        self.assertEqual(settings.root, root.resolve())
        self.assertEqual(settings.data, data.resolve())


if __name__ == "__main__":
    unittest.main()
