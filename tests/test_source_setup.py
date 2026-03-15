import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernels.vendor_whisper import ensure_vendor_whisper_on_path


class TestVendoredWhisperLayout(unittest.TestCase):
    def test_vendor_repo_exists(self):
        repo = ensure_vendor_whisper_on_path()
        self.assertTrue(repo.is_dir())
        self.assertTrue((repo / "whisper" / "__init__.py").is_file())
        self.assertTrue((repo / "tests" / "jfk.flac").is_file())


if __name__ == "__main__":
    unittest.main()
