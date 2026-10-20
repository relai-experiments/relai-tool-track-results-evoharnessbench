import unittest
from unittest.mock import patch

from eog_tools.bootstrap import require_sdk_version


class BootstrapTests(unittest.TestCase):
    def test_requires_matrix_capable_sdk(self):
        with patch("importlib.metadata.version", return_value="0.19.1"):
            with self.assertRaisesRegex(RuntimeError, ">=0.19.2 required"):
                require_sdk_version()
        with patch("importlib.metadata.version", return_value="0.19.2"):
            self.assertEqual(require_sdk_version(), "0.19.2")


if __name__ == "__main__":
    unittest.main()
