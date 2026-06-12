import tomllib
import unittest
from pathlib import Path


class PackageMetadataTests(unittest.TestCase):
    def test_pyproject_uses_src_layout_for_library_distribution(self):
        root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(metadata["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(metadata["project"]["name"], "sortmoments")
        self.assertEqual(metadata["tool"]["setuptools"]["package-dir"][""], "src")
        self.assertEqual(metadata["tool"]["setuptools"]["packages"]["find"]["where"], ["src"])
        self.assertNotIn(
            "scripts",
            metadata["project"],
            "PyPI package should be the Python library surface; the no-GUI CLI stays repo/source based.",
        )
        self.assertTrue((root / "src" / "sortmoments" / "__init__.py").exists())
        self.assertFalse(
            (root / "src" / "sortmoments" / "cli.py").exists(),
            "The CLI implementation should remain the repo/source workflow, not part of the PyPI library package.",
        )
        self.assertTrue((root / "sortmoments_cli.py").exists())


if __name__ == "__main__":
    unittest.main()
