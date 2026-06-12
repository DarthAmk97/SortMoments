import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class SortMomentsCliTests(unittest.TestCase):
    def make_image_folder(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "family.jpg").write_bytes(b"not really an image")
        (root / "notes.txt").write_text("ignore", encoding="utf-8")
        return temp, root

    def test_optional_organize_subcommand_is_accepted(self):
        import sortmoments_cli

        temp, input_dir = self.make_image_folder()
        self.addCleanup(temp.cleanup)

        stream = io.StringIO()
        with redirect_stdout(stream):
            code = sortmoments_cli.main([
                "organize",
                str(input_dir),
                "--dry-run",
            ])

        self.assertEqual(code, 0)
        self.assertIn("Dry run complete", stream.getvalue())

    def test_dry_run_validates_and_prints_plan_without_importing_processor(self):
        import sortmoments_cli

        temp, input_dir = self.make_image_folder()
        self.addCleanup(temp.cleanup)
        out_dir = input_dir / "sorted"

        with patch.dict(sys.modules, {"processphotos": None}):
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = sortmoments_cli.main([
                    str(input_dir),
                    "--output-folder", str(out_dir),
                    "--dry-run",
                ])

        self.assertEqual(code, 0)
        output = stream.getvalue()
        self.assertIn("Dry run", output)
        self.assertIn(str(input_dir), output)
        self.assertIn(str(out_dir), output)
        self.assertIn("1 image", output)

    def test_config_values_are_loaded_and_cli_values_override_them(self):
        import sortmoments_cli

        temp, input_dir = self.make_image_folder()
        self.addCleanup(temp.cleanup)
        configured_output = input_dir / "configured-output"
        cli_output = input_dir / "cli-output"
        config_path = input_dir / "sortmoments.json"
        config_path.write_text(json.dumps({
            "input_folder": str(input_dir),
            "output_folder": str(configured_output),
            "batch_size": 3,
            "workers": 2,
            "similarity_threshold": 0.61,
            "prefer_gpu": False,
            "keep_temp": True,
        }), encoding="utf-8")

        calls = []
        fake_processor = types.ModuleType("sortmoments.pipeline")
        fake_processor.detect_and_embed_faces = lambda *args, **kwargs: calls.append(("detect", args, kwargs)) or {"face": "embedding"}
        fake_processor.reorganize_by_person = lambda *args, **kwargs: calls.append(("reorganize", args, kwargs)) or ({"rename_0": []}, str(cli_output))
        fake_processor.clean_filenames = lambda *args, **kwargs: calls.append(("clean", args, kwargs)) or 0

        with patch.dict(sys.modules, {"sortmoments.pipeline": fake_processor}):
            code = sortmoments_cli.main([
                "--config", str(config_path),
                "--output-folder", str(cli_output),
                "--yes",
            ])

        self.assertEqual(code, 0)
        detect_call = next(call for call in calls if call[0] == "detect")
        reorganize_call = next(call for call in calls if call[0] == "reorganize")
        self.assertEqual(Path(detect_call[1][0]), input_dir)
        self.assertEqual(detect_call[2]["batch_size"], 3)
        self.assertEqual(detect_call[2]["max_workers"], 2)
        self.assertFalse(detect_call[2]["prefer_gpu"])
        self.assertEqual(Path(reorganize_call[2]["final_output_folder"]), cli_output)
        self.assertEqual(reorganize_call[2]["similarity_threshold"], 0.61)

    def test_confirm_decline_stops_before_processing(self):
        import sortmoments_cli

        temp, input_dir = self.make_image_folder()
        self.addCleanup(temp.cleanup)

        fake_processor = types.ModuleType("sortmoments.pipeline")
        fake_processor.detect_and_embed_faces = lambda *args, **kwargs: self.fail("processing should not start")
        fake_processor.reorganize_by_person = lambda *args, **kwargs: self.fail("processing should not start")
        fake_processor.clean_filenames = lambda *args, **kwargs: self.fail("processing should not start")

        with patch.dict(sys.modules, {"sortmoments.pipeline": fake_processor}), patch("builtins.input", return_value="no"):
            code = sortmoments_cli.main([str(input_dir), "--confirm"])

        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
