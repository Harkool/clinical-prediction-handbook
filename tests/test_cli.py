"""Integration checks for exported predictions and reader-facing failures."""
import importlib.util
import csv
from contextlib import contextmanager
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[1]
spec = importlib.util.spec_from_file_location("handbook_run", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@contextmanager
def test_tempdir():
    base = ROOT / "outputs" / "test-tmp"
    base.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(dir=str(base))
    # Verify the exact generated target before TemporaryDirectory removes it.
    if Path(temporary.name).absolute().parent != base.absolute():
        raise RuntimeError("Temporary test directory escaped the workspace")
    with temporary as folder:
        yield folder


class CLITests(unittest.TestCase):
    def test_invalid_chapter_fails(self):
        result = subprocess.run([sys.executable, str(ROOT / "run.py"), "chapter", "32"],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_python_dependencies_fail_doctor(self):
        with mock.patch.object(runner.importlib.util, "find_spec", return_value=None), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.doctor(SimpleNamespace()), 1)

    def test_can_call_from_another_directory(self):
        with test_tempdir() as folder:
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--help"],
                                    cwd=folder, capture_output=True)
        self.assertEqual(result.returncode, 0)

    def test_missing_results_do_not_pass(self):
        with test_tempdir() as folder:
            with self.assertRaises(ValueError):
                runner.verify_output(Path(folder))

    def test_failed_math_replaces_old_pass(self):
        with test_tempdir() as folder:
            fixture = Path(folder)
            (fixture / "math").mkdir()
            (fixture / "math/catalog.json").write_text(json.dumps([{
                "id": "21", "chapter": 21, "script": "math/failure.py", "dependencies": []}]), encoding="utf-8")
            (fixture / "math/failure.py").write_text('raise RuntimeError("intentional failure fixture")\n', encoding="utf-8")
            previous = fixture / "outputs/quick/math/21/result.json"
            previous.parent.mkdir(parents=True)
            previous.write_text('{"status":"PASS","old_run":true}\n', encoding="utf-8")
            with mock.patch.object(runner, "ROOT", fixture), redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    runner.run_math("21", SimpleNamespace(mode="quick"))
            actual = json.loads(previous.read_text(encoding="utf-8"))
            self.assertEqual(actual["status"], "FAIL")
            self.assertNotIn("old_run", actual)

    def test_failed_course_entry_replaces_old_execution_status(self):
        with test_tempdir() as folder:
            fixture = Path(folder)
            with mock.patch.object(runner, "ROOT", fixture), mock.patch.object(runner, "execute", side_effect=RuntimeError("source error")):
                with self.assertRaises(RuntimeError):
                    runner.run_course_cli("all", SimpleNamespace(mode="quick"))
            status = json.loads((fixture/"outputs/quick/course_execution.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAIL")
            self.assertEqual(json.loads((fixture/"outputs/quick/verification.json").read_text(encoding="utf-8"))["status"], "STALE")


class PredictionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / "outputs" / "quick"
        if not (cls.source / "results/predictions.csv").exists():
            raise unittest.SkipTest("Run python run.py course --mode quick first")

    def copied_outputs(self, folder):
        target = Path(folder)
        for part in ("data", "results"):
            shutil.copytree(self.source / part, target / part)
        return target

    def test_predictions_independently_reconstructed(self):
        result = runner.verify_output(self.source)
        self.assertEqual(result["n"], 3200)
        self.assertEqual(result["events"], 384)
        self.assertAlmostEqual(result["C"], 0.771238384824811, places=9)
        self.assertLess(result["max_prediction_error"], 1e-10)

    def test_stale_chapter_or_changed_source_cannot_be_verified(self):
        with test_tempdir() as folder:
            target = self.copied_outputs(folder)
            manifest = target/"manifest.csv"
            manifest.write_text("chapter,status\n8,STALE\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "STALE"):
                runner.verify_output(target)
            manifest.write_text("chapter,status\n8,PASS_EXECUTED\n", encoding="utf-8")
            sources = target/"results/context_sources.csv"
            expected = ("course_core.py", "course_validation.py", "course.py", "metrics.py", "preprocessing.py", "validation.py", "predictions.py", "baseline.py")
            rows = [{"file": name, "sha256": "0"*64} for name in expected]
            with sources.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["file", "sha256"])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "源码已改变"):
                runner.verify_output(target)

    def test_changed_prediction_is_rejected(self):
        with test_tempdir() as folder:
            target = self.copied_outputs(folder)
            path = target / "results/predictions.csv"
            rows = runner.read_csv(path)
            rows[0]["prediction"] = "0.8"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "独立重算"):
                runner.verify_output(target)

    def test_swapped_patient_order_is_rejected(self):
        with test_tempdir() as folder:
            target = self.copied_outputs(folder)
            path = target / "results/predictions.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            path.write_text("\n".join(lines)+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "行顺序"):
                runner.verify_output(target)

    def test_nonfinite_metrics_are_rejected(self):
        with test_tempdir() as folder:
            target = self.copied_outputs(folder)
            path = target / "results/apparent_metrics.csv"
            path.write_text('metric,estimate\nC,NaN\nBrier,Infinity\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "非有限"):
                runner.verify_output(target)

    def test_nonfinite_predictor_is_rejected(self):
        with test_tempdir() as folder:
            target = self.copied_outputs(folder)
            path = target / "data/main_case.csv"
            rows = runner.read_csv(path)
            rows[0]["age"] = "NaN"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "非有限"):
                runner.verify_output(target)


if __name__ == "__main__":
    unittest.main()
