import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from summarize_paper_results import build_summary, write_markdown


class PaperResultSummaryTests(unittest.TestCase):
    def write_audit(self, path):
        rows = [
            {
                "source": "timm",
                "backbone": "mobilenetv4_conv_small",
                "seed": "42",
                "status": "paper-ready",
                "selected_test_mae": "3.6130",
                "mae_raw": "3.6042",
                "mae_flip": "3.5965",
                "mae_multi": "3.6130",
            },
            {
                "source": "torchvision",
                "backbone": "mobilenet_v3_large",
                "seed": "42",
                "status": "blocked",
                "selected_test_mae": "3.7000",
                "mae_raw": "3.7000",
                "mae_flip": "3.6900",
                "mae_multi": "3.7000",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_summary_tracks_ready_and_missing_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = Path(tmpdir) / "audit.csv"
            self.write_audit(audit)

            summary = build_summary(
                [audit],
                candidates=["torchvision/mobilenet_v3_large", "timm/mobilenetv4_conv_small"],
                seeds=[42, 3407, 2026],
            )

        by_name = {row["candidate"]: row for row in summary}
        self.assertEqual("partial", by_name["timm/mobilenetv4_conv_small"]["status"])
        self.assertEqual("42", by_name["timm/mobilenetv4_conv_small"]["ready_seeds"])
        self.assertEqual("3407,2026", by_name["timm/mobilenetv4_conv_small"]["missing_seeds"])
        self.assertEqual("3.6130", by_name["timm/mobilenetv4_conv_small"]["mean_selected_test_mae"])
        self.assertEqual("", by_name["timm/mobilenetv4_conv_small"]["std_selected_test_mae"])
        self.assertEqual("missing", by_name["torchvision/mobilenet_v3_large"]["status"])

    def test_markdown_warns_partial_rows_are_not_final(self):
        rows = [
            {
                "candidate": "timm/mobilenetv4_conv_small",
                "status": "partial",
                "ready_seeds": "42",
                "missing_seeds": "3407,2026",
                "mean_selected_test_mae": "3.6130",
                "std_selected_test_mae": "",
                "mean_mae_raw": "3.6042",
                "mean_mae_flip": "3.5965",
                "mean_mae_multi": "3.6130",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "summary.md"
            write_markdown(rows, output, seeds=[42, 3407, 2026])
            content = output.read_text(encoding="utf-8")

        self.assertIn("not final paper mean/std", content)
        self.assertIn("timm/mobilenetv4_conv_small", content)
        self.assertIn("3407,2026", content)


if __name__ == "__main__":
    unittest.main()
