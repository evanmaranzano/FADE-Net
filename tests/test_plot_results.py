import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from plot_results import experiment_id_from_log_path, infer_batch_log_path, load_real_data


class PlotResultsTests(unittest.TestCase):
    def test_experiment_id_from_metadata_log_name(self):
        self.assertEqual(
            "FADE-Net_A7_timm_mobilenetv4_conv_small_seed42",
            experiment_id_from_log_path("training_log_FADE-Net_A7_timm_mobilenetv4_conv_small_seed42.csv"),
        )
        self.assertIsNone(experiment_id_from_log_path("training_log_seed42.csv"))

    def test_infer_batch_log_path_preserves_experiment_identity(self):
        self.assertEqual(
            "batch_log_FADE-Net_A9_seed42.csv",
            infer_batch_log_path("training_log_FADE-Net_A9_seed42.csv"),
        )
        self.assertEqual("batch_log_seed42.csv", infer_batch_log_path("training_log_seed42.csv"))

    def test_load_real_data_accepts_explicit_log_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                epoch_log = Path("training_log_FADE-Net_A7_seed42.csv")
                batch_log = Path("batch_log_FADE-Net_A7_seed42.csv")
                pd.DataFrame([{"Epoch": 1, "Val_MAE": 3.0}]).to_csv(epoch_log, index=False)
                pd.DataFrame([{"Epoch": 1, "Total_Loss": 1.0}]).to_csv(batch_log, index=False)

                df_epoch, df_batch, seed, experiment_id = load_real_data(log_path=str(epoch_log))
            finally:
                os.chdir(old_cwd)

        self.assertEqual(1, len(df_epoch))
        self.assertEqual(1, len(df_batch))
        self.assertEqual("FADE-Net_A7_seed42", seed)
        self.assertEqual("FADE-Net_A7_seed42", experiment_id)


if __name__ == "__main__":
    unittest.main()
