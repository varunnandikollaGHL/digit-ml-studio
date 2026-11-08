"""Sanity tests for the public five-model assignment artifacts."""
import json
from pathlib import Path
import unittest
import joblib
import pandas as pd
from metrics_utils import FEATURES, TARGET, validate_test_csv, evaluate

ROOT = Path(__file__).resolve().parents[1]

class ArtifactTests(unittest.TestCase):
    def test_dataset_contract(self):
        df = pd.read_csv(ROOT / "test_data.csv")
        validated = validate_test_csv(df)
        self.assertEqual(len(validated), 360)
        self.assertEqual(len(FEATURES), 64)
        self.assertEqual(set(validated[TARGET].unique()), set(range(10)))

    def test_all_saved_models_evaluate(self):
        meta = json.loads((ROOT / "metadata.json").read_text())
        frame = validate_test_csv(pd.read_csv(ROOT / "test_data.csv"))
        self.assertEqual(len(meta["model_index"]), 5)
        for name,path in meta["model_index"].items():
            with self.subTest(model=name):
                model=joblib.load(ROOT / path)
                scores=evaluate(model,frame)
                self.assertEqual(set(scores), {"Accuracy","AUC","Precision","Recall","F1","MCC"})
                self.assertTrue(0 <= scores["Accuracy"] <= 1)
                self.assertIsNotNone(scores["AUC"])

    def test_reject_missing_target(self):
        df=pd.read_csv(ROOT/"test_data.csv").drop(columns=[TARGET])
        with self.assertRaises(ValueError):
            validate_test_csv(df)

if __name__ == "__main__":
    unittest.main()
