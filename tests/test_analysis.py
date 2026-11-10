import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.analysis import FEATURES, candidates, load_data


class MusicTests(unittest.TestCase):
    def test_test_values_do_not_change_preprocessing(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(size=(40, 8)), columns=FEATURES)
        y = np.array(["Hip-Hop", "Rock"] * 20)
        model = clone(candidates()["regresion_logistica"]).fit(X, y)
        center = model.named_steps["scaler"].mean_.copy()
        model.predict(pd.DataFrame(np.full((5, 8), 1e6), columns=FEATURES))
        np.testing.assert_allclose(center, X.mean())
        np.testing.assert_array_equal(center, model.named_steps["scaler"].mean_)

    def test_identifier_and_label_not_features(self):
        self.assertNotIn("track_id", FEATURES)
        self.assertNotIn("genre_top", FEATURES)

    def test_many_to_many_join_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            pd.DataFrame({"track_id": [1, 1], "genre_top": ["Rock", "Hip-Hop"]}).to_csv(
                p / "t.csv", index=False
            )
            e = pd.DataFrame(np.zeros((2, 8)), columns=FEATURES)
            e["track_id"] = [1, 2]
            e.to_json(p / "e.json")
            with self.assertRaisesRegex(ValueError, "único"):
                load_data(p / "t.csv", p / "e.json")


if __name__ == "__main__":
    unittest.main()
