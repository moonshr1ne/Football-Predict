import shutil
import tempfile
import unittest
from pathlib import Path

from football_predictor.aliases import parse_matchup
from football_predictor.autocheck import AutoChecker
from football_predictor.data_store import DataStore
from football_predictor.learning import OnlineLearner
from football_predictor.predictor import MatchPredictor


def make_store(tmp_dir):
    root = Path(tmp_dir) / "project"
    source_data = Path(__file__).resolve().parents[1] / "data"
    shutil.copytree(source_data, root / "data")
    return DataStore(root)


class PredictorTests(unittest.TestCase):
    def test_predict_england_ghana_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            home, away = parse_matchup("Англия, Гана", store.resolver)
            prediction = MatchPredictor(store).predict(home, away, remember=False)
            self.assertIn(prediction.market_pick, {"П1", "X", "П2"})
            self.assertEqual(len(prediction.exact_scores), 2)
            self.assertGreater(prediction.predicted_corners, 0)
            self.assertEqual(prediction.match_context["competition"], "FIFA World Cup")
            self.assertIn("formation", prediction.home_tactics)
            self.assertIn("tactical_matchup", prediction.to_dict())

    def test_learning_records_review(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            review = OnlineLearner(store).record_result("England", "Ghana", "2099-01-01", 2, 1, corners_total=10)
            self.assertEqual(review["actual_score"], "2-1")
            self.assertIn("outcome_hit", review)

    def test_auto_checker_reviews_pending_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            MatchPredictor(store).predict("England", "Ghana", match_date="2099-01-02", remember=True)
            summary = AutoChecker(store, provider=FakeProvider()).check_pending(today="2099-01-03")
            predictions = store.load_predictions()
            self.assertEqual(summary["learned"], 1)
            self.assertEqual(predictions[0]["status"], "reviewed")
            self.assertEqual(predictions[0]["review"]["actual_score"], "2-1")


class FakeProvider:
    def get_finished_result(self, home_team, away_team, match_date):
        return {
            "date": match_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": 2,
            "away_goals": 1,
            "home_corners": 6,
            "away_corners": 4,
            "source": "fake",
        }


if __name__ == "__main__":
    unittest.main()
