import shutil
import tempfile
import unittest
from pathlib import Path

from football_predictor.aliases import parse_matchup
from football_predictor.autocheck import AutoChecker
from football_predictor.data_store import DataStore
from football_predictor.learning import OnlineLearner
from football_predictor.predictor import MatchPredictor
from football_predictor.providers import EspnWorldCupProvider


def make_store(tmp_dir):
    root = Path(tmp_dir) / "project"
    source_data = Path(__file__).resolve().parents[1] / "data"
    shutil.copytree(source_data, root / "data")
    store = DataStore(root)
    store.save_predictions([])
    return store


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
            self.assertIn("data_quality", prediction.to_dict())
            for score in prediction.exact_scores:
                home_goals, away_goals = map(int, score.split("-"))
                if prediction.market_pick == "П1":
                    self.assertGreater(home_goals, away_goals)
                elif prediction.market_pick == "П2":
                    self.assertGreater(away_goals, home_goals)
                else:
                    self.assertEqual(home_goals, away_goals)

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

    def test_espn_fixture_maps_scores_to_requested_order(self):
        provider = EspnWorldCupProvider()
        fixture = provider._fixture_from_event(sample_espn_event(), "Uruguay", "Spain")
        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["date"], "2026-06-24")
        self.assertTrue(fixture["completed"])
        self.assertEqual(fixture["home_goals"], 1)
        self.assertEqual(fixture["away_goals"], 2)
        self.assertEqual(fixture["home_corners"], 4)
        self.assertEqual(fixture["away_corners"], 6)
        self.assertEqual(fixture["away_possession"], 58)


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


def sample_espn_event():
    return {
        "id": "123",
        "date": "2026-06-24T19:00Z",
        "competitions": [
            {
                "status": {
                    "type": {
                        "name": "STATUS_FINAL",
                        "state": "post",
                        "completed": True,
                        "shortDetail": "FT",
                    }
                },
                "competitors": [
                    {
                        "homeAway": "home",
                        "score": "2",
                        "team": {"displayName": "Spain", "shortDisplayName": "Spain", "abbreviation": "ESP"},
                        "statistics": [
                            {"name": "wonCorners", "displayValue": "6"},
                            {"name": "possessionPct", "displayValue": "58"},
                            {"name": "totalShots", "displayValue": "14"},
                            {"name": "shotsOnTarget", "displayValue": "5"},
                        ],
                    },
                    {
                        "homeAway": "away",
                        "score": "1",
                        "team": {"displayName": "Uruguay", "shortDisplayName": "Uruguay", "abbreviation": "URU"},
                        "statistics": [
                            {"name": "wonCorners", "displayValue": "4"},
                            {"name": "possessionPct", "displayValue": "42"},
                            {"name": "totalShots", "displayValue": "9"},
                            {"name": "shotsOnTarget", "displayValue": "3"},
                        ],
                    },
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
