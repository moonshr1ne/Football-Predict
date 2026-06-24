import json
import shutil
import tempfile
import unittest
from pathlib import Path

from football_predictor.aliases import has_cyrillic, parse_matchup
from football_predictor.autocheck import AutoChecker
from football_predictor.data_store import DataStore
from football_predictor.learning import OnlineLearner
from football_predictor.models import TeamStats
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

    def test_world_cup_russian_aliases_use_real_profiles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            home, away = parse_matchup("Колумбия, ДР Конго", store.resolver)
            self.assertEqual(home, "Colombia")
            self.assertEqual(away, "Congo DR")

            prediction = MatchPredictor(store).predict(home, away, remember=False)
            data = prediction.to_dict()
            self.assertEqual(data["home_stats"]["sample_size"], 10)
            self.assertEqual(data["away_stats"]["sample_size"], 10)
            self.assertFalse(data["home_tactics"]["is_fallback"])
            self.assertFalse(data["away_tactics"]["is_fallback"])
            self.assertNotEqual(data["home_tactics"]["possession_intent"], 0.55)
            self.assertNotEqual(data["away_tactics"]["possession_intent"], 0.55)

    def test_all_world_cup_participants_have_russian_aliases(self):
        data_dir = Path(__file__).resolve().parents[1] / "data"
        participants = json.loads((data_dir / "participants.json").read_text(encoding="utf-8"))
        aliases = json.loads((data_dir / "team_aliases.json").read_text(encoding="utf-8"))
        missing = [
            item["team"]
            for item in participants
            if item["team"] not in aliases or not any(has_cyrillic(alias) for alias in aliases[item["team"]])
        ]
        self.assertEqual(missing, [])

    def test_unknown_russian_team_does_not_fallback_silently(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            with self.assertRaisesRegex(ValueError, "Не распознал"):
                parse_matchup("Нарния, Гана", store.resolver)

    def test_exact_scores_can_use_learned_score_frequency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            state = store.load_model_state()
            state["history"] = [{"actual_outcome": "П1", "actual_score": "2-0"} for _ in range(80)]
            store.save_model_state(state)
            home_stats = TeamStats(team="Home", sample_size=10, wins=7, draws=2, losses=1, clean_sheets=6)
            away_stats = TeamStats(team="Away", sample_size=10, wins=2, draws=2, losses=6, failed_to_score=4)
            scores = MatchPredictor(store)._top_scores(1.35, 0.55, "П1", home_stats, away_stats)
            self.assertIn("2-0", scores)

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
