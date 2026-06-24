import json
import shutil
import tempfile
import unittest
from pathlib import Path

from football_predictor.aliases import has_cyrillic, parse_matchup
from football_predictor.autocheck import AutoChecker
from football_predictor.data_store import DataStore
from football_predictor.features import build_team_stats
from football_predictor.learning import OnlineLearner
from football_predictor.models import MatchRecord, TeamStats
from football_predictor.predictor import MatchPredictor
from football_predictor.providers import EspnWorldCupProvider
from football_predictor.sync import formation_guess


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
            self.assertEqual(len(prediction.exact_scores), 3)
            self.assertEqual(len(prediction.exact_score_probabilities), 3)
            self.assertIn("1X", {item["code"] for item in prediction.markets})
            self.assertIn("X2", {item["code"] for item in prediction.markets})
            self.assertGreater(prediction.predicted_corners, 0)
            self.assertEqual(prediction.match_context["competition"], "FIFA World Cup")
            self.assertIn("formation", prediction.home_tactics)
            self.assertIn("tactical_matchup", prediction.to_dict())
            self.assertIn("result_summary", prediction.to_dict())
            self.assertIn("goal_total", prediction.to_dict())
            self.assertIn("foul_forecast", prediction.to_dict())
            self.assertIn("over_2_5", prediction.to_dict()["goal_total"]["probabilities"])
            self.assertIn("under_2_5", prediction.to_dict()["goal_total"]["probabilities"])
            self.assertGreater(prediction.to_dict()["foul_forecast"]["expected"], 0)
            self.assertIn("team_reports", prediction.to_dict())
            self.assertIn(home, prediction.to_dict()["team_reports"])
            self.assertIn("lineup_reports", prediction.to_dict())
            self.assertIn(home, prediction.to_dict()["lineup_reports"])
            self.assertIn("data_quality", prediction.to_dict())
            for item in prediction.exact_score_probabilities:
                self.assertGreaterEqual(item["probability"], 0)
                self.assertLessEqual(item["probability"], 1)
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
            scores = [item["score"] for item in MatchPredictor(store)._top_scores(1.35, 0.55, "П1", home_stats, away_stats)]
            self.assertIn("2-0", scores)

    def test_team_stats_include_fouls_from_recent_matches(self):
        matches = [
            MatchRecord("2099-01-01", "Home", "Away", home_goals=1, away_goals=0, home_fouls=12, away_fouls=16),
            MatchRecord("2099-01-02", "Away", "Home", home_goals=0, away_goals=2, home_fouls=11, away_fouls=13),
        ]
        stats = build_team_stats(matches, "Home")
        self.assertEqual(stats.foul_samples, 2)
        self.assertEqual(stats.avg_fouls_for, 12.5)
        self.assertEqual(stats.avg_fouls_against, 13.5)
        self.assertEqual(stats.avg_total_fouls, 26.0)

    def test_referee_profile_lifts_foul_forecast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            store.save_referee_profiles({"Strict Ref": {"matches": 5, "avg_fouls": 34.0, "source": "test"}})
            predictor = MatchPredictor(store)
            low = TeamStats(team="Low", sample_size=10, wins=5, draws=3, losses=2, foul_samples=6, fouls_for=54, fouls_against=60)
            normal = TeamStats(team="Normal", sample_size=10, wins=4, draws=3, losses=3, foul_samples=6, fouls_for=60, fouls_against=60)
            shared_args = (
                low,
                normal,
                {"pressing": 0.50, "directness": 0.48, "tempo": 0.50, "defensive_solidity": 0.60},
                {"pressing": 0.50, "directness": 0.48, "tempo": 0.50, "defensive_solidity": 0.60},
            )
            no_ref_forecast = predictor._foul_forecast(
                *shared_args,
                None,
                store.load_model_state()["weights"],
            )
            forecast = predictor._foul_forecast(
                *shared_args,
                {"referee": {"name": "Strict Ref", "source": "test"}},
                store.load_model_state()["weights"],
            )
            self.assertGreater(forecast["expected"], no_ref_forecast["expected"])
            self.assertEqual(forecast["referee"]["name"], "Strict Ref")

    def test_high_total_matchup_gets_high_score_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            prediction = MatchPredictor(store).predict("Norway", "Senegal", remember=False)
            score_totals = [sum(map(int, score.split("-"))) for score in prediction.exact_scores]
            self.assertGreaterEqual(prediction.goal_total["probabilities"]["over_3_5"], 0.33)
            self.assertTrue(any(total >= 4 for total in score_totals))

    def test_over_total_does_not_rank_one_nil_first(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            predictor = MatchPredictor(store)
            stats = TeamStats(team="A", sample_size=10, wins=5, draws=3, losses=2)
            goal_total = predictor._goal_total_forecast(1.75, 1.14)
            scores = predictor._top_scores(
                1.75,
                1.14,
                "П1",
                stats,
                stats,
                goal_total,
                {"П1": 0.46, "X": 0.27, "П2": 0.27},
            )
            self.assertGreater(goal_total["probabilities"]["over_2_5"], 0.52)
            self.assertGreaterEqual(sum(map(int, scores[0]["score"].split("-"))), 3)

    def test_close_match_exact_scores_can_include_draw(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            predictor = MatchPredictor(store)
            stats = TeamStats(team="A", sample_size=10, wins=4, draws=3, losses=3)
            goal_total = predictor._goal_total_forecast(1.25, 1.18)
            scores = predictor._top_scores(
                1.25,
                1.18,
                "П1",
                stats,
                stats,
                goal_total,
                {"П1": 0.38, "X": 0.30, "П2": 0.32},
            )
            self.assertIn("X", {item["outcome"] for item in scores})
            self.assertIn("П1", {item["outcome"] for item in scores})

    def test_confirmed_lineup_changes_formation_and_key_player_strength(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            fixture = {
                "date": "2099-01-01",
                "completed": False,
                "in_progress": False,
                "lineups": {
                    "France": {
                        "confirmed": True,
                        "formation": "4-3-3",
                        "starters": [{"name": "Michael Olise"}, {"name": "Ousmane Dembélé"}],
                        "bench": [],
                    },
                    "Iraq": {
                        "confirmed": True,
                        "formation": "4-1-4-1",
                        "starters": [{"name": "Aymen Hussein"}],
                        "bench": [],
                    },
                },
                "key_players": {
                    "France": [{"name": "Kylian Mbappé", "impact": 0.20, "roles": ["finisher"]}],
                    "Iraq": [],
                },
            }
            prediction = MatchPredictor(store).predict("France", "Iraq", fixture=fixture, remember=False)
            data = prediction.to_dict()
            self.assertEqual(data["home_tactics"]["formation"], "4-3-3")
            self.assertEqual(data["home_tactics"]["formation_source"], "confirmed-lineup")
            self.assertLess(data["lineup_reports"]["France"]["availability_score"], 1.0)
            self.assertIn("Kylian Mbappé", [item["name"] for item in data["lineup_reports"]["France"]["missing_key_players"]])

    def test_dominant_favorite_can_predict_three_nil(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            prediction = MatchPredictor(store).predict("France", "Iraq", remember=False)
            self.assertIn("3-0", prediction.exact_scores)

    def test_strong_favorite_keeps_two_nil_candidate(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            prediction = MatchPredictor(store).predict("Argentina", "Algeria", remember=False)
            self.assertIn("2-0", prediction.exact_scores)

    def test_formation_guess_has_defensive_and_possession_shapes(self):
        self.assertEqual(
            formation_guess(
                possession=0.34,
                directness=0.58,
                defensive_solidity=0.42,
                attack_width=0.50,
                central_progression=0.42,
                transition_attack=0.48,
                pressing=0.50,
                line_height=0.42,
                shots=7.0,
                shots_against=15.0,
                goals_for=0.7,
                goals_against=1.6,
            ),
            "5-4-1",
        )
        self.assertEqual(
            formation_guess(
                possession=0.66,
                directness=0.44,
                defensive_solidity=0.58,
                attack_width=0.54,
                central_progression=0.68,
                transition_attack=0.50,
                pressing=0.62,
                line_height=0.64,
                shots=13.0,
                shots_against=8.0,
                goals_for=1.9,
                goals_against=0.8,
            ),
            "4-3-3",
        )

    def test_result_summary_completed_fixture(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = make_store(tmp_dir)
            fixture = {
                "date": "2099-01-01",
                "home_goals": 2,
                "away_goals": 1,
                "home_corners": 5,
                "away_corners": 4,
                "home_fouls": 12,
                "away_fouls": 14,
                "completed": True,
            }
            prediction = MatchPredictor(store).predict("England", "Ghana", fixture=fixture, remember=False)
            data = prediction.to_dict()
            self.assertEqual(data["result_summary"]["status"], "completed")
            self.assertEqual(data["result_summary"]["actual"]["score"], "2-1")
            self.assertEqual(data["result_summary"]["actual"]["fouls"], 26.0)
            self.assertIn("fouls", data["result_summary"]["predicted"])
            self.assertIn("predicted", data["result_summary"])

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
        self.assertFalse(fixture["in_progress"])
        self.assertEqual(fixture["home_corners"], 4)
        self.assertEqual(fixture["away_corners"], 6)
        self.assertEqual(fixture["away_possession"], 58)
        self.assertEqual(fixture["home_fouls"], 13)
        self.assertEqual(fixture["away_fouls"], 9)

    def test_espn_fixture_keeps_live_score_for_running_match(self):
        event = sample_espn_event()
        status_type = event["competitions"][0]["status"]["type"]
        status_type.update({"name": "STATUS_SECOND_HALF", "state": "in", "completed": False, "shortDetail": "62'"})
        provider = EspnWorldCupProvider()
        fixture = provider._fixture_from_event(event, "Uruguay", "Spain")
        self.assertIsNotNone(fixture)
        self.assertFalse(fixture["completed"])
        self.assertTrue(fixture["in_progress"])
        self.assertEqual(fixture["home_goals"], 1)
        self.assertEqual(fixture["away_goals"], 2)

    def test_espn_summary_extracts_referee(self):
        provider = EspnWorldCupProvider()
        referee = provider._referee_from_summary(sample_espn_summary())
        self.assertEqual(referee["name"], "Drew Fischer")
        self.assertEqual(referee["source"], "espn-summary-officials")


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
                            {"name": "foulsCommitted", "displayValue": "9"},
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
                            {"name": "foulsCommitted", "displayValue": "13"},
                        ],
                    },
                ],
            }
        ],
    }


def sample_espn_summary():
    return {
        "gameInfo": {
            "officials": [
                {
                    "fullName": "Drew Fischer",
                    "displayName": "Drew Fischer",
                    "position": {"name": "Referee", "displayName": "Referee", "id": "1"},
                    "order": 1,
                }
            ]
        }
    }


if __name__ == "__main__":
    unittest.main()
