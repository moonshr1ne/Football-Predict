from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from football_predictor.barcelona_model import BarcelonaModel
from football_predictor.barcelona_provider import BarcelonaProvider
from football_predictor.barcelona_service import BarcelonaService
from football_predictor.barcelona_store import BarcelonaStore


def fixture(
    fixture_id: str,
    kickoff: str,
    opponent: str,
    barca_home: bool = True,
    home_goals: int | None = None,
    away_goals: int | None = None,
) -> dict:
    return {
        "fixture_id": fixture_id,
        "date": kickoff[:10],
        "kickoff": kickoff,
        "home_team": "Barcelona" if barca_home else opponent,
        "away_team": opponent if barca_home else "Barcelona",
        "home_team_id": "83" if barca_home else "999",
        "away_team_id": "999" if barca_home else "83",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "completed": home_goals is not None and away_goals is not None,
        "in_progress": False,
        "competition_code": "esp.1",
        "competition": "Ла Лига",
        "stage": "LALIGA",
        "home_stats": {},
        "away_stats": {},
    }


def starters(prefix: str) -> list[dict]:
    positions = [
        "Goalkeeper",
        "Right Back",
        "Center Right Defender",
        "Center Left Defender",
        "Left Back",
        "Defensive Midfielder",
        "Right Midfielder",
        "Attacking Midfielder",
        "Attacking Midfielder Right",
        "Forward",
        "Attacking Midfielder Left",
    ]
    return [
        {"name": f"{prefix} Player {slot}", "position": position, "formation_place": str(slot)}
        for slot, position in enumerate(positions, start=1)
    ]


class BarcelonaModeTests(unittest.TestCase):
    def test_russian_opponent_alias_and_nearest_upcoming_fixture(self):
        provider = BarcelonaProvider()
        now = datetime.now(timezone.utc)
        matches = [
            fixture("past", (now - timedelta(days=30)).isoformat(), "Real Madrid", home_goals=2, away_goals=1),
            fixture("future", (now + timedelta(days=30)).isoformat(), "Real Madrid"),
        ]
        opponent = provider.resolve_opponent("Реал Мадрид", matches)
        self.assertEqual(opponent, "Real Madrid")
        self.assertEqual(provider.find_fixture(opponent, matches)["fixture_id"], "future")

    def test_espn_list_pickcenter_does_not_break_summary_parsing(self):
        provider = BarcelonaProvider()
        market = provider._market_from_summary(
            {
                "pickcenter": [
                    {
                        "homeTeamOdds": {"moneyLine": -150},
                        "drawOdds": {"moneyLine": 300},
                        "awayTeamOdds": {"moneyLine": 450},
                        "overUnder": 2.5,
                    }
                ]
            }
        )
        self.assertEqual(market["total_line"], 2.5)
        self.assertEqual(set(market["probabilities"]), {"home", "draw", "away"})

    def test_prediction_snapshot_is_one_per_fixture_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BarcelonaStore(Path(tmp))
            future = fixture("match-1", "2099-05-01T18:00:00Z", "Real Madrid")
            first = store.save_prediction({"fixture_id": "match-1", "fixture": future, "exact_score": {"score": "2-1"}})
            second = store.save_prediction({"fixture_id": "match-1", "fixture": future, "exact_score": {"score": "0-4"}})
            self.assertEqual(first["prediction_id"], second["prediction_id"])
            self.assertEqual(second["exact_score"]["score"], "2-1")

    def test_result_cannot_enter_target_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = BarcelonaStore(Path(tmp))
            matches = []
            base = datetime(2025, 1, 1, tzinfo=timezone.utc)
            for index in range(10):
                matches.append(
                    fixture(
                        f"old-{index}",
                        (base + timedelta(days=index * 7)).isoformat(),
                        f"Opponent {index}",
                        barca_home=index % 2 == 0,
                        home_goals=2 if index % 2 == 0 else 0,
                        away_goals=0 if index % 2 == 0 else 2,
                    )
                )
            target = fixture("target", (base + timedelta(days=90)).isoformat(), "Real Madrid", home_goals=9, away_goals=0)
            universe = matches + [target]
            model = BarcelonaModel(store)
            first = model._training_examples(universe)[-1]
            changed = copy.deepcopy(universe)
            changed[-1]["home_goals"] = 0
            changed[-1]["away_goals"] = 8
            # Use a fresh store to avoid intentionally cached examples.
            second = BarcelonaModel(BarcelonaStore(Path(tmp) / "other"))._training_examples(changed)[-1]
            self.assertEqual(first["x"], second["x"])
            self.assertEqual(first["context"], second["context"])
            self.assertNotEqual(first["barca_goals"], second["barca_goals"])

    def test_exact_score_respects_outcome_and_clear_total_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BarcelonaModel(BarcelonaStore(Path(tmp)))
            matrix = model._score_matrix(2.4, 1.2)
            totals = model._total_probabilities(matrix)
            score, probability = model._best_consistent_score(matrix, "barcelona", totals)
            barca, opponent = map(int, score.split("-"))
            self.assertGreater(barca, opponent)
            if totals["over_2_5"] >= 0.58:
                self.assertGreaterEqual(barca + opponent, 3)
            self.assertGreater(probability, 0)

    def test_laliga_round_is_derived_from_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = BarcelonaService(store=BarcelonaStore(Path(tmp)), provider=BarcelonaProvider())
            matches = [
                fixture("one", "2099-08-10T18:00:00Z", "Elche"),
                fixture("two", "2099-08-17T18:00:00Z", "Sevilla"),
            ]
            self.assertEqual(service._with_derived_stage(matches[1], matches)["stage"], "Тур 2")

    def test_projected_lineup_has_unique_xi_positions_and_formation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BarcelonaModel(BarcelonaStore(Path(tmp)))
            base = datetime(2025, 1, 1, tzinfo=timezone.utc)
            history = []
            for index in range(10):
                match = fixture(
                    f"lineup-{index}",
                    (base + timedelta(days=index * 7)).isoformat(),
                    f"Opponent {index}",
                    home_goals=2,
                    away_goals=0,
                )
                match["lineups"] = {
                    "Barcelona": {
                        "confirmed": True,
                        "formation": "4-3-3",
                        "starters": starters("Barca"),
                        "source": "espn-summary",
                    }
                }
                history.append(match)
            target = fixture("target-lineup", (base + timedelta(days=80)).isoformat(), "Real Madrid")

            projected = model._projected_lineup("Barcelona", target, history)

            self.assertTrue(projected["available"])
            self.assertEqual(projected["formation"], "4-3-3")
            self.assertEqual(len(projected["players"]), 11)
            self.assertEqual(len({player["name"] for player in projected["players"]}), 11)
            self.assertTrue(all(player["position"] for player in projected["players"]))
            self.assertTrue(all(0 < player["probability"] <= 1 for player in projected["players"]))

    def test_lineup_report_distinguishes_official_and_predicted(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BarcelonaModel(BarcelonaStore(Path(tmp)))
            target = fixture("official-lineup", "2099-05-01T18:00:00Z", "Real Madrid")
            target["lineups"] = {
                "Barcelona": {
                    "confirmed": True,
                    "formation": "4-3-3",
                    "starters": starters("Official"),
                    "source": "espn-summary",
                }
            }

            official = model._lineup_report("Barcelona", target, [])
            predicted = model._lineup_report("Real Madrid", target, [])

            self.assertTrue(official["official_available"])
            self.assertEqual(official["display_lineup"]["type"], "confirmed")
            self.assertEqual(len(official["display_lineup"]["players"]), 11)
            self.assertFalse(predicted["official_available"])
            self.assertEqual(predicted["display_lineup"]["type"], "predicted")
            self.assertIn("Официального состава пока нет", predicted["message"])

    def test_web_uses_single_opponent_input(self):
        root = Path(__file__).resolve().parents[1]
        index = (root / "web" / "index.html").read_text(encoding="utf-8")
        app = (root / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="opponent"', index)
        self.assertNotIn('id="matchup"', index)
        self.assertIn("opponent=", app)
        self.assertNotIn("home_venue", app)
        self.assertIn("squad-list", app)
        self.assertIn("Официального состава пока нет", app)


if __name__ == "__main__":
    unittest.main()
