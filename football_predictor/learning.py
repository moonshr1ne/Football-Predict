from __future__ import annotations

from datetime import datetime, timezone

from .data_store import DataStore
from .models import MatchRecord
from .predictor import MatchPredictor


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "П1"
    if home_goals == away_goals:
        return "X"
    return "П2"


class OnlineLearner:
    def __init__(self, store: DataStore):
        self.store = store

    def record_result(
        self,
        home_team: str,
        away_team: str,
        date: str,
        home_goals: int,
        away_goals: int,
        corners_total: float | None = None,
        home_corners: float | None = None,
        away_corners: float | None = None,
        home_possession: float | None = None,
        away_possession: float | None = None,
        home_shots: float | None = None,
        away_shots: float | None = None,
        home_shots_on_target: float | None = None,
        away_shots_on_target: float | None = None,
        home_fouls: float | None = None,
        away_fouls: float | None = None,
        referee: str | None = None,
        competition: str = "",
        stage: str = "",
        neutral: bool = True,
        source: str = "manual_result",
        baseline_prediction: dict | None = None,
    ) -> dict:
        prediction = baseline_prediction or MatchPredictor(self.store).predict(
            home_team,
            away_team,
            neutral=neutral,
            remember=False,
        ).to_dict()
        state = self.store.load_model_state()
        lr = float(state.get("learning_rate", 0.08))
        weights = state["weights"]

        if corners_total is not None and home_corners is None and away_corners is None:
            home_corners = round(corners_total / 2, 2)
            away_corners = round(corners_total - home_corners, 2)

        actual_corners_total = None
        if home_corners is not None and away_corners is not None:
            actual_corners_total = float(home_corners) + float(away_corners)
        actual_fouls_total = None
        if home_fouls is not None and away_fouls is not None:
            actual_fouls_total = float(home_fouls) + float(away_fouls)

        expected_goals = prediction.get("expected_goals", {})
        expected_home_goals = float(expected_goals.get(home_team, 1.15))
        expected_away_goals = float(expected_goals.get(away_team, 1.15))
        predicted_corners = float(prediction.get("predicted_corners", 9.2))
        predicted_fouls = float((prediction.get("foul_forecast") or {}).get("expected", 24.0))

        home_error = home_goals - expected_home_goals
        away_error = away_goals - expected_away_goals
        total_error = (home_goals + away_goals) - (expected_home_goals + expected_away_goals)
        side_error = home_error - away_error

        weights["goal_scale"] = self._clamp(weights.get("goal_scale", 1.0) + lr * total_error * 0.025, 0.72, 1.32)
        weights["home_advantage_goals"] = self._clamp(
            weights.get("home_advantage_goals", 0.18) + (0 if neutral else lr * side_error * 0.03),
            -0.05,
            0.40,
        )
        weights["form_to_goals"] = self._clamp(weights.get("form_to_goals", 0.20) + lr * side_error * 0.01, 0.02, 0.50)
        tactical_matchup = prediction.get("tactical_matchup", {})
        tactical_edge = float(tactical_matchup.get("edge", 0.0))
        if abs(tactical_edge) > 0.02:
            weights["tactics_to_goals"] = self._clamp(
                weights.get("tactics_to_goals", 0.24) + lr * side_error * tactical_edge * 0.035,
                0.02,
                0.55,
            )

        corner_error = None
        if actual_corners_total is not None:
            corner_error = actual_corners_total - predicted_corners
            weights["corner_bias"] = self._clamp(weights.get("corner_bias", 0.0) + lr * corner_error * 0.12, -2.4, 2.4)
            corner_boost = float(tactical_matchup.get("corner_boost", 0.0))
            if abs(corner_boost) > 0.03:
                weights["tactics_to_corners"] = self._clamp(
                    weights.get("tactics_to_corners", 1.15) + lr * corner_error * corner_boost * 0.025,
                    0.20,
                    2.40,
                )

        foul_error = None
        if actual_fouls_total is not None:
            foul_error = actual_fouls_total - predicted_fouls
            weights["foul_bias"] = self._clamp(weights.get("foul_bias", 0.0) + lr * foul_error * 0.10, -6.0, 6.0)

        actual_outcome = outcome_label(home_goals, away_goals)
        score = f"{home_goals}-{away_goals}"
        review = {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "prediction_id": prediction.get("prediction_id"),
            "predicted_outcome": prediction.get("market_pick"),
            "actual_outcome": actual_outcome,
            "outcome_hit": prediction.get("market_pick") == actual_outcome,
            "predicted_scores": prediction.get("exact_scores", []),
            "actual_score": score,
            "score_hit": score in prediction.get("exact_scores", []),
            "predicted_corners": round(predicted_corners, 2),
            "actual_corners": None if actual_corners_total is None else round(actual_corners_total, 2),
            "corner_error": None if corner_error is None else round(corner_error, 2),
            "predicted_fouls": round(predicted_fouls, 2),
            "actual_fouls": None if actual_fouls_total is None else round(actual_fouls_total, 2),
            "foul_error": None if foul_error is None else round(foul_error, 2),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state.setdefault("history", []).append(review)
        state["history"] = state["history"][-1000:]
        self.store.save_model_state(state)

        self.store.add_or_update_match(
            MatchRecord(
                date=date,
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                home_corners=home_corners,
                away_corners=away_corners,
                home_possession=home_possession,
                away_possession=away_possession,
                home_shots=home_shots,
                away_shots=away_shots,
                home_shots_on_target=home_shots_on_target,
                away_shots_on_target=away_shots_on_target,
                home_fouls=home_fouls,
                away_fouls=away_fouls,
                referee=referee,
                competition=competition,
                stage=stage,
                neutral=neutral,
                source=source,
            )
        )
        return review

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
