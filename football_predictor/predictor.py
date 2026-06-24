from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .data_store import DataStore
from .features import build_team_stats
from .models import TeamStats


@dataclass
class Prediction:
    home_team: str
    away_team: str
    match_date: str | None
    neutral: bool
    market_pick: str
    confidence: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    expected_home_goals: float
    expected_away_goals: float
    predicted_corners: float
    exact_scores: list[str]
    home_stats: TeamStats
    away_stats: TeamStats
    home_context: dict[str, Any]
    away_context: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date,
            "neutral": self.neutral,
            "market_pick": self.market_pick,
            "confidence": round(self.confidence, 3),
            "probabilities": {
                "П1": round(self.home_win_probability, 3),
                "X": round(self.draw_probability, 3),
                "П2": round(self.away_win_probability, 3),
            },
            "expected_goals": {
                self.home_team: round(self.expected_home_goals, 2),
                self.away_team: round(self.expected_away_goals, 2),
            },
            "predicted_corners": round(self.predicted_corners, 2),
            "exact_scores": self.exact_scores,
            "home_stats": self.home_stats.as_dict(),
            "away_stats": self.away_stats.as_dict(),
            "home_context": self.home_context,
            "away_context": self.away_context,
            "warnings": self.warnings,
        }

    def short_text(self) -> str:
        scores = ", ".join(self.exact_scores)
        return f"{self.market_pick}, средние угловые: {self.predicted_corners:.2f}, точные счеты: {scores}"


class MatchPredictor:
    def __init__(self, store: DataStore):
        self.store = store

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        remember: bool = True,
        match_date: str | None = None,
    ) -> Prediction:
        matches = self.store.load_matches()
        home_stats = build_team_stats(matches, home_team)
        away_stats = build_team_stats(matches, away_team)
        context = self.store.load_context()
        home_context = context.get(home_team, {})
        away_context = context.get(away_team, {})
        state = self.store.load_model_state()
        weights = state["weights"]
        warnings = self._warnings(home_stats, away_stats)

        home_xg, away_xg = self._expected_goals(home_stats, away_stats, home_context, away_context, weights, neutral)
        home_win, draw, away_win = self._outcome_probabilities(home_xg, away_xg)
        probabilities = {"П1": home_win, "X": draw, "П2": away_win}
        market_pick = max(probabilities, key=probabilities.get)
        confidence = probabilities[market_pick]
        corners = self._expected_corners(home_stats, away_stats, home_xg + away_xg, weights)
        exact_scores = self._top_scores(home_xg, away_xg, limit=2)

        prediction = Prediction(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            neutral=neutral,
            market_pick=market_pick,
            confidence=confidence,
            home_win_probability=home_win,
            draw_probability=draw,
            away_win_probability=away_win,
            expected_home_goals=home_xg,
            expected_away_goals=away_xg,
            predicted_corners=corners,
            exact_scores=exact_scores,
            home_stats=home_stats,
            away_stats=away_stats,
            home_context=home_context,
            away_context=away_context,
            warnings=warnings,
        )
        if remember and match_date:
            self.store.save_prediction(prediction.to_dict())
        return prediction

    def _expected_goals(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_context: dict[str, Any],
        away_context: dict[str, Any],
        weights: dict[str, float],
        neutral: bool,
    ) -> tuple[float, float]:
        home_base = 0.54 * home_stats.avg_goals_for + 0.46 * away_stats.avg_goals_against
        away_base = 0.54 * away_stats.avg_goals_for + 0.46 * home_stats.avg_goals_against

        elo_home = float(home_context.get("elo") or 1500)
        elo_away = float(away_context.get("elo") or 1500)
        elo_diff = max(-1.4, min(1.4, (elo_home - elo_away) / 400))
        form_diff = max(-1.2, min(1.2, home_stats.points_per_match - away_stats.points_per_match))
        motivation_diff = self._motivation(home_context) - self._motivation(away_context)
        injury_diff = self._injury_impact(home_context) - self._injury_impact(away_context)

        home_advantage = 0.0 if neutral else weights["home_advantage_goals"]
        adjustment = (
            weights["elo_to_goals"] * elo_diff
            + weights["form_to_goals"] * form_diff
            + weights["motivation_to_goals"] * motivation_diff
            - weights["injury_to_goals"] * injury_diff
            + home_advantage
        )
        goal_scale = weights.get("goal_scale", 1.0)
        home_xg = max(0.15, (home_base + adjustment) * goal_scale)
        away_xg = max(0.15, (away_base - adjustment * 0.78) * goal_scale)
        return min(home_xg, 4.6), min(away_xg, 4.6)

    def _expected_corners(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        total_xg: float,
        weights: dict[str, float],
    ) -> float:
        samples = [value for value in (home_stats.avg_total_corners, away_stats.avg_total_corners) if value is not None]
        base = sum(samples) / len(samples) if samples else 9.2
        tempo = max(-0.6, min(1.1, (total_xg - 2.35) * 0.42))
        return max(4.0, min(16.0, base + tempo + weights.get("corner_bias", 0.0)))

    def _outcome_probabilities(self, home_xg: float, away_xg: float) -> tuple[float, float, float]:
        home_win = draw = away_win = 0.0
        for home_goals in range(8):
            for away_goals in range(8):
                probability = self._poisson(home_goals, home_xg) * self._poisson(away_goals, away_xg)
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away_win += probability
        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total

    def _top_scores(self, home_xg: float, away_xg: float, limit: int = 2) -> list[str]:
        grid = []
        for home_goals in range(6):
            for away_goals in range(6):
                probability = self._poisson(home_goals, home_xg) * self._poisson(away_goals, away_xg)
                grid.append((probability, f"{home_goals}-{away_goals}"))
        return [score for _, score in sorted(grid, reverse=True)[:limit]]

    @staticmethod
    def _poisson(k: int, rate: float) -> float:
        return math.exp(-rate) * rate**k / math.factorial(k)

    @staticmethod
    def _motivation(context: dict[str, Any]) -> float:
        motivation = context.get("motivation", {})
        if isinstance(motivation, dict):
            return float(motivation.get("level", 0.5))
        return 0.5

    @staticmethod
    def _injury_impact(context: dict[str, Any]) -> float:
        total = 0.0
        for injury in context.get("injuries", []):
            status = str(injury.get("status", "")).lower()
            if status in {"fit", "available", "ok"}:
                continue
            total += float(injury.get("impact", 0.0))
        return min(total, 2.0)

    @staticmethod
    def _warnings(home_stats: TeamStats, away_stats: TeamStats) -> list[str]:
        warnings = []
        if home_stats.sample_size < 10 or away_stats.sample_size < 10:
            warnings.append("В базе меньше 10 последних матчей для одной из команд.")
        if not home_stats.corner_samples or not away_stats.corner_samples:
            warnings.append("По угловым есть неполная статистика, часть оценки построена на среднем темпе.")
        if any(match.source == "demo_seed" for match in home_stats.recent + away_stats.recent):
            warnings.append("В стартовой базе есть демо-матчи: для боевого прогноза обновите данные через API или вручную.")
        return warnings
