from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .data_store import DataStore
from .features import build_team_stats
from .models import MatchRecord, TeamStats
from .tactics import corner_tactical_boost, summarize_matchup, tactical_edge, tactical_tempo


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
    exact_score_probabilities: list[dict[str, Any]]
    markets: list[dict[str, Any]]
    home_stats: TeamStats
    away_stats: TeamStats
    home_context: dict[str, Any]
    away_context: dict[str, Any]
    match_context: dict[str, Any]
    home_tactics: dict[str, Any]
    away_tactics: dict[str, Any]
    tactical_matchup: dict[str, Any]
    fixture: dict[str, Any] | None
    result_summary: dict[str, Any]
    data_quality: dict[str, Any]
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
            "exact_score_probabilities": self.exact_score_probabilities,
            "markets": self.markets,
            "home_stats": self.home_stats.as_dict(),
            "away_stats": self.away_stats.as_dict(),
            "home_context": self.home_context,
            "away_context": self.away_context,
            "match_context": self.match_context,
            "home_tactics": self.home_tactics,
            "away_tactics": self.away_tactics,
            "tactical_matchup": self.tactical_matchup,
            "fixture": self.fixture,
            "result_summary": self.result_summary,
            "data_quality": self.data_quality,
            "warnings": self.warnings,
        }

    def short_text(self) -> str:
        scores = ", ".join(
            f"{item['score']} ({item['probability']:.1%})" for item in self.exact_score_probabilities
        )
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
        fixture: dict[str, Any] | None = None,
        extra_warnings: list[str] | None = None,
        matches_override: list[MatchRecord] | None = None,
    ) -> Prediction:
        matches = matches_override if matches_override is not None else self.store.load_matches()
        home_stats = build_team_stats(matches, home_team)
        away_stats = build_team_stats(matches, away_team)
        context = self.store.load_context()
        home_context = context.get(home_team, {})
        away_context = context.get(away_team, {})
        match_context = self.store.load_match_context()
        home_tactics = self.store.team_tactics(home_team)
        away_tactics = self.store.team_tactics(away_team)
        tactical_matchup = summarize_matchup(home_team, away_team, home_tactics, away_tactics)
        state = self.store.load_model_state()
        weights = state["weights"]
        warnings = self._warnings(home_stats, away_stats, home_tactics, away_tactics)
        data_quality = self._data_quality(home_team, away_team, home_stats, away_stats)
        if extra_warnings:
            warnings.extend(extra_warnings)

        home_xg, away_xg = self._expected_goals(
            home_stats,
            away_stats,
            home_context,
            away_context,
            home_tactics,
            away_tactics,
            match_context,
            weights,
            neutral,
        )
        home_win, draw, away_win = self._outcome_probabilities(home_xg, away_xg)
        probabilities = {"П1": home_win, "X": draw, "П2": away_win}
        market_pick = max(probabilities, key=probabilities.get)
        confidence = probabilities[market_pick]
        markets = self._markets(home_team, away_team, probabilities)
        corners = self._expected_corners(home_stats, away_stats, home_tactics, away_tactics, home_xg + away_xg, weights)
        exact_score_probabilities = self._top_scores(
            home_xg,
            away_xg,
            market_pick,
            home_stats,
            away_stats,
            limit=3,
        )
        exact_scores = [item["score"] for item in exact_score_probabilities]
        result_summary = self._result_summary(
            market_pick,
            exact_score_probabilities,
            corners,
            fixture,
            home_team,
            away_team,
        )

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
            exact_score_probabilities=exact_score_probabilities,
            markets=markets,
            home_stats=home_stats,
            away_stats=away_stats,
            home_context=home_context,
            away_context=away_context,
            match_context=match_context,
            home_tactics=home_tactics,
            away_tactics=away_tactics,
            tactical_matchup=tactical_matchup,
            fixture=fixture,
            result_summary=result_summary,
            data_quality=data_quality,
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
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
        match_context: dict[str, Any],
        weights: dict[str, float],
        neutral: bool,
    ) -> tuple[float, float]:
        home_base = 0.54 * home_stats.avg_goals_for + 0.46 * away_stats.avg_goals_against
        away_base = 0.54 * away_stats.avg_goals_for + 0.46 * home_stats.avg_goals_against

        elo_home = float(home_context.get("elo") or 1500)
        elo_away = float(away_context.get("elo") or 1500)
        elo_diff = max(-1.4, min(1.4, (elo_home - elo_away) / 400))
        form_diff = max(-1.2, min(1.2, home_stats.points_per_match - away_stats.points_per_match))
        motivation_diff = self._motivation(home_context, match_context) - self._motivation(away_context, match_context)
        injury_diff = self._injury_impact(home_context) - self._injury_impact(away_context)
        lineup_diff = self._lineup_strength(home_context, match_context) - self._lineup_strength(away_context, match_context)
        tactics_diff = tactical_edge(home_tactics, away_tactics)
        tempo = tactical_tempo(home_tactics, away_tactics)
        intensity = max(0.0, min(1.0, float(match_context.get("importance", 1.0))))

        home_advantage = 0.0 if neutral else weights["home_advantage_goals"]
        adjustment = (
            weights["elo_to_goals"] * elo_diff
            + weights["form_to_goals"] * form_diff
            + weights["motivation_to_goals"] * motivation_diff
            + weights.get("lineup_to_goals", 0.10) * lineup_diff
            + weights.get("tactics_to_goals", 0.24) * tactics_diff
            - weights["injury_to_goals"] * injury_diff
            + home_advantage
        )
        intensity_boost = weights.get("world_cup_intensity_goals", 0.05) * intensity
        goal_scale = weights.get("goal_scale", 1.0)
        home_xg = max(0.15, (home_base + adjustment + tempo + intensity_boost) * goal_scale)
        away_xg = max(0.15, (away_base - adjustment * 0.78 + tempo + intensity_boost) * goal_scale)
        return min(home_xg, 4.6), min(away_xg, 4.6)

    def _expected_corners(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
        total_xg: float,
        weights: dict[str, float],
    ) -> float:
        samples = [value for value in (home_stats.avg_total_corners, away_stats.avg_total_corners) if value is not None]
        base = sum(samples) / len(samples) if samples else 9.2
        tempo = max(-0.6, min(1.1, (total_xg - 2.35) * 0.42))
        tactics = corner_tactical_boost(home_tactics, away_tactics) * weights.get("tactics_to_corners", 1.15)
        return max(4.0, min(16.0, base + tempo + tactics + weights.get("corner_bias", 0.0)))

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

    def _top_scores(
        self,
        home_xg: float,
        away_xg: float,
        market_pick: str,
        home_stats: TeamStats,
        away_stats: TeamStats,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        score_counts_by_outcome = {
            outcome: self._historical_score_counts(outcome) for outcome in ("П1", "X", "П2")
        }
        candidate_counts = {outcome: len(self._candidate_scores(outcome)) for outcome in ("П1", "X", "П2")}
        grid = []
        for home_goals in range(7):
            for away_goals in range(7):
                score = f"{home_goals}-{away_goals}"
                outcome = self._outcome_from_score(home_goals, away_goals)
                weight = (
                    self._poisson(home_goals, home_xg)
                    * self._poisson(away_goals, away_xg)
                    * self._empirical_score_factor(
                        score,
                        score_counts_by_outcome[outcome],
                        candidate_counts[outcome],
                    )
                    * self._team_zero_factor(home_goals, away_goals, home_stats, away_stats)
                    * self._tournament_total_factor(home_goals, away_goals, home_xg + away_xg)
                )
                grid.append((weight, outcome, score))

        total_weight = sum(weight for weight, _, _ in grid)
        picked = [(weight, score) for weight, outcome, score in grid if outcome == market_pick]
        if not picked or total_weight <= 0:
            fallback = ["1-1", "0-0", "2-2"] if market_pick == "X" else ["1-0", "2-1", "2-0"]
            return [{"score": score, "probability": 0.0} for score in fallback[:limit]]
        return [
            {"score": score, "probability": round(weight / total_weight, 4)}
            for weight, score in sorted(picked, reverse=True)[:limit]
        ]

    @staticmethod
    def _outcome_from_score(home_goals: int, away_goals: int) -> str:
        if home_goals > away_goals:
            return "П1"
        if home_goals == away_goals:
            return "X"
        return "П2"

    @staticmethod
    def _markets(home_team: str, away_team: str, probabilities: dict[str, float]) -> list[dict[str, Any]]:
        home_win = probabilities["П1"]
        draw = probabilities["X"]
        away_win = probabilities["П2"]
        return [
            {"code": "П1", "label": f"Победа {home_team}", "probability": round(home_win, 3)},
            {"code": "X", "label": "Ничья", "probability": round(draw, 3)},
            {"code": "П2", "label": f"Победа {away_team}", "probability": round(away_win, 3)},
            {"code": "1X", "label": f"{home_team} не проиграет", "probability": round(home_win + draw, 3)},
            {"code": "X2", "label": f"{away_team} не проиграет", "probability": round(away_win + draw, 3)},
        ]

    def _result_summary(
        self,
        market_pick: str,
        exact_score_probabilities: list[dict[str, Any]],
        predicted_corners: float,
        fixture: dict[str, Any] | None,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        predicted = {
            "outcome": market_pick,
            "outcome_label": self._market_label(market_pick, home_team, away_team),
            "scores": exact_score_probabilities,
            "corners": round(predicted_corners, 2),
        }
        if not fixture:
            return {
                "status": "unknown",
                "predicted": predicted,
                "actual": None,
                "message": "Матч не найден в расписании, фактический счет пока неизвестен.",
            }

        home_goals = fixture.get("home_goals")
        away_goals = fixture.get("away_goals")
        has_score = home_goals is not None and away_goals is not None
        if fixture.get("completed") and has_score:
            actual_score = f"{int(home_goals)}-{int(away_goals)}"
            actual_outcome = self._outcome_from_score(int(home_goals), int(away_goals))
            actual_corners = self._fixture_total_corners(fixture)
            return {
                "status": "completed",
                "predicted": predicted,
                "actual": {
                    "score": actual_score,
                    "outcome": actual_outcome,
                    "outcome_label": self._market_label(actual_outcome, home_team, away_team),
                    "corners": actual_corners,
                },
                "outcome_hit": market_pick == actual_outcome,
                "score_hit": actual_score in [item["score"] for item in exact_score_probabilities],
                "corner_error": None if actual_corners is None else round(predicted_corners - actual_corners, 2),
                "message": "Матч завершен, факт уже доступен.",
            }

        if fixture.get("in_progress"):
            current_score = f"{int(home_goals)}-{int(away_goals)}" if has_score else None
            return {
                "status": "live",
                "predicted": predicted,
                "actual": {"score": current_score, "outcome": None, "corners": None},
                "message": "Матч сейчас идет, финальный счет еще не известен.",
            }

        return {
            "status": "scheduled",
            "predicted": predicted,
            "actual": None,
            "message": "Матч еще не начался.",
        }

    @staticmethod
    def _fixture_total_corners(fixture: dict[str, Any]) -> float | None:
        home = fixture.get("home_corners")
        away = fixture.get("away_corners")
        if home is None or away is None:
            return None
        return round(float(home) + float(away), 2)

    @staticmethod
    def _market_label(code: str, home_team: str, away_team: str) -> str:
        if code == "П1":
            return f"Победа {home_team}"
        if code == "П2":
            return f"Победа {away_team}"
        return "Ничья"

    @staticmethod
    def _candidate_scores(market_pick: str) -> list[tuple[int, int]]:
        scores = []
        for home_goals in range(7):
            for away_goals in range(7):
                if market_pick == "П1" and home_goals > away_goals:
                    scores.append((home_goals, away_goals))
                elif market_pick == "П2" and away_goals > home_goals:
                    scores.append((home_goals, away_goals))
                elif market_pick == "X" and home_goals == away_goals:
                    scores.append((home_goals, away_goals))
        return scores

    def _historical_score_counts(self, market_pick: str) -> Counter[str]:
        history = self.store.load_model_state().get("history", [])
        return Counter(
            item.get("actual_score")
            for item in history
            if item.get("actual_outcome") == market_pick and item.get("actual_score")
        )

    @staticmethod
    def _empirical_score_factor(score: str, score_counts: Counter[str], candidate_count: int) -> float:
        if not score_counts:
            return 1.0
        total = sum(score_counts.values())
        smoothing = 0.35
        prior = (score_counts.get(score, 0) + smoothing) / (total + smoothing * max(candidate_count, 1))
        average_prior = 1.0 / max(candidate_count, 1)
        return max(0.35, min(6.00, 0.50 + 1.30 * (prior / average_prior)))

    @staticmethod
    def _team_zero_factor(
        home_goals: int,
        away_goals: int,
        home_stats: TeamStats,
        away_stats: TeamStats,
    ) -> float:
        home_zero = MatchPredictor._zero_goal_probability(home_stats, away_stats)
        away_zero = MatchPredictor._zero_goal_probability(away_stats, home_stats)
        factor = 1.0
        factor *= 0.74 + (home_zero if home_goals == 0 else 1.0 - home_zero)
        factor *= 0.74 + (away_zero if away_goals == 0 else 1.0 - away_zero)
        return max(0.64, min(1.42, factor))

    @staticmethod
    def _zero_goal_probability(attacking: TeamStats, defending: TeamStats) -> float:
        attacking_blank = attacking.failed_to_score / attacking.sample_size if attacking.sample_size else 0.25
        defending_clean = defending.clean_sheets / defending.sample_size if defending.sample_size else 0.25
        return max(0.05, min(0.72, 0.5 * attacking_blank + 0.5 * defending_clean))

    @staticmethod
    def _tournament_total_factor(home_goals: int, away_goals: int, total_xg: float) -> float:
        total_goals = home_goals + away_goals
        if total_xg < 2.15 and total_goals <= 2:
            return 1.08
        if total_xg > 3.15 and total_goals >= 3:
            return 1.06
        if total_goals >= 5 and total_xg < 2.7:
            return 0.72
        return 1.0

    @staticmethod
    def _poisson(k: int, rate: float) -> float:
        return math.exp(-rate) * rate**k / math.factorial(k)

    @staticmethod
    def _motivation(context: dict[str, Any], match_context: dict[str, Any]) -> float:
        motivation = context.get("motivation", {})
        if isinstance(motivation, dict):
            level = float(motivation.get("level", 0.5))
        else:
            level = 0.5
        floor = float(match_context.get("motivation_floor", 0.5))
        return max(level, floor)

    @staticmethod
    def _lineup_strength(context: dict[str, Any], match_context: dict[str, Any]) -> float:
        if "lineup_strength" in context:
            strength = float(context["lineup_strength"])
        else:
            strength = float(match_context.get("lineup_strength_floor", 0.92))
        return max(0.0, min(1.0, strength))

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
    def _warnings(
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
    ) -> list[str]:
        warnings = []
        if home_stats.sample_size < 10 or away_stats.sample_size < 10:
            warnings.append("У одной из команд пока меньше 10 матчей в доступной базе; используются все найденные матчи ЧМ-2026 и локальная история.")
        if not home_stats.corner_samples or not away_stats.corner_samples:
            warnings.append("По угловым есть неполная статистика, часть оценки построена на среднем темпе.")
        if any(match.source == "demo_seed" for match in home_stats.recent + away_stats.recent):
            warnings.append("В стартовой базе есть демо-матчи: для боевого прогноза обновите данные через API или вручную.")
        if home_tactics.get("is_fallback") or away_tactics.get("is_fallback"):
            warnings.append("Для одной из команд нет тактического профиля, используется нейтральный шаблон.")
        warnings.append("World Cup mode: мотивация и сила состава считаются высокими для обеих команд, если вы явно не внесли травмы или изменения состава.")
        return warnings

    def _data_quality(
        self,
        home_team: str,
        away_team: str,
        home_stats: TeamStats,
        away_stats: TeamStats,
    ) -> dict[str, Any]:
        participants = self.store.load_participants()
        sync_state = self.store.load_sync_state()
        backtest = self.store.load_backtest()
        by_team = backtest.get("by_team", {}) if isinstance(backtest, dict) else {}

        home_rich_matches = min(home_stats.corner_samples, home_stats.possession_samples, home_stats.shot_samples)
        away_rich_matches = min(away_stats.corner_samples, away_stats.possession_samples, away_stats.shot_samples)
        sample_score = min(1.0, (home_stats.sample_size + away_stats.sample_size) / 20.0)
        rich_score = min(1.0, (home_rich_matches + away_rich_matches) / 20.0)
        learned_score = min(1.0, float(backtest.get("matches", 0) or 0) / 100.0) if isinstance(backtest, dict) else 0.0

        return {
            "score": round(sample_score * 0.55 + rich_score * 0.30 + learned_score * 0.15, 3),
            "match_sample_score": round(sample_score, 3),
            "rich_stat_score": round(rich_score, 3),
            "participants": len(participants),
            "last_full_sync_at": sync_state.get("last_full_sync_at"),
            "home_matches": home_stats.sample_size,
            "away_matches": away_stats.sample_size,
            "home_rich_matches": home_rich_matches,
            "away_rich_matches": away_rich_matches,
            "home_corner_samples": home_stats.corner_samples,
            "away_corner_samples": away_stats.corner_samples,
            "home_possession_samples": home_stats.possession_samples,
            "away_possession_samples": away_stats.possession_samples,
            "home_shot_samples": home_stats.shot_samples,
            "away_shot_samples": away_stats.shot_samples,
            "backtest": {
                "matches": backtest.get("matches", 0) if isinstance(backtest, dict) else 0,
                "outcome_accuracy": backtest.get("outcome_accuracy") if isinstance(backtest, dict) else None,
                "exact_score_accuracy": backtest.get("exact_score_accuracy") if isinstance(backtest, dict) else None,
                "corner_mae": backtest.get("corner_mae") if isinstance(backtest, dict) else None,
                "corner_within_one_rate": backtest.get("corner_within_one_rate") if isinstance(backtest, dict) else None,
                "targets": backtest.get("targets", {}) if isinstance(backtest, dict) else {},
                "target_status": backtest.get("target_status", {}) if isinstance(backtest, dict) else {},
                "trained_match_keys": backtest.get("trained_match_keys") if isinstance(backtest, dict) else None,
                "updated_at": backtest.get("updated_at") if isinstance(backtest, dict) else None,
            },
            "home_backtest": by_team.get(home_team, {}),
            "away_backtest": by_team.get(away_team, {}),
        }
