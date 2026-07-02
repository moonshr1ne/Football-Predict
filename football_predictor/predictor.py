from __future__ import annotations

import math
import copy
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

from .data_store import DEFAULT_MODEL_STATE, DEFAULT_TACTICAL_PROFILE, DataStore
from .features import build_team_stats
from .models import MatchRecord, TeamStats
from .providers import normalize_provider_name
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
    goal_total: dict[str, Any]
    predicted_corners: float
    foul_forecast: dict[str, Any]
    exact_scores: list[str]
    exact_score_probabilities: list[dict[str, Any]]
    markets: list[dict[str, Any]]
    recommended_bets: dict[str, Any]
    home_stats: TeamStats
    away_stats: TeamStats
    home_context: dict[str, Any]
    away_context: dict[str, Any]
    team_reports: dict[str, Any]
    lineup_reports: dict[str, Any]
    match_context: dict[str, Any]
    home_tactics: dict[str, Any]
    away_tactics: dict[str, Any]
    tactical_matchup: dict[str, Any]
    h2h_report: dict[str, Any]
    round_info: dict[str, Any]
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
            "goal_total": self.goal_total,
            "predicted_corners": round(self.predicted_corners, 2),
            "corner_forecast": {
                "point_estimate": int(math.floor(self.predicted_corners + 0.5)),
                "expected": round(self.predicted_corners, 2),
            },
            "foul_forecast": self.foul_forecast,
            "exact_scores": self.exact_scores,
            "exact_score_probabilities": self.exact_score_probabilities,
            "markets": self.markets,
            "recommended_bets": self.recommended_bets,
            "home_stats": self.home_stats.as_dict(),
            "away_stats": self.away_stats.as_dict(),
            "home_context": self.home_context,
            "away_context": self.away_context,
            "team_reports": self.team_reports,
            "lineup_reports": self.lineup_reports,
            "match_context": self.match_context,
            "home_tactics": self.home_tactics,
            "away_tactics": self.away_tactics,
            "tactical_matchup": self.tactical_matchup,
            "h2h_report": self.h2h_report,
            "round_info": self.round_info,
            "fixture": self.fixture,
            "result_summary": self.result_summary,
            "data_quality": self.data_quality,
            "warnings": self.warnings,
        }

    def short_text(self) -> str:
        scores = ", ".join(
            f"{item['score']} ({item['probability']:.1%})" for item in self.exact_score_probabilities
        )
        fouls = self.foul_forecast.get("expected")
        foul_text = "" if fouls is None else f", фолы: {float(fouls):.2f}"
        return f"{self.market_pick}, средние угловые: {self.predicted_corners:.2f}{foul_text}, точный счет: {scores}"


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
        model_state_override: dict[str, Any] | None = None,
        temporal_snapshot: bool = False,
    ) -> Prediction:
        neutral = True
        all_matches = matches_override if matches_override is not None else self.store.load_matches()
        target_date = match_date or (fixture or {}).get("date")
        round_info = self._round_info(fixture, all_matches, home_team, away_team, target_date)
        matches = self._pre_match_history(all_matches, home_team, away_team, target_date, fixture)
        prediction_fixture = self._fixture_without_result(fixture)
        home_stats = build_team_stats(matches, home_team)
        away_stats = build_team_stats(matches, away_team)
        context = self.store.load_context()
        historical_cutoff = temporal_snapshot or bool(
            target_date and any(match.is_finished() and match.date >= target_date for match in all_matches)
        )
        raw_home_context = {} if historical_cutoff else context.get(home_team, {})
        raw_away_context = {} if historical_cutoff else context.get(away_team, {})
        match_context = self.store.load_match_context()
        lineup_reports = self._lineup_reports(home_team, away_team, prediction_fixture, match_context)
        home_context = self._context_with_lineup_report(raw_home_context, lineup_reports.get(home_team, {}))
        away_context = self._context_with_lineup_report(raw_away_context, lineup_reports.get(away_team, {}))
        home_tactics = self._prediction_tactics(home_team, matches, target_date)
        away_tactics = self._prediction_tactics(away_team, matches, target_date)
        home_tactics = self._apply_lineup_tactics(home_team, home_tactics, lineup_reports.get(home_team, {}), prediction_fixture)
        away_tactics = self._apply_lineup_tactics(away_team, away_tactics, lineup_reports.get(away_team, {}), prediction_fixture)
        tactical_matchup = summarize_matchup(home_team, away_team, home_tactics, away_tactics)
        h2h_report = self._head_to_head_report(matches, home_team, away_team, target_date)
        raw_state = model_state_override if model_state_override is not None else self.store.load_model_state()
        state = self._state_for_prediction(
            raw_state,
            matches,
            all_matches,
            target_date,
            trusted_snapshot=temporal_snapshot,
        )
        weights = state["weights"]
        warnings = self._warnings(home_stats, away_stats, home_tactics, away_tactics)
        warnings.extend(self._lineup_warnings(home_team, away_team, lineup_reports))
        data_quality = self._data_quality(home_team, away_team, home_stats, away_stats)
        if extra_warnings:
            warnings.extend(extra_warnings)
        if self._fixture_has_result_data(fixture):
            warnings.append("Счет и статистика этого матча исключены из входа модели; факт используется только для проверки после прогноза.")

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
            h2h_report,
        )
        home_xg, away_xg = self._calibrate_goal_rates(
            home_xg,
            away_xg,
            state,
            round_info,
        )
        base_home_win, base_draw, base_away_win = self._outcome_probabilities(home_xg, away_xg)
        outcome_features = self._outcome_features(
            home_team,
            away_team,
            home_stats,
            away_stats,
            home_tactics,
            away_tactics,
            home_xg,
            away_xg,
            neutral,
            h2h_report,
        )
        home_win, draw, away_win = self._apply_outcome_model(
            {"П1": base_home_win, "X": base_draw, "П2": base_away_win},
            outcome_features,
            state,
        )
        probabilities = {"П1": home_win, "X": draw, "П2": away_win}
        market_pick = max(probabilities, key=probabilities.get)
        confidence = probabilities[market_pick]
        markets = self._markets(home_team, away_team, probabilities)
        corners = self._expected_corners(
            home_stats,
            away_stats,
            home_tactics,
            away_tactics,
            home_xg + away_xg,
            weights,
            state,
            h2h_report,
        )
        corners, corner_calibration = self._calibrate_metric(
            corners,
            "corners",
            state,
            round_info,
            4.0,
            16.0,
        )
        foul_forecast = self._foul_forecast(
            home_stats,
            away_stats,
            home_tactics,
            away_tactics,
            prediction_fixture,
            weights,
            state,
            h2h_report,
        )
        goal_total = self._goal_total_forecast(home_xg, away_xg)
        goal_total["calibration_adjustment"] = round(
            float(state.get("_last_goal_calibration_adjustment", 0.0)),
            3,
        )
        goal_total["corner_calibration_adjustment"] = round(corner_calibration, 3)
        team_reports = {
            home_team: self._team_report(home_team, home_stats, home_tactics, home_context, home_xg, lineup_reports.get(home_team, {})),
            away_team: self._team_report(away_team, away_stats, away_tactics, away_context, away_xg, lineup_reports.get(away_team, {})),
        }
        exact_score_probabilities = self._top_scores(
            home_xg,
            away_xg,
            market_pick,
            home_stats,
            away_stats,
            goal_total,
            probabilities,
            limit=1,
            state=state,
            h2h_report=h2h_report,
            round_info=round_info,
        )
        exact_scores = [item["score"] for item in exact_score_probabilities]
        if exact_scores:
            goal_total["aligned_score"] = exact_scores[0]
            goal_total["alignment"] = self._score_total_alignment(exact_scores[0], goal_total)
        recommended_bets = self._recommended_bets(
            home_team,
            away_team,
            market_pick,
            probabilities,
            goal_total,
            corners,
            foul_forecast,
        )
        result_summary = self._result_summary(
            market_pick,
            exact_score_probabilities,
            corners,
            goal_total,
            foul_forecast,
            fixture,
            home_team,
            away_team,
        )

        prediction = Prediction(
            home_team=home_team,
            away_team=away_team,
            match_date=target_date,
            neutral=neutral,
            market_pick=market_pick,
            confidence=confidence,
            home_win_probability=home_win,
            draw_probability=draw,
            away_win_probability=away_win,
            expected_home_goals=home_xg,
            expected_away_goals=away_xg,
            goal_total=goal_total,
            predicted_corners=corners,
            foul_forecast=foul_forecast,
            exact_scores=exact_scores,
            exact_score_probabilities=exact_score_probabilities,
            markets=markets,
            recommended_bets=recommended_bets,
            home_stats=home_stats,
            away_stats=away_stats,
            home_context=home_context,
            away_context=away_context,
            team_reports=team_reports,
            lineup_reports=lineup_reports,
            match_context=match_context,
            home_tactics=home_tactics,
            away_tactics=away_tactics,
            tactical_matchup=tactical_matchup,
            h2h_report=h2h_report,
            round_info=round_info,
            fixture=fixture,
            result_summary=result_summary,
            data_quality=data_quality,
            warnings=warnings,
        )
        if remember and target_date:
            self.store.save_prediction(prediction.to_dict())
        return prediction

    @classmethod
    def _pre_match_history(
        cls,
        matches: list[MatchRecord],
        home_team: str,
        away_team: str,
        target_date: str | None,
        fixture: dict[str, Any] | None,
    ) -> list[MatchRecord]:
        fixture_id = str((fixture or {}).get("fixture_id") or "")
        filtered: list[MatchRecord] = []
        for match in matches:
            if fixture_id and match.fixture_id and str(match.fixture_id) == fixture_id:
                continue
            if cls._same_pair(match, home_team, away_team) and target_date and match.date == target_date:
                continue
            if target_date and match.date >= target_date:
                continue
            filtered.append(match)
        return filtered

    @staticmethod
    def _same_pair(match: MatchRecord, home_team: str, away_team: str) -> bool:
        return (
            (match.home_team == home_team and match.away_team == away_team)
            or (match.home_team == away_team and match.away_team == home_team)
        )

    @classmethod
    def _round_info(
        cls,
        fixture: dict[str, Any] | None,
        matches: list[MatchRecord],
        home_team: str,
        away_team: str,
        target_date: str | None,
    ) -> dict[str, Any]:
        competition = str((fixture or {}).get("competition") or "")
        source = "fixture" if competition else None
        if not competition and target_date:
            stored = next(
                (
                    match
                    for match in matches
                    if match.date == target_date and cls._same_pair(match, home_team, away_team)
                ),
                None,
            )
            if stored:
                competition = stored.competition or ""
                source = "match-history"

        normalized = competition.lower()
        rounds = (
            (("round of 64",), "1/32", "1/32 финала"),
            (("round of 32",), "1/16", "1/16 финала"),
            (("round of 16",), "1/8", "1/8 финала"),
            (("quarterfinal", "quarter-final"), "1/4", "1/4 финала"),
            (("semifinal", "semi-final"), "1/2", "1/2 финала"),
            (("third place", "3rd place"), "3rd", "Матч за 3-е место"),
            (("final",), "final", "Финал"),
        )
        for aliases, code, label in rounds:
            if any(alias in normalized for alias in aliases):
                return {
                    "code": code,
                    "label": label,
                    "knockout": True,
                    "competition": competition,
                    "source": source,
                }
        if "group" in normalized:
            group_name = competition.split(",", 1)[1].strip() if "," in competition else "Групповой этап"
            return {
                "code": "group",
                "label": group_name,
                "knockout": False,
                "competition": competition,
                "source": source,
            }
        return {
            "code": "playoff" if target_date and target_date >= "2026-06-28" else "unknown",
            "label": "Плей-офф" if target_date and target_date >= "2026-06-28" else "Стадия не определена",
            "knockout": bool(target_date and target_date >= "2026-06-28"),
            "competition": competition or None,
            "source": source,
        }

    @classmethod
    def _fixture_without_result(cls, fixture: dict[str, Any] | None) -> dict[str, Any] | None:
        if not fixture:
            return None
        clean = copy.deepcopy(fixture)
        for key in (
            "home_goals",
            "away_goals",
            "home_corners",
            "away_corners",
            "home_fouls",
            "away_fouls",
            "home_possession",
            "away_possession",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
        ):
            clean.pop(key, None)
        clean["completed"] = False
        clean["in_progress"] = False
        clean["source_in_progress"] = bool(fixture.get("in_progress"))
        clean["prediction_input_sanitized"] = cls._fixture_has_result_data(fixture)
        return clean

    @staticmethod
    def _fixture_has_result_data(fixture: dict[str, Any] | None) -> bool:
        if not fixture:
            return False
        return any(
            fixture.get(key) is not None
            for key in (
                "home_goals",
                "away_goals",
                "home_corners",
                "away_corners",
                "home_fouls",
                "away_fouls",
                "home_possession",
                "away_possession",
                "home_shots",
                "away_shots",
            )
        )

    def _state_for_prediction(
        self,
        state: dict[str, Any],
        pre_match_history: list[MatchRecord],
        all_matches: list[MatchRecord],
        target_date: str | None,
        trusted_snapshot: bool = False,
    ) -> dict[str, Any]:
        if trusted_snapshot:
            return copy.deepcopy(state)
        if not target_date:
            return state
        finished_dates = [match.date for match in all_matches if match.is_finished()]
        if finished_dates and max(finished_dates) < target_date:
            return state

        clean = copy.deepcopy(state)
        clean["weights"] = copy.deepcopy(DEFAULT_MODEL_STATE["weights"])
        clean["outcome_model"] = {}
        clean["stat_profiles"] = self._stat_profiles_from_matches(pre_match_history)
        clean["score_profiles"] = self._score_profiles_from_matches(pre_match_history)
        clean["history"] = self._history_from_matches(pre_match_history)
        training = clean.setdefault("training", {})
        training["prediction_cutoff"] = target_date
        training["leak_guard"] = "pre_match_history_only"
        return clean

    @classmethod
    def _stat_profiles_from_matches(cls, matches: list[MatchRecord]) -> dict[str, Any]:
        corner_totals: list[float] = []
        corner_teams: dict[str, list[float]] = {}
        corner_for: dict[str, list[float]] = {}
        corner_against: dict[str, list[float]] = {}
        foul_totals: list[float] = []
        foul_teams: dict[str, list[float]] = {}
        foul_for: dict[str, list[float]] = {}
        foul_against: dict[str, list[float]] = {}
        foul_referees: dict[str, list[float]] = {}

        for match in matches:
            if match.home_corners is not None and match.away_corners is not None:
                total_corners = float(match.home_corners) + float(match.away_corners)
                corner_totals.append(total_corners)
                corner_teams.setdefault(match.home_team, []).append(total_corners)
                corner_teams.setdefault(match.away_team, []).append(total_corners)
                corner_for.setdefault(match.home_team, []).append(float(match.home_corners))
                corner_for.setdefault(match.away_team, []).append(float(match.away_corners))
                corner_against.setdefault(match.home_team, []).append(float(match.away_corners))
                corner_against.setdefault(match.away_team, []).append(float(match.home_corners))
            if match.home_fouls is not None and match.away_fouls is not None:
                total_fouls = float(match.home_fouls) + float(match.away_fouls)
                foul_totals.append(total_fouls)
                foul_teams.setdefault(match.home_team, []).append(total_fouls)
                foul_teams.setdefault(match.away_team, []).append(total_fouls)
                foul_for.setdefault(match.home_team, []).append(float(match.home_fouls))
                foul_for.setdefault(match.away_team, []).append(float(match.away_fouls))
                foul_against.setdefault(match.home_team, []).append(float(match.away_fouls))
                foul_against.setdefault(match.away_team, []).append(float(match.home_fouls))
                if match.referee:
                    foul_referees.setdefault(match.referee, []).append(total_fouls)

        return {
            "corners": {
                "global": cls._mean_profile(corner_totals),
                "global_median": cls._median_profile(corner_totals),
                "teams": {
                    team: {
                        "matches": len(values),
                        "avg_total": cls._mean_profile(values),
                        "median_total": cls._median_profile(values),
                        "avg_for": cls._mean_profile(corner_for.get(team, [])),
                        "avg_against": cls._mean_profile(corner_against.get(team, [])),
                    }
                    for team, values in sorted(corner_teams.items())
                },
            },
            "fouls": {
                "global": cls._mean_profile(foul_totals),
                "global_median": cls._median_profile(foul_totals),
                "teams": {
                    team: {
                        "matches": len(values),
                        "avg_total": cls._mean_profile(values),
                        "median_total": cls._median_profile(values),
                        "avg_for": cls._mean_profile(foul_for.get(team, [])),
                        "avg_against": cls._mean_profile(foul_against.get(team, [])),
                    }
                    for team, values in sorted(foul_teams.items())
                },
                "referees": {
                    referee: {"matches": len(values), "avg_fouls": cls._mean_profile(values), "source": "pre-match-history"}
                    for referee, values in sorted(foul_referees.items())
                },
            },
        }

    @classmethod
    def _score_profiles_from_matches(cls, matches: list[MatchRecord]) -> dict[str, Any]:
        by_outcome: dict[str, Counter[str]] = {}
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            outcome = cls._outcome_from_score(int(match.home_goals), int(match.away_goals))
            score = f"{int(match.home_goals)}-{int(match.away_goals)}"
            by_outcome.setdefault(outcome, Counter())[score] += 1
        return {
            "mode": "pre_match_outcome_top1",
            "by_outcome": {
                outcome: cls._profile_items_from_counter(counter)
                for outcome, counter in sorted(by_outcome.items())
            },
            "by_outcome_bucket": {},
        }

    @classmethod
    def _history_from_matches(cls, matches: list[MatchRecord]) -> list[dict[str, Any]]:
        history = []
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            actual_outcome = cls._outcome_from_score(int(match.home_goals), int(match.away_goals))
            actual_score = f"{int(match.home_goals)}-{int(match.away_goals)}"
            history.append({"actual_outcome": actual_outcome, "actual_score": actual_score})
        return history[-5000:]

    @staticmethod
    def _mean_profile(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    @staticmethod
    def _median_profile(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return round(ordered[middle], 2)
        return round((ordered[middle - 1] + ordered[middle]) / 2.0, 2)

    @staticmethod
    def _profile_items_from_counter(counter: Counter[str]) -> list[dict[str, Any]]:
        total = sum(counter.values()) or 1
        return [
            {"score": score, "count": count, "probability": round(count / total, 4)}
            for score, count in counter.most_common()
        ]

    def _prediction_tactics(
        self,
        team: str,
        matches: list[MatchRecord],
        target_date: str | None,
    ) -> dict[str, Any]:
        if not target_date:
            return self.store.team_tactics(team)

        stats = build_team_stats(matches, team, limit=10)
        if stats.sample_size < 2:
            fallback = copy.deepcopy(DEFAULT_TACTICAL_PROFILE)
            fallback.update(
                {
                    "team": team,
                    "sample_size": stats.sample_size,
                    "formation_source": "pre-match-fallback",
                    "formation_confidence": 0.20,
                    "is_fallback": True,
                }
            )
            return fallback

        possession = (stats.avg_possession if stats.avg_possession is not None else 50.0) / 100.0
        shots = stats.avg_shots_for if stats.avg_shots_for is not None else 10.0
        shots_against = stats.avg_shots_against if stats.avg_shots_against is not None else 10.0
        shots_on_target = stats.avg_shots_on_target_for if stats.avg_shots_on_target_for is not None else 3.2
        corners_for = stats.avg_corners_for if stats.avg_corners_for is not None else 4.5
        corners_against = stats.avg_corners_against if stats.avg_corners_against is not None else 4.5
        goals_for = stats.avg_goals_for
        goals_against = stats.avg_goals_against

        chance_creation = self._clamp01(0.35 + shots / 80.0 + shots_on_target / 40.0 + goals_for / 16.0)
        defensive_solidity = self._clamp01(0.86 - goals_against / 4.5 - shots_against / 34.0)
        attack_width = self._clamp01(0.40 + corners_for / 25.0)
        set_piece_threat = self._clamp01(0.38 + corners_for / 30.0 + goals_for / 20.0)
        directness = self._clamp01(0.45 + shots / 70.0 - possession * 0.35)
        transition_attack = self._clamp01(0.38 + directness * 0.25 + goals_for / 8.0)
        transition_defense = self._clamp01(
            0.35 + defensive_solidity * 0.55 + (1.0 - corners_against / 12.0) * 0.18
        )
        pressing = self._clamp01(0.42 + shots / 70.0 + (1.0 - possession) * 0.12)
        tempo = self._clamp01(0.38 + (shots + shots_against) / 80.0)
        central_progression = self._clamp01(0.34 + possession * 0.32 + shots_on_target / 35.0)
        line_height = self._clamp01(0.40 + pressing * 0.20 + possession * 0.15)

        formation_history = []
        formation_counts: Counter[str] = Counter()
        for match in stats.recent:
            formation = match.formation_for(team)
            if not formation:
                continue
            weight = max(0.35, 1.0 - len(formation_history) * 0.10)
            formation_counts[formation] += weight
            formation_history.append(
                {
                    "date": match.date,
                    "opponent": match.away_team if match.home_team == team else match.home_team,
                    "formation": formation,
                    "source": "confirmed-lineup" if match.lineup_confirmed_for(team) else "match-record",
                }
            )

        estimated_formation = self._formation_from_signals(
            possession,
            directness,
            defensive_solidity,
            attack_width,
            central_progression,
            transition_attack,
            pressing,
            line_height,
            shots,
            shots_against,
            goals_for,
            goals_against,
        )
        formation = estimated_formation
        formation_source = "estimated-from-pre-match-stats"
        formation_confidence = min(0.82, 0.32 + stats.sample_size * 0.05)
        if formation_counts:
            formation, formation_weight = formation_counts.most_common(1)[0]
            consistency = formation_weight / (sum(formation_counts.values()) or 1.0)
            formation_source = "confirmed-lineups-before-match"
            formation_confidence = min(0.94, 0.52 + consistency * 0.30 + len(formation_history) * 0.02)

        return {
            "team": team,
            "formation": formation,
            "formation_source": formation_source,
            "formation_confidence": round(formation_confidence, 3),
            "formation_history": formation_history[:6],
            "estimated_formation": estimated_formation,
            "style": self._style_from_signals(possession, directness, pressing),
            "build_up": "short positional build-up" if possession >= 0.60 else "direct build-up" if directness >= 0.62 else "mixed build-up",
            "primary_attack": "wide attacks" if attack_width >= max(central_progression, transition_attack) else "quick transitions" if transition_attack >= central_progression else "central progression",
            "defensive_block": "high" if line_height >= 0.62 else "low-mid" if line_height <= 0.45 else "mid",
            "possession_intent": round(possession, 3),
            "pressing": round(pressing, 3),
            "line_height": round(line_height, 3),
            "defensive_solidity": round(defensive_solidity, 3),
            "attack_width": round(attack_width, 3),
            "central_progression": round(central_progression, 3),
            "directness": round(directness, 3),
            "chance_creation": round(chance_creation, 3),
            "transition_attack": round(transition_attack, 3),
            "transition_defense": round(transition_defense, 3),
            "set_piece_threat": round(set_piece_threat, 3),
            "tempo": round(tempo, 3),
            "sample_size": stats.sample_size,
            "source": "strict-pre-match-history",
            "is_fallback": False,
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _style_from_signals(possession: float, directness: float, pressing: float) -> str:
        if possession >= 0.60:
            return "possession control"
        if directness >= 0.63:
            return "direct transitions"
        if pressing >= 0.62:
            return "pressing and territory"
        return "balanced tournament football"

    @staticmethod
    def _formation_from_signals(
        possession: float,
        directness: float,
        defensive_solidity: float,
        attack_width: float,
        central_progression: float,
        transition_attack: float,
        pressing: float,
        line_height: float,
        shots: float,
        shots_against: float,
        goals_for: float,
        goals_against: float,
    ) -> str:
        if possession <= 0.38 and shots_against >= 13.0 and goals_for <= 1.1:
            return "5-4-1"
        if defensive_solidity >= 0.64 and attack_width >= 0.60 and line_height <= 0.56:
            return "3-4-2-1"
        if possession >= 0.61 and central_progression >= 0.60:
            return "4-3-3"
        if pressing >= 0.66 and shots >= 13.0 and central_progression >= 0.56:
            return "4-3-3"
        if directness >= 0.62 and transition_attack >= 0.60:
            return "4-2-3-1"
        if possession <= 0.44 and directness >= 0.56 and attack_width >= 0.56:
            return "4-4-2"
        if defensive_solidity >= 0.66 or (goals_against <= 0.7 and shots_against <= 8.0):
            return "4-3-3" if central_progression >= attack_width else "4-2-3-1"
        if central_progression >= attack_width and possession >= 0.52:
            return "4-3-3"
        return "4-2-3-1"

    @classmethod
    def _head_to_head_report(
        cls,
        matches: list[MatchRecord],
        home_team: str,
        away_team: str,
        target_date: str | None,
    ) -> dict[str, Any]:
        try:
            cutoff = date.fromisoformat(target_date) if target_date else date.today()
        except ValueError:
            cutoff = date.today()
        direct = sorted(
            [match for match in matches if match.is_finished() and cls._same_pair(match, home_team, away_team)],
            key=lambda item: item.date,
            reverse=True,
        )[:10]
        if not direct:
            return {
                "matches": 0,
                "recent_matches": 0,
                "older_matches": 0,
                "effective_matches": 0.0,
                "impact": 0.0,
                "home_goal_edge": 0.0,
                "average_total_goals": None,
                "average_corners": None,
                "average_fouls": None,
                "history": [],
                "policy": "no prior meetings",
            }

        total_weight = 0.0
        recent_count = 0
        older_count = 0
        home_goals_sum = away_goals_sum = 0.0
        home_points_sum = 0.0
        goal_total_sum = 0.0
        corner_sum = corner_weight = 0.0
        foul_sum = foul_weight = 0.0
        history = []

        for match in direct:
            try:
                match_day = date.fromisoformat(match.date)
                age_days = max(0, (cutoff - match_day).days)
            except ValueError:
                age_days = 1460
            if age_days <= 730:
                weight = 1.0 - 0.35 * age_days / 730.0
                recent_count += 1
                influence = "main"
            else:
                weight = 0.20 * math.exp(-(age_days - 730) / 1460.0)
                older_count += 1
                influence = "auxiliary"
            if "world cup" in (match.competition or "").lower():
                weight *= 1.12

            if match.home_team == home_team:
                home_goals = int(match.home_goals or 0)
                away_goals = int(match.away_goals or 0)
            else:
                home_goals = int(match.away_goals or 0)
                away_goals = int(match.home_goals or 0)
            points = 3.0 if home_goals > away_goals else 1.0 if home_goals == away_goals else 0.0
            total_weight += weight
            home_goals_sum += home_goals * weight
            away_goals_sum += away_goals * weight
            home_points_sum += points * weight
            goal_total_sum += (home_goals + away_goals) * weight

            if match.home_corners is not None and match.away_corners is not None:
                corner_sum += (float(match.home_corners) + float(match.away_corners)) * weight
                corner_weight += weight
            if match.home_fouls is not None and match.away_fouls is not None:
                foul_sum += (float(match.home_fouls) + float(match.away_fouls)) * weight
                foul_weight += weight
            history.append(
                {
                    "date": match.date,
                    "score": f"{home_goals}-{away_goals}",
                    "competition": match.competition,
                    "age_days": age_days,
                    "influence": influence,
                    "weight": round(weight, 3),
                }
            )

        denominator = total_weight or 1.0
        home_avg = home_goals_sum / denominator
        away_avg = away_goals_sum / denominator
        impact = min(1.0, total_weight / 3.0)
        if recent_count == 0:
            impact = min(impact, 0.22)
        return {
            "matches": len(direct),
            "recent_matches": recent_count,
            "older_matches": older_count,
            "effective_matches": round(total_weight, 2),
            "impact": round(impact, 3),
            "home_goal_edge": round(max(-2.0, min(2.0, home_avg - away_avg)), 3),
            "home_points_per_match": round(home_points_sum / denominator, 2),
            "average_score": {home_team: round(home_avg, 2), away_team: round(away_avg, 2)},
            "average_total_goals": round(goal_total_sum / denominator, 2),
            "average_corners": None if not corner_weight else round(corner_sum / corner_weight, 2),
            "average_fouls": None if not foul_weight else round(foul_sum / foul_weight, 2),
            "history": history[:6],
            "policy": "last 2 years are primary; older meetings are auxiliary",
        }

    def _lineup_reports(
        self,
        home_team: str,
        away_team: str,
        fixture: dict[str, Any] | None,
        match_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            team: self._lineup_report(team, fixture, match_context)
            for team in (home_team, away_team)
        }

    def _lineup_report(
        self,
        team: str,
        fixture: dict[str, Any] | None,
        match_context: dict[str, Any],
    ) -> dict[str, Any]:
        floor = float(match_context.get("lineup_strength_floor", 0.92))
        lineup = (fixture or {}).get("lineups", {}).get(team) if fixture else None
        key_players = self._key_players_for_team(team, fixture)
        if not lineup or not lineup.get("confirmed"):
            return {
                "team": team,
                "status": "not_released",
                "availability_score": round(floor, 3),
                "confidence": 0.25,
                "formation": None,
                "formation_source": "recent-match-projection",
                "key_players": key_players,
                "starting_key_players": [],
                "benched_key_players": [],
                "missing_key_players": [],
                "starters": [],
                "message": "Состав еще не вышел; используется базовая сила ЧМ и схема по последним матчам.",
            }

        starters = lineup.get("starters") or []
        bench = lineup.get("bench") or []
        starter_names = {normalize_provider_name(player.get("name", "")) for player in starters}
        squad_names = starter_names | {normalize_provider_name(player.get("name", "")) for player in bench}
        starting_key_players = []
        benched_key_players = []
        missing_key_players = []
        penalty = 0.0
        for player in key_players:
            normalized = normalize_provider_name(player.get("name", ""))
            impact = float(player.get("impact", 0.08))
            if normalized in starter_names:
                starting_key_players.append(player)
            elif normalized in squad_names:
                item = dict(player)
                item["status"] = "bench"
                benched_key_players.append(item)
                penalty += impact * 0.55
            else:
                item = dict(player)
                item["status"] = "absent"
                missing_key_players.append(item)
                penalty += impact

        availability = max(0.58, min(1.03, 1.0 - penalty))
        return {
            "team": team,
            "status": "confirmed",
            "availability_score": round(availability, 3),
            "confidence": 0.92,
            "formation": lineup.get("formation"),
            "formation_source": "current-lineup",
            "key_players": key_players,
            "starting_key_players": starting_key_players,
            "benched_key_players": benched_key_players,
            "missing_key_players": missing_key_players,
            "starters": [player.get("name") for player in starters[:11] if player.get("name")],
            "message": "Состав подтвержден источником; схема и сила состава применены к прогнозу.",
        }

    def _key_players_for_team(self, team: str, fixture: dict[str, Any] | None) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        for player in self.store.load_key_players().get(team, []):
            key = normalize_provider_name(player.get("name", ""))
            if key:
                combined[key] = dict(player)
        for player in (fixture or {}).get("key_players", {}).get(team, []):
            key = normalize_provider_name(player.get("name", ""))
            if not key:
                continue
            existing = combined.get(key, {})
            merged = dict(player)
            merged["impact"] = max(float(existing.get("impact", 0.0)), float(player.get("impact", 0.0)))
            merged["roles"] = sorted(set(existing.get("roles", []) + player.get("roles", [])))
            combined[key] = merged
        return sorted(combined.values(), key=lambda item: float(item.get("impact", 0.0)), reverse=True)[:8]

    def _context_with_lineup_report(self, context: dict[str, Any], lineup_report: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(context)
        merged["lineup_status"] = lineup_report.get("status", "not_released")
        merged["lineup_report"] = lineup_report
        if lineup_report.get("status") != "confirmed":
            return merged
        auto_strength = float(lineup_report.get("availability_score", 1.0))
        if "lineup_strength" in merged:
            merged["lineup_strength"] = min(float(merged["lineup_strength"]), auto_strength)
        else:
            merged["lineup_strength"] = auto_strength
        auto_absences = []
        for player in lineup_report.get("missing_key_players", []):
            auto_absences.append({"player": player.get("name"), "status": "absent", "impact": player.get("impact", 0.08)})
        for player in lineup_report.get("benched_key_players", []):
            auto_absences.append({"player": player.get("name"), "status": "bench", "impact": float(player.get("impact", 0.08)) * 0.55})
        if auto_absences:
            merged["auto_absences"] = auto_absences
        return merged

    def _apply_lineup_tactics(
        self,
        team: str,
        tactics: dict[str, Any],
        lineup_report: dict[str, Any],
        fixture: dict[str, Any] | None,
    ) -> dict[str, Any]:
        adjusted = copy.deepcopy(tactics)
        formation = lineup_report.get("formation")
        if lineup_report.get("status") == "confirmed" and formation:
            adjusted["formation"] = formation
            adjusted["formation_source"] = "live-lineup" if (fixture or {}).get("source_in_progress") else "confirmed-lineup"
            adjusted["formation_confidence"] = 0.96
            adjusted["starters"] = lineup_report.get("starters", [])
        adjusted["lineup_status"] = lineup_report.get("status", "not_released")
        adjusted["lineup_availability"] = lineup_report.get("availability_score")
        adjusted["team"] = team
        return adjusted

    def _outcome_features(
        self,
        home_team: str,
        away_team: str,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
        home_xg: float,
        away_xg: float,
        neutral: bool,
        h2h_report: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        goal_diff_delta = (home_stats.avg_goals_for - home_stats.avg_goals_against) - (
            away_stats.avg_goals_for - away_stats.avg_goals_against
        )
        points_delta = home_stats.points_per_match - away_stats.points_per_match
        class_delta = self._team_class_score(home_stats) - self._team_class_score(away_stats)
        xg_delta = home_xg - away_xg
        tactic_delta = tactical_edge(home_tactics, away_tactics)
        attack_delta = home_stats.avg_goals_for - away_stats.avg_goals_for
        defense_delta = away_stats.avg_goals_against - home_stats.avg_goals_against
        h2h_report = h2h_report or {}
        h2h_edge = float(h2h_report.get("home_goal_edge") or 0.0) * float(h2h_report.get("impact") or 0.0)
        features = {
            "bias": 1.0,
            "xg_delta": xg_delta,
            "abs_xg_delta": abs(xg_delta),
            "goal_diff_delta": goal_diff_delta,
            "points_delta": points_delta,
            "class_delta": class_delta,
            "attack_delta": attack_delta,
            "defense_delta": defense_delta,
            "tactic_delta": tactic_delta,
            "h2h_edge": h2h_edge,
            "h2h_sample": min(1.0, float(h2h_report.get("effective_matches") or 0.0) / 3.0),
            "neutral": 1.0 if neutral else 0.0,
            f"home:{home_team}": 1.0,
            f"away:{away_team}": 1.0,
            f"home_form:{home_team}": points_delta,
            f"away_form:{away_team}": -points_delta,
        }
        return {key: float(value) for key, value in features.items() if value}

    @staticmethod
    def _apply_outcome_model(
        base_probabilities: dict[str, float],
        features: dict[str, float],
        state: dict[str, Any],
    ) -> tuple[float, float, float]:
        model = state.get("outcome_model") or {}
        weights = model.get("weights") or {}
        labels = model.get("labels") or ["П1", "X", "П2"]
        if not weights or not features:
            return base_probabilities["П1"], base_probabilities["X"], base_probabilities["П2"]

        scores: dict[str, float] = {}
        for label in labels:
            label_weights = weights.get(label, {})
            scores[label] = sum(float(label_weights.get(key, 0.0)) * value for key, value in features.items())
        if max(abs(value) for value in scores.values()) < 1e-9:
            return base_probabilities["П1"], base_probabilities["X"], base_probabilities["П2"]

        temperature = max(0.18, float(model.get("temperature", 0.72)))
        top_score = max(scores.values())
        exp_scores = {label: math.exp((score - top_score) / temperature) for label, score in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        model_probabilities = {label: value / total for label, value in exp_scores.items()}
        blend = max(0.0, min(1.0, float(model.get("blend", 1.0))))
        mixed = {
            label: model_probabilities.get(label, 0.0) * blend + base_probabilities.get(label, 0.0) * (1.0 - blend)
            for label in ("П1", "X", "П2")
        }
        mixed_total = sum(mixed.values()) or 1.0
        return mixed["П1"] / mixed_total, mixed["X"] / mixed_total, mixed["П2"] / mixed_total

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
        h2h_report: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        home_base = 0.54 * home_stats.avg_goals_for + 0.46 * away_stats.avg_goals_against
        away_base = 0.54 * away_stats.avg_goals_for + 0.46 * home_stats.avg_goals_against

        elo_home = float(home_context.get("elo") or 1500)
        elo_away = float(away_context.get("elo") or 1500)
        elo_diff = max(-1.4, min(1.4, (elo_home - elo_away) / 400))
        form_diff = max(-1.2, min(1.2, home_stats.points_per_match - away_stats.points_per_match))
        class_diff = self._team_class_score(home_stats) - self._team_class_score(away_stats)
        motivation_diff = self._motivation(home_context, match_context) - self._motivation(away_context, match_context)
        injury_diff = self._injury_impact(home_context) - self._injury_impact(away_context)
        lineup_diff = self._lineup_strength(home_context, match_context) - self._lineup_strength(away_context, match_context)
        tactics_diff = tactical_edge(home_tactics, away_tactics)
        tempo = tactical_tempo(home_tactics, away_tactics)
        home_chance_edge = self._chance_edge(home_stats, away_stats, home_tactics, away_tactics)
        away_chance_edge = self._chance_edge(away_stats, home_stats, away_tactics, home_tactics)
        intensity = max(0.0, min(1.0, float(match_context.get("importance", 1.0))))

        home_advantage = 0.0 if neutral else weights["home_advantage_goals"]
        adjustment = (
            weights["elo_to_goals"] * elo_diff
            + weights["form_to_goals"] * form_diff
            + weights.get("class_to_goals", 0.10) * class_diff
            + weights["motivation_to_goals"] * motivation_diff
            + weights.get("lineup_to_goals", 0.24) * lineup_diff
            + weights.get("tactics_to_goals", 0.24) * tactics_diff
            - weights["injury_to_goals"] * injury_diff
            + home_advantage
        )
        intensity_boost = weights.get("world_cup_intensity_goals", 0.05) * intensity
        chance_weight = weights.get("chance_to_goals", 0.06)
        goal_scale = weights.get("goal_scale", 1.0)
        home_xg = max(0.15, (home_base + adjustment + tempo + intensity_boost + chance_weight * home_chance_edge) * goal_scale)
        away_xg = max(0.15, (away_base - adjustment * 0.78 + tempo + intensity_boost + chance_weight * away_chance_edge) * goal_scale)

        h2h_report = h2h_report or {}
        h2h_impact = float(h2h_report.get("impact") or 0.0)
        if h2h_impact > 0:
            edge = float(h2h_report.get("home_goal_edge") or 0.0)
            edge_adjustment = max(-0.28, min(0.28, edge * weights.get("h2h_to_goals", 0.16) * h2h_impact))
            home_xg += edge_adjustment
            away_xg -= edge_adjustment * 0.78
            historical_total = h2h_report.get("average_total_goals")
            if historical_total is not None:
                total_shift = max(-0.20, min(0.20, (float(historical_total) - (home_xg + away_xg)) * 0.10 * h2h_impact))
                home_share = home_xg / max(0.30, home_xg + away_xg)
                home_xg += total_shift * home_share
                away_xg += total_shift * (1.0 - home_share)
        return max(0.15, min(home_xg, 4.6)), max(0.15, min(away_xg, 4.6))

    def _calibrate_goal_rates(
        self,
        home_xg: float,
        away_xg: float,
        state: dict[str, Any],
        round_info: dict[str, Any],
    ) -> tuple[float, float]:
        total = home_xg + away_xg
        calibrated, adjustment = self._calibrate_metric(
            total,
            "goals",
            state,
            round_info,
            0.45,
            8.5,
        )
        state["_last_goal_calibration_adjustment"] = adjustment
        home_share = home_xg / max(total, 0.30)
        return (
            max(0.15, min(4.6, calibrated * home_share)),
            max(0.15, min(4.6, calibrated * (1.0 - home_share))),
        )

    @classmethod
    def _calibrate_metric(
        cls,
        value: float,
        metric: str,
        state: dict[str, Any],
        round_info: dict[str, Any],
        low: float,
        high: float,
    ) -> tuple[float, float]:
        profiles = ((state.get("calibration_profiles") or {}).get(metric) or {})
        candidates: list[tuple[float, float]] = []

        global_profile = profiles.get("global") or {}
        global_count = int(global_profile.get("matches", 0) or 0)
        global_error = global_profile.get("median_error")
        if global_error is not None and global_count >= 12:
            candidates.append((float(global_error), min(1.0, global_count / 40.0)))

        stage = "knockout" if round_info.get("knockout") else "group"
        stage_profile = (profiles.get("by_stage") or {}).get(stage) or {}
        stage_count = int(stage_profile.get("matches", 0) or 0)
        stage_error = stage_profile.get("median_error")
        if stage_error is not None and stage_count >= 6:
            candidates.append((float(stage_error), 0.75 * min(1.0, stage_count / 24.0)))

        bucket = cls._calibration_bin(metric, value)
        bin_profile = (profiles.get("by_bin") or {}).get(bucket) or {}
        bin_count = int(bin_profile.get("matches", 0) or 0)
        bin_error = bin_profile.get("median_error")
        if bin_error is not None and bin_count >= 8:
            candidates.append((float(bin_error), 0.90 * min(1.0, bin_count / 24.0)))

        if not candidates:
            return value, 0.0
        total_weight = sum(weight for _, weight in candidates) or 1.0
        correction = sum(error * weight for error, weight in candidates) / total_weight
        correction = max(-1.25, min(1.25, correction))
        return max(low, min(high, value + correction)), correction

    @staticmethod
    def _calibration_bin(metric: str, value: float) -> str:
        width = {"goals": 0.5, "corners": 1.0, "fouls": 2.0}.get(metric, 1.0)
        center = round(value / width) * width
        return f"{center:.1f}"

    def _expected_corners(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
        total_xg: float,
        weights: dict[str, float],
        state: dict[str, Any] | None = None,
        h2h_report: dict[str, Any] | None = None,
    ) -> float:
        samples = [value for value in (home_stats.avg_total_corners, away_stats.avg_total_corners) if value is not None]
        base = sum(samples) / len(samples) if samples else 9.2
        tempo = max(-0.6, min(1.1, (total_xg - 2.35) * 0.42))
        tactics = corner_tactical_boost(home_tactics, away_tactics) * weights.get("tactics_to_corners", 1.15)
        expected = base + tempo + tactics + weights.get("corner_bias", 0.0)

        corner_profile = ((state or {}).get("stat_profiles") or {}).get("corners", {})
        team_profiles = corner_profile.get("teams") or {}
        home_profile = team_profiles.get(home_stats.team, {})
        away_profile = team_profiles.get(away_stats.team, {})
        directional_values = [
            (home_profile.get("avg_for"), away_profile.get("avg_against")),
            (away_profile.get("avg_for"), home_profile.get("avg_against")),
        ]
        directional_estimates = [
            (float(first) + float(second)) / 2.0
            for first, second in directional_values
            if first is not None and second is not None
        ]
        profile_values = [
            profile.get("median_total") if profile.get("median_total") is not None else profile.get("avg_total")
            for profile in (home_profile, away_profile)
            if profile.get("avg_total") is not None
        ]
        if directional_estimates:
            profile_estimate = sum(directional_estimates)
            median_estimate = sum(float(value) for value in profile_values) / len(profile_values) if profile_values else profile_estimate
            profile_estimate = profile_estimate * 0.72 + median_estimate * 0.28
            profile_weight = min(
                0.90,
                0.52 + 0.025 * min(int(home_profile.get("matches", 0)), int(away_profile.get("matches", 0))),
            )
            expected = expected * (1.0 - profile_weight) + profile_estimate * profile_weight
        elif profile_values:
            profile_estimate = sum(float(value) for value in profile_values) / len(profile_values)
            expected = expected * 0.36 + profile_estimate * 0.64
        elif corner_profile.get("global") is not None:
            global_center = corner_profile.get("global_median") or corner_profile["global"]
            expected = expected * 0.58 + float(global_center) * 0.42

        global_center = corner_profile.get("global_median")
        if global_center is None:
            global_center = corner_profile.get("global")
        if global_center is not None:
            sample_reliability = min(1.0, (home_stats.corner_samples + away_stats.corner_samples) / 14.0)
            expected = expected * sample_reliability + float(global_center) * (1.0 - sample_reliability)

        h2h_corners = (h2h_report or {}).get("average_corners")
        h2h_impact = float((h2h_report or {}).get("impact") or 0.0)
        if h2h_corners is not None and h2h_impact > 0:
            blend = min(0.20, 0.20 * h2h_impact)
            expected = expected * (1.0 - blend) + float(h2h_corners) * blend

        return max(4.0, min(16.0, expected))

    def _foul_forecast(
        self,
        home_stats: TeamStats,
        away_stats: TeamStats,
        home_tactics: dict[str, Any],
        away_tactics: dict[str, Any],
        fixture: dict[str, Any] | None,
        weights: dict[str, float],
        state: dict[str, Any] | None = None,
        h2h_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        global_average = self._global_foul_average(state)
        weighted_team_values = []
        for stats in (home_stats, away_stats):
            if stats.avg_total_fouls is None:
                continue
            weight = max(1.0, min(6.0, stats.foul_samples))
            weighted_team_values.append((stats.avg_total_fouls, weight))
        if weighted_team_values:
            weighted_sum = sum(value * weight for value, weight in weighted_team_values)
            weight_sum = sum(weight for _, weight in weighted_team_values)
            team_average = weighted_sum / weight_sum
        else:
            team_average = global_average

        sample_confidence = min(1.0, (home_stats.foul_samples + away_stats.foul_samples) / 12.0)
        team_estimate = team_average * sample_confidence + global_average * (1.0 - sample_confidence)
        pressing = (float(home_tactics.get("pressing", 0.55)) + float(away_tactics.get("pressing", 0.55))) / 2.0
        directness = (float(home_tactics.get("directness", 0.50)) + float(away_tactics.get("directness", 0.50))) / 2.0
        tempo = (float(home_tactics.get("tempo", 0.55)) + float(away_tactics.get("tempo", 0.55))) / 2.0
        defensive_solidity = (
            float(home_tactics.get("defensive_solidity", 0.55))
            + float(away_tactics.get("defensive_solidity", 0.55))
        ) / 2.0
        style_adjustment = (
            (pressing - 0.55) * 6.2
            + (directness - 0.50) * 3.4
            + (tempo - 0.55) * 3.0
            - (defensive_solidity - 0.55) * 2.2
        )
        expected = team_estimate + style_adjustment + float(weights.get("foul_bias", 0.0))

        referee = self._fixture_referee(fixture)
        stat_foul_profile = ((state or {}).get("stat_profiles") or {}).get("fouls", {})
        trained_referees = stat_foul_profile.get("referees") or {}
        trained_teams = stat_foul_profile.get("teams") or {}
        referee_profile = trained_referees.get(referee.get("name") if referee else None)
        if not referee_profile and state is None:
            referee_profile = self.store.referee_profile(referee.get("name") if referee else None)
        referee_profile = referee_profile or {}
        referee_average = referee_profile.get("avg_fouls")
        referee_matches = int(referee_profile.get("matches", 0) or 0)
        if referee_average is not None:
            referee_reliability = referee_matches / (referee_matches + 5.0)
            shrunk_referee_average = (
                float(referee_average) * referee_reliability
                + global_average * (1.0 - referee_reliability)
            )
            referee_weight = min(0.78, 0.28 + referee_matches * 0.07)
            expected = expected * (1.0 - referee_weight) + shrunk_referee_average * referee_weight
        else:
            home_profile = trained_teams.get(home_stats.team, {})
            away_profile = trained_teams.get(away_stats.team, {})
            directional_values = [
                (home_profile.get("avg_for"), away_profile.get("avg_against")),
                (away_profile.get("avg_for"), home_profile.get("avg_against")),
            ]
            directional_estimates = [
                (float(first) + float(second)) / 2.0
                for first, second in directional_values
                if first is not None and second is not None
            ]
            profile_values = [
                profile.get("median_total") if profile.get("median_total") is not None else profile.get("avg_total")
                for profile in (home_profile, away_profile)
                if profile.get("avg_total") is not None
            ]
            if directional_estimates:
                profile_estimate = sum(directional_estimates)
                median_estimate = sum(float(value) for value in profile_values) / len(profile_values) if profile_values else profile_estimate
                profile_estimate = profile_estimate * 0.68 + median_estimate * 0.32
                profile_matches = min(
                    int(home_profile.get("matches", 0)),
                    int(away_profile.get("matches", 0)),
                )
                profile_reliability = profile_matches / (profile_matches + 6.0)
                profile_estimate = (
                    profile_estimate * profile_reliability
                    + global_average * (1.0 - profile_reliability)
                )
                expected = expected * 0.26 + profile_estimate * 0.74
            elif profile_values:
                profile_estimate = sum(float(value) for value in profile_values) / len(profile_values)
                profile_matches = min(
                    int(profile.get("matches", 0))
                    for profile in (home_profile, away_profile)
                    if profile.get("avg_total") is not None
                )
                profile_reliability = profile_matches / (profile_matches + 6.0)
                profile_estimate = (
                    profile_estimate * profile_reliability
                    + global_average * (1.0 - profile_reliability)
                )
                expected = expected * 0.28 + profile_estimate * 0.72
            elif stat_foul_profile.get("global") is not None:
                global_center = stat_foul_profile.get("global_median") or stat_foul_profile["global"]
                expected = expected * 0.55 + float(global_center) * 0.45

        h2h_fouls = (h2h_report or {}).get("average_fouls")
        h2h_impact = float((h2h_report or {}).get("impact") or 0.0)
        if h2h_fouls is not None and h2h_impact > 0:
            blend = min(0.16, 0.16 * h2h_impact)
            expected = expected * (1.0 - blend) + float(h2h_fouls) * blend

        expected = max(14.0, min(40.0, expected))
        probabilities = {}
        for line in (20.5, 24.5, 28.5):
            key = str(line).replace(".", "_")
            over = self._foul_probability_over(expected, line)
            probabilities[f"over_{key}"] = round(over, 4)
            probabilities[f"under_{key}"] = round(1.0 - over, 4)

        return {
            "expected": round(expected, 2),
            "point_estimate": int(math.floor(expected + 0.5)),
            "label": self._foul_label(expected),
            "team_average": round(team_average, 2),
            "global_average": round(global_average, 2),
            "home_average": None if home_stats.avg_total_fouls is None else round(home_stats.avg_total_fouls, 2),
            "away_average": None if away_stats.avg_total_fouls is None else round(away_stats.avg_total_fouls, 2),
            "home_samples": home_stats.foul_samples,
            "away_samples": away_stats.foul_samples,
            "style_adjustment": round(style_adjustment, 2),
            "sample_confidence": round(sample_confidence, 3),
            "referee": {
                "name": referee.get("name") if referee else None,
                "avg_fouls": None if referee_average is None else round(float(referee_average), 2),
                "matches": referee_matches,
                "source": (referee or {}).get("source") or referee_profile.get("source"),
            },
            "probabilities": probabilities,
        }

    def _global_foul_average(self, state: dict[str, Any] | None = None) -> float:
        foul_profile = (((state or {}).get("stat_profiles") or {}).get("fouls") or {})
        trained_global = foul_profile.get("global_median")
        if trained_global is None:
            trained_global = foul_profile.get("global")
        if trained_global is not None:
            return float(trained_global)
        totals = [
            float(match.home_fouls) + float(match.away_fouls)
            for match in self.store.load_matches()
            if match.home_fouls is not None and match.away_fouls is not None
        ]
        return sum(totals) / len(totals) if totals else 24.0

    @staticmethod
    def _fixture_referee(fixture: dict[str, Any] | None) -> dict[str, Any] | None:
        referee = (fixture or {}).get("referee")
        if isinstance(referee, dict):
            return referee
        if isinstance(referee, str):
            return {"name": referee, "source": "fixture"}
        return None

    @staticmethod
    def _foul_probability_over(expected: float, line: float) -> float:
        probability = 1.0 / (1.0 + math.exp(-(expected - line) / 3.6))
        return max(0.03, min(0.97, probability))

    @staticmethod
    def _foul_label(expected: float) -> str:
        if expected >= 30.0:
            return "жесткий матч"
        if expected >= 26.0:
            return "фолов выше среднего"
        if expected <= 20.0:
            return "аккуратный матч"
        return "обычный уровень фолов"

    def _recommended_bets(
        self,
        home_team: str,
        away_team: str,
        market_pick: str,
        outcome_probabilities: dict[str, float],
        goal_total: dict[str, Any],
        predicted_corners: float,
        foul_forecast: dict[str, Any],
    ) -> dict[str, Any]:
        winner = {
            "type": "winner",
            "label": "Победитель",
            "pick": self._market_label(market_pick, home_team, away_team),
            "code": market_pick,
            "probability": round(float(outcome_probabilities.get(market_pick, 0.0)), 4),
        }
        goals = self._goal_total_bet(goal_total)
        corners = self._corner_total_bet(predicted_corners)
        fouls = self._foul_total_bet(foul_forecast)
        items = [winner, goals, corners, fouls]
        return {
            "summary": " + ".join(item["pick"] for item in items if item.get("pick")),
            "items": items,
        }

    @staticmethod
    def _goal_total_bet(goal_total: dict[str, Any]) -> dict[str, Any]:
        probabilities = goal_total.get("probabilities", {}) if goal_total else {}
        candidates = [
            ("ТБ 2.5 голов", 2.5, "over", probabilities.get("over_2_5")),
            ("ТМ 2.5 голов", 2.5, "under", probabilities.get("under_2_5")),
            ("ТБ 3.5 голов", 3.5, "over", probabilities.get("over_3_5")),
            ("ТМ 3.5 голов", 3.5, "under", probabilities.get("under_3_5")),
        ]
        pick, line, side, probability = max(
            candidates,
            key=lambda item: float(item[3] if item[3] is not None else 0.0),
        )
        return {
            "type": "goals_total",
            "label": "Тотал голов",
            "pick": pick,
            "line": line,
            "side": side,
            "probability": round(float(probability or 0.0), 4),
            "expected": goal_total.get("expected") if goal_total else None,
        }

    @staticmethod
    def _corner_total_bet(expected_corners: float) -> dict[str, Any]:
        line = max(5.5, min(13.5, math.floor(float(expected_corners)) + 0.5))
        over = MatchPredictor._total_probability_over(float(expected_corners), line, spread=2.05)
        if over >= 0.5:
            pick = f"ТБ {line:.1f} угловых"
            side = "over"
            probability = over
        else:
            pick = f"ТМ {line:.1f} угловых"
            side = "under"
            probability = 1.0 - over
        return {
            "type": "corners_total",
            "label": "Тотал угловых",
            "pick": pick,
            "line": line,
            "side": side,
            "probability": round(probability, 4),
            "expected": round(float(expected_corners), 2),
        }

    @staticmethod
    def _foul_total_bet(foul_forecast: dict[str, Any]) -> dict[str, Any]:
        expected = float((foul_forecast or {}).get("expected") or 24.0)
        probabilities = (foul_forecast or {}).get("probabilities", {})
        line = min((20.5, 24.5, 28.5), key=lambda value: abs(expected - value))
        key = str(line).replace(".", "_")
        over = probabilities.get(f"over_{key}")
        if over is None:
            over = MatchPredictor._total_probability_over(expected, line, spread=3.6)
        over = float(over)
        if over >= 0.5:
            pick = f"ТБ {line:.1f} фолов"
            side = "over"
            probability = over
        else:
            pick = f"ТМ {line:.1f} фолов"
            side = "under"
            probability = 1.0 - over
        return {
            "type": "fouls_total",
            "label": "Тотал фолов",
            "pick": pick,
            "line": line,
            "side": side,
            "probability": round(probability, 4),
            "expected": round(expected, 2),
        }

    @staticmethod
    def _total_probability_over(expected: float, line: float, spread: float) -> float:
        probability = 1.0 / (1.0 + math.exp(-(expected - line) / spread))
        return max(0.03, min(0.97, probability))

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
        goal_total: dict[str, Any] | None = None,
        outcome_probabilities: dict[str, float] | None = None,
        limit: int = 3,
        state: dict[str, Any] | None = None,
        h2h_report: dict[str, Any] | None = None,
        round_info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        score_counts_by_outcome = {
            outcome: self._historical_score_counts(outcome, state) for outcome in ("П1", "X", "П2")
        }
        candidate_counts = {outcome: len(self._candidate_scores(outcome)) for outcome in ("П1", "X", "П2")}
        goal_total = goal_total or self._goal_total_forecast(home_xg, away_xg)
        if outcome_probabilities is None:
            home_win, draw, away_win = self._outcome_probabilities(home_xg, away_xg)
            outcome_probabilities = {"П1": home_win, "X": draw, "П2": away_win}
        profile_candidate = self._profile_score_candidate(
            market_pick,
            home_xg,
            away_xg,
            state,
            round_info,
        )
        profile_score = profile_candidate.get("score") if profile_candidate else None
        grid = []
        for home_goals in range(9):
            for away_goals in range(9):
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
                    * self._score_total_factor(home_goals, away_goals, home_xg, away_xg, goal_total)
                    * self._mismatch_score_factor(home_goals, away_goals, home_xg, away_xg, market_pick)
                    * self._outcome_score_factor(outcome, market_pick, outcome_probabilities)
                    * (1.32 if score == profile_score else 1.0)
                    * self._h2h_score_factor(score, h2h_report)
                )
                grid.append((weight, outcome, score))

        total_weight = sum(weight for weight, _, _ in grid)
        if total_weight <= 0:
            fallback = ["1-1", "0-0", "2-2"] if market_pick == "X" else ["1-0", "2-1", "2-0"]
            return [{"score": score, "probability": 0.0} for score in fallback[:limit]]
        selected = self._select_score_mix(grid, market_pick, home_xg, away_xg, goal_total, outcome_probabilities, limit)
        return [
            {
                "score": score,
                "outcome": outcome,
                "probability": self._calibrated_exact_score_probability(
                    weight / total_weight,
                    outcome,
                    outcome_probabilities,
                ),
            }
            for weight, outcome, score in selected
        ]

    @staticmethod
    def _calibrated_exact_score_probability(
        score_probability: float,
        outcome: str,
        outcome_probabilities: dict[str, float],
        profile_probability: bool = False,
    ) -> float:
        outcome_probability = max(0.0, min(1.0, float(outcome_probabilities.get(outcome, 0.0))))
        probability = max(0.0, min(1.0, float(score_probability or 0.0)))
        if profile_probability:
            probability *= outcome_probability
        else:
            probability = min(probability, outcome_probability)
        return round(probability, 4)

    def _profile_score_candidate(
        self,
        market_pick: str,
        home_xg: float,
        away_xg: float,
        state: dict[str, Any] | None = None,
        round_info: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        profiles = ((state or self.store.load_model_state()).get("score_profiles") or {})
        bucket = self._score_profile_bucket(market_pick, home_xg, away_xg)
        stage = "knockout" if (round_info or {}).get("knockout") else "group"
        by_stage_bucket = profiles.get("by_outcome_stage_bucket") or {}
        by_stage = profiles.get("by_outcome_stage") or {}
        by_bucket = profiles.get("by_outcome_bucket") or {}
        by_outcome = profiles.get("by_outcome") or {}
        stage_bucket_candidates = by_stage_bucket.get(f"{market_pick}|{stage}|{bucket}") or []
        stage_candidates = by_stage.get(f"{market_pick}|{stage}") or []
        candidates = (
            (stage_bucket_candidates if self._profile_sample(stage_bucket_candidates) >= 6 else [])
            or (stage_candidates if self._profile_sample(stage_candidates) >= 8 else [])
            or by_bucket.get(f"{market_pick}|{bucket}")
            or by_outcome.get(market_pick)
            or []
        )
        if not candidates:
            return None
        candidate = self._select_profile_score(candidates, bucket, home_xg, away_xg)
        score = str(candidate.get("score", ""))
        if "-" not in score:
            return None
        home_goals, away_goals = [int(value) for value in score.split("-", 1)]
        return {
            "score": score,
            "outcome": self._outcome_from_score(home_goals, away_goals),
            "probability": round(float(candidate.get("probability", 0.0)), 4),
        }

    @staticmethod
    def _profile_sample(candidates: list[dict[str, Any]]) -> int:
        return sum(int(item.get("count", 0) or 0) for item in candidates)

    @classmethod
    def _select_profile_score(
        cls,
        candidates: list[dict[str, Any]],
        bucket: str,
        home_xg: float,
        away_xg: float,
    ) -> dict[str, Any]:
        favorite_xg = max(home_xg, away_xg)
        total_xg = home_xg + away_xg
        top = candidates[0]
        top_probability = float(top.get("probability", 0.0) or 0.0)
        if bucket == "dominant" and favorite_xg >= 2.35 and total_xg >= 3.0:
            for candidate in candidates:
                score = str(candidate.get("score", ""))
                if "-" not in score:
                    continue
                total_goals = cls._score_total(score)
                probability = float(candidate.get("probability", 0.0) or 0.0)
                if total_goals >= 3 and probability >= top_probability * 0.82:
                    return candidate
        if bucket == "open" and total_xg >= 3.25:
            underdog_xg = min(home_xg, away_xg)
            if underdog_xg >= 1.0:
                for candidate in candidates:
                    score = str(candidate.get("score", ""))
                    probability = float(candidate.get("probability", 0.0) or 0.0)
                    if "-" in score and cls._score_total(score) >= 4 and cls._both_score(score) and probability >= top_probability * 0.68:
                        return candidate
            for candidate in candidates:
                score = str(candidate.get("score", ""))
                probability = float(candidate.get("probability", 0.0) or 0.0)
                if "-" in score and cls._score_total(score) >= 4 and probability >= top_probability * 0.75:
                    return candidate
        return top

    @staticmethod
    def _score_profile_bucket(market_pick: str, home_xg: float, away_xg: float) -> str:
        total_xg = home_xg + away_xg
        favorite_xg = max(home_xg, away_xg)
        underdog_xg = min(home_xg, away_xg)
        margin = abs(home_xg - away_xg)
        if market_pick == "X":
            return "draw_high" if total_xg >= 2.70 else "draw_low"
        if favorite_xg >= 2.15 and underdog_xg <= 1.05 and margin >= 1.00:
            return "dominant"
        if total_xg >= 3.20:
            return "open"
        if margin <= 0.45:
            return "narrow"
        return "normal"

    def _goal_total_forecast(self, home_xg: float, away_xg: float) -> dict[str, Any]:
        total_xg = home_xg + away_xg
        total_probs = self._total_goal_probabilities(total_xg, max_goals=9)
        over_1_5 = sum(prob for goals, prob in total_probs.items() if goals >= 2)
        over_2_5 = sum(prob for goals, prob in total_probs.items() if goals >= 3)
        over_3_5 = sum(prob for goals, prob in total_probs.items() if goals >= 4)
        over_4_5 = sum(prob for goals, prob in total_probs.items() if goals >= 5)
        under_2_5 = 1.0 - over_2_5
        under_3_5 = 1.0 - over_3_5
        under_4_5 = 1.0 - over_4_5
        buckets = {
            "0-1": sum(prob for goals, prob in total_probs.items() if goals <= 1),
            "2-3": sum(prob for goals, prob in total_probs.items() if 2 <= goals <= 3),
            "4+": over_3_5,
            "5+": over_4_5,
        }
        top_totals = sorted(total_probs.items(), key=lambda item: item[1], reverse=True)[:3]
        if over_3_5 >= 0.44:
            label = "верховой матч"
        elif over_2_5 >= 0.58:
            label = "скорее 3+ гола"
        elif buckets["0-1"] >= 0.34:
            label = "низовой матч"
        else:
            label = "умеренный тотал"
        return {
            "expected": round(total_xg, 2),
            "point_estimate": int(math.floor(total_xg + 0.5)),
            "label": label,
            "most_likely_totals": [
                {"goals": goals, "probability": round(probability, 4)}
                for goals, probability in top_totals
            ],
            "probabilities": {
                "under_1_5": round(buckets["0-1"], 4),
                "over_1_5": round(over_1_5, 4),
                "over_2_5": round(over_2_5, 4),
                "under_2_5": round(under_2_5, 4),
                "over_3_5": round(over_3_5, 4),
                "under_3_5": round(under_3_5, 4),
                "over_4_5": round(over_4_5, 4),
                "under_4_5": round(under_4_5, 4),
            },
            "buckets": {key: round(value, 4) for key, value in buckets.items()},
        }

    def _total_goal_probabilities(self, total_xg: float, max_goals: int = 9) -> dict[int, float]:
        probabilities = {goals: self._poisson(goals, total_xg) for goals in range(max_goals)}
        probabilities[max_goals] = max(0.0, 1.0 - sum(probabilities.values()))
        return probabilities

    def _score_total_factor(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
        goal_total: dict[str, Any],
    ) -> float:
        total_goals = home_goals + away_goals
        likely_totals = {
            int(item["goals"]): float(item["probability"])
            for item in goal_total.get("most_likely_totals", [])
            if item.get("goals") is not None
        }
        if likely_totals:
            best_probability = max(likely_totals.values()) or 1.0
            total_probability = likely_totals.get(total_goals, self._poisson(total_goals, home_xg + away_xg))
            factor = 0.76 + 0.54 * min(1.25, total_probability / best_probability)
        else:
            factor = 1.0

        probabilities = goal_total.get("probabilities", {})
        over_2_5 = float(probabilities.get("over_2_5", 0.0))
        under_2_5 = float(probabilities.get("under_2_5", 0.0))
        over_3_5 = float(probabilities.get("over_3_5", 0.0))
        under_3_5 = float(probabilities.get("under_3_5", 0.0))

        if over_2_5 >= 0.52:
            if total_goals <= 1:
                factor *= 0.52
            elif total_goals == 2:
                factor *= 0.82
            else:
                factor *= 1.12
        elif under_2_5 >= 0.56:
            if total_goals <= 2:
                factor *= 1.10
            elif total_goals >= 4:
                factor *= 0.68

        if over_3_5 >= 0.31:
            if total_goals >= 4:
                factor *= 1.16
            elif total_goals <= 1:
                factor *= 0.70
        elif under_3_5 >= 0.64 and total_goals >= 5:
            factor *= 0.62

        return max(0.30, min(1.85, factor))

    @staticmethod
    def _outcome_score_factor(outcome: str, market_pick: str, probabilities: dict[str, float]) -> float:
        if outcome == market_pick:
            return 1.08
        favorite_probability = float(probabilities.get(market_pick, 0.0))
        outcome_probability = float(probabilities.get(outcome, 0.0))
        gap = favorite_probability - outcome_probability
        if outcome == "X" and (outcome_probability >= 0.27 or gap <= 0.12):
            return 0.92
        if gap <= 0.04:
            return 0.72
        if outcome_probability >= 0.25:
            return 0.34
        return 0.24

    @staticmethod
    def _mismatch_score_factor(
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float,
        market_pick: str,
    ) -> float:
        favorite_xg = max(home_xg, away_xg)
        underdog_xg = min(home_xg, away_xg)
        margin = abs(home_xg - away_xg)
        if favorite_xg < 2.05 or margin < 1.25:
            return 1.0

        favorite_goals = home_goals if home_xg >= away_xg else away_goals
        underdog_goals = away_goals if home_xg >= away_xg else home_goals
        expected_pick = "П1" if home_xg >= away_xg else "П2"
        if market_pick != expected_pick:
            return 1.0

        factor = 1.0
        if underdog_xg <= 0.85 and underdog_goals == 0:
            if favorite_goals == 1:
                factor *= 0.66
            elif favorite_goals == 2:
                factor *= 1.02
            elif favorite_goals >= 3:
                factor *= 1.30
        if favorite_xg >= 2.35 and favorite_goals >= 3:
            factor *= 1.12
        return max(0.55, min(1.70, factor))

    def _select_score_mix(
        self,
        candidates: list[tuple[float, str, str]],
        market_pick: str,
        home_xg: float,
        away_xg: float,
        goal_total: dict[str, Any],
        outcome_probabilities: dict[str, float],
        limit: int,
    ) -> list[tuple[float, str, str]]:
        ordered = sorted(candidates, reverse=True)
        if limit <= 1:
            market_candidates = [item for item in ordered if item[1] == market_pick]
            aligned = [
                item
                for item in market_candidates
                if self._score_total_compatible(item[2], goal_total)
            ]
            return (aligned or market_candidates or ordered)[:1]
        selected = ordered[:limit]
        probabilities = goal_total.get("probabilities", {})
        over_2_5 = float(probabilities.get("over_2_5", 0.0))
        over_3_5 = float(probabilities.get("over_3_5", 0.0))
        over_4_5 = float(probabilities.get("over_4_5", 0.0))
        btts = (1.0 - math.exp(-home_xg)) * (1.0 - math.exp(-away_xg))

        if over_2_5 >= 0.52 and selected and self._score_total(selected[0][2]) <= 1:
            richer = next(
                (
                    item
                    for item in ordered
                    if self._score_total(item[2]) >= 3 and item[0] >= selected[0][0] * 0.54
                ),
                None,
            )
            selected = self._promote_score_candidate(selected, richer)

        if not any(outcome == market_pick for _, outcome, _ in selected):
            market_candidate = next((item for item in ordered if item[1] == market_pick), None)
            selected = self._include_score_candidate(selected, market_candidate)

        draw_probability = float(outcome_probabilities.get("X", 0.0))
        market_probability = float(outcome_probabilities.get(market_pick, 0.0))
        if (
            market_pick != "X"
            and (draw_probability >= 0.29 or market_probability - draw_probability <= 0.10)
            and not any(outcome == "X" for _, outcome, _ in selected)
        ):
            draw_candidate = next((item for item in ordered if item[1] == "X"), None)
            selected = self._include_score_candidate(selected, draw_candidate)

        selected = self._remove_weak_opposite_scores(selected, ordered, market_pick, outcome_probabilities, limit)

        if over_3_5 >= 0.38 and not any(self._score_total(score) >= 4 for _, _, score in selected):
            pool = ordered
            if btts >= 0.56:
                pool = [item for item in ordered if self._both_score(item[2]) and self._score_total(item[2]) >= 4] or ordered
            high = next((item for item in pool if self._score_total(item[2]) >= 4), None)
            selected = self._include_score_candidate(selected, high)

        if over_4_5 >= 0.26 and btts >= 0.60 and not any(self._score_total(score) >= 5 for _, _, score in selected):
            high5_pool = [
                item
                for item in ordered
                if self._both_score(item[2]) and self._score_total(item[2]) >= 5
            ]
            high5 = self._competitive_high_total_candidate(high5_pool, min(home_xg, away_xg))
            selected = self._include_score_candidate(selected, high5)

        favorite_xg = max(home_xg, away_xg)
        underdog_xg = min(home_xg, away_xg)
        if favorite_xg >= 2.25 and underdog_xg <= 0.85:
            clean_big = next(
                (
                    item
                    for item in ordered
                    if self._is_big_clean_favorite_score(item[2], home_xg >= away_xg, market_pick)
                ),
                None,
            )
            selected = self._include_score_candidate(selected, clean_big)

        if not any(outcome == market_pick for _, outcome, _ in selected):
            market_candidate = next((item for item in ordered if item[1] == market_pick), None)
            selected = self._include_score_candidate(selected, market_candidate)

        return sorted(selected, reverse=True)[:limit]

    @classmethod
    def _score_total_compatible(cls, score: str, goal_total: dict[str, Any]) -> bool:
        total = cls._score_total(score)
        probabilities = goal_total.get("probabilities", {})
        over_2_5 = float(probabilities.get("over_2_5", 0.0))
        under_2_5 = float(probabilities.get("under_2_5", 0.0))
        over_3_5 = float(probabilities.get("over_3_5", 0.0))
        if over_3_5 >= 0.52:
            return total >= 4
        if over_2_5 >= 0.58:
            return total >= 3
        if under_2_5 >= 0.62:
            return total <= 2
        return True

    @classmethod
    def _score_total_alignment(cls, score: str, goal_total: dict[str, Any]) -> dict[str, Any]:
        total = cls._score_total(score)
        probabilities = goal_total.get("probabilities", {})
        over_2_5 = float(probabilities.get("over_2_5", 0.0))
        under_2_5 = float(probabilities.get("under_2_5", 0.0))
        if over_2_5 >= 0.58:
            side = "over_2_5"
            consistent = total >= 3
        elif under_2_5 >= 0.62:
            side = "under_2_5"
            consistent = total <= 2
        else:
            side = "balanced"
            consistent = True
        return {
            "consistent": consistent,
            "score_total": total,
            "dominant_total_side": side,
        }

    @staticmethod
    def _h2h_score_factor(score: str, h2h_report: dict[str, Any] | None) -> float:
        report = h2h_report or {}
        impact = float(report.get("impact") or 0.0)
        if impact <= 0:
            return 1.0
        score_weight = sum(
            float(item.get("weight") or 0.0)
            for item in report.get("history", [])
            if item.get("score") == score
        )
        if score_weight <= 0:
            return 1.0
        effective = max(1.0, float(report.get("effective_matches") or 1.0))
        return min(1.35, 1.0 + 0.35 * impact * score_weight / effective)

    @staticmethod
    def _remove_weak_opposite_scores(
        selected: list[tuple[float, str, str]],
        ordered: list[tuple[float, str, str]],
        market_pick: str,
        outcome_probabilities: dict[str, float],
        limit: int,
    ) -> list[tuple[float, str, str]]:
        if market_pick == "X":
            return selected
        market_probability = float(outcome_probabilities.get(market_pick, 0.0))

        def allowed(item: tuple[float, str, str]) -> bool:
            outcome = item[1]
            if outcome in {market_pick, "X"}:
                return True
            return market_probability - float(outcome_probabilities.get(outcome, 0.0)) <= 0.04

        next_selected = [item for item in selected if allowed(item)]
        for item in ordered:
            if len(next_selected) >= limit:
                break
            if not allowed(item):
                continue
            if any(score == item[2] for _, _, score in next_selected):
                continue
            next_selected.append(item)
        return next_selected or selected

    @staticmethod
    def _include_score_candidate(
        selected: list[tuple[float, str, str]],
        candidate: tuple[float, str, str] | None,
    ) -> list[tuple[float, str, str]]:
        if candidate is None or any(score == candidate[2] for _, _, score in selected):
            return selected
        next_selected = selected[:]
        next_selected[-1] = candidate
        return next_selected

    @staticmethod
    def _promote_score_candidate(
        selected: list[tuple[float, str, str]],
        candidate: tuple[float, str, str] | None,
    ) -> list[tuple[float, str, str]]:
        if candidate is None or any(score == candidate[2] for _, _, score in selected):
            return selected
        next_selected = selected[:]
        next_selected[0] = candidate
        return next_selected

    @staticmethod
    def _competitive_high_total_candidate(
        candidates: list[tuple[float, str, str]],
        underdog_xg: float,
    ) -> tuple[float, str, str] | None:
        if not candidates:
            return None
        best = candidates[0]
        if underdog_xg < 0.95:
            return best
        for candidate in candidates:
            home, away = [int(value) for value in candidate[2].split("-")]
            if min(home, away) >= 2 and candidate[0] >= best[0] * 0.70:
                return candidate
        return best

    @staticmethod
    def _score_total(score: str) -> int:
        home, away = score.split("-")
        return int(home) + int(away)

    @staticmethod
    def _both_score(score: str) -> bool:
        home, away = score.split("-")
        return int(home) > 0 and int(away) > 0

    @staticmethod
    def _is_big_clean_favorite_score(score: str, home_favorite: bool, market_pick: str) -> bool:
        home, away = [int(value) for value in score.split("-")]
        if home_favorite and market_pick == "П1":
            return home >= 3 and away == 0
        if not home_favorite and market_pick == "П2":
            return away >= 3 and home == 0
        return False

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

    def _team_report(
        self,
        team: str,
        stats: TeamStats,
        tactics: dict[str, Any],
        context: dict[str, Any],
        expected_goals: float,
        lineup_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attack_score = self._rating01(
            0.28 * min(stats.avg_goals_for / 3.0, 1.0)
            + 0.18 * (0.5 if stats.avg_shots_for is None else min(stats.avg_shots_for / 18.0, 1.0))
            + 0.18 * (0.5 if stats.avg_shots_on_target_for is None else min(stats.avg_shots_on_target_for / 7.0, 1.0))
            + 0.24 * float(tactics.get("chance_creation", 0.55))
            + 0.12 * float(tactics.get("set_piece_threat", 0.50))
        )
        defense_score = self._rating01(
            0.30 * max(0.0, 1.0 - stats.avg_goals_against / 3.0)
            + 0.18 * (0.5 if stats.avg_shots_against is None else max(0.0, 1.0 - stats.avg_shots_against / 20.0))
            + 0.26 * float(tactics.get("defensive_solidity", 0.55))
            + 0.14 * float(tactics.get("transition_defense", 0.55))
            + 0.12 * (stats.clean_sheets / stats.sample_size if stats.sample_size else 0.25)
        )
        form_score = self._rating01(
            0.56 * min(stats.points_per_match / 3.0, 1.0)
            + 0.24 * min(max((stats.avg_goals_for - stats.avg_goals_against + 1.5) / 3.0, 0.0), 1.0)
            + 0.20 * (1.0 - (stats.failed_to_score / stats.sample_size if stats.sample_size else 0.25))
        )
        class_score = self._rating01(0.50 + self._team_class_score(stats) / 3.2)
        strengths = self._team_strengths(stats, tactics, attack_score, defense_score, form_score)
        risks = self._team_risks(stats, tactics, context)
        lineup_report = lineup_report or {}
        return {
            "team": team,
            "level": self._level_label(class_score),
            "class_score": round(class_score, 3),
            "overall_score": round(self._rating01(class_score * 0.38 + attack_score * 0.24 + defense_score * 0.18 + form_score * 0.20), 3),
            "attack_score": round(attack_score, 3),
            "defense_score": round(defense_score, 3),
            "form_score": round(form_score, 3),
            "expected_goals": round(expected_goals, 2),
            "formation": tactics.get("formation"),
            "formation_confidence": tactics.get("formation_confidence"),
            "lineup_status": lineup_report.get("status", "not_released"),
            "lineup_strength": lineup_report.get("availability_score"),
            "missing_key_players": lineup_report.get("missing_key_players", []),
            "benched_key_players": lineup_report.get("benched_key_players", []),
            "starting_key_players": lineup_report.get("starting_key_players", []),
            "strengths": strengths,
            "risks": risks,
            "data_note": f"{stats.sample_size} матчей, rich {min(stats.corner_samples, stats.possession_samples, stats.shot_samples, stats.foul_samples)}",
        }

    @staticmethod
    def _rating01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _level_label(value: float) -> str:
        if value >= 0.74:
            return "топ-уровень"
        if value >= 0.62:
            return "сильная сборная"
        if value >= 0.48:
            return "средний уровень"
        return "зона риска"

    @staticmethod
    def _team_strengths(stats: TeamStats, tactics: dict[str, Any], attack_score: float, defense_score: float, form_score: float) -> list[str]:
        strengths: list[str] = []
        if attack_score >= 0.66:
            strengths.append("создание моментов")
        if defense_score >= 0.66:
            strengths.append("защита")
        if form_score >= 0.68:
            strengths.append("форма")
        if float(tactics.get("set_piece_threat", 0.50)) >= 0.62:
            strengths.append("стандарты")
        if float(tactics.get("tempo", 0.55)) >= 0.66:
            strengths.append("темп")
        if stats.clean_sheets >= max(2, stats.sample_size // 3):
            strengths.append("сухие матчи")
        return strengths[:4] or ["нет ярко выраженного преимущества"]

    @staticmethod
    def _team_risks(stats: TeamStats, tactics: dict[str, Any], context: dict[str, Any]) -> list[str]:
        risks: list[str] = []
        if stats.sample_size < 10:
            risks.append("мало матчей")
        if stats.failed_to_score >= max(2, stats.sample_size // 4):
            risks.append("риск без гола")
        if stats.avg_goals_against >= 1.45:
            risks.append("пропускает")
        if float(tactics.get("defensive_solidity", 0.55)) <= 0.38:
            risks.append("нестабильная оборона")
        injuries = [item for item in context.get("injuries", []) if str(item.get("status", "")).lower() not in {"fit", "available", "ok"}]
        if injuries:
            risks.append("травмы/риски состава")
        return risks[:4] or ["явных рисков нет"]

    def _result_summary(
        self,
        market_pick: str,
        exact_score_probabilities: list[dict[str, Any]],
        predicted_corners: float,
        goal_total: dict[str, Any],
        foul_forecast: dict[str, Any],
        fixture: dict[str, Any] | None,
        home_team: str,
        away_team: str,
    ) -> dict[str, Any]:
        predicted = {
            "outcome": market_pick,
            "outcome_label": self._market_label(market_pick, home_team, away_team),
            "scores": exact_score_probabilities,
            "corners": round(predicted_corners, 2),
            "corner_point_estimate": int(math.floor(predicted_corners + 0.5)),
            "goal_total": goal_total,
            "fouls": foul_forecast,
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
            actual_fouls = self._fixture_total_fouls(fixture)
            corner_point = int(math.floor(predicted_corners + 0.5))
            foul_expected = float(foul_forecast.get("expected", 0.0))
            foul_point = int(foul_forecast.get("point_estimate", math.floor(foul_expected + 0.5)))
            return {
                "status": "completed",
                "predicted": predicted,
                "actual": {
                    "score": actual_score,
                    "outcome": actual_outcome,
                    "outcome_label": self._market_label(actual_outcome, home_team, away_team),
                    "corners": actual_corners,
                    "fouls": actual_fouls,
                },
                "outcome_hit": market_pick == actual_outcome,
                "score_hit": actual_score in [item["score"] for item in exact_score_probabilities],
                "corner_error": None if actual_corners is None else round(corner_point - actual_corners, 2),
                "expected_corner_error": None if actual_corners is None else round(predicted_corners - actual_corners, 2),
                "foul_error": None if actual_fouls is None else round(foul_point - actual_fouls, 2),
                "expected_foul_error": None if actual_fouls is None else round(foul_expected - actual_fouls, 2),
                "message": "Матч завершен, факт уже доступен.",
            }

        if fixture.get("in_progress"):
            current_score = f"{int(home_goals)}-{int(away_goals)}" if has_score else None
            return {
                "status": "live",
                "predicted": predicted,
                "actual": {"score": current_score, "outcome": None, "corners": None, "fouls": None},
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
    def _fixture_total_fouls(fixture: dict[str, Any]) -> float | None:
        home = fixture.get("home_fouls")
        away = fixture.get("away_fouls")
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
        for home_goals in range(9):
            for away_goals in range(9):
                if market_pick == "П1" and home_goals > away_goals:
                    scores.append((home_goals, away_goals))
                elif market_pick == "П2" and away_goals > home_goals:
                    scores.append((home_goals, away_goals))
                elif market_pick == "X" and home_goals == away_goals:
                    scores.append((home_goals, away_goals))
        return scores

    def _historical_score_counts(self, market_pick: str, state: dict[str, Any] | None = None) -> Counter[str]:
        history = (state or self.store.load_model_state()).get("history", [])
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
    def _team_class_score(stats: TeamStats) -> float:
        if not stats.sample_size:
            return 0.0
        goal_diff = (stats.goals_for - stats.goals_against) / stats.sample_size
        clean_rate = stats.clean_sheets / stats.sample_size
        blank_rate = stats.failed_to_score / stats.sample_size
        attack = stats.avg_goals_for - 1.25
        defense = 1.15 - stats.avg_goals_against
        points = stats.points_per_match - 1.55
        score = 0.34 * goal_diff + 0.22 * attack + 0.18 * defense + 0.16 * points + 0.12 * clean_rate - 0.14 * blank_rate
        return max(-1.6, min(1.6, score))

    @staticmethod
    def _chance_edge(
        attacking_stats: TeamStats,
        defending_stats: TeamStats,
        attacking_tactics: dict[str, Any],
        defending_tactics: dict[str, Any],
    ) -> float:
        return max(
            -1.6,
            min(
                1.6,
                MatchPredictor._attack_signal(attacking_stats, attacking_tactics)
                + MatchPredictor._defensive_weakness(defending_stats, defending_tactics),
            ),
        )

    @staticmethod
    def _attack_signal(stats: TeamStats, tactics: dict[str, Any]) -> float:
        shot_sample = min(1.0, stats.shot_samples / 5.0)
        shots = 0.0 if stats.avg_shots_for is None else (stats.avg_shots_for - 11.0) / 12.0
        shots_on_target = 0.0 if stats.avg_shots_on_target_for is None else (stats.avg_shots_on_target_for - 3.8) / 5.0
        goals = (stats.avg_goals_for - 1.25) / 2.4
        chance_creation = float(tactics.get("chance_creation", 0.55)) - 0.55
        tempo = float(tactics.get("tempo", 0.55)) - 0.55
        set_pieces = float(tactics.get("set_piece_threat", 0.50)) - 0.50
        signal = 0.38 * goals + shot_sample * (0.28 * shots + 0.30 * shots_on_target)
        signal += 0.72 * chance_creation + 0.34 * tempo + 0.22 * set_pieces
        return max(-1.25, min(1.25, signal))

    @staticmethod
    def _defensive_weakness(stats: TeamStats, tactics: dict[str, Any]) -> float:
        shot_sample = min(1.0, stats.shot_samples / 5.0)
        goals_allowed = (stats.avg_goals_against - 1.10) / 2.3
        shots_allowed = 0.0 if stats.avg_shots_against is None else (stats.avg_shots_against - 11.0) / 14.0
        clean_rate = stats.clean_sheets / stats.sample_size if stats.sample_size else 0.25
        solidity = 0.55 - float(tactics.get("defensive_solidity", 0.55))
        transition = 0.55 - float(tactics.get("transition_defense", 0.55))
        signal = 0.42 * goals_allowed + shot_sample * 0.28 * shots_allowed + 0.62 * solidity + 0.26 * transition
        signal -= 0.18 * clean_rate
        return max(-1.25, min(1.25, signal))

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
        for injury in context.get("injuries", []) + context.get("auto_absences", []):
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

    @staticmethod
    def _lineup_warnings(home_team: str, away_team: str, lineup_reports: dict[str, Any]) -> list[str]:
        warnings = []
        not_released = [
            team
            for team in (home_team, away_team)
            if lineup_reports.get(team, {}).get("status") != "confirmed"
        ]
        if not_released:
            warnings.append(
                "Составы еще не подтверждены для: "
                + ", ".join(not_released)
                + ". За час до матча модель попробует подтянуть стартовые составы и пересчитать схему/xG."
            )
        impacted = []
        for team, report in lineup_reports.items():
            missing = report.get("missing_key_players", [])
            benched = report.get("benched_key_players", [])
            if missing or benched:
                impacted.append(
                    f"{team}: вне старта/заявки "
                    + ", ".join(player.get("name", "") for player in (missing + benched)[:4])
                )
        if impacted:
            warnings.append("Состав влияет на прогноз: " + "; ".join(impacted) + ".")
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

        home_rich_matches = min(
            home_stats.corner_samples,
            home_stats.possession_samples,
            home_stats.shot_samples,
            home_stats.foul_samples,
        )
        away_rich_matches = min(
            away_stats.corner_samples,
            away_stats.possession_samples,
            away_stats.shot_samples,
            away_stats.foul_samples,
        )
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
            "home_foul_samples": home_stats.foul_samples,
            "away_foul_samples": away_stats.foul_samples,
            "backtest": {
                "matches": backtest.get("matches", 0) if isinstance(backtest, dict) else 0,
                "outcome_accuracy": backtest.get("outcome_accuracy") if isinstance(backtest, dict) else None,
                "exact_score_accuracy": backtest.get("exact_score_accuracy") if isinstance(backtest, dict) else None,
                "goal_mae": backtest.get("goal_mae") if isinstance(backtest, dict) else None,
                "expected_goal_mae": backtest.get("expected_goal_mae") if isinstance(backtest, dict) else None,
                "goal_within_0_7_rate": backtest.get("goal_within_0_7_rate") if isinstance(backtest, dict) else None,
                "corner_mae": backtest.get("corner_mae") if isinstance(backtest, dict) else None,
                "expected_corner_mae": backtest.get("expected_corner_mae") if isinstance(backtest, dict) else None,
                "corner_within_1_5_rate": backtest.get("corner_within_1_5_rate") if isinstance(backtest, dict) else None,
                "foul_mae": backtest.get("foul_mae") if isinstance(backtest, dict) else None,
                "expected_foul_mae": backtest.get("expected_foul_mae") if isinstance(backtest, dict) else None,
                "foul_within_2_rate": backtest.get("foul_within_2_rate") if isinstance(backtest, dict) else None,
                "playoff": backtest.get("playoff", {}) if isinstance(backtest, dict) else {},
                "playoff_target_status": backtest.get("playoff_target_status", {}) if isinstance(backtest, dict) else {},
                "targets": backtest.get("targets", {}) if isinstance(backtest, dict) else {},
                "target_status": backtest.get("target_status", {}) if isinstance(backtest, dict) else {},
                "trained_match_keys": backtest.get("trained_match_keys") if isinstance(backtest, dict) else None,
                "training": backtest.get("training", {}) if isinstance(backtest, dict) else {},
                "evaluation_mode": backtest.get("evaluation_mode")
                or (backtest.get("training", {}) if isinstance(backtest, dict) else {}).get("evaluation_mode"),
                "result_leakage_guard": bool(backtest.get("result_leakage_guard")) if isinstance(backtest, dict) else False,
                "updated_at": backtest.get("updated_at") if isinstance(backtest, dict) else None,
            },
            "home_backtest": by_team.get(home_team, {}),
            "away_backtest": by_team.get(away_team, {}),
        }
