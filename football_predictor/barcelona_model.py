from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median, pstdev
from typing import Any

from .barcelona_provider import BARCELONA_ID, BARCELONA_NAME
from .barcelona_store import BarcelonaStore
from .providers import normalize_provider_name


FEATURE_NAMES = [
    "barcelona_home",
    "champions_league",
    "knockout",
    "barca_points_10",
    "barca_goals_for_10",
    "barca_goals_against_10",
    "barca_win_rate_10",
    "barca_clean_sheet_10",
    "barca_over_2_5_10",
    "barca_comp_points",
    "opponent_points_10",
    "opponent_goals_for_10",
    "opponent_goals_against_10",
    "opponent_win_rate_10",
    "opponent_clean_sheet_10",
    "opponent_over_2_5_10",
    "opponent_comp_points",
    "elo_difference",
    "barca_rest_days",
    "opponent_rest_days",
    "h2h_barca_points",
    "h2h_barca_goals",
    "h2h_opponent_goals",
    "barca_possession",
    "barca_shots",
    "barca_shots_on_target",
    "barca_corners_for",
    "barca_corners_against",
    "barca_fouls_for",
    "opponent_possession",
    "opponent_shots",
    "opponent_shots_on_target",
    "opponent_corners_for",
    "opponent_corners_against",
    "opponent_fouls_for",
    "barca_lineup_strength",
    "opponent_lineup_strength",
    "referee_foul_tendency",
    "market_barcelona_win",
    "market_draw",
    "market_opponent_win",
    "market_goal_line",
]


class BarcelonaModel:
    def __init__(self, store: BarcelonaStore):
        self.store = store
        self._models_cache_key: str | None = None
        self._models_cache: dict[str, Any] | None = None

    def predict(self, fixture: dict[str, Any], remember: bool = True) -> dict[str, Any]:
        universe = self.store.load_universe()
        cutoff = str(fixture.get("kickoff") or fixture.get("date") or "")
        opponent = self.opponent_name(fixture)
        history = self._history_before(universe, cutoff)
        features, context = self._features(fixture, history)
        examples = self._training_examples(universe, cutoff=cutoff)
        models = self._fit_models(examples)
        forecast = self._forecast(features, context, models, examples)

        barca_profile = self.team_profile(BARCELONA_NAME, fixture, history)
        opponent_profile = self.team_profile(opponent, fixture, history)
        lineup = self._lineup_reports(fixture, history)
        referee = self._referee_report(fixture, history)
        backtest = self.store.load_backtest()
        payload = self._payload(
            fixture,
            opponent,
            forecast,
            context,
            barca_profile,
            opponent_profile,
            lineup,
            referee,
            backtest,
            len(examples),
        )
        if remember:
            saved = self.store.save_prediction(payload)
            saved["prediction_snapshot"] = {
                "immutable": True,
                "source": "pre_match_snapshot",
                "created_at": saved.get("created_at"),
                "message": "Предматчевый снимок сохранен и не изменится после начала матча.",
            }
            return saved
        payload["prediction_snapshot"] = {
            "immutable": False,
            "source": "preview",
            "message": "Предпросмотр без сохранения.",
        }
        return payload

    def lineup_reports(self, fixture: dict[str, Any]) -> dict[str, Any]:
        cutoff = str(fixture.get("kickoff") or fixture.get("date") or "")
        history = self._history_before(self.store.load_universe(), cutoff)
        return self._lineup_reports(fixture, history)

    def _lineup_reports(self, fixture: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        opponent = self.opponent_name(fixture)
        return {
            "barcelona": self._lineup_report(BARCELONA_NAME, fixture, history),
            "opponent": self._lineup_report(opponent, fixture, history),
        }

    def result_payload(self, fixture: dict[str, Any], saved: dict[str, Any] | None = None) -> dict[str, Any]:
        if saved:
            payload = dict(saved)
            payload["fixture"] = fixture
            payload["fixture_status"] = self._fixture_status(fixture)
            payload["result_summary"] = self._result_summary(fixture, saved)
            payload["prediction_snapshot"] = {
                "immutable": True,
                "source": "pre_match_snapshot",
                "created_at": saved.get("created_at"),
                "message": "Показан исходный прогноз, сохраненный до начала матча.",
            }
            return payload
        return {
            "prediction_available": False,
            "fixture": fixture,
            "fixture_id": fixture.get("fixture_id"),
            "opponent": self.opponent_name(fixture),
            "fixture_status": self._fixture_status(fixture),
            "result_summary": self._result_summary(fixture, None),
            "message": "До начала матча прогноз не был сохранен. Модель не создает прогноз задним числом.",
        }

    def train_and_backtest(self, force: bool = False) -> dict[str, Any]:
        universe = self.store.load_universe()
        examples = self._training_examples(universe)
        fingerprint = f"{len(examples)}:{examples[-1]['fixture_id'] if examples else ''}"
        cached = self.store.load_backtest()
        if not force and cached.get("fingerprint") == fingerprint:
            return cached
        if len(examples) < 35:
            payload = {
                "fingerprint": fingerprint,
                "honest_walk_forward": True,
                "sample_size": 0,
                "message": "Недостаточно матчей для бэктеста.",
            }
            self.store.save_backtest(payload)
            return payload

        predictions: list[dict[str, Any]] = []
        bundle: dict[str, Any] | None = None
        for index in range(30, len(examples)):
            # Retraining in chronological blocks keeps the test strictly future
            # while avoiding hundreds of almost identical model fits.
            if bundle is None or index % 10 == 0:
                bundle = self._fit_models(examples[:index], use_cache=False)
            item = examples[index]
            forecast = self._forecast(item["x"], item["context"], bundle, examples[:index])
            actual_outcome = self._outcome(item["barca_goals"], item["opponent_goals"])
            actual_total = item["barca_goals"] + item["opponent_goals"]
            predictions.append(
                {
                    "outcome_hit": forecast["outcome_pick"] == actual_outcome,
                    "exact_hit": forecast["exact_score"] == f"{item['barca_goals']}-{item['opponent_goals']}",
                    "goal_error": abs(forecast["total_expected"] - actual_total),
                    "corner_error": None if item["corners"] is None else abs(forecast["corners"] - item["corners"]),
                    "foul_error": None if item["fouls"] is None else abs(forecast["fouls"] - item["fouls"]),
                }
            )

        corner_errors = [item["corner_error"] for item in predictions if item["corner_error"] is not None]
        foul_errors = [item["foul_error"] for item in predictions if item["foul_error"] is not None]
        payload = {
            "fingerprint": fingerprint,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "honest_walk_forward": True,
            "sample_size": len(predictions),
            "training_matches": len(examples),
            "outcome_accuracy": _rate(item["outcome_hit"] for item in predictions),
            "exact_score_accuracy": _rate(item["exact_hit"] for item in predictions),
            "goal_total_mae": _average(item["goal_error"] for item in predictions),
            "corner_mae": _average(corner_errors),
            "foul_mae": _average(foul_errors),
            "corners_within_1_5": _rate(error <= 1.5 for error in corner_errors),
            "fouls_within_2_5": _rate(error <= 2.5 for error in foul_errors),
            "evaluation": "Каждый матч предсказан только по данным, доступным до его стартового времени.",
        }
        self.store.save_backtest(payload)
        model_state = self.store.load_model_state()
        model_state.update(
            {
                "version": 2,
                "trained_at": payload["generated_at"],
                "training_matches": len(examples),
                "feature_names": FEATURE_NAMES,
                "algorithm": "Leak-free Poisson + nearest-match score similarity + robust stat boosting",
                "backtest_fingerprint": fingerprint,
            }
        )
        self.store.save_model_state(model_state)
        return payload

    def _training_examples(self, universe: list[dict[str, Any]], cutoff: str | None = None) -> list[dict[str, Any]]:
        barca_matches = [
            item
            for item in universe
            if item.get("completed")
            and item.get("home_goals") is not None
            and item.get("away_goals") is not None
            and self._has_team(item, BARCELONA_NAME)
        ]
        barca_matches.sort(key=lambda item: item.get("kickoff", ""))
        fingerprint = self._example_fingerprint(barca_matches)
        model_state = self.store.load_model_state()
        cache = model_state.get("example_cache") or {}
        if cache.get("fingerprint") == fingerprint and cache.get("feature_version") == 4:
            cached_examples = cache.get("examples") or []
            if cutoff is None:
                return cached_examples
            return [item for item in cached_examples if str(item.get("kickoff") or item.get("date") or "") < cutoff]

        examples: list[dict[str, Any]] = []
        for fixture in barca_matches:
            fixture_cutoff = str(fixture.get("kickoff") or fixture.get("date") or "")
            history = self._history_before(universe, fixture_cutoff)
            if len([item for item in history if self._has_team(item, BARCELONA_NAME)]) < 8:
                continue
            x, context = self._features(fixture, history)
            barca_home = self._barca_home(fixture)
            barca_goals = int(fixture["home_goals"] if barca_home else fixture["away_goals"])
            opponent_goals = int(fixture["away_goals"] if barca_home else fixture["home_goals"])
            home_stats = fixture.get("home_stats") or {}
            away_stats = fixture.get("away_stats") or {}
            examples.append(
                {
                    "fixture_id": str(fixture.get("fixture_id")),
                    "date": fixture.get("date"),
                    "kickoff": fixture.get("kickoff"),
                    "x": x,
                    "context": context,
                    "barca_goals": barca_goals,
                    "opponent_goals": opponent_goals,
                    "outcome": self._outcome(barca_goals, opponent_goals),
                    "corners": _sum_stats(home_stats, away_stats, "wonCorners", fixture, "home_corners", "away_corners"),
                    "fouls": _sum_stats(home_stats, away_stats, "foulsCommitted", fixture, "home_fouls", "away_fouls"),
                }
            )
        model_state["example_cache"] = {
            "fingerprint": fingerprint,
            "feature_version": 4,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "examples": examples,
        }
        self.store.save_model_state(model_state)
        if cutoff is None:
            return examples
        return [item for item in examples if str(item.get("kickoff") or item.get("date") or "") < cutoff]

    def _fit_models(self, examples: list[dict[str, Any]], use_cache: bool = True) -> dict[str, Any]:
        if len(examples) < 24:
            return {}
        cache_key = f"{len(examples)}:{examples[-1].get('fixture_id', '')}"
        if use_cache and self._models_cache_key == cache_key and self._models_cache is not None:
            return self._models_cache
        try:
            import numpy as np
            from sklearn.ensemble import HistGradientBoostingRegressor
        except Exception:
            return {}

        x = np.asarray([item["x"] for item in examples], dtype=float)
        bundle: dict[str, Any] = {}
        for key in ("corners", "fouls"):
            rows = [item for item in examples if item[key] is not None]
            if len(rows) < 24:
                continue
            sx = np.asarray([item["x"] for item in rows], dtype=float)
            sy = np.asarray([item[key] for item in rows], dtype=float)
            bundle[key] = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=100,
                learning_rate=0.055,
                max_leaf_nodes=6,
                min_samples_leaf=7,
                l2_regularization=2.0,
                random_state=83,
            ).fit(sx, sy)
        if use_cache:
            self._models_cache_key = cache_key
            self._models_cache = bundle
        return bundle

    def _forecast(
        self,
        features: list[float],
        context: dict[str, Any],
        models: dict[str, Any],
        examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        baseline_barca = context["baseline_barca_goals"]
        baseline_opponent = context["baseline_opponent_goals"]
        # The walk-forward selector found the analytical intensity more stable
        # than flexible goal regressors on a one-club sample. Rich context still
        # enters these baselines through form, Elo, venue, H2H and lineup strength.
        barca_xg = baseline_barca
        opponent_xg = baseline_opponent
        market = context.get("market") or {}
        market_total = market.get("total_line")
        if market_total is not None:
            raw_total = barca_xg + opponent_xg
            adjusted_total = 0.72 * raw_total + 0.28 * (float(market_total) + 0.05)
            scale = adjusted_total / raw_total if raw_total else 1.0
            barca_xg *= scale
            opponent_xg *= scale

        score_matrix = self._score_matrix(barca_xg, opponent_xg)
        poisson_probs = self._outcome_probabilities(score_matrix)
        outcome_probs = dict(poisson_probs)
        market_probs = market.get("probabilities") or {}
        if all(key in market_probs for key in ("barcelona", "draw", "opponent")):
            outcome_probs = _normalize(
                {
                    key: 0.45 * poisson_probs[key] + 0.55 * float(market_probs[key])
                    for key in poisson_probs
                }
            )

        outcome_pick = max(outcome_probs, key=outcome_probs.get)
        total_probabilities = self._total_probabilities(score_matrix)
        exact_matrix = self._similar_score_adjustment(score_matrix, examples, features, context)
        exact_score, exact_probability = self._best_consistent_score(
            exact_matrix,
            outcome_pick,
            total_probabilities,
        )

        recent_corner_values = [float(item["corners"]) for item in examples[-50:] if item.get("corners") is not None]
        recent_foul_values = [float(item["fouls"]) for item in examples[-50:] if item.get("fouls") is not None]
        corner_rolling = median(recent_corner_values) if recent_corner_values else 9.5
        foul_rolling = median(recent_foul_values) if recent_foul_values else 24.0
        corner_context = context.get("recent_total_corners")
        foul_context = context.get("recent_total_fouls")
        corner_baseline = corner_rolling if corner_context is None else 0.90 * corner_rolling + 0.10 * float(corner_context)
        foul_baseline = foul_rolling if foul_context is None else 0.85 * foul_rolling + 0.15 * float(foul_context)
        if context.get("referee_avg_fouls") is not None:
            foul_baseline = 0.85 * foul_baseline + 0.15 * float(context["referee_avg_fouls"])
        corners = self._stat_model_prediction(models.get("corners"), features, corner_baseline, 3.0, 18.0, model_weight=0.15)
        fouls = self._stat_model_prediction(models.get("fouls"), features, foul_baseline, 10.0, 42.0, model_weight=0.15)

        corner_sigma = max(1.8, _mad(recent_corner_values, corners) * 1.45)
        foul_sigma = max(3.0, _mad(recent_foul_values, fouls) * 1.45)
        return {
            "barca_xg": round(barca_xg, 2),
            "opponent_xg": round(opponent_xg, 2),
            "total_expected": round(barca_xg + opponent_xg, 2),
            "outcome_pick": outcome_pick,
            "outcome_probabilities": outcome_probs,
            "exact_score": exact_score,
            "exact_probability": exact_probability,
            "total_probabilities": total_probabilities,
            "corners": int(round(corners)),
            "corners_expected": round(corners, 2),
            "corner_sigma": round(corner_sigma, 2),
            "fouls": int(round(fouls)),
            "fouls_expected": round(fouls, 2),
            "foul_sigma": round(foul_sigma, 2),
        }

    def _features(self, fixture: dict[str, Any], history: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
        opponent = self.opponent_name(fixture)
        competition = str(fixture.get("competition_code") or "")
        barca_form = self._team_form(BARCELONA_NAME, history)
        barca_comp = self._team_form(BARCELONA_NAME, history, competition)
        opponent_form = self._team_form(opponent, history)
        opponent_comp = self._team_form(opponent, history, competition)
        elo = self._elo_ratings(history)
        barca_elo = elo.get(normalize_provider_name(BARCELONA_NAME), 1700.0)
        opponent_elo = elo.get(normalize_provider_name(opponent), 1500.0)
        h2h = self._h2h(history, opponent, str(fixture.get("kickoff") or fixture.get("date") or ""))
        barca_rich = self._rich_form(BARCELONA_NAME, history)
        opponent_rich = self._rich_form(opponent, history)
        barca_lineup = self._lineup_strength(BARCELONA_NAME, fixture, history)
        opponent_lineup = self._lineup_strength(opponent, fixture, history)
        referee_avg_fouls = self._referee_average_fouls(fixture, history)
        market = self._market_context(fixture)
        barca_home = self._barca_home(fixture)

        league_home, league_away = self._league_goal_rates(history, competition)
        if barca_home:
            baseline_barca = 0.48 * barca_form["gf"] + 0.29 * opponent_form["ga"] + 0.23 * league_home
            baseline_opponent = 0.46 * opponent_form["gf"] + 0.31 * barca_form["ga"] + 0.23 * league_away
        else:
            baseline_barca = 0.48 * barca_form["gf"] + 0.29 * opponent_form["ga"] + 0.23 * league_away
            baseline_opponent = 0.46 * opponent_form["gf"] + 0.31 * barca_form["ga"] + 0.23 * league_home
        elo_factor = math.exp((barca_elo - opponent_elo) / 1250.0)
        baseline_barca *= _clip(elo_factor, 0.74, 1.38) * _clip(barca_lineup, 0.82, 1.05)
        baseline_opponent /= _clip(elo_factor, 0.76, 1.34)
        if h2h["sample"] >= 2:
            baseline_barca = 0.9 * baseline_barca + 0.1 * h2h["gf"]
            baseline_opponent = 0.9 * baseline_opponent + 0.1 * h2h["ga"]
        baseline_barca = _clip(baseline_barca, 0.28, 4.4)
        baseline_opponent = _clip(baseline_opponent, 0.18, 3.5)

        features = [
            1.0 if barca_home else 0.0,
            1.0 if competition == "uefa.champions" else 0.0,
            1.0 if self._is_knockout(fixture) else 0.0,
            barca_form["points"], barca_form["gf"], barca_form["ga"], barca_form["win_rate"], barca_form["clean_rate"], barca_form["over25"],
            barca_comp["points"],
            opponent_form["points"], opponent_form["gf"], opponent_form["ga"], opponent_form["win_rate"], opponent_form["clean_rate"], opponent_form["over25"],
            opponent_comp["points"],
            (barca_elo - opponent_elo) / 400.0,
            self._rest_days(BARCELONA_NAME, fixture, history) / 10.0,
            self._rest_days(opponent, fixture, history) / 10.0,
            h2h["points"], h2h["gf"], h2h["ga"],
            barca_rich["possession"], barca_rich["shots"], barca_rich["shots_on_target"], barca_rich["corners_for"], barca_rich["corners_against"], barca_rich["fouls_for"],
            opponent_rich["possession"], opponent_rich["shots"], opponent_rich["shots_on_target"], opponent_rich["corners_for"], opponent_rich["corners_against"], opponent_rich["fouls_for"],
            barca_lineup,
            opponent_lineup,
            (referee_avg_fouls or 24.0) / 25.0,
            float((market.get("probabilities") or {}).get("barcelona", 0.5)),
            float((market.get("probabilities") or {}).get("draw", 0.25)),
            float((market.get("probabilities") or {}).get("opponent", 0.25)),
            float(market.get("total_line") or 2.75) / 3.0,
        ]
        context = {
            "opponent": opponent,
            "barca_home": barca_home,
            "baseline_barca_goals": baseline_barca,
            "baseline_opponent_goals": baseline_opponent,
            "barca_elo": round(barca_elo),
            "opponent_elo": round(opponent_elo),
            "elo_difference": round(barca_elo - opponent_elo),
            "barca_form": barca_form,
            "opponent_form": opponent_form,
            "h2h": h2h,
            "recent_total_corners": self._recent_match_total(history, BARCELONA_NAME, opponent, "wonCorners"),
            "recent_total_fouls": self._recent_match_total(history, BARCELONA_NAME, opponent, "foulsCommitted"),
            "referee_avg_fouls": referee_avg_fouls,
            "market": market,
        }
        return [float(value) for value in features], context

    def team_profile(self, team: str, fixture: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        recent = self._team_matches(team, history)[:10]
        form = self._team_form(team, history)
        rich = self._rich_form(team, history)
        formations = [self._team_formation(item, team) for item in recent]
        formations = [value for value in formations if value]
        current = self._fixture_lineup(fixture, team)
        predicted_formation = current.get("formation") if current.get("confirmed") else (Counter(formations).most_common(1)[0][0] if formations else "нет данных")
        possession = rich["possession"]
        attack = _clip((rich["shots"] / 18.0 + rich["shots_on_target"] / 7.0 + form["gf"] / 2.4) / 3.0, 0.2, 0.95)
        defense = _clip(1.0 - form["ga"] / 3.2, 0.2, 0.94)
        width = _clip((rich["corners_for"] / 7.0 + rich.get("crosses", 15.0) / 24.0) / 2.0, 0.2, 0.92)
        pressing = _clip(0.42 + form["points"] * 0.08 + (possession - 0.5) * 0.35, 0.3, 0.9)
        tempo = _clip(0.28 + rich["shots"] / 42.0 + rich["shots_on_target"] / 52.0, 0.3, 0.9)
        return {
            "team": team,
            "formation": predicted_formation,
            "formation_source": "confirmed_lineup" if current.get("confirmed") else "mode_last_matches",
            "formation_sample": dict(Counter(formations)),
            "sample_size": len(recent),
            "form": form,
            "metrics": {
                "possession": round(possession * 100),
                "pressing": round(pressing * 100),
                "attack": round(attack * 100),
                "defense": round(defense * 100),
                "width": round(width * 100),
                "tempo": round(tempo * 100),
            },
            "style": self._style_label(possession, tempo, width),
            "recent_matches": [self._match_row(item, team) for item in recent],
            "averages": {
                "goals_for": round(form["gf"], 2),
                "goals_against": round(form["ga"], 2),
                "shots": round(rich["shots"], 2),
                "shots_on_target": round(rich["shots_on_target"], 2),
                "corners_for": round(rich["corners_for"], 2),
                "fouls_for": round(rich["fouls_for"], 2),
            },
        }

    def _payload(
        self,
        fixture: dict[str, Any],
        opponent: str,
        forecast: dict[str, Any],
        context: dict[str, Any],
        barca_profile: dict[str, Any],
        opponent_profile: dict[str, Any],
        lineup: dict[str, Any],
        referee: dict[str, Any],
        backtest: dict[str, Any],
        training_matches: int,
    ) -> dict[str, Any]:
        labels = {"barcelona": "Победа Барселоны", "draw": "Ничья", "opponent": f"Победа {opponent}"}
        probabilities = forecast["outcome_probabilities"]
        recommended = self._recommended_bet(forecast, opponent, backtest)
        payload = {
            "prediction_available": True,
            "fixture_id": str(fixture.get("fixture_id")),
            "fixture": fixture,
            "match_date": fixture.get("date"),
            "opponent": opponent,
            "competition": fixture.get("competition"),
            "stage": fixture.get("stage"),
            "venue": fixture.get("venue"),
            "barcelona_side": "home" if context["barca_home"] else "away",
            "outcome": {
                "pick": forecast["outcome_pick"],
                "label": labels[forecast["outcome_pick"]],
                "confidence": round(probabilities[forecast["outcome_pick"]], 4),
                "probabilities": {
                    "barcelona_win": round(probabilities["barcelona"], 4),
                    "draw": round(probabilities["draw"], 4),
                    "opponent_win": round(probabilities["opponent"], 4),
                    "barcelona_not_lose": round(probabilities["barcelona"] + probabilities["draw"], 4),
                    "opponent_not_lose": round(probabilities["opponent"] + probabilities["draw"], 4),
                },
            },
            "exact_score": {
                "score": forecast["exact_score"],
                "probability": round(forecast["exact_probability"], 4),
                "orientation": "Барселона — соперник",
            },
            "goals": {
                "barcelona_expected": forecast["barca_xg"],
                "opponent_expected": forecast["opponent_xg"],
                "total_expected": forecast["total_expected"],
                "point": int(round(forecast["total_expected"])),
                "probabilities": {key: round(value, 4) for key, value in forecast["total_probabilities"].items()},
            },
            "corners": self._stat_payload(forecast["corners"], forecast["corners_expected"], forecast["corner_sigma"], "угловых"),
            "fouls": self._stat_payload(forecast["fouls"], forecast["fouls_expected"], forecast["foul_sigma"], "фолов"),
            "barcelona_profile": barca_profile,
            "opponent_profile": opponent_profile,
            "lineups": lineup,
            "referee": referee,
            "h2h": context["h2h"],
            "strength": {
                "barcelona_elo": context["barca_elo"],
                "opponent_elo": context["opponent_elo"],
                "difference": context["elo_difference"],
            },
            "recommended_bet": recommended,
            "data_quality": {
                "training_matches": training_matches,
                "barcelona_last10": barca_profile["sample_size"],
                "opponent_last10": opponent_profile["sample_size"],
                "lineups_confirmed": bool(lineup["barcelona"]["confirmed"] and lineup["opponent"]["confirmed"]),
                "market_signal": bool(context.get("market", {}).get("probabilities")),
                "backtest": backtest,
                "leakage_guard": "Признаки обрезаны по времени стартового свистка; итог матча не входит в прогноз.",
            },
            "fixture_status": self._fixture_status(fixture),
            "result_summary": self._result_summary(fixture, None),
        }
        # The full fixture is needed for immutable start-time validation, but
        # historical result fields must never become model inputs here.
        return payload

    def _stat_payload(self, point: int, expected: float, sigma: float, label: str) -> dict[str, Any]:
        lower_line = math.floor(point - 1.5) + 0.5
        upper_line = math.ceil(point + 1.5) - 0.5
        return {
            "point": point,
            "expected": expected,
            "interval_70": [max(0, round(expected - 1.04 * sigma, 1)), round(expected + 1.04 * sigma, 1)],
            "markets": {
                f"over_{str(lower_line).replace('.', '_')}": round(1.0 - _normal_cdf(lower_line, expected, sigma), 4),
                f"under_{str(upper_line).replace('.', '_')}": round(_normal_cdf(upper_line, expected, sigma), 4),
            },
            "label": label,
        }

    def _recommended_bet(self, forecast: dict[str, Any], opponent: str, backtest: dict[str, Any]) -> dict[str, Any]:
        probs = forecast["outcome_probabilities"]
        candidates = [
            ("Барселона не проиграет", probs["barcelona"] + probs["draw"], "1X"),
            (f"{opponent} не проиграет", probs["opponent"] + probs["draw"], "X2"),
            ("ТМ 4.5 голов", forecast["total_probabilities"]["under_4_5"], "goals_under_4_5"),
            ("ТБ 1.5 голов", forecast["total_probabilities"]["over_1_5"], "goals_over_1_5"),
            (
                f"ТБ {math.floor(forecast['corners_expected'] - 2.5) + 0.5} угловых",
                1.0 - _normal_cdf(math.floor(forecast["corners_expected"] - 2.5) + 0.5, forecast["corners_expected"], forecast["corner_sigma"]),
                "corners_safe_over",
            ),
            (
                f"ТМ {math.ceil(forecast['fouls_expected'] + 5.5) - 0.5} фолов",
                _normal_cdf(math.ceil(forecast["fouls_expected"] + 5.5) - 0.5, forecast["fouls_expected"], forecast["foul_sigma"]),
                "fouls_safe_under",
            ),
        ]
        label, probability, code = max(candidates, key=lambda item: item[1])
        return {
            "label": label,
            "code": code,
            "model_probability": round(probability, 4),
            "eligible": probability >= 0.75,
            "backtest_sample": backtest.get("sample_size", 0),
            "note": "Ставка показана только при расчетной вероятности не ниже 75%; это оценка модели, не гарантия.",
        }

    def _goal_model_prediction(self, pair: Any, features: list[float], baseline: float, model_weight: float) -> float:
        if not pair:
            return baseline
        try:
            hgb, forest = pair
            model_value = 0.58 * float(hgb.predict([features])[0]) + 0.42 * float(forest.predict([features])[0])
            return _clip(model_weight * model_value + (1.0 - model_weight) * baseline, 0.16, 4.8)
        except Exception:
            return baseline

    def _stat_model_prediction(
        self,
        model: Any,
        features: list[float],
        baseline: float,
        low: float,
        high: float,
        model_weight: float,
    ) -> float:
        if model is None:
            return _clip(baseline, low, high)
        try:
            value = float(model.predict([features])[0])
            return _clip(model_weight * value + (1.0 - model_weight) * baseline, low, high)
        except Exception:
            return _clip(baseline, low, high)

    def _score_matrix(self, barca_xg: float, opponent_xg: float) -> list[tuple[int, int, float]]:
        matrix: list[tuple[int, int, float]] = []
        for barca in range(8):
            for opponent in range(8):
                probability = _poisson(barca, barca_xg) * _poisson(opponent, opponent_xg)
                if (barca, opponent) in {(0, 0), (1, 1)}:
                    probability *= 1.06
                elif (barca, opponent) in {(1, 0), (0, 1)}:
                    probability *= 0.98
                matrix.append((barca, opponent, probability))
        total = sum(item[2] for item in matrix) or 1.0
        return [(a, b, p / total) for a, b, p in matrix]

    def _empirical_score_adjustment(
        self,
        matrix: list[tuple[int, int, float]],
        examples: list[dict[str, Any]],
    ) -> list[tuple[int, int, float]]:
        counts = Counter((int(item["barca_goals"]), int(item["opponent_goals"])) for item in examples)
        sample = len(examples)
        adjusted = []
        for barca, opponent, probability in matrix:
            smoothed_frequency = (counts[(barca, opponent)] + 0.5) / (sample + 32.0)
            relative_prior = smoothed_frequency * 64.0
            adjusted.append((barca, opponent, probability * relative_prior**0.15))
        total = sum(item[2] for item in adjusted) or 1.0
        return [(barca, opponent, probability / total) for barca, opponent, probability in adjusted]

    def _similar_score_adjustment(
        self,
        matrix: list[tuple[int, int, float]],
        examples: list[dict[str, Any]],
        features: list[float],
        context: dict[str, Any],
    ) -> list[tuple[int, int, float]]:
        if len(examples) < 12:
            return matrix

        def similarity_vector(item_features: list[float], item_context: dict[str, Any]) -> list[float]:
            return [
                item_features[0] * 2.0,
                item_features[1] * 1.2,
                item_features[17],
                item_features[4],
                item_features[5],
                item_features[11],
                item_features[12],
                float(item_context["baseline_barca_goals"]),
                float(item_context["baseline_opponent_goals"]),
            ]

        target = similarity_vector(features, context)
        vectors = [similarity_vector(item["x"], item["context"]) for item in examples]
        scales = [max(0.15, pstdev([vector[index] for vector in vectors])) for index in range(len(target))]
        ranked = sorted(
            zip(examples, vectors),
            key=lambda pair: sum(
                ((pair[1][index] - target[index]) / scales[index]) ** 2
                for index in range(len(target))
            ),
        )[:25]
        counts = Counter((int(item["barca_goals"]), int(item["opponent_goals"])) for item, _ in ranked)
        adjusted = []
        for barca, opponent, probability in matrix:
            relative_prior = ((counts[(barca, opponent)] + 0.5) / (len(ranked) + 32.0)) * 64.0
            adjusted.append((barca, opponent, probability * relative_prior**0.4))
        total = sum(item[2] for item in adjusted) or 1.0
        return [(barca, opponent, probability / total) for barca, opponent, probability in adjusted]

    def _example_fingerprint(self, matches: list[dict[str, Any]]) -> str:
        if not matches:
            return "0"
        rich = sum(bool(item.get("details_loaded")) for item in matches)
        last = matches[-1]
        return f"{len(matches)}:{last.get('fixture_id', '')}:{last.get('home_goals')}:{last.get('away_goals')}:{rich}"

    def _outcome_probabilities(self, matrix: list[tuple[int, int, float]]) -> dict[str, float]:
        result = {"barcelona": 0.0, "draw": 0.0, "opponent": 0.0}
        for barca, opponent, probability in matrix:
            result[self._outcome(barca, opponent)] += probability
        return _normalize(result)

    def _total_probabilities(self, matrix: list[tuple[int, int, float]]) -> dict[str, float]:
        result: dict[str, float] = {}
        for line in (1.5, 2.5, 3.5, 4.5):
            over = sum(probability for a, b, probability in matrix if a + b > line)
            key = str(line).replace(".", "_")
            result[f"over_{key}"] = over
            result[f"under_{key}"] = 1.0 - over
        return result

    def _best_consistent_score(
        self,
        matrix: list[tuple[int, int, float]],
        outcome: str,
        totals: dict[str, float],
    ) -> tuple[str, float]:
        candidates = [item for item in matrix if self._outcome(item[0], item[1]) == outcome]
        if totals["over_2_5"] >= 0.55:
            aligned = [item for item in candidates if item[0] + item[1] >= 3]
            candidates = aligned or candidates
        elif totals["under_2_5"] >= 0.58:
            aligned = [item for item in candidates if item[0] + item[1] <= 2]
            candidates = aligned or candidates
        barca, opponent, probability = max(candidates, key=lambda item: item[2])
        return f"{barca}-{opponent}", probability

    def _history_before(self, universe: list[dict[str, Any]], cutoff: str) -> list[dict[str, Any]]:
        return sorted(
            [
                item
                for item in universe
                if item.get("completed")
                and item.get("home_goals") is not None
                and item.get("away_goals") is not None
                and str(item.get("kickoff") or item.get("date") or "") < cutoff
            ],
            key=lambda item: item.get("kickoff", ""),
        )

    def _team_matches(self, team: str, history: list[dict[str, Any]], competition: str | None = None) -> list[dict[str, Any]]:
        result = [
            item
            for item in history
            if self._has_team(item, team)
            and (competition is None or item.get("competition_code") == competition)
        ]
        return sorted(result, key=lambda item: item.get("kickoff", ""), reverse=True)

    def _team_form(self, team: str, history: list[dict[str, Any]], competition: str | None = None) -> dict[str, float]:
        matches = self._team_matches(team, history, competition)[:10]
        if not matches and competition:
            return self._team_form(team, history, None)
        if not matches:
            return {"sample": 0, "points": 1.25, "gf": 1.25, "ga": 1.25, "win_rate": 0.35, "clean_rate": 0.25, "over25": 0.5}
        points = gf = ga = wins = cleans = over25 = 0.0
        for fixture in matches:
            scored, conceded = self._team_score(fixture, team)
            gf += scored
            ga += conceded
            points += 3 if scored > conceded else 1 if scored == conceded else 0
            wins += scored > conceded
            cleans += conceded == 0
            over25 += scored + conceded >= 3
        n = len(matches)
        return {
            "sample": n,
            "points": points / n,
            "gf": gf / n,
            "ga": ga / n,
            "win_rate": wins / n,
            "clean_rate": cleans / n,
            "over25": over25 / n,
        }

    def _rich_form(self, team: str, history: list[dict[str, Any]]) -> dict[str, float]:
        matches = [item for item in self._team_matches(team, history) if item.get("home_stats") and item.get("away_stats")][:10]
        defaults = {"possession": 0.5, "shots": 11.5, "shots_on_target": 4.2, "corners_for": 4.8, "corners_against": 4.8, "fouls_for": 12.0, "crosses": 16.0}
        if not matches:
            return defaults
        values: dict[str, list[float]] = defaultdict(list)
        mapping = {
            "possession": "possessionPct",
            "shots": "totalShots",
            "shots_on_target": "shotsOnTarget",
            "corners_for": "wonCorners",
            "fouls_for": "foulsCommitted",
            "crosses": "totalCrosses",
        }
        for fixture in matches:
            own, other = self._team_stats(fixture, team)
            for label, key in mapping.items():
                if own.get(key) is not None:
                    value = float(own[key])
                    values[label].append(value / 100.0 if label == "possession" else value)
            if other.get("wonCorners") is not None:
                values["corners_against"].append(float(other["wonCorners"]))
        return {key: mean(values[key]) if values.get(key) else default for key, default in defaults.items()}

    def _elo_ratings(self, history: list[dict[str, Any]]) -> dict[str, float]:
        ratings: defaultdict[str, float] = defaultdict(lambda: 1500.0)
        for fixture in history:
            home = normalize_provider_name(str(fixture.get("home_team") or ""))
            away = normalize_provider_name(str(fixture.get("away_team") or ""))
            if not home or not away:
                continue
            home_rating = ratings[home]
            away_rating = ratings[away]
            expected = 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + 55.0)) / 400.0))
            hg, ag = int(fixture["home_goals"]), int(fixture["away_goals"])
            actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            margin = max(1.0, math.log(abs(hg - ag) + 1.0) * 1.35)
            k = (28.0 if fixture.get("competition_code") == "uefa.champions" else 22.0) * margin
            change = k * (actual - expected)
            ratings[home] += change
            ratings[away] -= change
        return dict(ratings)

    def _h2h(self, history: list[dict[str, Any]], opponent: str, cutoff: str) -> dict[str, Any]:
        matches = [item for item in history if self._has_team(item, BARCELONA_NAME) and self._has_team(item, opponent)]
        matches = sorted(matches, key=lambda item: item.get("kickoff", ""), reverse=True)[:10]
        if not matches:
            return {"sample": 0, "points": 1.5, "gf": 1.5, "ga": 1.0, "matches": [], "weighting": "none"}
        cutoff_dt = _parse_time(cutoff)
        total_weight = points = gf = ga = 0.0
        rows = []
        for fixture in matches:
            age_days = max(0, (cutoff_dt - _parse_time(str(fixture.get("kickoff") or fixture.get("date")))).days)
            weight = math.exp(-age_days / 730.0)
            scored, conceded = self._team_score(fixture, BARCELONA_NAME)
            total_weight += weight
            points += weight * (3 if scored > conceded else 1 if scored == conceded else 0)
            gf += weight * scored
            ga += weight * conceded
            rows.append({"date": fixture.get("date"), "score": f"{scored}-{conceded}", "weight": round(weight, 2)})
        return {
            "sample": len(matches),
            "points": points / total_weight,
            "gf": gf / total_weight,
            "ga": ga / total_weight,
            "matches": rows,
            "weighting": "Экспоненциальное затухание; матчи старше 2 лет имеют вспомогательный вес.",
        }

    def _league_goal_rates(self, history: list[dict[str, Any]], competition: str) -> tuple[float, float]:
        selected = [item for item in history if item.get("competition_code") == competition][-500:]
        if not selected:
            return 1.45, 1.2
        return mean(float(item["home_goals"]) for item in selected), mean(float(item["away_goals"]) for item in selected)

    def _rest_days(self, team: str, fixture: dict[str, Any], history: list[dict[str, Any]]) -> float:
        matches = self._team_matches(team, history)
        if not matches:
            return 10.0
        days = (_parse_time(str(fixture.get("kickoff") or fixture.get("date"))) - _parse_time(str(matches[0].get("kickoff") or matches[0].get("date")))).total_seconds() / 86400
        return _clip(days, 2.0, 21.0)

    def _lineup_strength(self, team: str, fixture: dict[str, Any], history: list[dict[str, Any]]) -> float:
        current = self._fixture_lineup(fixture, team)
        if not current.get("confirmed"):
            projection = self._projected_lineup(team, fixture, history)
            return _clip(0.94 + 0.06 * float(projection.get("confidence", 0.0)), 0.94, 1.0)
        counts = self._regular_player_counts(team, history)
        if not counts:
            return 1.0
        top = sorted(counts.values(), reverse=True)[:11]
        denominator = sum(top) or 1.0
        selected = sum(counts.get(normalize_provider_name(player.get("name", "")), 0.0) for player in current.get("starters", []))
        return _clip(selected / denominator, 0.72, 1.05)

    def _lineup_report(self, team: str, fixture: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        current = self._fixture_lineup(fixture, team)
        counts = self._regular_player_counts(team, history)
        regulars = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:11]
        display_names = self._regular_player_names(team, history)
        projection = self._projected_lineup(team, fixture, history)
        official_players = self._prepared_confirmed_players(current.get("starters") or [])
        official_available = bool(current.get("confirmed") and len(official_players) >= 11)
        display_lineup = {
            "type": "confirmed" if official_available else "predicted",
            "players": official_players if official_available else projection["players"],
            "formation": current.get("formation") if official_available else projection.get("formation"),
            "confidence": 1.0 if official_available else projection.get("confidence", 0.0),
        }
        return {
            "team": team,
            "confirmed": official_available,
            "official_available": official_available,
            "formation": display_lineup["formation"],
            "strength": round(self._lineup_strength(team, fixture, history), 3) if official_available else None,
            "starters": current.get("starters", []),
            "regular_core": [display_names.get(key, key) for key, _ in regulars],
            "source": current.get("source", "not-released"),
            "official_lineup": {
                "available": official_available,
                "formation": current.get("formation") if official_available else None,
                "players": official_players if official_available else [],
                "source": current.get("source", "not-released"),
            },
            "predicted_lineup": projection,
            "display_lineup": display_lineup,
            "squad_context": {
                "applied": projection.get("active_roster_applied", False),
                "active_players": projection.get("current_roster_size", 0),
                "source": projection.get("roster_source"),
                "source_url": projection.get("roster_source_url"),
                "recent_signings": projection.get("recent_signings", []),
                "filtered_departures": projection.get("filtered_departures", []),
            },
            "message": (
                "Официальный стартовый состав опубликован ESPN."
                if official_available
                else (
                    "Официального состава пока нет. Ниже показан прогноз по форме, сезонной основе и текущей заявке."
                    if projection.get("active_roster_applied")
                    else "Официального состава пока нет. Ниже показан прогноз модели по последним 10 матчам."
                )
            ),
        }

    def _projected_lineup(self, team: str, fixture: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        recent = self._team_matches(team, history)[:10]
        target_competition = fixture.get("competition_code")
        squad = self._current_squad(fixture, team)
        squad_players = squad.get("players") or []
        active_players = {
            normalize_provider_name(str(player.get("name") or "")): player
            for player in squad_players
            if player.get("name")
        }
        active_keys = set(active_players)
        slot_scores: defaultdict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        player_scores: defaultdict[str, float] = defaultdict(float)
        player_names: dict[str, str] = {
            key: str(player.get("name")) for key, player in active_players.items()
        }
        slot_positions: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
        formation_scores: defaultdict[str, float] = defaultdict(float)
        filtered_players: Counter[str] = Counter()
        usable_matches = 0
        total_match_weight = 0.0

        def add_lineup(match: dict[str, Any], weight: float, include_formation: bool = False) -> bool:
            nonlocal total_match_weight
            lineup = self._fixture_lineup(match, team)
            starters = lineup.get("starters") or []
            if len(starters) < 8:
                return False
            total_match_weight += weight
            if include_formation and lineup.get("formation"):
                formation_scores[str(lineup["formation"])] += weight
            for player in starters:
                name = str(player.get("name") or "").strip()
                if not name:
                    continue
                key = normalize_provider_name(name)
                if active_keys and key not in active_keys:
                    filtered_players[name] += 1
                    continue
                slot = self._formation_slot(player)
                player_names.setdefault(key, name)
                player_scores[key] += weight
                if slot is None:
                    continue
                slot_scores[slot][key] += weight
                raw_position = str(player.get("position") or "").strip()
                if raw_position:
                    slot_positions[(slot, key)][raw_position] += 1
            return True

        for index, match in enumerate(recent):
            recency_weight = max(0.38, 1.0 - index * 0.07)
            competition_weight = 1.12 if match.get("competition_code") == target_competition else 1.0
            if add_lineup(match, recency_weight * competition_weight, include_formation=True):
                usable_matches += 1

        # Last-ten form can overvalue a short rotation spell. A lighter season
        # prior keeps established starters ahead in important fixtures.
        for index, match in enumerate(self._team_matches(team, history)[:30]):
            season_weight = max(0.09, 0.18 - index * 0.003)
            add_lineup(match, season_weight, include_formation=False)

        high_stakes = self._is_high_stakes_fixture(fixture)
        if high_stakes:
            opponent = self._other_team(fixture, team)
            head_to_head = [
                match
                for match in self._team_matches(team, history)
                if self._fixture_has_team(match, opponent)
            ][:6]
            for index, match in enumerate(head_to_head):
                add_lineup(match, max(0.18, 0.58 - index * 0.08), include_formation=False)

        # New signings have no Barcelona start history yet. Keep them in the
        # positional candidate pool with a conservative prior until match data
        # replaces it.
        if active_players:
            newcomer_score = max(0.16, total_match_weight * 0.025)
            for key, player in active_players.items():
                if key in player_scores:
                    continue
                raw_position = str(player.get("position") or "")
                compatible_slots = self._generic_position_slots(raw_position)
                if not compatible_slots:
                    continue
                player_scores[key] = newcomer_score
                for slot in compatible_slots:
                    slot_scores[slot][key] += newcomer_score
                    slot_positions[(slot, key)][raw_position] += 1

        formation = max(formation_scores, key=formation_scores.get) if formation_scores else None
        assignments: dict[int, tuple[str, float]] = {}
        assigned_players: set[str] = set()
        ranked_pairs = sorted(
            (
                (score, slot, player)
                for slot, candidates in slot_scores.items()
                for player, score in candidates.items()
            ),
            reverse=True,
        )
        for score, slot, player in ranked_pairs:
            if slot in assignments or player in assigned_players:
                continue
            assignments[slot] = (player, score)
            assigned_players.add(player)

        # ESPN occasionally omits formationPlace for one player. Fill remaining
        # slots from the strongest unassigned regulars without duplicating names.
        remaining_players = [
            (score, player)
            for player, score in sorted(player_scores.items(), key=lambda item: item[1], reverse=True)
            if player not in assigned_players
        ]
        for slot in range(1, 12):
            if slot in assignments or not remaining_players:
                continue
            score, player = remaining_players.pop(0)
            assignments[slot] = (player, score)
            assigned_players.add(player)

        players = []
        confidences = []
        for slot in range(1, 12):
            assigned = assignments.get(slot)
            if not assigned:
                continue
            player, score = assigned
            slot_total = sum(slot_scores.get(slot, {}).values()) or score or 1.0
            slot_share = score / slot_total
            appearance_rate = score / total_match_weight if total_match_weight else 0.0
            probability = _clip(0.55 * slot_share + 0.45 * appearance_rate, 0.05, 0.98)
            position_counts = slot_positions.get((slot, player), Counter())
            raw_position = position_counts.most_common(1)[0][0] if position_counts else ""
            alternatives = [
                {
                    "name": player_names.get(candidate, candidate),
                    "probability": round(_clip(candidate_score / slot_total, 0.03, 0.95), 3),
                }
                for candidate, candidate_score in sorted(
                    slot_scores.get(slot, {}).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if candidate != player
            ][:2]
            players.append(
                {
                    "name": player_names.get(player, player),
                    "position": self._translated_position(raw_position, slot),
                    "raw_position": raw_position,
                    "formation_place": str(slot),
                    "probability": round(probability, 3),
                    "alternatives": alternatives,
                    "source": "last-10-lineup-projection",
                }
            )
            confidences.append(probability)

        return {
            "available": len(players) >= 8,
            "formation": formation,
            "players": players,
            "confidence": round(mean(confidences), 3) if confidences else 0.0,
            "sample_matches": usable_matches,
            "active_roster_applied": bool(active_players),
            "current_roster_size": len(active_players),
            "roster_source": squad.get("source"),
            "roster_source_url": squad.get("source_url"),
            "recent_signings": squad.get("recent_signings") or [],
            "filtered_departures": [name for name, _ in filtered_players.most_common(8)],
            "selection_context": (
                "Класико или топ-матч: усилен вес сезонной основы и предыдущих очных встреч."
                if high_stakes
                else "Свежие старты дополнены сезонной частотой и актуальной заявкой."
            ),
            "method": (
                "Текущий ростер фильтрует кандидатов; затем учитываются последние 10 матчей, сезонная основа и важность встречи."
                if active_players
                else "Учитываются последние 10 матчей, сезонная основа и важность встречи; актуальный ростер команды пока не загружен."
            ),
        }

    def _current_squad(self, fixture: dict[str, Any], team: str) -> dict[str, Any]:
        contexts = fixture.get("squad_context") or {}
        query = normalize_provider_name(team)
        for name, payload in contexts.items():
            if normalize_provider_name(str(name)) == query and isinstance(payload, dict):
                return payload
        return {}

    def _generic_position_slots(self, raw_position: str) -> list[int]:
        normalized = normalize_provider_name(raw_position)
        if "goalkeeper" in normalized:
            return [1]
        if "defender" in normalized:
            return [2, 3, 5, 6]
        if "midfielder" in normalized:
            return [4, 8, 10]
        if any(token in normalized for token in ("forward", "striker", "winger")):
            return [7, 9, 11]
        return []

    def _is_high_stakes_fixture(self, fixture: dict[str, Any]) -> bool:
        teams = {
            normalize_provider_name(str(fixture.get("home_team") or "")),
            normalize_provider_name(str(fixture.get("away_team") or "")),
        }
        return teams == {"barcelona", "real madrid"} or (
            fixture.get("competition_code") == "uefa.champions" and self._is_knockout(fixture)
        )

    def _other_team(self, fixture: dict[str, Any], team: str) -> str:
        query = normalize_provider_name(team)
        home = str(fixture.get("home_team") or "")
        away = str(fixture.get("away_team") or "")
        return away if normalize_provider_name(home) == query else home

    def _fixture_has_team(self, fixture: dict[str, Any], team: str) -> bool:
        query = normalize_provider_name(team)
        return any(
            normalize_provider_name(str(fixture.get(key) or "")) == query
            for key in ("home_team", "away_team")
        )

    def _prepared_confirmed_players(self, starters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        for player in starters:
            slot = self._formation_slot(player)
            prepared.append(
                {
                    "name": player.get("name"),
                    "position": self._translated_position(str(player.get("position") or ""), slot),
                    "raw_position": player.get("position"),
                    "formation_place": None if slot is None else str(slot),
                    "probability": 1.0,
                    "alternatives": [],
                    "source": "espn-confirmed-lineup",
                }
            )
        return sorted(prepared, key=lambda item: int(item.get("formation_place") or 99))

    def _formation_slot(self, player: dict[str, Any]) -> int | None:
        raw = player.get("formation_place")
        try:
            slot = int(str(raw))
            return slot if 1 <= slot <= 11 else None
        except (TypeError, ValueError):
            return None

    def _translated_position(self, raw_position: str, slot: int | None) -> str:
        normalized = normalize_provider_name(raw_position)
        if "goalkeeper" in normalized:
            return "Вратарь"
        if "right back" in normalized:
            return "Правый защитник"
        if "left back" in normalized:
            return "Левый защитник"
        if "center" in normalized and "defender" in normalized:
            return "Центральный защитник"
        if "defender" in normalized:
            return "Защитник"
        if "attacking midfielder left" in normalized:
            return "Левый вингер"
        if "attacking midfielder right" in normalized:
            return "Правый вингер"
        if "attacking midfielder" in normalized:
            return "Атакующий полузащитник"
        if "defensive midfielder" in normalized:
            return "Опорный полузащитник"
        if "left midfielder" in normalized:
            return "Левый полузащитник"
        if "right midfielder" in normalized:
            return "Правый полузащитник"
        if "midfielder" in normalized:
            return "Центральный полузащитник"
        if any(token in normalized for token in ("forward", "striker")):
            return "Нападающий"
        slot_labels = {
            1: "Вратарь",
            2: "Правый защитник",
            3: "Левый защитник",
            4: "Центральный полузащитник",
            5: "Центральный защитник",
            6: "Центральный защитник",
            7: "Правый вингер",
            8: "Центральный полузащитник",
            9: "Нападающий",
            10: "Атакующий полузащитник",
            11: "Левый вингер",
        }
        return slot_labels.get(slot, raw_position or "Позиция не определена")

    def _regular_player_counts(self, team: str, history: list[dict[str, Any]]) -> dict[str, float]:
        counts: defaultdict[str, float] = defaultdict(float)
        for index, fixture in enumerate(self._team_matches(team, history)[:8]):
            lineup = self._fixture_lineup(fixture, team)
            weight = 1.0 - index * 0.06
            for player in lineup.get("starters", []):
                key = normalize_provider_name(player.get("name", ""))
                if key:
                    counts[key] += weight
        return dict(counts)

    def _regular_player_names(self, team: str, history: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = {}
        for fixture in self._team_matches(team, history)[:8]:
            for player in self._fixture_lineup(fixture, team).get("starters", []):
                if player.get("name"):
                    result[normalize_provider_name(player["name"])] = player["name"]
        return result

    def _fixture_lineup(self, fixture: dict[str, Any], team: str) -> dict[str, Any]:
        lineups = fixture.get("lineups") or {}
        query = normalize_provider_name(team)
        for name, lineup in lineups.items():
            if normalize_provider_name(name) == query:
                return lineup
        return {"team": team, "confirmed": False, "formation": None, "starters": [], "source": "not-released"}

    def _team_formation(self, fixture: dict[str, Any], team: str) -> str | None:
        return self._fixture_lineup(fixture, team).get("formation")

    def _referee_report(self, fixture: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        referee = fixture.get("referee") or {}
        name = referee.get("name") if isinstance(referee, dict) else None
        if not name:
            return {"name": None, "matches": 0, "avg_fouls": None, "message": "Судья еще не назначен или не опубликован."}
        values = []
        for item in history:
            item_ref = item.get("referee") or {}
            if normalize_provider_name(str(item_ref.get("name") or "")) != normalize_provider_name(str(name)):
                continue
            total = _sum_stats(item.get("home_stats") or {}, item.get("away_stats") or {}, "foulsCommitted", item, "home_fouls", "away_fouls")
            if total is not None:
                values.append(total)
        return {
            "name": name,
            "matches": len(values),
            "avg_fouls": round(mean(values), 2) if values else None,
            "source": referee.get("source", "espn-summary-officials"),
            "message": "Среднее рассчитано только по загруженным протоколам Ла Лиги и ЛЧ.",
        }

    def _referee_average_fouls(self, fixture: dict[str, Any], history: list[dict[str, Any]]) -> float | None:
        referee = fixture.get("referee") or {}
        name = referee.get("name") if isinstance(referee, dict) else None
        if not name:
            return None
        values = []
        for item in history:
            item_referee = item.get("referee") or {}
            if normalize_provider_name(str(item_referee.get("name") or "")) != normalize_provider_name(str(name)):
                continue
            total = _sum_stats(
                item.get("home_stats") or {},
                item.get("away_stats") or {},
                "foulsCommitted",
                item,
                "home_fouls",
                "away_fouls",
            )
            if total is not None:
                values.append(total)
        return mean(values) if values else None

    def _market_context(self, fixture: dict[str, Any]) -> dict[str, Any]:
        market = fixture.get("market") or {}
        probabilities = market.get("probabilities") or {}
        if not probabilities:
            return {}
        barca_home = self._barca_home(fixture)
        return {
            "probabilities": {
                "barcelona": probabilities.get("home" if barca_home else "away"),
                "draw": probabilities.get("draw"),
                "opponent": probabilities.get("away" if barca_home else "home"),
            },
            "total_line": market.get("total_line"),
            "provider": market.get("provider"),
            "source": market.get("source"),
        }

    def _recent_match_total(self, history: list[dict[str, Any]], barca: str, opponent: str, stat: str) -> float | None:
        values = []
        for team in (barca, opponent):
            for item in self._team_matches(team, history)[:10]:
                total = _sum_stats(item.get("home_stats") or {}, item.get("away_stats") or {}, stat, item, "home_corners" if stat == "wonCorners" else "home_fouls", "away_corners" if stat == "wonCorners" else "away_fouls")
                if total is not None:
                    values.append(total)
        return median(values) if values else None

    def _match_row(self, fixture: dict[str, Any], team: str) -> dict[str, Any]:
        scored, conceded = self._team_score(fixture, team)
        opponent = fixture.get("away_team") if normalize_provider_name(str(fixture.get("home_team"))) == normalize_provider_name(team) else fixture.get("home_team")
        return {
            "date": fixture.get("date"),
            "opponent": opponent,
            "score": f"{scored}-{conceded}",
            "result": "W" if scored > conceded else "D" if scored == conceded else "L",
            "competition": fixture.get("competition"),
            "formation": self._team_formation(fixture, team),
        }

    def _team_score(self, fixture: dict[str, Any], team: str) -> tuple[int, int]:
        is_home = normalize_provider_name(str(fixture.get("home_team") or "")) == normalize_provider_name(team)
        return (
            int(fixture["home_goals"] if is_home else fixture["away_goals"]),
            int(fixture["away_goals"] if is_home else fixture["home_goals"]),
        )

    def _team_stats(self, fixture: dict[str, Any], team: str) -> tuple[dict[str, Any], dict[str, Any]]:
        is_home = normalize_provider_name(str(fixture.get("home_team") or "")) == normalize_provider_name(team)
        return (
            fixture.get("home_stats") or {} if is_home else fixture.get("away_stats") or {},
            fixture.get("away_stats") or {} if is_home else fixture.get("home_stats") or {},
        )

    def opponent_name(self, fixture: dict[str, Any]) -> str:
        return str(fixture.get("away_team") if self._barca_home(fixture) else fixture.get("home_team"))

    def _barca_home(self, fixture: dict[str, Any]) -> bool:
        return str(fixture.get("home_team_id")) == BARCELONA_ID or normalize_provider_name(str(fixture.get("home_team") or "")) == "barcelona"

    def _has_team(self, fixture: dict[str, Any], team: str) -> bool:
        query = normalize_provider_name(team)
        return any(normalize_provider_name(str(fixture.get(key) or "")) == query for key in ("home_team", "away_team"))

    def _is_knockout(self, fixture: dict[str, Any]) -> bool:
        stage = normalize_provider_name(str(fixture.get("stage") or ""))
        return any(token in stage for token in ("round of", "quarter", "semi", "final", "1 8", "1 4", "1 2"))

    def _outcome(self, barca: int, opponent: int) -> str:
        return "barcelona" if barca > opponent else "draw" if barca == opponent else "opponent"

    def _style_label(self, possession: float, tempo: float, width: float) -> str:
        if possession >= 0.59:
            base = "позиционный контроль"
        elif tempo >= 0.65:
            base = "вертикальная игра"
        else:
            base = "смешанный темп"
        return f"{base}, {'акцент на фланги' if width >= 0.62 else 'комбинации через центр'}"

    def _fixture_status(self, fixture: dict[str, Any]) -> dict[str, Any]:
        if fixture.get("completed"):
            state = "finished"
            label = "Матч завершен"
        elif fixture.get("in_progress"):
            state = "live"
            label = "Матч идет"
        else:
            state = "scheduled"
            label = "Матч еще не начался"
        return {"state": state, "label": label, "detail": fixture.get("status_detail") or fixture.get("status") or ""}

    def _result_summary(self, fixture: dict[str, Any], prediction: dict[str, Any] | None) -> dict[str, Any]:
        barca_home = self._barca_home(fixture)
        barca_goals = fixture.get("home_goals") if barca_home else fixture.get("away_goals")
        opponent_goals = fixture.get("away_goals") if barca_home else fixture.get("home_goals")
        actual = None if barca_goals is None or opponent_goals is None else f"{barca_goals}-{opponent_goals}"
        return {
            "prediction": (prediction or {}).get("exact_score", {}).get("score") if prediction else None,
            "actual": actual,
            "orientation": "Барселона — соперник",
            "status": self._fixture_status(fixture)["state"],
        }


def _sum_stats(home: dict[str, Any], away: dict[str, Any], stat: str, fixture: dict[str, Any], home_fallback: str, away_fallback: str) -> float | None:
    first = home.get(stat, fixture.get(home_fallback))
    second = away.get(stat, fixture.get(away_fallback))
    if first is None or second is None:
        return None
    return float(first) + float(second)


def _poisson(goals: int, expected: float) -> float:
    return math.exp(-expected) * expected**goals / math.factorial(goals)


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values()) or 1.0
    return {key: value / total for key, value in values.items()}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normal_cdf(value: float, expected: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if value >= expected else 0.0
    return 0.5 * (1.0 + math.erf((value - expected) / (sigma * math.sqrt(2.0))))


def _mad(values: list[float], center: float) -> float:
    if not values:
        return 2.0
    return median(abs(value - center) for value in values) or 1.0


def _average(values: Any) -> float | None:
    items = list(values)
    return round(mean(items), 3) if items else None


def _rate(values: Any) -> float | None:
    items = list(values)
    return round(sum(bool(value) for value in items) / len(items), 4) if items else None


def _parse_time(raw: str) -> datetime:
    value = raw if "T" in raw else f"{raw}T12:00:00Z"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
