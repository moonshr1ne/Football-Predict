from __future__ import annotations

import copy
import math
import threading
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .data_store import DEFAULT_MODEL_STATE, DataStore
from .models import MatchRecord
from .providers import EspnWorldCupProvider, ProviderError
from .tactics import clamp01


MODEL_TRAINING_LOCK = threading.Lock()


class WorldCupDataSync:
    def __init__(self, store: DataStore, provider: EspnWorldCupProvider | None = None):
        self.store = store
        self.provider = provider or EspnWorldCupProvider()

    def sync_finished(self, end_date: str | None = None) -> dict[str, Any]:
        end_date = end_date or date.today().isoformat()
        try:
            fixtures = self.provider.fixtures(start_date="2026-06-01", end_date=end_date)
        except ProviderError:
            return {
                "participants": len(self.store.load_participants()),
                "recent_imported": 0,
                "imported": 0,
                "profiles_updated": 0,
                "referees_updated": 0,
                "trained": 0,
                "source": "espn-world-cup",
                "error": "provider unavailable",
            }

        existing_matches = {
            (match.date, match.home_team, match.away_team): match
            for match in self.store.load_matches()
        }
        imported = 0
        for fixture in fixtures:
            if not fixture.get("completed"):
                continue
            existing = existing_matches.get((fixture.get("date"), fixture.get("home_team"), fixture.get("away_team")))
            if existing is None or not (
                existing.home_formation
                and existing.away_formation
                and existing.home_lineup_confirmed
                and existing.away_lineup_confirmed
            ) or not existing.referee:
                fixture = self.provider.enrich_fixture(fixture)
            record = self._record_from_fixture(fixture)
            self.store.add_or_update_match(record)
            imported += 1

        profiles_updated = self._update_tactical_profiles()
        referees_updated = self._update_referee_profiles()
        trained = self._train_model_from_imported_matches()
        self._save_backtest_summary()
        return {
            "participants": len(self.store.load_participants()),
            "recent_imported": 0,
            "imported": imported,
            "profiles_updated": profiles_updated,
            "referees_updated": referees_updated,
            "trained": trained,
            "source": "espn-world-cup",
        }

    def sync_all(self, force: bool = False, max_age_hours: int = 6) -> dict[str, Any]:
        finished_summary = self.sync_finished()
        if not force and not self._full_sync_due(max_age_hours):
            finished_summary["skipped_full_sync"] = True
            return finished_summary

        errors: list[str] = []
        try:
            participants = self.provider.participants()
        except ProviderError as exc:
            participants = self.store.load_participants()
            errors.append(f"participants: {exc}")
        self.store.save_participants(participants)

        recent_imported = 0
        for participant in participants:
            team_id = participant.get("team_id")
            if not team_id:
                continue
            try:
                fixtures = self.provider.team_recent_fixtures(str(team_id), limit=10)
            except Exception as exc:
                errors.append(f"{participant.get('team')}: {exc}")
                continue
            for fixture in fixtures:
                self.store.add_or_update_match(self._record_from_fixture(fixture))
                recent_imported += 1

        profiles_updated = self._update_tactical_profiles()
        referees_updated = self._update_referee_profiles()
        trained = self._train_model_from_imported_matches()
        backtest = self._save_backtest_summary()
        self.store.save_sync_state(
            {
                "last_full_sync_at": datetime.now(timezone.utc).isoformat(),
                "participants": len(participants),
                "recent_imported": recent_imported,
                "referees_updated": referees_updated,
                "backtest": backtest,
            }
        )
        return {
            "participants": len(participants),
            "recent_imported": recent_imported,
            "imported": finished_summary.get("imported", 0),
            "profiles_updated": profiles_updated,
            "referees_updated": referees_updated,
            "trained": trained,
            "backtest": backtest,
            "errors": errors[:10],
            "source": "espn-world-cup+espn-team-schedule",
        }

    def _record_from_fixture(self, fixture: dict[str, Any]) -> MatchRecord:
        return MatchRecord(
            date=fixture["date"],
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            fixture_id=fixture.get("fixture_id"),
            home_goals=fixture.get("home_goals"),
            away_goals=fixture.get("away_goals"),
            home_corners=fixture.get("home_corners"),
            away_corners=fixture.get("away_corners"),
            home_possession=fixture.get("home_possession"),
            away_possession=fixture.get("away_possession"),
            home_shots=fixture.get("home_shots"),
            away_shots=fixture.get("away_shots"),
            home_shots_on_target=fixture.get("home_shots_on_target"),
            away_shots_on_target=fixture.get("away_shots_on_target"),
            home_fouls=fixture.get("home_fouls"),
            away_fouls=fixture.get("away_fouls"),
            referee=self._fixture_referee_name(fixture),
            competition=fixture.get("competition") or "ESPN soccer",
            stage=fixture.get("status_detail", ""),
            neutral=True,
            source=fixture.get("source", "espn-world-cup"),
            home_formation=fixture.get("home_formation"),
            away_formation=fixture.get("away_formation"),
            home_lineup_confirmed=bool(fixture.get("home_formation")),
            away_lineup_confirmed=bool(fixture.get("away_formation")),
        )

    @staticmethod
    def _fixture_referee_name(fixture: dict[str, Any]) -> str | None:
        referee = fixture.get("referee")
        if isinstance(referee, dict):
            return referee.get("name")
        if isinstance(referee, str):
            return referee
        return None

    def _update_tactical_profiles(self) -> int:
        matches = [
            match
            for match in self.store.load_matches()
            if match.is_finished()
            and match.home_possession is not None
            and match.away_possession is not None
            and match.home_shots is not None
            and match.away_shots is not None
        ]
        aggregates: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        weights_by_team: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        formation_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        formation_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for match in sorted(matches, key=lambda item: item.date, reverse=True):
            for team in (match.home_team, match.away_team):
                recency_rank = counts[team]
                weight = max(0.35, 1.0 - recency_rank * 0.07)
                counts[team] += 1
                weights_by_team[team] += weight
                gf = match.goals_for(team) or 0
                ga = match.goals_against(team) or 0
                corners_for = match.corners_for(team) or 0.0
                corners_against = match.corners_against(team) or 0.0
                possession = match.possession_for(team)
                shots_for = match.shots_for(team) or 0.0
                shots_against = match.shots_against(team) or 0.0
                sot = match.shots_on_target_for(team) or 0.0

                aggregates[team]["goals_for"] += gf * weight
                aggregates[team]["goals_against"] += ga * weight
                aggregates[team]["corners_for"] += corners_for * weight
                aggregates[team]["corners_against"] += corners_against * weight
                aggregates[team]["possession"] += (possession if possession is not None else 50.0) * weight
                aggregates[team]["shots_for"] += shots_for * weight
                aggregates[team]["shots_against"] += shots_against * weight
                aggregates[team]["sot"] += sot * weight
                formation = match.formation_for(team)
                if formation:
                    formation_weights[team][formation] += weight
                    formation_history[team].append(
                        {
                            "date": match.date,
                            "opponent": match.away_team if match.home_team == team else match.home_team,
                            "formation": formation,
                            "source": "confirmed-lineup" if match.lineup_confirmed_for(team) else "match-record",
                        }
                    )

        profiles = self.store.load_tactical_profiles()
        updated = 0
        for team, count in counts.items():
            if count <= 0:
                continue
            total_weight = weights_by_team[team] or count
            avg = {key: value / total_weight for key, value in aggregates[team].items()}
            existing = profiles.setdefault(team, {})
            existing.update(
                self._profile_from_averages(
                    existing,
                    avg,
                    count,
                    formation_weights.get(team, {}),
                    formation_history.get(team, []),
                )
            )
            profiles[team] = existing
            updated += 1

        self.store.save_tactical_profiles(profiles)
        return updated

    def _update_referee_profiles(self) -> int:
        aggregates: dict[str, dict[str, Any]] = {}
        for match in sorted(self.store.load_matches(), key=lambda item: item.date, reverse=True):
            if (
                not match.is_finished()
                or not match.referee
                or match.home_fouls is None
                or match.away_fouls is None
            ):
                continue
            total_fouls = float(match.home_fouls) + float(match.away_fouls)
            profile = aggregates.setdefault(
                match.referee,
                {
                    "name": match.referee,
                    "matches": 0,
                    "total_fouls": 0.0,
                    "recent": [],
                    "source": "espn-world-cup-derived",
                },
            )
            profile["matches"] += 1
            profile["total_fouls"] += total_fouls
            if len(profile["recent"]) < 10:
                profile["recent"].append(
                    {
                        "date": match.date,
                        "match": f"{match.home_team} - {match.away_team}",
                        "fouls": round(total_fouls, 2),
                    }
                )

        profiles = {}
        for name, profile in sorted(aggregates.items()):
            matches = int(profile["matches"])
            profiles[name] = {
                "name": name,
                "matches": matches,
                "avg_fouls": round(float(profile["total_fouls"]) / matches, 2),
                "recent": profile["recent"],
                "source": profile["source"],
            }
        self.store.save_referee_profiles(profiles)
        return len(profiles)

    def _train_model_from_imported_matches(self) -> int:
        matches = self._training_matches(espn_only=False)
        state = self.store.load_model_state()
        available = {self._match_key(match): self._match_fingerprint(match) for match in matches}
        trained_fingerprints = state.get("trained_match_fingerprints") or {}
        changed_keys = {
            key
            for key, fingerprint in available.items()
            if trained_fingerprints.get(key) != fingerprint
        }
        training = state.get("training") or {}
        needs_upgrade = training.get("evaluation_mode") != "walk_forward_strict_date"
        if not changed_keys and not needs_upgrade:
            return 0

        with MODEL_TRAINING_LOCK:
            state = self.store.load_model_state()
            trained_fingerprints = state.get("trained_match_fingerprints") or {}
            changed_keys = {
                key
                for key, fingerprint in available.items()
                if trained_fingerprints.get(key) != fingerprint
            }
            training = state.get("training") or {}
            needs_upgrade = training.get("evaluation_mode") != "walk_forward_strict_date"
            if not changed_keys and not needs_upgrade:
                return 0
            self.retrain_model_from_history(epochs=12, automatic=True)
            return len(changed_keys) if changed_keys else len(matches)

    def retrain_model_from_history(self, epochs: int = 2, automatic: bool = False) -> dict[str, Any]:
        return self._retrain_walk_forward(epochs=epochs, automatic=automatic)

    def _retrain_walk_forward(self, epochs: int, automatic: bool) -> dict[str, Any]:
        from .predictor import MatchPredictor

        epochs = max(2, min(int(epochs or 2), 80))
        matches = self._training_matches(espn_only=False)
        started_at = datetime.now(timezone.utc).isoformat()
        base_state = copy.deepcopy(DEFAULT_MODEL_STATE)
        rows = self._strict_feature_rows(matches, base_state)
        reviews = self._walk_forward_reviews(matches, rows, base_state, epochs)

        state = copy.deepcopy(DEFAULT_MODEL_STATE)
        state["weights"] = copy.deepcopy(DEFAULT_MODEL_STATE["weights"])
        state["stat_profiles"] = self._fit_stat_profiles(matches)
        state["score_profiles"] = self._score_profiles_from_rows(rows)
        state["calibration_profiles"] = self._fit_calibration_profiles(rows)
        state["outcome_model"] = self._fit_outcome_model_from_rows(rows, epochs)
        state["history"] = reviews[-5000:]
        state["trained_match_keys"] = sorted({self._match_key(match) for match in matches})
        state["trained_match_fingerprints"] = {
            self._match_key(match): self._match_fingerprint(match)
            for match in matches
        }
        completed_at = datetime.now(timezone.utc).isoformat()
        state["training"] = {
            "mode": "automatic_walk_forward" if automatic else "walk_forward",
            "status": "complete",
            "epochs": epochs,
            "unique_matches": len(matches),
            "training_rows": len(rows),
            "evaluation_matches": len(reviews),
            "evaluation_scope": "FIFA World Cup 2026 finals",
            "evaluation_mode": "walk_forward_strict_date",
            "same_day_policy": "all matches on a date are predicted before any result from that date is learned",
            "result_leakage_guard": True,
            "started_at": started_at,
            "completed_at": completed_at,
        }
        self.store.save_model_state(state)
        backtest = self._save_backtest_summary()
        summary = {
            "trained": epochs * len(rows),
            "unique_matches": len(matches),
            "epochs": epochs,
            "automatic": automatic,
            "kept_previous": False,
            "backtest": backtest,
            "source": "strict-walk-forward",
        }
        sync_state = self.store.load_sync_state()
        sync_state["last_retrain_at"] = completed_at
        sync_state["last_retrain"] = summary
        sync_state["backtest"] = backtest
        self.store.save_sync_state(sync_state)
        return summary

    def _strict_feature_rows(
        self,
        matches: list[MatchRecord],
        base_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        from .predictor import MatchPredictor

        predictor = MatchPredictor(self.store)
        rows: list[dict[str, Any]] = []
        past: list[MatchRecord] = []
        for match_date, group in self._matches_by_date(matches):
            snapshot = copy.deepcopy(base_state)
            snapshot["stat_profiles"] = self._fit_stat_profiles(past)
            snapshot["score_profiles"] = self._score_profiles_from_rows(rows)
            snapshot["calibration_profiles"] = self._fit_calibration_profiles(rows)
            snapshot["history"] = MatchPredictor._history_from_matches(past)
            for match in group:
                prediction = predictor.predict(
                    match.home_team,
                    match.away_team,
                    neutral=match.neutral,
                    remember=False,
                    match_date=match_date,
                    fixture=self._training_fixture(match),
                    matches_override=past,
                    model_state_override=snapshot,
                    temporal_snapshot=True,
                )
                features = predictor._outcome_features(
                    match.home_team,
                    match.away_team,
                    prediction.home_stats,
                    prediction.away_stats,
                    prediction.home_tactics,
                    prediction.away_tactics,
                    prediction.expected_home_goals,
                    prediction.expected_away_goals,
                    match.neutral,
                    prediction.h2h_report,
                )
                rows.append(
                    {
                        "key": self._match_key(match),
                        "date": match.date,
                        "match": match,
                        "features": features,
                        "label": MatchPredictor._outcome_from_score(
                            int(match.home_goals or 0),
                            int(match.away_goals or 0),
                        ),
                        "home_xg": prediction.expected_home_goals,
                        "away_xg": prediction.expected_away_goals,
                        "round_info": prediction.round_info,
                        "predicted_goals": prediction.expected_home_goals + prediction.expected_away_goals,
                        "actual_goals": int(match.home_goals or 0) + int(match.away_goals or 0),
                        "predicted_corners": prediction.predicted_corners,
                        "actual_corners": (
                            None
                            if match.home_corners is None or match.away_corners is None
                            else float(match.home_corners) + float(match.away_corners)
                        ),
                        "predicted_fouls": (prediction.foul_forecast or {}).get("expected"),
                        "actual_fouls": (
                            None
                            if match.home_fouls is None or match.away_fouls is None
                            else float(match.home_fouls) + float(match.away_fouls)
                        ),
                    }
                )
            past.extend(group)
        return rows

    def _walk_forward_reviews(
        self,
        matches: list[MatchRecord],
        rows: list[dict[str, Any]],
        base_state: dict[str, Any],
        epochs: int,
    ) -> list[dict[str, Any]]:
        from .predictor import MatchPredictor

        predictor = MatchPredictor(self.store)
        rows_by_key = {row["key"]: row for row in rows}
        world_cup_keys = {
            self._match_key(match)
            for match in matches
            if self._is_world_cup_finals_match(match)
        }
        evaluation_keys = world_cup_keys or {self._match_key(match) for match in matches}
        reviews: list[dict[str, Any]] = []
        prior_rows: list[dict[str, Any]] = []
        past: list[MatchRecord] = []

        for match_date, group in self._matches_by_date(matches):
            snapshot = copy.deepcopy(base_state)
            snapshot["stat_profiles"] = self._fit_stat_profiles(past)
            snapshot["score_profiles"] = self._score_profiles_from_rows(prior_rows)
            snapshot["calibration_profiles"] = self._fit_calibration_profiles(prior_rows)
            snapshot["history"] = MatchPredictor._history_from_matches(past)
            snapshot["outcome_model"] = self._fit_outcome_model_from_rows(prior_rows, epochs)
            snapshot["training"] = {
                "evaluation_mode": "walk_forward_strict_date",
                "trained_until": past[-1].date if past else None,
            }

            for match in group:
                key = self._match_key(match)
                if key not in evaluation_keys:
                    continue
                prediction = predictor.predict(
                    match.home_team,
                    match.away_team,
                    neutral=match.neutral,
                    remember=False,
                    match_date=match_date,
                    fixture=self._training_fixture(match),
                    matches_override=past,
                    model_state_override=snapshot,
                    temporal_snapshot=True,
                )
                actual_score = f"{int(match.home_goals or 0)}-{int(match.away_goals or 0)}"
                actual_outcome = MatchPredictor._outcome_from_score(
                    int(match.home_goals or 0),
                    int(match.away_goals or 0),
                )
                actual_corners = (
                    None
                    if match.home_corners is None or match.away_corners is None
                    else float(match.home_corners) + float(match.away_corners)
                )
                actual_fouls = (
                    None
                    if match.home_fouls is None or match.away_fouls is None
                    else float(match.home_fouls) + float(match.away_fouls)
                )
                predicted_fouls = prediction.foul_forecast.get("expected")
                predicted_foul_count = int(
                    prediction.foul_forecast.get(
                        "point_estimate",
                        math.floor(float(predicted_fouls or 0.0) + 0.5),
                    )
                )
                predicted_corner_count = int(math.floor(prediction.predicted_corners + 0.5))
                predicted_goals = prediction.expected_home_goals + prediction.expected_away_goals
                predicted_goal_count = int(prediction.goal_total.get("point_estimate", math.floor(predicted_goals + 0.5)))
                actual_goals = int(match.home_goals or 0) + int(match.away_goals or 0)
                reviews.append(
                    {
                        "date": match.date,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                        "predicted_outcome": prediction.market_pick,
                        "probabilities": {
                            "П1": round(prediction.home_win_probability, 4),
                            "X": round(prediction.draw_probability, 4),
                            "П2": round(prediction.away_win_probability, 4),
                        },
                        "expected_goals": {
                            match.home_team: round(prediction.expected_home_goals, 3),
                            match.away_team: round(prediction.expected_away_goals, 3),
                        },
                        "predicted_goals": round(predicted_goals, 3),
                        "predicted_goal_count": predicted_goal_count,
                        "actual_goals": actual_goals,
                        "goal_error": predicted_goal_count - actual_goals,
                        "expected_goal_error": round(predicted_goals - actual_goals, 3),
                        "round_info": prediction.round_info,
                        "actual_outcome": actual_outcome,
                        "outcome_hit": prediction.market_pick == actual_outcome,
                        "predicted_scores": prediction.exact_scores[:1],
                        "actual_score": actual_score,
                        "score_hit": bool(prediction.exact_scores and prediction.exact_scores[0] == actual_score),
                        "predicted_corners": round(prediction.predicted_corners, 2),
                        "predicted_corner_count": predicted_corner_count,
                        "actual_corners": None if actual_corners is None else round(actual_corners, 2),
                        "corner_error": None if actual_corners is None else round(predicted_corner_count - actual_corners, 2),
                        "expected_corner_error": None if actual_corners is None else round(prediction.predicted_corners - actual_corners, 2),
                        "predicted_fouls": None if predicted_fouls is None else round(float(predicted_fouls), 2),
                        "predicted_foul_count": predicted_foul_count,
                        "actual_fouls": None if actual_fouls is None else round(actual_fouls, 2),
                        "foul_error": None if actual_fouls is None or predicted_fouls is None else round(predicted_foul_count - actual_fouls, 2),
                        "expected_foul_error": None if actual_fouls is None or predicted_fouls is None else round(float(predicted_fouls) - actual_fouls, 2),
                        "training_mode": "walk_forward_strict_date",
                        "training_cutoff": past[-1].date if past else None,
                        "training_key": key,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            prior_rows.extend(rows_by_key[self._match_key(match)] for match in group)
            past.extend(group)
        return reviews

    def _fit_outcome_model_from_rows(
        self,
        rows: list[dict[str, Any]],
        epochs: int,
    ) -> dict[str, Any]:
        from .predictor import MatchPredictor

        labels = [
            MatchPredictor._outcome_from_score(1, 0),
            MatchPredictor._outcome_from_score(1, 1),
            MatchPredictor._outcome_from_score(0, 1),
        ]
        weights: dict[str, dict[str, float]] = {label: {} for label in labels}
        if not rows:
            return {
                "type": "multiclass_logistic",
                "labels": labels,
                "epochs": 0,
                "learning_rate": 0.035,
                "blend": 0.0,
                "temperature": 1.0,
                "training_rows": 0,
                "weights": weights,
            }

        learning_rate = 0.035
        l2 = 0.0004
        for _ in range(max(1, epochs)):
            for row in rows:
                features = row["features"]
                actual = row["label"]
                scores = {
                    label: sum(weights[label].get(feature, 0.0) * value for feature, value in features.items())
                    for label in labels
                }
                top = max(scores.values())
                exponentials = {label: math.exp(max(-30.0, min(30.0, scores[label] - top))) for label in labels}
                total = sum(exponentials.values()) or 1.0
                probabilities = {label: exponentials[label] / total for label in labels}
                for label in labels:
                    error = (1.0 if label == actual else 0.0) - probabilities[label]
                    label_weights = weights[label]
                    for feature, value in features.items():
                        current = label_weights.get(feature, 0.0)
                        label_weights[feature] = current * (1.0 - learning_rate * l2) + learning_rate * error * value

        compact_weights = {
            label: {
                feature: round(weight, 6)
                for feature, weight in sorted(feature_weights.items())
                if abs(weight) >= 1e-7
            }
            for label, feature_weights in weights.items()
        }
        return {
            "type": "multiclass_logistic",
            "labels": labels,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "blend": 0.48,
            "temperature": 1.0,
            "training_rows": len(rows),
            "weights": compact_weights,
        }

    def _score_profiles_from_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        from .predictor import MatchPredictor

        by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
        by_outcome_bucket: dict[str, Counter[str]] = defaultdict(Counter)
        by_outcome_stage: dict[str, Counter[str]] = defaultdict(Counter)
        by_outcome_stage_bucket: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            match = row["match"]
            outcome = row["label"]
            score = f"{int(match.home_goals or 0)}-{int(match.away_goals or 0)}"
            bucket = MatchPredictor._score_profile_bucket(outcome, row["home_xg"], row["away_xg"])
            stage = "knockout" if (row.get("round_info") or {}).get("knockout") else "group"
            by_outcome[outcome][score] += 1
            by_outcome_bucket[f"{outcome}|{bucket}"][score] += 1
            by_outcome_stage[f"{outcome}|{stage}"][score] += 1
            by_outcome_stage_bucket[f"{outcome}|{stage}|{bucket}"][score] += 1
        return {
            "mode": "strict_pre_match_stage_feature_buckets",
            "by_outcome": {
                outcome: self._profile_items(counter)
                for outcome, counter in sorted(by_outcome.items())
            },
            "by_outcome_bucket": {
                key: self._profile_items(counter)
                for key, counter in sorted(by_outcome_bucket.items())
            },
            "by_outcome_stage": {
                key: self._profile_items(counter)
                for key, counter in sorted(by_outcome_stage.items())
            },
            "by_outcome_stage_bucket": {
                key: self._profile_items(counter)
                for key, counter in sorted(by_outcome_stage_bucket.items())
            },
        }

    def _fit_calibration_profiles(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        profiles: dict[str, Any] = {}
        definitions = {
            "goals": ("predicted_goals", "actual_goals"),
            "corners": ("predicted_corners", "actual_corners"),
            "fouls": ("predicted_fouls", "actual_fouls"),
        }
        for metric, (predicted_key, actual_key) in definitions.items():
            residuals: list[float] = []
            by_stage: dict[str, list[float]] = defaultdict(list)
            by_bin: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                predicted = row.get(predicted_key)
                actual = row.get(actual_key)
                if predicted is None or actual is None:
                    continue
                residual = float(actual) - float(predicted)
                residuals.append(residual)
                stage = "knockout" if (row.get("round_info") or {}).get("knockout") else "group"
                by_stage[stage].append(residual)
                by_bin[self._calibration_bin(metric, float(predicted))].append(residual)
            profiles[metric] = {
                "global": self._residual_profile(residuals),
                "by_stage": {
                    stage: self._residual_profile(values)
                    for stage, values in sorted(by_stage.items())
                },
                "by_bin": {
                    bucket: self._residual_profile(values)
                    for bucket, values in sorted(by_bin.items())
                },
            }
        return profiles

    @staticmethod
    def _calibration_bin(metric: str, value: float) -> str:
        width = {"goals": 0.5, "corners": 1.0, "fouls": 2.0}.get(metric, 1.0)
        center = round(value / width) * width
        return f"{center:.1f}"

    def _residual_profile(self, values: list[float]) -> dict[str, Any]:
        return {
            "matches": len(values),
            "median_error": self._median(values),
            "mean_error": self._mean(values),
        }

    @staticmethod
    def _matches_by_date(matches: list[MatchRecord]) -> list[tuple[str, list[MatchRecord]]]:
        grouped: dict[str, list[MatchRecord]] = defaultdict(list)
        for match in matches:
            grouped[match.date].append(match)
        return [
            (match_date, sorted(group, key=lambda item: (item.home_team, item.away_team)))
            for match_date, group in sorted(grouped.items())
        ]

    @staticmethod
    def _is_world_cup_finals_match(match: MatchRecord) -> bool:
        competition = (match.competition or "").lower()
        return (
            "world cup" in competition
            and "qualifying" not in competition
            and match.date >= "2026-06-01"
            and match.source.startswith("espn-world-cup")
        )

    def _fit_stat_profiles(self, matches: list[MatchRecord]) -> dict[str, Any]:
        corner_totals: list[float] = []
        corner_teams: dict[str, list[float]] = defaultdict(list)
        corner_for: dict[str, list[float]] = defaultdict(list)
        corner_against: dict[str, list[float]] = defaultdict(list)
        foul_totals: list[float] = []
        foul_teams: dict[str, list[float]] = defaultdict(list)
        foul_for: dict[str, list[float]] = defaultdict(list)
        foul_against: dict[str, list[float]] = defaultdict(list)
        foul_referees: dict[str, list[float]] = defaultdict(list)

        for match in matches:
            if match.home_corners is not None and match.away_corners is not None:
                total_corners = float(match.home_corners) + float(match.away_corners)
                corner_totals.append(total_corners)
                corner_teams[match.home_team].append(total_corners)
                corner_teams[match.away_team].append(total_corners)
                corner_for[match.home_team].append(float(match.home_corners))
                corner_for[match.away_team].append(float(match.away_corners))
                corner_against[match.home_team].append(float(match.away_corners))
                corner_against[match.away_team].append(float(match.home_corners))
            if match.home_fouls is not None and match.away_fouls is not None:
                total_fouls = float(match.home_fouls) + float(match.away_fouls)
                foul_totals.append(total_fouls)
                foul_teams[match.home_team].append(total_fouls)
                foul_teams[match.away_team].append(total_fouls)
                foul_for[match.home_team].append(float(match.home_fouls))
                foul_for[match.away_team].append(float(match.away_fouls))
                foul_against[match.home_team].append(float(match.away_fouls))
                foul_against[match.away_team].append(float(match.home_fouls))
                if match.referee:
                    foul_referees[match.referee].append(total_fouls)

        return {
            "corners": {
                "global": self._mean(corner_totals),
                "global_median": self._median(corner_totals),
                "teams": {
                    team: {
                        "matches": len(values),
                        "avg_total": self._mean(values),
                        "median_total": self._median(values),
                        "avg_for": self._mean(corner_for[team]),
                        "avg_against": self._mean(corner_against[team]),
                    }
                    for team, values in sorted(corner_teams.items())
                },
            },
            "fouls": {
                "global": self._mean(foul_totals),
                "global_median": self._median(foul_totals),
                "teams": {
                    team: {
                        "matches": len(values),
                        "avg_total": self._mean(values),
                        "median_total": self._median(values),
                        "avg_for": self._mean(foul_for[team]),
                        "avg_against": self._mean(foul_against[team]),
                    }
                    for team, values in sorted(foul_teams.items())
                },
                "referees": {
                    referee: {"matches": len(values), "avg_fouls": self._mean(values), "source": "trained-history"}
                    for referee, values in sorted(foul_referees.items())
                },
            },
        }

    @staticmethod
    def _profile_items(counter: Counter[str]) -> list[dict[str, Any]]:
        total = sum(counter.values()) or 1
        return [
            {"score": score, "count": count, "probability": round(count / total, 4)}
            for score, count in counter.most_common()
        ]

    @staticmethod
    def _training_fixture(match: MatchRecord) -> dict[str, Any]:
        fixture: dict[str, Any] = {
            "date": match.date,
            "competition": match.competition,
            "completed": False,
            "in_progress": False,
            "referee": {"name": match.referee, "source": match.source} if match.referee else None,
            "lineups": {},
        }
        if match.home_formation:
            fixture["lineups"][match.home_team] = {
                "confirmed": bool(match.home_lineup_confirmed),
                "formation": match.home_formation,
                "starters": [],
                "bench": [],
            }
        if match.away_formation:
            fixture["lineups"][match.away_team] = {
                "confirmed": bool(match.away_lineup_confirmed),
                "formation": match.away_formation,
                "starters": [],
                "bench": [],
            }
        return fixture

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 2)

    @staticmethod
    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return round(ordered[middle], 2)
        return round((ordered[middle - 1] + ordered[middle]) / 2.0, 2)

    def _training_matches(self, espn_only: bool = False) -> list[MatchRecord]:
        matches = []
        for match in self.store.load_matches():
            if not match.is_finished() or match.source == "demo_seed":
                continue
            if espn_only and not match.source.startswith("espn"):
                continue
            matches.append(match)
        return sorted(matches, key=lambda item: (item.date, item.home_team, item.away_team))

    @staticmethod
    def _match_key(match: MatchRecord) -> str:
        return f"{match.date}|{match.home_team}|{match.away_team}"

    @staticmethod
    def _match_fingerprint(match: MatchRecord) -> str:
        values = (
            match.home_goals,
            match.away_goals,
            match.home_corners,
            match.away_corners,
            match.home_fouls,
            match.away_fouls,
            match.home_possession,
            match.away_possession,
            match.home_shots,
            match.away_shots,
            match.home_shots_on_target,
            match.away_shots_on_target,
            match.referee,
            match.home_formation,
            match.away_formation,
            match.home_lineup_confirmed,
            match.away_lineup_confirmed,
        )
        return "|".join("" if value is None else str(value) for value in values)

    def _save_backtest_summary(self) -> dict[str, Any]:
        state = self.store.load_model_state()
        history = state.get("history", [])
        total = len(history)
        if not total:
            backtest = {"matches": 0}
            self.store.save_backtest(backtest)
            return backtest

        def review_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
            count = len(items)
            goal_errors = [abs(float(item["goal_error"])) for item in items if item.get("goal_error") is not None]
            expected_goal_errors = [
                abs(float(item["expected_goal_error"]))
                for item in items
                if item.get("expected_goal_error") is not None
            ]
            corner_errors = [abs(float(item["corner_error"])) for item in items if item.get("corner_error") is not None]
            expected_corner_errors = [
                abs(float(item["expected_corner_error"]))
                for item in items
                if item.get("expected_corner_error") is not None
            ]
            foul_errors = [abs(float(item["foul_error"])) for item in items if item.get("foul_error") is not None]
            expected_foul_errors = [
                abs(float(item["expected_foul_error"]))
                for item in items
                if item.get("expected_foul_error") is not None
            ]
            return {
                "matches": count,
                "outcome_accuracy": None if not count else sum(1 for item in items if item.get("outcome_hit")) / count,
                "exact_score_accuracy": None if not count else sum(1 for item in items if item.get("score_hit")) / count,
                "goal_mae": None if not goal_errors else sum(goal_errors) / len(goal_errors),
                "expected_goal_mae": None if not expected_goal_errors else sum(expected_goal_errors) / len(expected_goal_errors),
                "goal_within_0_7_rate": None if not goal_errors else sum(error <= 0.7 for error in goal_errors) / len(goal_errors),
                "corner_mae": None if not corner_errors else sum(corner_errors) / len(corner_errors),
                "expected_corner_mae": None if not expected_corner_errors else sum(expected_corner_errors) / len(expected_corner_errors),
                "corner_within_1_5_rate": None if not corner_errors else sum(error <= 1.5 for error in corner_errors) / len(corner_errors),
                "foul_mae": None if not foul_errors else sum(foul_errors) / len(foul_errors),
                "expected_foul_mae": None if not expected_foul_errors else sum(expected_foul_errors) / len(expected_foul_errors),
                "foul_within_2_rate": None if not foul_errors else sum(error <= 2.0 for error in foul_errors) / len(foul_errors),
            }

        metrics = review_metrics(history)
        playoff_metrics = review_metrics(
            [item for item in history if (item.get("round_info") or {}).get("knockout")]
        )
        outcome_accuracy = float(metrics["outcome_accuracy"] or 0.0)
        exact_score_accuracy = float(metrics["exact_score_accuracy"] or 0.0)
        goal_mae = metrics["goal_mae"]
        corner_mae = metrics["corner_mae"]
        foul_mae = metrics["foul_mae"]
        targets = {
            "outcome_accuracy": 0.80,
            "exact_score_accuracy": 0.20,
            "goal_mae": 0.70,
            "corner_mae": 1.50,
            "foul_mae": 2.00,
        }
        by_team: dict[str, dict[str, int]] = defaultdict(lambda: {"matches": 0, "outcome_hits": 0})
        for item in history:
            for team in (item.get("home_team"), item.get("away_team")):
                if not team:
                    continue
                by_team[team]["matches"] += 1
                if item.get("outcome_hit"):
                    by_team[team]["outcome_hits"] += 1

        backtest = {
            "matches": total,
            "evaluation_mode": (state.get("training") or {}).get("evaluation_mode", "unknown"),
            "evaluation_scope": (state.get("training") or {}).get("evaluation_scope"),
            "result_leakage_guard": bool((state.get("training") or {}).get("result_leakage_guard")),
            "outcome_accuracy": round(outcome_accuracy, 3),
            "exact_score_accuracy": round(exact_score_accuracy, 3),
            "goal_mae": None if goal_mae is None else round(goal_mae, 2),
            "expected_goal_mae": None if metrics["expected_goal_mae"] is None else round(metrics["expected_goal_mae"], 2),
            "goal_within_0_7_rate": None if metrics["goal_within_0_7_rate"] is None else round(metrics["goal_within_0_7_rate"], 3),
            "corner_mae": None if corner_mae is None else round(corner_mae, 2),
            "expected_corner_mae": None if metrics["expected_corner_mae"] is None else round(metrics["expected_corner_mae"], 2),
            "corner_within_1_5_rate": None if metrics["corner_within_1_5_rate"] is None else round(metrics["corner_within_1_5_rate"], 3),
            "foul_mae": None if foul_mae is None else round(foul_mae, 2),
            "expected_foul_mae": None if metrics["expected_foul_mae"] is None else round(metrics["expected_foul_mae"], 2),
            "foul_within_2_rate": None if metrics["foul_within_2_rate"] is None else round(metrics["foul_within_2_rate"], 3),
            "playoff": {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in playoff_metrics.items()
            },
            "playoff_target_status": {
                "outcome_accuracy": playoff_metrics["outcome_accuracy"] is not None
                and playoff_metrics["outcome_accuracy"] >= targets["outcome_accuracy"],
                "exact_score_accuracy": playoff_metrics["exact_score_accuracy"] is not None
                and playoff_metrics["exact_score_accuracy"] >= targets["exact_score_accuracy"],
                "goal_mae": playoff_metrics["goal_mae"] is not None
                and playoff_metrics["goal_mae"] <= targets["goal_mae"],
                "corner_mae": playoff_metrics["corner_mae"] is not None
                and playoff_metrics["corner_mae"] <= targets["corner_mae"],
                "foul_mae": playoff_metrics["foul_mae"] is not None
                and playoff_metrics["foul_mae"] <= targets["foul_mae"],
            },
            "targets": targets,
            "target_status": {
                "outcome_accuracy": outcome_accuracy >= targets["outcome_accuracy"],
                "exact_score_accuracy": exact_score_accuracy >= targets["exact_score_accuracy"],
                "goal_mae": goal_mae is not None and goal_mae <= targets["goal_mae"],
                "corner_mae": corner_mae is not None and corner_mae <= targets["corner_mae"],
                "foul_mae": foul_mae is not None and foul_mae <= targets["foul_mae"],
            },
            "trained_match_keys": len(state.get("trained_match_keys", [])),
            "training": state.get("training", {}),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "by_team": {
                team: {
                    "matches": values["matches"],
                    "outcome_accuracy": round(values["outcome_hits"] / values["matches"], 3),
                }
                for team, values in sorted(by_team.items())
            },
        }
        self.store.save_backtest(backtest)
        return backtest

    def _full_sync_due(self, max_age_hours: int) -> bool:
        sync_state = self.store.load_sync_state()
        last_sync = sync_state.get("last_full_sync_at")
        if not last_sync:
            return True
        try:
            last = datetime.fromisoformat(last_sync)
        except ValueError:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last > timedelta(hours=max_age_hours)

    def _profile_from_averages(
        self,
        existing: dict[str, Any],
        avg: dict[str, float],
        sample_size: int,
        formation_weights: dict[str, float] | None = None,
        formation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        possession = avg.get("possession", 50.0) / 100.0
        shots = avg.get("shots_for", 10.0)
        shots_against = avg.get("shots_against", 10.0)
        sot = avg.get("sot", 3.0)
        corners_for = avg.get("corners_for", 4.5)
        corners_against = avg.get("corners_against", 4.5)
        goals_for = avg.get("goals_for", 1.1)
        goals_against = avg.get("goals_against", 1.1)

        chance_creation = clamp01(0.35 + shots / 80.0 + sot / 40.0 + goals_for / 16.0)
        defensive_solidity = clamp01(0.86 - goals_against / 4.5 - shots_against / 34.0)
        attack_width = clamp01(0.40 + corners_for / 25.0)
        set_piece_threat = clamp01(0.38 + corners_for / 30.0 + goals_for / 20.0)
        directness = clamp01(0.45 + shots / 70.0 - possession * 0.35)
        transition_attack = clamp01(0.38 + directness * 0.25 + goals_for / 8.0)
        transition_defense = clamp01(0.35 + defensive_solidity * 0.55 + (1.0 - corners_against / 12.0) * 0.18)
        pressing = clamp01(0.42 + shots / 70.0 + (1.0 - possession) * 0.12)
        tempo = clamp01(0.38 + (shots + shots_against) / 80.0)
        central_progression = clamp01(0.34 + possession * 0.32 + sot / 35.0)
        line_height = clamp01(0.40 + pressing * 0.20 + possession * 0.15)

        derived_formation = formation_guess(
            possession=possession,
            directness=directness,
            defensive_solidity=defensive_solidity,
            attack_width=attack_width,
            central_progression=central_progression,
            transition_attack=transition_attack,
            pressing=pressing,
            line_height=line_height,
            shots=shots,
            shots_against=shots_against,
            goals_for=goals_for,
            goals_against=goals_against,
        )
        manual_formation = existing.get("formation") if existing.get("formation_source") == "manual" else None
        formation_source = "estimated-from-match-stats"
        formation_confidence = round(min(0.82, 0.32 + sample_size * 0.12), 3)
        selected_formation = derived_formation
        formation_history = (formation_history or [])[:6]
        if formation_weights:
            selected_formation, selected_weight = max(formation_weights.items(), key=lambda item: item[1])
            total_formation_weight = sum(formation_weights.values()) or 1.0
            consistency = selected_weight / total_formation_weight
            formation_source = "confirmed-lineups-last-matches"
            formation_confidence = round(min(0.96, 0.50 + consistency * 0.30 + min(sample_size, 6) * 0.03), 3)
        if manual_formation:
            selected_formation = manual_formation
            formation_source = "manual"
            formation_confidence = 1.0

        return {
            "formation": selected_formation,
            "formation_source": formation_source,
            "formation_confidence": formation_confidence,
            "formation_history": formation_history,
            "estimated_formation": derived_formation,
            "style": style_guess(possession, directness, pressing),
            "build_up": build_up_guess(possession, directness),
            "primary_attack": primary_attack_guess(attack_width, central_progression, transition_attack),
            "defensive_block": block_guess(line_height),
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
            "sample_size": sample_size,
            "source": "espn-world-cup-derived",
            "notes": [f"Estimated from {sample_size} recent rich match(es); lineup formations override stat estimates when ESPN rosters are available."],
        }


def formation_guess(
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


def style_guess(possession: float, directness: float, pressing: float) -> str:
    if possession >= 0.60:
        return "possession control"
    if directness >= 0.63:
        return "direct transitions"
    if pressing >= 0.62:
        return "pressing and territory"
    return "balanced tournament football"


def build_up_guess(possession: float, directness: float) -> str:
    if possession >= 0.60:
        return "short build-up and positional control"
    if directness >= 0.62:
        return "early forward passes and second balls"
    return "mixed build-up"


def primary_attack_guess(width: float, central: float, transition: float) -> str:
    if width >= central and width >= transition:
        return "wide attacks and corners"
    if transition >= central:
        return "quick transitions"
    return "central progression"


def block_guess(line_height: float) -> str:
    if line_height >= 0.62:
        return "high"
    if line_height <= 0.45:
        return "low-mid"
    return "mid"
