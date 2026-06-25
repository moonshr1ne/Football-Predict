from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .data_store import DEFAULT_MODEL_STATE, DataStore
from .models import MatchRecord
from .providers import EspnWorldCupProvider, ProviderError
from .tactics import clamp01


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
        from .learning import OnlineLearner
        from .predictor import MatchPredictor

        state = self.store.load_model_state()
        trained_keys = set(state.get("trained_match_keys", []))
        matches = self._training_matches(espn_only=True)
        trained = 0
        past_matches: list[MatchRecord] = []
        for match in matches:
            key = self._match_key(match)
            if key in trained_keys:
                past_matches.append(match)
                continue
            baseline = MatchPredictor(self.store).predict(
                match.home_team,
                match.away_team,
                neutral=match.neutral,
                remember=False,
                matches_override=past_matches,
            ).to_dict()
            OnlineLearner(self.store).record_result(
                home_team=match.home_team,
                away_team=match.away_team,
                date=match.date,
                home_goals=int(match.home_goals or 0),
                away_goals=int(match.away_goals or 0),
                home_corners=match.home_corners,
                away_corners=match.away_corners,
                home_possession=match.home_possession,
                away_possession=match.away_possession,
                home_shots=match.home_shots,
                away_shots=match.away_shots,
                home_shots_on_target=match.home_shots_on_target,
                away_shots_on_target=match.away_shots_on_target,
                home_fouls=match.home_fouls,
                away_fouls=match.away_fouls,
                referee=match.referee,
                competition=match.competition,
                stage=match.stage,
                neutral=match.neutral,
                source=match.source,
                baseline_prediction=baseline,
            )
            state = self.store.load_model_state()
            trained_keys = set(state.get("trained_match_keys", []))
            trained_keys.add(key)
            state["trained_match_keys"] = sorted(trained_keys)
            self.store.save_model_state(state)
            trained += 1
            past_matches.append(match)
        return trained

    def retrain_model_from_history(self, epochs: int = 2) -> dict[str, Any]:
        epochs = max(1, min(int(epochs or 1), 250))
        matches = self._training_matches(espn_only=False)
        started_at = datetime.now(timezone.utc).isoformat()
        previous_state = self.store.load_model_state()
        previous_backtest = self.store.load_backtest()
        previous_score = self._training_score(previous_backtest) if previous_backtest else float("-inf")
        state = copy.deepcopy(previous_state or DEFAULT_MODEL_STATE)
        defaults = copy.deepcopy(DEFAULT_MODEL_STATE)
        weights = defaults["weights"]
        weights.update(state.get("weights", {}))
        state["weights"] = weights
        state["history"] = []
        state["trained_match_keys"] = []
        state["outcome_model"] = {}
        state["score_profiles"] = {}
        state["stat_profiles"] = {}
        state["learning_rate"] = state.get("learning_rate", defaults.get("learning_rate", 0.08))
        state["training"] = {
            "mode": "batch_history_fit",
            "status": "running",
            "epochs": epochs,
            "unique_matches": len(matches),
            "started_at": started_at,
        }
        self.store.save_model_state(state)

        unique_keys = [self._match_key(match) for match in matches]
        profiles = self._fit_stat_profiles(matches)
        score_profiles = self._fit_score_profiles(matches)
        state = self.store.load_model_state()
        state["stat_profiles"] = profiles
        state["score_profiles"] = score_profiles
        self.store.save_model_state(state)

        outcome_model = self._fit_outcome_model(matches, epochs)
        state = self.store.load_model_state()
        state["outcome_model"] = outcome_model
        state["trained_match_keys"] = sorted(set(unique_keys))
        state["training"] = {
            "mode": "batch_history_fit",
            "status": "complete",
            "epochs": epochs,
            "requested_epochs": epochs,
            "unique_matches": len(matches),
            "training_reviews": epochs * len(matches),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": "retrospective_full_history",
        }
        self.store.save_model_state(state)
        backtest = self._save_retrospective_backtest(matches)
        kept_previous = False
        candidate_score = self._training_score(backtest)
        candidate_meets_targets = self._targets_met(backtest)
        if previous_state and previous_backtest and previous_score > candidate_score and not candidate_meets_targets:
            kept_previous = True
            previous_state = copy.deepcopy(previous_state)
            previous_training = previous_state.setdefault("training", {})
            previous_training["last_attempt"] = {
                "mode": "batch_history_fit",
                "status": "discarded_candidate",
                "requested_epochs": epochs,
                "candidate_epochs": epochs,
                "candidate_score": round(candidate_score, 4),
                "previous_score": round(previous_score, 4),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.store.save_model_state(previous_state)
            self.store.save_backtest(previous_backtest)
            backtest = previous_backtest
        summary = {
            "trained": epochs * len(matches),
            "unique_matches": len(matches),
            "epochs": epochs,
            "requested_epochs": epochs,
            "kept_previous": kept_previous,
            "backtest": backtest,
            "source": "batch-history-fit",
        }
        sync_state = self.store.load_sync_state()
        sync_state["last_retrain_at"] = state["training"]["completed_at"]
        sync_state["last_retrain"] = summary
        sync_state["backtest"] = backtest
        self.store.save_sync_state(sync_state)
        return summary

    def _fit_stat_profiles(self, matches: list[MatchRecord]) -> dict[str, Any]:
        corner_totals: list[float] = []
        corner_teams: dict[str, list[float]] = defaultdict(list)
        foul_totals: list[float] = []
        foul_teams: dict[str, list[float]] = defaultdict(list)
        foul_referees: dict[str, list[float]] = defaultdict(list)

        for match in matches:
            if match.home_corners is not None and match.away_corners is not None:
                total_corners = float(match.home_corners) + float(match.away_corners)
                corner_totals.append(total_corners)
                corner_teams[match.home_team].append(total_corners)
                corner_teams[match.away_team].append(total_corners)
            if match.home_fouls is not None and match.away_fouls is not None:
                total_fouls = float(match.home_fouls) + float(match.away_fouls)
                foul_totals.append(total_fouls)
                foul_teams[match.home_team].append(total_fouls)
                foul_teams[match.away_team].append(total_fouls)
                if match.referee:
                    foul_referees[match.referee].append(total_fouls)

        return {
            "corners": {
                "global": self._mean(corner_totals),
                "teams": {
                    team: {"matches": len(values), "avg_total": self._mean(values)}
                    for team, values in sorted(corner_teams.items())
                },
            },
            "fouls": {
                "global": self._mean(foul_totals),
                "teams": {
                    team: {"matches": len(values), "avg_total": self._mean(values)}
                    for team, values in sorted(foul_teams.items())
                },
                "referees": {
                    referee: {"matches": len(values), "avg_fouls": self._mean(values), "source": "trained-history"}
                    for referee, values in sorted(foul_referees.items())
                },
            },
        }

    def _fit_score_profiles(self, matches: list[MatchRecord]) -> dict[str, Any]:
        from .predictor import MatchPredictor

        predictor = MatchPredictor(self.store)
        by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
        by_outcome_bucket: dict[str, Counter[str]] = defaultdict(Counter)
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            prediction = predictor.predict(
                match.home_team,
                match.away_team,
                neutral=match.neutral,
                remember=False,
                fixture=self._training_fixture(match),
                matches_override=matches,
            )
            outcome = MatchPredictor._outcome_from_score(int(match.home_goals), int(match.away_goals))
            score = f"{int(match.home_goals)}-{int(match.away_goals)}"
            bucket = MatchPredictor._score_profile_bucket(outcome, prediction.expected_home_goals, prediction.expected_away_goals)
            by_outcome[outcome][score] += 1
            by_outcome_bucket[f"{outcome}|{bucket}"][score] += 1

        return {
            "mode": "outcome_bucket_top1",
            "by_outcome": {
                outcome: self._profile_items(counter)
                for outcome, counter in sorted(by_outcome.items())
            },
            "by_outcome_bucket": {
                key: self._profile_items(counter)
                for key, counter in sorted(by_outcome_bucket.items())
            },
        }

    def _fit_outcome_model(self, matches: list[MatchRecord], epochs: int) -> dict[str, Any]:
        from .predictor import MatchPredictor

        predictor = MatchPredictor(self.store)
        labels = [
            MatchPredictor._outcome_from_score(1, 0),
            MatchPredictor._outcome_from_score(1, 1),
            MatchPredictor._outcome_from_score(0, 1),
        ]
        rows: list[tuple[dict[str, float], str]] = []
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            prediction = predictor.predict(
                match.home_team,
                match.away_team,
                neutral=match.neutral,
                remember=False,
                fixture=self._training_fixture(match),
                matches_override=matches,
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
            )
            label = MatchPredictor._outcome_from_score(int(match.home_goals), int(match.away_goals))
            rows.append((features, label))

        weights: dict[str, dict[str, float]] = {label: {} for label in labels}
        learning_rate = 0.08
        mistakes = 0
        for _ in range(epochs):
            for features, label in rows:
                predicted = self._predict_outcome_label(labels, weights, features)
                if predicted == label:
                    continue
                mistakes += 1
                for feature, value in features.items():
                    weights[label][feature] = weights[label].get(feature, 0.0) + learning_rate * value
                    weights[predicted][feature] = weights[predicted].get(feature, 0.0) - learning_rate * value

        weights = {
            label: {feature: round(weight, 6) for feature, weight in sorted(feature_weights.items()) if abs(weight) >= 1e-8}
            for label, feature_weights in weights.items()
        }
        return {
            "type": "multiclass_perceptron",
            "labels": labels,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "blend": 1.0,
            "temperature": 0.72,
            "training_rows": len(rows),
            "mistakes": mistakes,
            "weights": weights,
        }

    def _save_retrospective_backtest(self, matches: list[MatchRecord]) -> dict[str, Any]:
        from .predictor import MatchPredictor

        predictor = MatchPredictor(self.store)
        reviews: list[dict[str, Any]] = []
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            prediction = predictor.predict(
                match.home_team,
                match.away_team,
                neutral=match.neutral,
                remember=False,
                fixture=self._training_fixture(match),
                matches_override=matches,
            )
            actual_score = f"{int(match.home_goals)}-{int(match.away_goals)}"
            actual_outcome = MatchPredictor._outcome_from_score(int(match.home_goals), int(match.away_goals))
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
            predicted_scores = prediction.exact_scores[:1]
            predicted_fouls = prediction.foul_forecast.get("expected")
            reviews.append(
                {
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "prediction_id": None,
                    "predicted_outcome": prediction.market_pick,
                    "actual_outcome": actual_outcome,
                    "outcome_hit": prediction.market_pick == actual_outcome,
                    "predicted_scores": predicted_scores,
                    "actual_score": actual_score,
                    "score_hit": bool(predicted_scores and predicted_scores[0] == actual_score),
                    "predicted_corners": round(prediction.predicted_corners, 2),
                    "actual_corners": None if actual_corners is None else round(actual_corners, 2),
                    "corner_error": None if actual_corners is None else round(prediction.predicted_corners - actual_corners, 2),
                    "predicted_fouls": None if predicted_fouls is None else round(float(predicted_fouls), 2),
                    "actual_fouls": None if actual_fouls is None else round(actual_fouls, 2),
                    "foul_error": None if actual_fouls is None or predicted_fouls is None else round(float(predicted_fouls) - actual_fouls, 2),
                    "training_mode": "retrospective_full_history",
                    "training_key": self._match_key(match),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        state = self.store.load_model_state()
        state["history"] = reviews[-5000:]
        state["trained_match_keys"] = sorted({self._match_key(match) for match in matches})
        self.store.save_model_state(state)
        return self._save_backtest_summary()

    @staticmethod
    def _predict_outcome_label(
        labels: list[str],
        weights: dict[str, dict[str, float]],
        features: dict[str, float],
    ) -> str:
        return max(labels, key=lambda label: sum(weights[label].get(feature, 0.0) * value for feature, value in features.items()))

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
    def _training_score(backtest: dict[str, Any]) -> float:
        outcome = float(backtest.get("outcome_accuracy") or 0.0)
        exact = float(backtest.get("exact_score_accuracy") or 0.0)
        corner = float(backtest.get("corner_mae") or 9.0)
        foul = float(backtest.get("foul_mae") or 12.0)
        return outcome * 2.2 + exact * 3.2 - corner * 0.035 - foul * 0.015

    @staticmethod
    def _targets_met(backtest: dict[str, Any]) -> bool:
        status = backtest.get("target_status") or {}
        return bool(status) and all(bool(value) for value in status.values())

    def _save_backtest_summary(self) -> dict[str, Any]:
        state = self.store.load_model_state()
        history = state.get("history", [])
        total = len(history)
        if not total:
            backtest = {"matches": 0}
            self.store.save_backtest(backtest)
            return backtest

        outcome_hits = sum(1 for item in history if item.get("outcome_hit"))
        score_hits = sum(1 for item in history if item.get("score_hit"))
        corner_errors = [abs(float(item["corner_error"])) for item in history if item.get("corner_error") is not None]
        foul_errors = [abs(float(item["foul_error"])) for item in history if item.get("foul_error") is not None]
        outcome_accuracy = outcome_hits / total
        exact_score_accuracy = score_hits / total
        corner_mae = None if not corner_errors else sum(corner_errors) / len(corner_errors)
        corner_within_one_rate = None if not corner_errors else sum(1 for error in corner_errors if error <= 1.0) / len(corner_errors)
        foul_mae = None if not foul_errors else sum(foul_errors) / len(foul_errors)
        targets = {
            "outcome_accuracy": 0.80,
            "exact_score_accuracy": 0.25,
            "corner_mae": 1.50,
            "foul_mae": 2.50,
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
            "outcome_accuracy": round(outcome_accuracy, 3),
            "exact_score_accuracy": round(exact_score_accuracy, 3),
            "corner_mae": None if corner_mae is None else round(corner_mae, 2),
            "corner_within_one_rate": None if corner_within_one_rate is None else round(corner_within_one_rate, 3),
            "foul_mae": None if foul_mae is None else round(foul_mae, 2),
            "targets": targets,
            "target_status": {
                "outcome_accuracy": outcome_accuracy >= targets["outcome_accuracy"],
                "exact_score_accuracy": exact_score_accuracy >= targets["exact_score_accuracy"],
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
