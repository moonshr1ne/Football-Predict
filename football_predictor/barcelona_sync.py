from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .barcelona_provider import BARCELONA_ID, BARCELONA_NAME, BarcelonaProvider
from .barcelona_store import BarcelonaStore
from .providers import ProviderError, normalize_provider_name


_SYNC_LOCK = threading.RLock()


class BarcelonaDataSync:
    def __init__(self, store: BarcelonaStore | None = None, provider: BarcelonaProvider | None = None):
        self.store = store or BarcelonaStore()
        self.provider = provider or BarcelonaProvider()

    def sync(self, force: bool = False) -> dict[str, Any]:
        with _SYNC_LOCK:
            state = self.store.load_sync_state()
            existing = self.store.load_universe()
            fresh = self._is_fresh(state.get("last_sync"), hours=4)
            full = force or len(existing) < 100
            if fresh and not force and existing:
                return {
                    "skipped": True,
                    "fixtures": len(existing),
                    "barcelona_matches": len(self.store.load_matches()),
                    "details_loaded": sum(bool(item.get("details_loaded")) for item in self.store.load_matches()),
                    "last_sync": state.get("last_sync"),
                }

            windows = self.provider.season_windows() if full else self.provider.season_windows(seasons=1)
            incoming = self.provider.fetch_universe(windows)
            universe = self.store.merge_universe(incoming)

            barcelona = [item for item in universe if self._is_barcelona_fixture(item)]
            completed_missing = [
                item
                for item in barcelona
                if item.get("completed") and not self._rich_enough(item)
            ]
            enriched, errors = self._enrich_many(completed_missing)
            if enriched:
                universe = self.store.merge_universe(enriched)

            # Refresh the closest match so lineups and referee appear as soon as
            # ESPN publishes them, normally near kickoff.
            closest = self._closest_active_barcelona_fixture(universe)
            if closest and self._within_hours(closest, 48):
                try:
                    refreshed = self.provider.enrich_fixture(closest)
                    universe = self.store.merge_universe([refreshed])
                except ProviderError:
                    errors += 1

            barcelona = [item for item in universe if self._is_barcelona_fixture(item)]
            self.store.save_matches(barcelona)
            now = datetime.now(timezone.utc).isoformat()
            state = {
                "last_sync": now,
                "full_history": full or bool(state.get("full_history")),
                "fixtures": len(universe),
                "barcelona_matches": len(barcelona),
                "details_loaded": sum(bool(item.get("details_loaded")) for item in barcelona),
                "errors": errors,
                "leagues": ["esp.1", "uefa.champions"],
            }
            self.store.save_sync_state(state)
            return {"skipped": False, **state}

    def ensure_recent_rich(self, team: str, cutoff: str, limit: int = 10) -> dict[str, Any]:
        """Download the actual last-ten protocols used for this prediction."""
        with _SYNC_LOCK:
            universe = self.store.load_universe()
            candidates = [
                item
                for item in universe
                if item.get("completed")
                and self._before_cutoff(item, cutoff)
                and self._has_team(item, team)
            ]
            candidates.sort(key=lambda item: item.get("kickoff", ""), reverse=True)
            selected = candidates[:limit]
            missing = [item for item in selected if not self._rich_enough(item)]
            enriched, errors = self._enrich_many(missing)
            if enriched:
                universe = self.store.merge_universe(enriched)
                barcelona = [item for item in universe if self._is_barcelona_fixture(item)]
                self.store.save_matches(barcelona)
            return {"requested": len(selected), "enriched": len(enriched), "errors": errors}

    def refresh_fixture(self, fixture: dict[str, Any]) -> dict[str, Any]:
        with _SYNC_LOCK:
            enriched = self.provider.enrich_fixture(fixture)
            universe = self.store.merge_universe([enriched])
            self.store.save_matches([item for item in universe if self._is_barcelona_fixture(item)])
            return next(
                (item for item in universe if str(item.get("fixture_id")) == str(fixture.get("fixture_id"))),
                enriched,
            )

    def review_predictions(self) -> dict[str, int]:
        universe = {str(item.get("fixture_id")): item for item in self.store.load_universe()}
        checked = reviewed = pending = 0
        for prediction in self.store.load_predictions():
            if prediction.get("status") != "pending":
                continue
            pending += 1
            fixture = universe.get(str(prediction.get("fixture_id")))
            if not fixture or not fixture.get("completed"):
                continue
            checked += 1
            review = self._review(prediction, fixture)
            self.store.review_prediction(str(prediction.get("fixture_id")), review)
            reviewed += 1
        return {"checked": checked, "reviewed": reviewed, "pending": max(0, pending - reviewed)}

    def _enrich_many(self, fixtures: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        if not fixtures:
            return [], 0
        enriched: list[dict[str, Any]] = []
        errors = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self.provider.enrich_fixture, fixture): fixture for fixture in fixtures}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result.get("details_loaded"):
                        enriched.append(result)
                except Exception:
                    errors += 1
        return enriched, errors

    def _review(self, prediction: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
        barca_home = str(fixture.get("home_team_id")) == BARCELONA_ID or normalize_provider_name(
            str(fixture.get("home_team") or "")
        ) == "barcelona"
        barca_goals = fixture.get("home_goals") if barca_home else fixture.get("away_goals")
        opponent_goals = fixture.get("away_goals") if barca_home else fixture.get("home_goals")
        score = f"{barca_goals}-{opponent_goals}"
        actual_outcome = "barcelona" if barca_goals > opponent_goals else "draw" if barca_goals == opponent_goals else "opponent"
        actual_corners = _sum_optional(
            (fixture.get("home_stats") or {}).get("wonCorners", fixture.get("home_corners")),
            (fixture.get("away_stats") or {}).get("wonCorners", fixture.get("away_corners")),
        )
        actual_fouls = _sum_optional(
            (fixture.get("home_stats") or {}).get("foulsCommitted", fixture.get("home_fouls")),
            (fixture.get("away_stats") or {}).get("foulsCommitted", fixture.get("away_fouls")),
        )
        return {
            "actual_score": score,
            "actual_outcome": actual_outcome,
            "actual_corners": actual_corners,
            "actual_fouls": actual_fouls,
            "outcome_hit": prediction.get("outcome", {}).get("pick") == actual_outcome,
            "exact_score_hit": prediction.get("exact_score", {}).get("score") == score,
            "corner_error": _absolute_error(prediction.get("corners", {}).get("point"), actual_corners),
            "foul_error": _absolute_error(prediction.get("fouls", {}).get("point"), actual_fouls),
            "goal_total_error": _absolute_error(
                prediction.get("goals", {}).get("total_expected"),
                None if barca_goals is None or opponent_goals is None else barca_goals + opponent_goals,
            ),
            "reviewed_from": "espn-final-protocol",
        }

    def _rich_enough(self, fixture: dict[str, Any]) -> bool:
        home = fixture.get("home_stats") or {}
        away = fixture.get("away_stats") or {}
        return bool(
            fixture.get("details_loaded")
            and int(fixture.get("details_version") or 0) >= 2
            and "totalShots" in home
            and "totalShots" in away
            and "wonCorners" in home
            and "foulsCommitted" in away
        )

    def _is_barcelona_fixture(self, fixture: dict[str, Any]) -> bool:
        return str(fixture.get("home_team_id")) == BARCELONA_ID or str(fixture.get("away_team_id")) == BARCELONA_ID or self._has_team(fixture, BARCELONA_NAME)

    def _has_team(self, fixture: dict[str, Any], team: str) -> bool:
        query = normalize_provider_name(team)
        return any(normalize_provider_name(str(fixture.get(key) or "")) == query for key in ("home_team", "away_team"))

    def _before_cutoff(self, fixture: dict[str, Any], cutoff: str) -> bool:
        return str(fixture.get("kickoff") or fixture.get("date") or "") < str(cutoff)

    def _closest_active_barcelona_fixture(self, universe: list[dict[str, Any]]) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        candidates = [item for item in universe if self._is_barcelona_fixture(item) and not item.get("completed")]
        if not candidates:
            return None

        def distance(item: dict[str, Any]) -> float:
            parsed = _parse_kickoff(item)
            return abs((parsed - now).total_seconds())

        return min(candidates, key=distance)

    def _within_hours(self, fixture: dict[str, Any], hours: int) -> bool:
        return abs((_parse_kickoff(fixture) - datetime.now(timezone.utc)).total_seconds()) <= hours * 3600

    def _is_fresh(self, raw: Any, hours: int) -> bool:
        if not raw:
            return False
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - parsed < timedelta(hours=hours)
        except ValueError:
            return False


def _parse_kickoff(fixture: dict[str, Any]) -> datetime:
    raw = str(fixture.get("kickoff") or f"{fixture.get('date')}T12:00:00Z")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _sum_optional(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return float(first) + float(second)


def _absolute_error(predicted: Any, actual: Any) -> float | None:
    if predicted is None or actual is None:
        return None
    return round(abs(float(predicted) - float(actual)), 2)
