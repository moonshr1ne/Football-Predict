from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .barcelona_model import BarcelonaModel
from .barcelona_provider import BARCELONA_NAME, BarcelonaProvider
from .barcelona_store import BarcelonaStore
from .barcelona_sync import BarcelonaDataSync
from .data_store import fixture_has_started
from .providers import ProviderError, normalize_provider_name


class BarcelonaService:
    def __init__(self, store: BarcelonaStore | None = None, provider: BarcelonaProvider | None = None):
        self.store = store or BarcelonaStore()
        self.provider = provider or BarcelonaProvider()
        self.syncer = BarcelonaDataSync(self.store, self.provider)
        self.model = BarcelonaModel(self.store)

    def predict(self, opponent_query: str, remember: bool = True) -> dict[str, Any]:
        sync_info = self.syncer.sync()
        universe = self.store.load_universe()
        opponent = self.provider.resolve_opponent(opponent_query, universe)
        fixture = self.provider.find_fixture(opponent, universe)
        fixture = self._with_derived_stage(fixture, universe)

        if self._refresh_needed(fixture):
            try:
                fixture = self.syncer.refresh_fixture(fixture)
            except ProviderError:
                pass
            fixture = self._with_derived_stage(fixture, self.store.load_universe())

        fixture = self._with_current_squad(fixture)

        saved = self.store.prediction_for_fixture(str(fixture.get("fixture_id")))
        if fixture_has_started(fixture):
            payload = self.model.result_payload(fixture, saved)
            payload["lineups"] = self.model.lineup_reports(fixture)
            payload["sync"] = sync_info
            return payload

        cutoff = str(fixture.get("kickoff") or fixture.get("date") or "")
        barca_rich = self.syncer.ensure_recent_rich(BARCELONA_NAME, cutoff, limit=10)
        opponent_rich = self.syncer.ensure_recent_rich(opponent, cutoff, limit=10)
        universe = self.store.load_universe()
        fixture = next(
            (item for item in universe if str(item.get("fixture_id")) == str(fixture.get("fixture_id"))),
            fixture,
        )
        fixture = self._with_derived_stage(fixture, universe)
        fixture = self._with_current_squad(fixture)

        self.model.train_and_backtest()
        if saved:
            payload = self.model.result_payload(fixture, saved)
            payload["lineups"] = self.model.lineup_reports(fixture)
        else:
            payload = self.model.predict(fixture, remember=remember)
        payload["sync"] = sync_info
        payload["data_refresh"] = {"barcelona": barca_rich, "opponent": opponent_rich}
        return payload

    def opponents(self) -> list[dict[str, Any]]:
        self.syncer.sync()
        universe = self.store.load_universe()
        seen: dict[str, dict[str, Any]] = {}
        for fixture in universe:
            if not self._has_barcelona(fixture):
                continue
            opponent = fixture.get("away_team") if self._barca_home(fixture) else fixture.get("home_team")
            logo = fixture.get("away_logo") if self._barca_home(fixture) else fixture.get("home_logo")
            if not opponent:
                continue
            key = normalize_provider_name(str(opponent))
            item = seen.setdefault(key, {"name": opponent, "logo": logo, "fixtures": []})
            item["fixtures"].append(
                {
                    "fixture_id": fixture.get("fixture_id"),
                    "date": fixture.get("date"),
                    "kickoff": fixture.get("kickoff"),
                    "competition": fixture.get("competition"),
                    "completed": fixture.get("completed"),
                }
            )
        return sorted(seen.values(), key=lambda item: normalize_provider_name(str(item["name"])))

    def auto_check(self) -> dict[str, Any]:
        sync = self.syncer.sync(force=True)
        review = self.syncer.review_predictions()
        backtest = self.model.train_and_backtest(force=True)
        return {"sync": sync, "review": review, "backtest": backtest}

    def status(self) -> dict[str, Any]:
        return {
            "mode": "barcelona",
            "leagues": ["La Liga", "UEFA Champions League"],
            "sync": self.store.load_sync_state(),
            "backtest": self.store.load_backtest(),
            "predictions": len(self.store.load_predictions()),
        }

    def background_refresh(self) -> dict[str, Any]:
        sync = self.syncer.sync()
        review = self.syncer.review_predictions()
        backtest = self.model.train_and_backtest()
        return {"sync": sync, "review": review, "backtest": backtest}

    def _refresh_needed(self, fixture: dict[str, Any]) -> bool:
        if fixture.get("completed") or fixture.get("in_progress"):
            return True
        kickoff = _parse_kickoff(fixture)
        return abs((kickoff - datetime.now(timezone.utc)).total_seconds()) <= 48 * 3600

    def _with_current_squad(self, fixture: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(fixture)
        try:
            squad = self.provider.fetch_current_squad()
        except ProviderError:
            return enriched
        enriched["squad_context"] = {BARCELONA_NAME: squad}
        return enriched

    def _has_barcelona(self, fixture: dict[str, Any]) -> bool:
        return self._barca_home(fixture) or normalize_provider_name(str(fixture.get("away_team") or "")) == "barcelona"

    def _barca_home(self, fixture: dict[str, Any]) -> bool:
        return normalize_provider_name(str(fixture.get("home_team") or "")) == "barcelona"

    def _with_derived_stage(self, fixture: dict[str, Any], universe: list[dict[str, Any]]) -> dict[str, Any]:
        enriched = dict(fixture)
        if fixture.get("competition_code") != "esp.1":
            return enriched
        match_date = str(fixture.get("date") or "")
        try:
            year = int(match_date[:4])
            month = int(match_date[5:7])
        except (TypeError, ValueError):
            return enriched
        start_year = year if month >= 7 else year - 1
        season_start = f"{start_year}-07-01"
        season_end = f"{start_year + 1}-06-30"
        league_matches = sorted(
            [
                item
                for item in universe
                if item.get("competition_code") == "esp.1"
                and self._has_barcelona(item)
                and season_start <= str(item.get("date") or "") <= season_end
            ],
            key=lambda item: (item.get("kickoff", ""), item.get("fixture_id", "")),
        )
        for index, item in enumerate(league_matches, start=1):
            if str(item.get("fixture_id")) == str(fixture.get("fixture_id")):
                enriched["stage"] = f"Тур {index}"
                break
        return enriched


def _parse_kickoff(fixture: dict[str, Any]) -> datetime:
    raw = str(fixture.get("kickoff") or f"{fixture.get('date')}T12:00:00Z")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
