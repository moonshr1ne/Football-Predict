from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class ApiFootballProvider:
    """Best-effort API-Football adapter.

    It is optional because API-Football/API-Sports requires a key. The rest of
    the app works locally without network access.
    """

    def __init__(self, api_key: str | None = None, host: str | None = None):
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        self.host = host or os.environ.get("API_FOOTBALL_HOST", "v3.football.api-sports.io")
        if not self.api_key:
            raise ProviderError("Нет API_FOOTBALL_KEY. Можно ввести результат вручную командой result.")

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
        url = f"https://{self.host}{path}?{query}"
        request = urllib.request.Request(url, headers={"x-apisports-key": self.api_key})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    def search_team_id(self, team_name: str) -> int:
        payload = self._get("/teams", {"search": team_name})
        candidates = payload.get("response", [])
        national = [item for item in candidates if item.get("team", {}).get("national")]
        chosen = national[0] if national else (candidates[0] if candidates else None)
        if not chosen:
            raise ProviderError(f"Не нашел сборную в API: {team_name}")
        return int(chosen["team"]["id"])

    def get_finished_result(self, home_team: str, away_team: str, date: str) -> dict[str, Any]:
        home_id = self.search_team_id(home_team)
        fixtures = self._get("/fixtures", {"team": home_id, "date": date}).get("response", [])
        for item in fixtures:
            teams = item.get("teams", {})
            home_name = teams.get("home", {}).get("name", "")
            away_name = teams.get("away", {}).get("name", "")
            if away_team.casefold() not in f"{home_name} {away_name}".casefold():
                continue
            status = item.get("fixture", {}).get("status", {}).get("short")
            if status not in {"FT", "AET", "PEN"}:
                raise ProviderError("Матч найден, но еще не завершен.")
            goals = item.get("goals", {})
            fixture_id = item.get("fixture", {}).get("id")
            corners = self._fixture_corners(fixture_id) if fixture_id else (None, None)
            return {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "home_corners": corners[0],
                "away_corners": corners[1],
                "source": "api-football",
            }
        raise ProviderError("Не нашел завершенный матч для этой пары и даты.")

    def _fixture_corners(self, fixture_id: int) -> tuple[float | None, float | None]:
        payload = self._get("/fixtures/statistics", {"fixture": fixture_id})
        corners: list[float | None] = []
        for team_stats in payload.get("response", [])[:2]:
            value = None
            for stat in team_stats.get("statistics", []):
                if str(stat.get("type", "")).casefold() == "corner kicks":
                    value = stat.get("value")
                    break
            corners.append(None if value is None else float(value))
        while len(corners) < 2:
            corners.append(None)
        return corners[0], corners[1]
