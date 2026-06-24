from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
import urllib.parse
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


FINISHED_STATES = {"post"}
FINISHED_STATUS_NAMES = {"STATUS_FINAL", "STATUS_FULL_TIME", "FT", "AET", "PEN"}


def normalize_provider_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return value.strip()


def iso_date(value: str) -> str:
    return value[:10]


def compact_date(value: str) -> str:
    return value.replace("-", "")


class EspnWorldCupProvider:
    """Keyless ESPN World Cup scoreboard provider.

    ESPN's endpoint is undocumented, so this provider is intentionally small and
    defensive. It is used for private local lookup of fixtures and final scores.
    """

    endpoint = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def find_fixture(
        self,
        home_team: str,
        away_team: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        start_date, end_date = self._date_window(start_date, end_date)
        payload = self._scoreboard(start_date, end_date)
        candidates = []
        for event in payload.get("events", []):
            fixture = self._fixture_from_event(event, home_team, away_team)
            if fixture:
                candidates.append(fixture)
        if not candidates:
            raise ProviderError(f"Не нашел матч ЧМ для пары {home_team} - {away_team}.")
        return self._best_candidate(candidates)

    def get_finished_result(self, home_team: str, away_team: str, date: str | None = None) -> dict[str, Any]:
        if date:
            fixture = self.find_fixture(home_team, away_team, start_date=date, end_date=date)
        else:
            fixture = self.find_fixture(home_team, away_team)
        if not fixture.get("completed"):
            raise ProviderError("Матч найден, но еще не завершен.")
        return {
            "date": fixture["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": fixture["home_goals"],
            "away_goals": fixture["away_goals"],
            "home_corners": None,
            "away_corners": None,
            "source": fixture["source"],
            "fixture": fixture,
        }

    def _scoreboard(self, start_date: str, end_date: str) -> dict[str, Any]:
        params = {"limit": 1000, "dates": f"{compact_date(start_date)}-{compact_date(end_date)}"}
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "national-football-predictor/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _date_window(self, start_date: str | None, end_date: str | None) -> tuple[str, str]:
        if start_date and end_date:
            return start_date, end_date
        today = date.today()
        year = today.year
        tournament_start = date(year, 6, 1)
        tournament_end = date(year, 7, 31)
        start = start_date or min(today - timedelta(days=7), tournament_start).isoformat()
        end = end_date or max(today + timedelta(days=45), tournament_end).isoformat()
        return start, end

    def _fixture_from_event(self, event: dict[str, Any], home_team: str, away_team: str) -> dict[str, Any] | None:
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])
        requested_home = self._find_competitor(competitors, home_team)
        requested_away = self._find_competitor(competitors, away_team)
        if not requested_home or not requested_away:
            return None

        status = competition.get("status") or event.get("status") or {}
        status_type = status.get("type", {})
        completed = bool(status_type.get("completed")) or status_type.get("state") in FINISHED_STATES
        status_name = status_type.get("name") or status_type.get("shortDetail") or status_type.get("description") or ""
        completed = completed or status_name in FINISHED_STATUS_NAMES

        event_date = event.get("date") or competition.get("date") or ""
        actual_home = self._home_away_competitor(competitors, "home")
        actual_away = self._home_away_competitor(competitors, "away")
        return {
            "fixture_id": str(event.get("id", "")),
            "date": iso_date(event_date),
            "kickoff": event_date,
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_team": self._team_name(actual_home) if actual_home else None,
            "actual_away_team": self._team_name(actual_away) if actual_away else None,
            "home_goals": self._score(requested_home) if completed else None,
            "away_goals": self._score(requested_away) if completed else None,
            "completed": completed,
            "status": status_name,
            "status_detail": status_type.get("detail") or status_type.get("shortDetail") or "",
            "source": "espn-world-cup",
        }

    def _best_candidate(self, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date()

        def sort_key(fixture: dict[str, Any]) -> tuple[int, int]:
            fixture_date = datetime.fromisoformat(fixture["date"]).date()
            days = (fixture_date - today).days
            future_priority = 0 if days >= 0 else 1
            return future_priority, abs(days)

        return sorted(fixtures, key=sort_key)[0]

    def _find_competitor(self, competitors: list[dict[str, Any]], team_name: str) -> dict[str, Any] | None:
        query = normalize_provider_name(team_name)
        for competitor in competitors:
            names = self._team_names(competitor)
            normalized = [normalize_provider_name(name) for name in names if name]
            if query in normalized:
                return competitor
            if any(query in name or name in query for name in normalized):
                return competitor
        return None

    def _team_names(self, competitor: dict[str, Any]) -> list[str]:
        team = competitor.get("team", {})
        return [
            team.get("displayName", ""),
            team.get("shortDisplayName", ""),
            team.get("name", ""),
            team.get("abbreviation", ""),
            competitor.get("displayName", ""),
        ]

    def _team_name(self, competitor: dict[str, Any]) -> str:
        team = competitor.get("team", {})
        return team.get("displayName") or team.get("shortDisplayName") or team.get("name") or ""

    def _home_away_competitor(self, competitors: list[dict[str, Any]], home_away: str) -> dict[str, Any] | None:
        for competitor in competitors:
            if competitor.get("homeAway") == home_away:
                return competitor
        return None

    def _score(self, competitor: dict[str, Any]) -> int | None:
        score = competitor.get("score")
        if score in (None, ""):
            return None
        return int(float(score))


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
