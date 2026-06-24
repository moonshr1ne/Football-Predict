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
LIVE_STATES = {"in"}
LIVE_STATUS_NAMES = {
    "STATUS_FIRST_HALF",
    "STATUS_HALFTIME",
    "STATUS_SECOND_HALF",
    "STATUS_EXTRA_TIME",
    "STATUS_PENALTY_SHOOTOUT",
    "1H",
    "HT",
    "2H",
    "ET",
    "P",
    "LIVE",
}


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
    summary_endpoint = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
    all_team_schedule_endpoint = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{team_id}/schedule"

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
        return self.enrich_fixture(self._best_candidate(candidates))

    def fixtures(self, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        start_date, end_date = self._date_window(start_date, end_date)
        payload = self._scoreboard(start_date, end_date)
        fixtures = []
        for event in payload.get("events", []):
            fixture = self._fixture_from_event_without_requested_order(event)
            if fixture:
                fixtures.append(fixture)
        return fixtures

    def participants(self) -> list[dict[str, Any]]:
        payload = self._scoreboard("2026-06-01", "2026-06-30")
        teams: dict[str, dict[str, Any]] = {}
        for event in payload.get("events", []):
            competition = (event.get("competitions") or [{}])[0]
            note = str(competition.get("altGameNote") or "")
            if "Group" not in note:
                continue
            for competitor in competition.get("competitors", []):
                team = competitor.get("team", {})
                name = team.get("displayName")
                team_id = team.get("id")
                if not name or not team_id:
                    continue
                teams[name] = {
                    "team": name,
                    "team_id": str(team_id),
                    "abbreviation": team.get("abbreviation", ""),
                    "logo": team.get("logo", ""),
                    "source": "espn-world-cup",
                }
        return sorted(teams.values(), key=lambda item: item["team"])

    def team_recent_fixtures(self, team_id: str, limit: int = 10) -> list[dict[str, Any]]:
        payload = self._team_schedule(team_id)
        fixtures = []
        for event in payload.get("events", []):
            fixture = self._fixture_from_event_without_requested_order(event, source="espn-team-schedule")
            if fixture and fixture.get("completed"):
                fixtures.append(fixture)
        fixtures.sort(key=lambda item: item.get("date", ""), reverse=True)
        return fixtures[:limit]

    def enrich_fixture(self, fixture: dict[str, Any]) -> dict[str, Any]:
        fixture_id = fixture.get("fixture_id")
        if not fixture_id:
            return fixture
        try:
            summary = self._summary(str(fixture_id))
        except ProviderError:
            return fixture

        lineups = self._lineups_from_summary(summary)
        home_team = fixture.get("home_team", "")
        away_team = fixture.get("away_team", "")
        home_lineup = self._matching_lineup(lineups, home_team) or self._empty_lineup(home_team)
        away_lineup = self._matching_lineup(lineups, away_team) or self._empty_lineup(away_team)

        key_players = self._key_players_from_summary(summary)
        enriched = dict(fixture)
        enriched["lineups"] = {
            home_team: home_lineup,
            away_team: away_lineup,
        }
        enriched["key_players"] = {
            home_team: self._matching_key_players(key_players, home_team),
            away_team: self._matching_key_players(key_players, away_team),
        }
        enriched["home_formation"] = home_lineup.get("formation")
        enriched["away_formation"] = away_lineup.get("formation")
        enriched["lineup_status"] = (
            "confirmed" if home_lineup.get("confirmed") and away_lineup.get("confirmed") else "not_released"
        )
        return enriched

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
            "home_corners": fixture.get("home_corners"),
            "away_corners": fixture.get("away_corners"),
            "home_possession": fixture.get("home_possession"),
            "away_possession": fixture.get("away_possession"),
            "home_shots": fixture.get("home_shots"),
            "away_shots": fixture.get("away_shots"),
            "home_shots_on_target": fixture.get("home_shots_on_target"),
            "away_shots_on_target": fixture.get("away_shots_on_target"),
            "home_fouls": fixture.get("home_fouls"),
            "away_fouls": fixture.get("away_fouls"),
            "source": fixture["source"],
            "fixture": fixture,
        }

    def _scoreboard(self, start_date: str, end_date: str) -> dict[str, Any]:
        params = {"limit": 1000, "dates": f"{compact_date(start_date)}-{compact_date(end_date)}"}
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "national-football-predictor/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"ESPN scoreboard unavailable: {exc}") from exc

    def _summary(self, event_id: str) -> dict[str, Any]:
        params = {"event": event_id}
        url = f"{self.summary_endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"User-Agent": "national-football-predictor/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"ESPN summary unavailable: {exc}") from exc

    def _team_schedule(self, team_id: str) -> dict[str, Any]:
        url = self.all_team_schedule_endpoint.format(team_id=urllib.parse.quote(str(team_id)))
        request = urllib.request.Request(url, headers={"User-Agent": "national-football-predictor/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"ESPN team schedule unavailable: {exc}") from exc

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

    def _fixture_from_event(
        self,
        event: dict[str, Any],
        home_team: str,
        away_team: str,
        source: str = "espn-world-cup",
    ) -> dict[str, Any] | None:
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
        status_detail = status_type.get("detail") or status_type.get("shortDetail") or ""
        in_progress = (
            not completed
            and (
                status_type.get("state") in LIVE_STATES
                or status_name in LIVE_STATUS_NAMES
                or bool(re.search(r"\d+\s*'", str(status_detail)))
            )
        )
        has_score = completed or in_progress

        event_date = event.get("date") or competition.get("date") or ""
        actual_home = self._home_away_competitor(competitors, "home")
        actual_away = self._home_away_competitor(competitors, "away")
        requested_home_stats = self._stats(requested_home)
        requested_away_stats = self._stats(requested_away)
        competition_label = (
            competition.get("altGameNote")
            or event.get("league", {}).get("name")
            or event.get("season", {}).get("displayName")
            or "ESPN soccer"
        )
        return {
            "fixture_id": str(event.get("id", "")),
            "date": iso_date(event_date),
            "kickoff": event_date,
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_team": self._team_name(actual_home) if actual_home else None,
            "actual_away_team": self._team_name(actual_away) if actual_away else None,
            "home_goals": self._score(requested_home) if has_score else None,
            "away_goals": self._score(requested_away) if has_score else None,
            "home_corners": requested_home_stats.get("wonCorners"),
            "away_corners": requested_away_stats.get("wonCorners"),
            "home_possession": requested_home_stats.get("possessionPct"),
            "away_possession": requested_away_stats.get("possessionPct"),
            "home_shots": requested_home_stats.get("totalShots"),
            "away_shots": requested_away_stats.get("totalShots"),
            "home_shots_on_target": requested_home_stats.get("shotsOnTarget"),
            "away_shots_on_target": requested_away_stats.get("shotsOnTarget"),
            "home_fouls": requested_home_stats.get("foulsCommitted"),
            "away_fouls": requested_away_stats.get("foulsCommitted"),
            "completed": completed,
            "in_progress": in_progress,
            "status": status_name,
            "status_detail": status_detail,
            "competition": competition_label,
            "source": source,
        }

    def _fixture_from_event_without_requested_order(
        self,
        event: dict[str, Any],
        source: str = "espn-world-cup",
    ) -> dict[str, Any] | None:
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors", [])
        actual_home = self._home_away_competitor(competitors, "home")
        actual_away = self._home_away_competitor(competitors, "away")
        if not actual_home or not actual_away:
            return None
        home_team = self._team_name(actual_home)
        away_team = self._team_name(actual_away)
        return self._fixture_from_event(event, home_team, away_team, source=source)

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
        if isinstance(score, dict):
            score = score.get("value", score.get("displayValue"))
        if score in (None, ""):
            return None
        return int(float(score))

    def _stats(self, competitor: dict[str, Any]) -> dict[str, float]:
        stats: dict[str, float] = {}
        for stat in competitor.get("statistics") or []:
            name = stat.get("name")
            value = stat.get("displayValue")
            if isinstance(value, dict):
                value = value.get("value", value.get("displayValue"))
            if name and value not in (None, ""):
                try:
                    stats[name] = float(str(value).replace("%", ""))
                except ValueError:
                    continue
        return stats

    def _lineups_from_summary(self, summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
        lineups: dict[str, dict[str, Any]] = {}
        for roster_block in summary.get("rosters") or []:
            team = roster_block.get("team", {})
            team_name = team.get("displayName") or team.get("name") or ""
            if not team_name:
                continue
            roster = roster_block.get("roster") or []
            starters = [self._player_from_roster_item(item) for item in roster if item.get("starter")]
            bench = [self._player_from_roster_item(item) for item in roster if not item.get("starter")]
            starters = [item for item in starters if item.get("name")]
            bench = [item for item in bench if item.get("name")]
            formation = self._formation_from_starters(starters)
            lineups[team_name] = {
                "team": team_name,
                "confirmed": len(starters) >= 11,
                "formation": formation,
                "starters": starters,
                "bench": bench,
                "source": "espn-summary-rosters",
            }
        return lineups

    def _player_from_roster_item(self, item: dict[str, Any]) -> dict[str, Any]:
        athlete = item.get("athlete") or {}
        position = athlete.get("position") or item.get("position") or {}
        return {
            "name": athlete.get("displayName") or athlete.get("shortName") or athlete.get("name") or "",
            "position": position.get("displayName") or position.get("name") or position.get("abbreviation") or "",
            "formation_place": item.get("formationPlace"),
        }

    def _formation_from_starters(self, starters: list[dict[str, Any]]) -> str | None:
        if len(starters) < 11:
            return None
        defenders = midfielders = forwards = 0
        defensive_midfielders = attacking_midfielders = 0
        for player in starters:
            category = self._position_category(player.get("position", ""))
            if category == "G":
                continue
            if category == "D":
                defenders += 1
            elif category == "F":
                forwards += 1
            else:
                midfielders += 1
                position = normalize_provider_name(str(player.get("position", "")))
                defensive_midfielders += 1 if "defensive" in position else 0
                attacking_midfielders += 1 if "attacking" in position else 0

        if defenders <= 0 or defenders + midfielders + forwards != 10:
            return None
        if defenders == 4 and midfielders == 5 and forwards == 1 and defensive_midfielders:
            return "4-1-4-1"
        if defenders == 4 and midfielders == 4 and forwards == 2 and attacking_midfielders:
            return "4-2-3-1"
        return f"{defenders}-{midfielders}-{forwards}"

    def _position_category(self, position: str) -> str:
        normalized = normalize_provider_name(position)
        if "goalkeeper" in normalized or normalized in {"g", "keeper"}:
            return "G"
        if "attacking midfielder left" in normalized or "attacking midfielder right" in normalized:
            return "F"
        if "back" in normalized or "defender" in normalized or normalized == "d":
            return "D"
        if "forward" in normalized or "striker" in normalized or "winger" in normalized or normalized == "f":
            return "F"
        return "M"

    def _matching_lineup(self, lineups: dict[str, dict[str, Any]], team_name: str) -> dict[str, Any] | None:
        query = normalize_provider_name(team_name)
        for name, lineup in lineups.items():
            normalized = normalize_provider_name(name)
            if query == normalized or query in normalized or normalized in query:
                return lineup
        return None

    def _empty_lineup(self, team_name: str) -> dict[str, Any]:
        return {
            "team": team_name,
            "confirmed": False,
            "formation": None,
            "starters": [],
            "bench": [],
            "source": "not-released",
        }

    def _key_players_from_summary(self, summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        by_team: dict[str, dict[str, dict[str, Any]]] = {}
        for team_block in summary.get("leaders") or []:
            team = team_block.get("team", {})
            team_name = team.get("displayName") or team.get("name") or ""
            if not team_name:
                continue
            players = by_team.setdefault(team_name, {})
            for category in team_block.get("leaders") or []:
                category_name = str(category.get("name") or category.get("displayName") or "")
                impact = self._leader_impact(category_name)
                if impact <= 0:
                    continue
                for leader in (category.get("leaders") or [])[:3]:
                    athlete = leader.get("athlete") or {}
                    name = athlete.get("displayName") or athlete.get("shortName") or athlete.get("name")
                    if not name:
                        continue
                    key = normalize_provider_name(name)
                    existing = players.setdefault(key, {"name": name, "impact": 0.0, "roles": []})
                    existing["impact"] = max(float(existing.get("impact", 0.0)), impact)
                    existing.setdefault("roles", []).append(category_name)
        return {
            team: sorted(players.values(), key=lambda item: item["impact"], reverse=True)[:6]
            for team, players in by_team.items()
        }

    def _leader_impact(self, category_name: str) -> float:
        normalized = normalize_provider_name(category_name)
        if any(token in normalized for token in ("goal", "shot", "assist")):
            return 0.13
        if any(token in normalized for token in ("pass", "chance")):
            return 0.08
        if any(token in normalized for token in ("defensive", "save")):
            return 0.07
        return 0.0

    def _matching_key_players(self, key_players: dict[str, list[dict[str, Any]]], team_name: str) -> list[dict[str, Any]]:
        query = normalize_provider_name(team_name)
        for name, players in key_players.items():
            normalized = normalize_provider_name(name)
            if query == normalized or query in normalized or normalized in query:
                return players
        return []


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
