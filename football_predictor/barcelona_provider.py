from __future__ import annotations

import json
import html
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from difflib import get_close_matches
from typing import Any

from .providers import EspnWorldCupProvider, ProviderError, normalize_provider_name


BARCELONA_ID = "83"
BARCELONA_NAME = "Barcelona"
OFFICIAL_SQUAD_URL = "https://www.fcbarcelona.com/en/football/first-team/players"
OFFICIAL_TRANSFERS_URL = "https://www.fcbarcelona.com/en/transfer-market/"

LEAGUES = {
    "esp.1": "Ла Лига",
    "uefa.champions": "Лига чемпионов УЕФА",
}

RU_ALIASES = {
    "барселона": "Barcelona",
    "барса": "Barcelona",
    "реал": "Real Madrid",
    "реал мадрид": "Real Madrid",
    "атлетико": "Atletico Madrid",
    "атлетико мадрид": "Atletico Madrid",
    "атлетик": "Athletic Club",
    "атлетик бильбао": "Athletic Club",
    "сосьедад": "Real Sociedad",
    "реал сосьедад": "Real Sociedad",
    "севилья": "Sevilla",
    "вильярреал": "Villarreal",
    "валенсия": "Valencia",
    "бетис": "Real Betis",
    "жирона": "Girona",
    "хетафе": "Getafe",
    "осасуна": "Osasuna",
    "селта": "Celta Vigo",
    "эспаньол": "Espanyol",
    "эльче": "Elche",
    "алавес": "Alavés",
    "мальорка": "Mallorca",
    "райо": "Rayo Vallecano",
    "леванте": "Levante",
    "овьедо": "Real Oviedo",
    "гранада": "Granada",
    "кадис": "Cádiz",
    "лас пальмас": "Las Palmas",
    "псж": "Paris Saint-Germain",
    "пари сен жермен": "Paris Saint-Germain",
    "манчестер сити": "Manchester City",
    "ман сити": "Manchester City",
    "манчестер юнайтед": "Manchester United",
    "арсенал": "Arsenal",
    "ливерпуль": "Liverpool",
    "челси": "Chelsea",
    "бавария": "Bayern Munich",
    "боруссия дортмунд": "Borussia Dortmund",
    "интер": "Internazionale",
    "интер милан": "Internazionale",
    "милан": "AC Milan",
    "ювентус": "Juventus",
    "наполи": "Napoli",
    "бенфика": "Benfica",
    "спортинг": "Sporting CP",
    "порту": "FC Porto",
    "аякс": "Ajax Amsterdam",
    "фейеноорд": "Feyenoord Rotterdam",
    "аталанта": "Atalanta",
    "марсель": "Marseille",
    "монако": "AS Monaco",
    "байер": "Bayer Leverkusen",
    "лейпциг": "RB Leipzig",
}


class BarcelonaProvider:
    user_agent = "barcelona-match-lab/1.0"

    def __init__(self, timeout: int = 25):
        self.timeout = timeout
        self.parser = EspnWorldCupProvider(timeout=timeout)
        self._squad_cache: dict[str, Any] | None = None
        self._squad_cached_at: datetime | None = None

    def fetch_current_squad(self, force: bool = False) -> dict[str, Any]:
        """Return the current first-team roster from FC Barcelona's site."""
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._squad_cache
            and self._squad_cached_at
            and now - self._squad_cached_at < timedelta(hours=2)
        ):
            return dict(self._squad_cache)

        page = self._get_text(OFFICIAL_SQUAD_URL, "official squad")
        players = self._official_squad_from_html(page)
        if len(players) < 18:
            raise ProviderError("FC Barcelona official squad page returned an incomplete roster.")

        recent_signings: list[str] = []
        try:
            transfers = self._get_text(OFFICIAL_TRANSFERS_URL, "official transfers")
            recent_signings = self._recent_signings_from_html(transfers)
        except ProviderError:
            pass

        indexed = {normalize_provider_name(player["name"]): player for player in players}
        for name in recent_signings:
            key = normalize_provider_name(name)
            if key not in indexed:
                player = {
                    "name": name,
                    "position": "Forward",
                    "number": None,
                    "source": "fcbarcelona-transfer-market",
                    "recent_signing": True,
                }
                players.append(player)
                indexed[key] = player
            else:
                indexed[key]["recent_signing"] = True

        payload = {
            "team": BARCELONA_NAME,
            "players": players,
            "active_names": [player["name"] for player in players],
            "recent_signings": recent_signings,
            "fetched_at": now.isoformat(),
            "source": "FC Barcelona official first-team squad",
            "source_url": OFFICIAL_SQUAD_URL,
        }
        self._squad_cache = payload
        self._squad_cached_at = now
        return dict(payload)

    def season_windows(self, today: date | None = None, seasons: int = 4) -> list[tuple[str, str]]:
        current = today or date.today()
        first_start_year = current.year if current.month >= 7 else current.year - 1
        years = range(first_start_year - seasons + 1, first_start_year + 1)
        return [(f"{year}-07-01", f"{year + 1}-06-30") for year in years]

    def fetch_universe(self, windows: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
        fixtures: dict[str, dict[str, Any]] = {}
        for start, end in windows or self.season_windows():
            for league_code in LEAGUES:
                for event in self._scoreboard(league_code, start, end).get("events", []):
                    fixture = self.fixture_from_event(event, league_code)
                    if fixture and fixture.get("fixture_id"):
                        fixtures[str(fixture["fixture_id"])] = fixture
        return sorted(fixtures.values(), key=lambda item: (item.get("kickoff", ""), item.get("fixture_id", "")))

    def fixture_from_event(self, event: dict[str, Any], league_code: str) -> dict[str, Any] | None:
        fixture = self.parser._fixture_from_event_without_requested_order(
            event,
            source=f"espn-{league_code}",
        )
        if not fixture:
            return None
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        home = self.parser._home_away_competitor(competitors, "home") or {}
        away = self.parser._home_away_competitor(competitors, "away") or {}
        status_type = ((competition.get("status") or event.get("status") or {}).get("type") or {})
        fixture.update(
            {
                "home_team_id": str((home.get("team") or {}).get("id") or ""),
                "away_team_id": str((away.get("team") or {}).get("id") or ""),
                "home_logo": (home.get("team") or {}).get("logo") or "",
                "away_logo": (away.get("team") or {}).get("logo") or "",
                "competition_code": league_code,
                "competition": LEAGUES[league_code],
                "stage": self._stage(event, competition, league_code),
                "venue": ((competition.get("venue") or {}).get("fullName") or ""),
                "attendance": competition.get("attendance"),
                "status_detail": status_type.get("detail") or status_type.get("shortDetail") or fixture.get("status_detail", ""),
                "details_loaded": False,
            }
        )
        return fixture

    def enrich_fixture(self, fixture: dict[str, Any]) -> dict[str, Any]:
        fixture_id = str(fixture.get("fixture_id") or "")
        if not fixture_id:
            return fixture
        league_code = str(fixture.get("competition_code") or "")
        if league_code not in LEAGUES:
            return fixture
        summary = self._summary(league_code, fixture_id)
        enriched = dict(fixture)
        teams = self._summary_team_stats(summary)
        home_name = str(fixture.get("home_team") or "")
        away_name = str(fixture.get("away_team") or "")
        enriched["home_stats"] = self._matching_value(teams, home_name) or {}
        enriched["away_stats"] = self._matching_value(teams, away_name) or {}

        lineups = self._summary_lineups(summary)
        home_lineup = self._matching_value(lineups, home_name) or self._empty_lineup(home_name)
        away_lineup = self._matching_value(lineups, away_name) or self._empty_lineup(away_name)
        enriched["lineups"] = {home_name: home_lineup, away_name: away_lineup}
        enriched["home_formation"] = home_lineup.get("formation")
        enriched["away_formation"] = away_lineup.get("formation")
        enriched["lineup_status"] = (
            "confirmed" if home_lineup.get("confirmed") and away_lineup.get("confirmed") else "not_released"
        )
        referee = self.parser._referee_from_summary(summary)
        if referee:
            enriched["referee"] = referee
        market = self._market_from_summary(summary)
        if market:
            enriched["market"] = market
        enriched["details_loaded"] = bool(teams or lineups or referee)
        enriched["details_version"] = 2
        enriched["summary_source"] = "espn-match-summary"
        return enriched

    def resolve_opponent(self, query: str, fixtures: list[dict[str, Any]]) -> str:
        raw = query.strip().strip("'\"")
        if not raw:
            raise ValueError("Введите название соперника Барселоны.")
        normalized = normalize_provider_name(raw)
        aliased = RU_ALIASES.get(raw.casefold()) or RU_ALIASES.get(normalized)

        candidates: dict[str, str] = {}
        for fixture in fixtures:
            for name in (fixture.get("home_team"), fixture.get("away_team")):
                if name and normalize_provider_name(str(name)) != "barcelona":
                    candidates[normalize_provider_name(str(name))] = str(name)
        if aliased:
            alias_norm = normalize_provider_name(aliased)
            if alias_norm in candidates:
                return candidates[alias_norm]
            return aliased
        if normalized in candidates:
            return candidates[normalized]
        partial = [name for key, name in candidates.items() if normalized in key or key in normalized]
        if len(partial) == 1:
            return partial[0]
        close = get_close_matches(normalized, list(candidates), n=1, cutoff=0.62)
        if close:
            return candidates[close[0]]
        raise ValueError(f"Не распознал соперника «{raw}». Начните вводить название из расписания.")

    def find_fixture(self, opponent: str, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = normalize_provider_name(opponent)
        candidates = [
            item
            for item in fixtures
            if self._contains_team(item, BARCELONA_NAME)
            and any(
                self._same_name(str(item.get(key) or ""), normalized)
                for key in ("home_team", "away_team")
            )
        ]
        if not candidates:
            raise ProviderError(f"Не нашел матч Барселоны против {opponent} в Ла Лиге или ЛЧ.")

        now = datetime.now(timezone.utc)

        def kickoff(item: dict[str, Any]) -> datetime:
            raw = str(item.get("kickoff") or f"{item.get('date')}T12:00:00Z")
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

        upcoming = [item for item in candidates if kickoff(item) > now and not item.get("completed")]
        if upcoming:
            return min(upcoming, key=kickoff)
        return max(candidates, key=kickoff)

    def _scoreboard(self, league_code: str, start: str, end: str) -> dict[str, Any]:
        params = {"limit": 1000, "dates": f"{start.replace('-', '')}-{end.replace('-', '')}"}
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/scoreboard?{urllib.parse.urlencode(params)}"
        return self._get(url, "scoreboard")

    def _summary(self, league_code: str, event_id: str) -> dict[str, Any]:
        params = urllib.parse.urlencode({"event": event_id})
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_code}/summary?{params}"
        return self._get(url, "match summary")

    def _get(self, url: str, label: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"ESPN {label} unavailable: {exc}") from exc

    def _get_text(self, url: str, label: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:
            raise ProviderError(f"{label} unavailable: {exc}") from exc

    def _official_squad_from_html(self, page: str) -> list[dict[str, Any]]:
        blocks = re.findall(
            r'<a[^>]+href="(?:https://www\.fcbarcelona\.com)?/en/football/first-team/players/[^\"]+"'
            r'[^>]*class="team-person[^\"]*"[^>]*>(.*?)</a>',
            page,
            flags=re.IGNORECASE | re.DOTALL,
        )
        players: list[dict[str, Any]] = []
        seen: set[str] = set()
        for block in blocks:
            first = self._class_text(block, "team-person__first-name")
            last = self._class_text(block, "team-person__last-name")
            position = self._class_text(block, "team-person__position-meta")
            number_match = re.search(r'class="team-person__number"[^>]*aria-label="([^\"]*)"', block, re.IGNORECASE)
            name = " ".join(value for value in (first, last) if value).strip()
            key = normalize_provider_name(name)
            if not key or not position or key in seen:
                continue
            seen.add(key)
            players.append(
                {
                    "name": name,
                    "position": position,
                    "number": number_match.group(1).strip() if number_match else None,
                    "source": "fcbarcelona-official-squad",
                    "recent_signing": False,
                }
            )
        return players

    def _recent_signings_from_html(self, page: str) -> list[str]:
        titles = [self._clean_html(value) for value in re.findall(r'class="thumbnail__title"[^>]*>(.*?)</div>', page, re.IGNORECASE | re.DOTALL)]
        signings: list[str] = []
        for title in titles:
            name = None
            match = re.fullmatch(r"FC Barcelona sign (.+)", title, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
            match = match or re.fullmatch(r"(.+) joins Barça", title, re.IGNORECASE)
            if match and name is None:
                name = match.group(1).strip()
            if re.fullmatch(r"Adeyemi, second signing", title, re.IGNORECASE):
                name = "Karim Adeyemi"
            if name and normalize_provider_name(name) not in {normalize_provider_name(item) for item in signings}:
                signings.append(name)
        return signings

    def _class_text(self, block: str, class_name: str) -> str:
        match = re.search(
            rf'class="{re.escape(class_name)}[^\"]*"[^>]*>(.*?)</(?:span|li)>',
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return self._clean_html(match.group(1)) if match else ""

    def _clean_html(self, value: str) -> str:
        return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value))).strip()

    def _summary_team_stats(self, summary: dict[str, Any]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for block in (summary.get("boxscore") or {}).get("teams") or []:
            team = block.get("team") or {}
            name = team.get("displayName") or team.get("name") or ""
            if not name:
                continue
            stats: dict[str, float] = {}
            for item in block.get("statistics") or []:
                key = item.get("name")
                value = item.get("displayValue", item.get("value"))
                if not key or value in (None, ""):
                    continue
                try:
                    stats[str(key)] = float(str(value).replace("%", ""))
                except ValueError:
                    continue
            result[name] = stats
        return result

    def _summary_lineups(self, summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for block in summary.get("rosters") or []:
            team = block.get("team") or {}
            name = team.get("displayName") or team.get("name") or ""
            if not name:
                continue
            roster = block.get("roster") or []
            starters = [self.parser._player_from_roster_item(item) for item in roster if item.get("starter")]
            bench = [self.parser._player_from_roster_item(item) for item in roster if not item.get("starter")]
            starters = [item for item in starters if item.get("name")]
            bench = [item for item in bench if item.get("name")]
            # ESPN exposes the submitted formation directly. It is much more
            # reliable than inferring a shape from broad player positions.
            formation = self._clean_formation(block.get("formation")) or self.parser._formation_from_starters(starters)
            result[name] = {
                "team": name,
                "confirmed": len(starters) >= 11,
                "formation": formation,
                "starters": starters,
                "bench": bench,
                "source": "espn-summary-rosters",
            }
        return result

    def _market_from_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        pickcenter = summary.get("pickcenter") or {}
        if isinstance(pickcenter, list):
            pickcenter = pickcenter[0] if pickcenter else {}
        if not isinstance(pickcenter, dict):
            pickcenter = {}
        if not pickcenter:
            odds = summary.get("odds") or []
            pickcenter = odds[0] if isinstance(odds, list) and odds else odds
        if not isinstance(pickcenter, dict):
            return {}
        home = (pickcenter.get("homeTeamOdds") or {}).get("moneyLine")
        away = (pickcenter.get("awayTeamOdds") or {}).get("moneyLine")
        draw = (pickcenter.get("drawOdds") or {}).get("moneyLine")
        probabilities = {
            "home": self._american_probability(home),
            "draw": self._american_probability(draw),
            "away": self._american_probability(away),
        }
        available = {key: value for key, value in probabilities.items() if value is not None}
        if len(available) == 3:
            total = sum(available.values()) or 1.0
            probabilities = {key: value / total for key, value in available.items()}
        else:
            probabilities = {}
        total_line = pickcenter.get("overUnder")
        if total_line is None:
            raw_line = (((pickcenter.get("total") or {}).get("over") or {}).get("close") or {}).get("line")
            match = re.search(r"\d+(?:\.\d+)?", str(raw_line or ""))
            total_line = float(match.group(0)) if match else None
        if not probabilities and total_line is None:
            return {}
        return {
            "probabilities": probabilities,
            "home_moneyline": home,
            "draw_moneyline": draw,
            "away_moneyline": away,
            "total_line": None if total_line is None else float(total_line),
            "provider": (pickcenter.get("provider") or {}).get("name"),
            "source": "espn-pre-match-market",
        }

    def _american_probability(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            odds = float(value)
        except (TypeError, ValueError):
            return None
        return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / (-odds + 100.0)

    def _clean_formation(self, value: Any) -> str | None:
        if not value:
            return None
        match = re.search(r"\d(?:-\d){2,4}", str(value))
        return match.group(0) if match else None

    def _matching_value(self, mapping: dict[str, Any], team_name: str) -> Any:
        query = normalize_provider_name(team_name)
        for name, value in mapping.items():
            candidate = normalize_provider_name(name)
            if query == candidate or query in candidate or candidate in query:
                return value
        return None

    def _empty_lineup(self, team: str) -> dict[str, Any]:
        return {"team": team, "confirmed": False, "formation": None, "starters": [], "bench": [], "source": "not-released"}

    def _contains_team(self, fixture: dict[str, Any], team: str) -> bool:
        query = normalize_provider_name(team)
        return any(
            self._same_name(str(fixture.get(key) or ""), query)
            for key in ("home_team", "away_team")
        )

    def _same_name(self, candidate: str, normalized_query: str) -> bool:
        normalized = normalize_provider_name(candidate)
        return normalized == normalized_query or normalized_query in normalized or normalized in normalized_query

    def _stage(self, event: dict[str, Any], competition: dict[str, Any], league_code: str) -> str:
        note = str(competition.get("altGameNote") or "").strip()
        if note:
            return note
        detail = str((((competition.get("status") or {}).get("type") or {}).get("detail") or ""))
        if league_code == "esp.1":
            match = re.search(r"(?:Jornada|Matchday)\s*(\d+)", detail, re.IGNORECASE)
            return f"Тур {match.group(1)}" if match else "Ла Лига"
        season_type = event.get("season", {}).get("type", {})
        return str(season_type.get("name") or season_type.get("abbreviation") or "Лига чемпионов")
