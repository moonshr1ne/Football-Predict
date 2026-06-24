from __future__ import annotations

from .models import MatchRecord, TeamStats


def last_matches(matches: list[MatchRecord], team: str, limit: int = 10) -> list[MatchRecord]:
    finished = [match for match in matches if match.involves(team) and match.is_finished()]
    return sorted(finished, key=lambda item: item.date, reverse=True)[:limit]


def build_team_stats(matches: list[MatchRecord], team: str, limit: int = 10) -> TeamStats:
    stats = TeamStats(team=team)
    stats.recent = last_matches(matches, team, limit=limit)
    for match in stats.recent:
        gf = match.goals_for(team)
        ga = match.goals_against(team)
        if gf is None or ga is None:
            continue
        stats.sample_size += 1
        stats.goals_for += gf
        stats.goals_against += ga
        if gf > ga:
            stats.wins += 1
        elif gf == ga:
            stats.draws += 1
        else:
            stats.losses += 1
        if ga == 0:
            stats.clean_sheets += 1
        if gf == 0:
            stats.failed_to_score += 1

        corners_for = match.corners_for(team)
        corners_against = match.corners_against(team)
        if corners_for is not None and corners_against is not None:
            stats.corner_samples += 1
            stats.corners_for += float(corners_for)
            stats.corners_against += float(corners_against)
    return stats
