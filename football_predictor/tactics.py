from __future__ import annotations

from typing import Any


TACTIC_NUMERIC_FIELDS = {
    "possession_intent",
    "pressing",
    "line_height",
    "defensive_solidity",
    "attack_width",
    "central_progression",
    "directness",
    "chance_creation",
    "transition_attack",
    "transition_defense",
    "set_piece_threat",
    "tempo",
}


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def normalized_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(profile)
    for field in TACTIC_NUMERIC_FIELDS:
        normalized[field] = clamp01(normalized.get(field), 0.5)
    return normalized


def attack_matchup(attacking: dict[str, Any], defending: dict[str, Any]) -> float:
    attacking = normalized_profile(attacking)
    defending = normalized_profile(defending)

    creation = attacking["chance_creation"] * 0.32
    central = attacking["central_progression"] * (1.0 - defending["defensive_solidity"]) * 0.20
    width = attacking["attack_width"] * (1.0 - defending["transition_defense"]) * 0.14
    transition = attacking["transition_attack"] * max(0.0, defending["line_height"] - 0.42) * 0.18
    set_pieces = attacking["set_piece_threat"] * (1.0 - defending["defensive_solidity"] * 0.65) * 0.10
    control = attacking["possession_intent"] * (1.0 - defending["pressing"] * 0.55) * 0.06
    pressure_cost = defending["pressing"] * (1.0 - attacking["directness"]) * 0.04

    return creation + central + width + transition + set_pieces + control - pressure_cost


def tactical_edge(home_tactics: dict[str, Any], away_tactics: dict[str, Any]) -> float:
    home_attack = attack_matchup(home_tactics, away_tactics)
    away_attack = attack_matchup(away_tactics, home_tactics)
    return max(-0.75, min(0.75, home_attack - away_attack))


def tactical_tempo(home_tactics: dict[str, Any], away_tactics: dict[str, Any]) -> float:
    home_tactics = normalized_profile(home_tactics)
    away_tactics = normalized_profile(away_tactics)
    tempo = (
        home_tactics["tempo"]
        + away_tactics["tempo"]
        + home_tactics["pressing"] * 0.5
        + away_tactics["pressing"] * 0.5
        + home_tactics["directness"] * 0.4
        + away_tactics["directness"] * 0.4
    ) / 3.8
    return max(-0.35, min(0.35, tempo - 0.52))


def corner_tactical_boost(home_tactics: dict[str, Any], away_tactics: dict[str, Any]) -> float:
    home_tactics = normalized_profile(home_tactics)
    away_tactics = normalized_profile(away_tactics)
    width = home_tactics["attack_width"] + away_tactics["attack_width"]
    set_pieces = home_tactics["set_piece_threat"] + away_tactics["set_piece_threat"]
    pressing = home_tactics["pressing"] + away_tactics["pressing"]
    directness = home_tactics["directness"] + away_tactics["directness"]
    return (width - 1.05) * 0.65 + (set_pieces - 1.0) * 0.35 + (pressing - 1.0) * 0.20 + (directness - 1.0) * 0.18


def summarize_matchup(home_team: str, away_team: str, home_tactics: dict[str, Any], away_tactics: dict[str, Any]) -> dict[str, Any]:
    edge = tactical_edge(home_tactics, away_tactics)
    tempo = tactical_tempo(home_tactics, away_tactics)
    corner_boost = corner_tactical_boost(home_tactics, away_tactics)
    if edge > 0.08:
        edge_text = f"{home_team} has the cleaner tactical route to chances."
    elif edge < -0.08:
        edge_text = f"{away_team} has the cleaner tactical route to chances."
    else:
        edge_text = "Tactical routes to chances are close."

    return {
        "edge": round(edge, 3),
        "tempo": round(tempo, 3),
        "corner_boost": round(corner_boost, 3),
        "summary": edge_text,
        "home_route": route_line(home_tactics),
        "away_route": route_line(away_tactics),
    }


def route_line(profile: dict[str, Any]) -> str:
    profile = normalized_profile(profile)
    primary = profile.get("primary_attack", "mixed")
    block = profile.get("defensive_block", "mid")
    formation = profile.get("formation", "unknown")
    control = "high control" if profile["possession_intent"] >= 0.62 else "directer play" if profile["directness"] >= 0.62 else "mixed control"
    return f"{formation}: {primary}; {block} block; {control}."
