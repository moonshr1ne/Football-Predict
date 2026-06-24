from __future__ import annotations

from .data_store import DataStore


def update_team_context(
    store: DataStore,
    team: str,
    motivation: float | None = None,
    lineup_strength: float | None = None,
    note: str | None = None,
    injury: str | None = None,
    clear_injuries: bool = False,
) -> dict:
    context = store.load_context()
    item = context.setdefault(team, {"injuries": [], "motivation": {"level": 0.5, "notes": []}, "notes": []})
    item.setdefault("injuries", [])
    item.setdefault("motivation", {"level": 0.5, "notes": []})
    item.setdefault("notes", [])

    if motivation is not None:
        item["motivation"]["level"] = max(0.0, min(1.0, float(motivation)))
    if lineup_strength is not None:
        item["lineup_strength"] = max(0.0, min(1.0, float(lineup_strength)))
    if note:
        item["notes"].append(note)
    if clear_injuries:
        item["injuries"] = []
    if injury:
        item["injuries"].append(_parse_injury(injury))

    store.save_context(context)
    return item


def update_team_tactics(store: DataStore, team: str, **updates) -> dict:
    profiles = store.load_tactical_profiles()
    profile = profiles.setdefault(team, {})
    numeric_fields = {
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
    for key, value in updates.items():
        if value is None:
            continue
        if key == "note":
            profile.setdefault("notes", []).append(value)
            continue
        if key in numeric_fields:
            profile[key] = max(0.0, min(1.0, float(value)))
        else:
            profile[key] = value
    profiles[team] = profile
    store.save_tactical_profiles(profiles)
    return store.team_tactics(team)


def _parse_injury(value: str) -> dict:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) == 1:
        return {"player": parts[0], "status": "unknown", "impact": 0.25}
    if len(parts) == 2:
        return {"player": parts[0], "status": parts[1], "impact": 0.25}
    return {"player": parts[0], "status": parts[1], "impact": float(parts[2])}
