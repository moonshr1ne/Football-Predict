from __future__ import annotations

from .data_store import DataStore


def update_team_context(
    store: DataStore,
    team: str,
    motivation: float | None = None,
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
    if note:
        item["notes"].append(note)
    if clear_injuries:
        item["injuries"] = []
    if injury:
        item["injuries"].append(_parse_injury(injury))

    store.save_context(context)
    return item


def _parse_injury(value: str) -> dict:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) == 1:
        return {"player": parts[0], "status": "unknown", "impact": 0.25}
    if len(parts) == 2:
        return {"player": parts[0], "status": parts[1], "impact": 0.25}
    return {"player": parts[0], "status": parts[1], "impact": float(parts[2])}
