from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aliases import TeamResolver
from .models import MatchRecord


DEFAULT_MODEL_STATE = {
    "version": 1,
    "learning_rate": 0.08,
    "weights": {
        "home_advantage_goals": 0.18,
        "elo_to_goals": 0.22,
        "form_to_goals": 0.20,
        "motivation_to_goals": 0.18,
        "injury_to_goals": 0.16,
        "corner_bias": 0.0,
        "goal_scale": 1.0,
    },
    "history": [],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class DataStore:
    def __init__(self, root: Path | None = None):
        self.root = root or project_root()
        self.data_dir = self.root / "data"
        self.alias_path = self.data_dir / "team_aliases.json"
        self.matches_path = self.data_dir / "matches.json"
        self.context_path = self.data_dir / "team_context.json"
        self.model_path = self.data_dir / "model_state.json"
        self.predictions_path = self.data_dir / "predictions.json"
        self.resolver = TeamResolver(self.alias_path)
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path, default in (
            (self.matches_path, []),
            (self.context_path, {}),
            (self.model_path, DEFAULT_MODEL_STATE),
            (self.predictions_path, []),
        ):
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_matches(self) -> list[MatchRecord]:
        return [MatchRecord.from_dict(item) for item in self._read_json(self.matches_path, [])]

    def save_matches(self, matches: list[MatchRecord]) -> None:
        ordered = sorted(matches, key=lambda item: (item.date, item.home_team, item.away_team))
        self._write_json(self.matches_path, [match.to_dict() for match in ordered])

    def add_or_update_match(self, record: MatchRecord) -> None:
        matches = self.load_matches()
        next_matches = []
        replaced = False
        key = (record.date, record.home_team, record.away_team)
        for match in matches:
            if (match.date, match.home_team, match.away_team) == key:
                next_matches.append(record)
                replaced = True
            else:
                next_matches.append(match)
        if not replaced:
            next_matches.append(record)
        self.save_matches(next_matches)

    def load_context(self) -> dict[str, Any]:
        return self._read_json(self.context_path, {})

    def save_context(self, context: dict[str, Any]) -> None:
        self._write_json(self.context_path, context)

    def team_context(self, team: str) -> dict[str, Any]:
        context = self.load_context()
        return context.get(team, {})

    def load_model_state(self) -> dict[str, Any]:
        state = self._read_json(self.model_path, DEFAULT_MODEL_STATE)
        merged = json.loads(json.dumps(DEFAULT_MODEL_STATE))
        merged.update({key: value for key, value in state.items() if key != "weights"})
        merged["weights"].update(state.get("weights", {}))
        return merged

    def save_model_state(self, state: dict[str, Any]) -> None:
        self._write_json(self.model_path, state)

    def save_prediction(self, prediction: dict[str, Any]) -> None:
        predictions = self._read_json(self.predictions_path, [])
        prediction = dict(prediction)
        prediction["created_at"] = datetime.now(timezone.utc).isoformat()
        predictions.append(prediction)
        self._write_json(self.predictions_path, predictions[-500:])

    def latest_prediction(self, home_team: str, away_team: str) -> dict[str, Any] | None:
        predictions = self._read_json(self.predictions_path, [])
        for prediction in reversed(predictions):
            if prediction.get("home_team") == home_team and prediction.get("away_team") == away_team:
                return prediction
        return None
