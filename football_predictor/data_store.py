from __future__ import annotations

import json
import time
import uuid
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
        "class_to_goals": 0.10,
        "chance_to_goals": 0.06,
        "motivation_to_goals": 0.18,
        "injury_to_goals": 0.16,
        "lineup_to_goals": 0.24,
        "tactics_to_goals": 0.24,
        "tactics_to_corners": 1.15,
        "world_cup_intensity_goals": 0.05,
        "corner_bias": 0.0,
        "goal_scale": 1.0,
    },
    "history": [],
    "trained_match_keys": [],
}

DEFAULT_KEY_PLAYERS = {
    "Argentina": [
        {"name": "Lionel Messi", "impact": 0.18, "roles": ["creator", "finisher"]},
        {"name": "Lautaro Martínez", "impact": 0.12, "roles": ["finisher"]},
        {"name": "Julián Álvarez", "impact": 0.11, "roles": ["pressing forward"]},
    ],
    "Brazil": [
        {"name": "Vinícius Júnior", "impact": 0.17, "roles": ["creator", "winger"]},
        {"name": "Rodrygo", "impact": 0.13, "roles": ["creator"]},
        {"name": "Bruno Guimarães", "impact": 0.10, "roles": ["midfield"]},
    ],
    "England": [
        {"name": "Harry Kane", "impact": 0.17, "roles": ["finisher"]},
        {"name": "Jude Bellingham", "impact": 0.16, "roles": ["creator", "midfield"]},
        {"name": "Bukayo Saka", "impact": 0.13, "roles": ["winger"]},
    ],
    "France": [
        {"name": "Kylian Mbappé", "impact": 0.20, "roles": ["creator", "finisher"]},
        {"name": "Michael Olise", "impact": 0.13, "roles": ["creator"]},
        {"name": "Ousmane Dembélé", "impact": 0.13, "roles": ["creator"]},
    ],
    "Germany": [
        {"name": "Florian Wirtz", "impact": 0.15, "roles": ["creator"]},
        {"name": "Jamal Musiala", "impact": 0.15, "roles": ["creator"]},
        {"name": "Kai Havertz", "impact": 0.11, "roles": ["finisher"]},
    ],
    "Norway": [
        {"name": "Erling Haaland", "impact": 0.20, "roles": ["finisher"]},
        {"name": "Martin Ødegaard", "impact": 0.16, "roles": ["creator"]},
    ],
    "Portugal": [
        {"name": "Bruno Fernandes", "impact": 0.15, "roles": ["creator"]},
        {"name": "Bernardo Silva", "impact": 0.13, "roles": ["creator"]},
        {"name": "Cristiano Ronaldo", "impact": 0.11, "roles": ["finisher"]},
    ],
    "Spain": [
        {"name": "Lamine Yamal", "impact": 0.18, "roles": ["creator", "winger"]},
        {"name": "Pedri", "impact": 0.14, "roles": ["midfield"]},
        {"name": "Rodri", "impact": 0.15, "roles": ["midfield", "control"]},
        {"name": "Nico Williams", "impact": 0.12, "roles": ["winger"]},
    ],
    "Uruguay": [
        {"name": "Federico Valverde", "impact": 0.15, "roles": ["midfield"]},
        {"name": "Darwin Núñez", "impact": 0.13, "roles": ["finisher"]},
        {"name": "Ronald Araújo", "impact": 0.11, "roles": ["defense"]},
    ],
}

DEFAULT_MATCH_CONTEXT = {
    "competition": "FIFA World Cup",
    "importance": 1.0,
    "motivation_floor": 0.92,
    "lineup_strength_floor": 0.92,
    "notes": [
        "World Cup mode: every match is treated as high-importance, with near-first-choice lineups unless overridden."
    ],
}

DEFAULT_TACTICAL_PROFILE = {
    "formation": "4-2-3-1",
    "style": "balanced",
    "build_up": "mixed",
    "primary_attack": "mixed",
    "defensive_block": "mid",
    "possession_intent": 0.55,
    "pressing": 0.55,
    "line_height": 0.52,
    "defensive_solidity": 0.55,
    "attack_width": 0.55,
    "central_progression": 0.55,
    "directness": 0.50,
    "chance_creation": 0.55,
    "transition_attack": 0.55,
    "transition_defense": 0.55,
    "set_piece_threat": 0.50,
    "tempo": 0.55,
    "notes": ["Neutral tactical fallback. Add a real profile for sharper predictions."],
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
        self.match_context_path = self.data_dir / "match_context.json"
        self.tactics_path = self.data_dir / "tactical_profiles.json"
        self.participants_path = self.data_dir / "participants.json"
        self.backtest_path = self.data_dir / "backtest.json"
        self.sync_state_path = self.data_dir / "sync_state.json"
        self.model_path = self.data_dir / "model_state.json"
        self.predictions_path = self.data_dir / "predictions.json"
        self.key_players_path = self.data_dir / "key_players.json"
        self.resolver = TeamResolver(self.alias_path)
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path, default in (
            (self.matches_path, []),
            (self.context_path, {}),
            (self.match_context_path, DEFAULT_MATCH_CONTEXT),
            (self.tactics_path, {}),
            (self.participants_path, []),
            (self.backtest_path, {}),
            (self.sync_state_path, {}),
            (self.model_path, DEFAULT_MODEL_STATE),
            (self.predictions_path, []),
            (self.key_players_path, DEFAULT_KEY_PLAYERS),
        ):
            if not path.exists():
                path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        last_error: json.JSONDecodeError | None = None
        permission_error: PermissionError | None = None
        for _ in range(8):
            try:
                text = path.read_text(encoding="utf-8")
                permission_error = None
            except PermissionError as exc:
                permission_error = exc
                time.sleep(0.05)
                continue
            if text.strip():
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    last_error = exc
            time.sleep(0.05)
        if permission_error:
            raise permission_error
        if last_error:
            raise last_error
        return default

    def _write_json(self, path: Path, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(payload, encoding="utf-8")
            last_error: PermissionError | None = None
            for attempt in range(40):
                try:
                    temp_path.replace(path)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(min(0.25, 0.03 + attempt * 0.01))
            if last_error:
                raise last_error
        finally:
            if temp_path.exists():
                temp_path.unlink()

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
                next_matches.append(_merge_match(match, record))
                replaced = True
            else:
                next_matches.append(match)
        if not replaced:
            next_matches.append(record)
        self.save_matches(next_matches)

    def load_participants(self) -> list[dict[str, Any]]:
        return self._read_json(self.participants_path, [])

    def save_participants(self, participants: list[dict[str, Any]]) -> None:
        ordered = sorted(participants, key=lambda item: item.get("team", ""))
        self._write_json(self.participants_path, ordered)

    def load_backtest(self) -> dict[str, Any]:
        return self._read_json(self.backtest_path, {})

    def save_backtest(self, backtest: dict[str, Any]) -> None:
        self._write_json(self.backtest_path, backtest)

    def load_sync_state(self) -> dict[str, Any]:
        return self._read_json(self.sync_state_path, {})

    def save_sync_state(self, sync_state: dict[str, Any]) -> None:
        self._write_json(self.sync_state_path, sync_state)

    def load_context(self) -> dict[str, Any]:
        return self._read_json(self.context_path, {})

    def save_context(self, context: dict[str, Any]) -> None:
        self._write_json(self.context_path, context)

    def team_context(self, team: str) -> dict[str, Any]:
        context = self.load_context()
        return context.get(team, {})

    def load_match_context(self) -> dict[str, Any]:
        context = self._read_json(self.match_context_path, DEFAULT_MATCH_CONTEXT)
        merged = json.loads(json.dumps(DEFAULT_MATCH_CONTEXT))
        merged.update(context)
        return merged

    def save_match_context(self, context: dict[str, Any]) -> None:
        merged = self.load_match_context()
        merged.update(context)
        self._write_json(self.match_context_path, merged)

    def load_tactical_profiles(self) -> dict[str, Any]:
        return self._read_json(self.tactics_path, {})

    def save_tactical_profiles(self, profiles: dict[str, Any]) -> None:
        self._write_json(self.tactics_path, profiles)

    def team_tactics(self, team: str) -> dict[str, Any]:
        profiles = self.load_tactical_profiles()
        profile = json.loads(json.dumps(DEFAULT_TACTICAL_PROFILE))
        profile.update(profiles.get(team, {}))
        profile["team"] = team
        profile["is_fallback"] = team not in profiles
        return profile

    def load_model_state(self) -> dict[str, Any]:
        state = self._read_json(self.model_path, DEFAULT_MODEL_STATE)
        merged = json.loads(json.dumps(DEFAULT_MODEL_STATE))
        merged.update({key: value for key, value in state.items() if key != "weights"})
        merged["weights"].update(state.get("weights", {}))
        return merged

    def load_key_players(self) -> dict[str, Any]:
        data = self._read_json(self.key_players_path, DEFAULT_KEY_PLAYERS)
        merged = json.loads(json.dumps(DEFAULT_KEY_PLAYERS))
        for team, players in data.items():
            merged[team] = players
        return merged

    def save_model_state(self, state: dict[str, Any]) -> None:
        self._write_json(self.model_path, state)

    def save_prediction(self, prediction: dict[str, Any]) -> None:
        predictions = self._read_json(self.predictions_path, [])
        prediction = dict(prediction)
        prediction.setdefault("prediction_id", uuid.uuid4().hex)
        prediction.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        prediction.setdefault("status", "pending" if prediction.get("match_date") else "recorded")
        for existing in predictions:
            same_match = (
                existing.get("home_team") == prediction.get("home_team")
                and existing.get("away_team") == prediction.get("away_team")
                and existing.get("match_date") == prediction.get("match_date")
                and existing.get("status") == "pending"
                and prediction.get("status") == "pending"
            )
            if same_match:
                prediction["prediction_id"] = existing.get("prediction_id", prediction["prediction_id"])
                prediction["created_at"] = existing.get("created_at", prediction["created_at"])
                existing.update(prediction)
                self._write_json(self.predictions_path, predictions[-500:])
                return
        predictions.append(prediction)
        self._write_json(self.predictions_path, predictions[-500:])

    def load_predictions(self) -> list[dict[str, Any]]:
        return self._read_json(self.predictions_path, [])

    def save_predictions(self, predictions: list[dict[str, Any]]) -> None:
        self._write_json(self.predictions_path, predictions[-500:])

    def update_prediction(self, prediction_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        predictions = self.load_predictions()
        updated_prediction = None
        for prediction in predictions:
            if prediction.get("prediction_id") == prediction_id:
                prediction.update(updates)
                updated_prediction = prediction
                break
        self.save_predictions(predictions)
        return updated_prediction

    def latest_prediction(
        self,
        home_team: str,
        away_team: str,
        match_date: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        predictions = self.load_predictions()
        for prediction in reversed(predictions):
            same_teams = prediction.get("home_team") == home_team and prediction.get("away_team") == away_team
            same_date = match_date is None or prediction.get("match_date") == match_date
            same_status = status is None or prediction.get("status") == status
            if same_teams and same_date and same_status:
                return prediction
        return None


def _merge_match(existing: MatchRecord, incoming: MatchRecord) -> MatchRecord:
    merged = existing.to_dict()
    incoming_data = incoming.to_dict()
    for key, value in incoming_data.items():
        if value not in (None, ""):
            merged[key] = value
    if existing.source == "espn-world-cup":
        merged["source"] = existing.source
    return MatchRecord.from_dict(merged)
