from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_store import fixture_has_started, project_root


_LOCK = threading.RLock()


class BarcelonaStore:
    """Isolated storage for Barcelona club forecasts.

    Keeping this data separate from the national-team model prevents World Cup
    priors and neutral-venue assumptions from leaking into club predictions.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or project_root()
        self.data_dir = self.root / "data"
        self.universe_path = self.data_dir / "barcelona_universe.json"
        self.matches_path = self.data_dir / "barcelona_matches.json"
        self.predictions_path = self.data_dir / "barcelona_predictions.json"
        self.backtest_path = self.data_dir / "barcelona_backtest.json"
        self.sync_path = self.data_dir / "barcelona_sync.json"
        self.model_path = self.data_dir / "barcelona_model.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path, default in (
            (self.universe_path, []),
            (self.matches_path, []),
            (self.predictions_path, []),
            (self.backtest_path, {}),
            (self.sync_path, {}),
            (self.model_path, {"version": 1}),
        ):
            if not path.exists():
                self._write_json(path, default)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        last_error: Exception | None = None
        for _ in range(8):
            try:
                raw = path.read_text(encoding="utf-8")
                return json.loads(raw) if raw.strip() else default
            except (PermissionError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error:
            raise last_error
        return default

    def _write_json(self, path: Path, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            last_error: Exception | None = None
            for attempt in range(40):
                try:
                    temporary.replace(path)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(min(0.25, 0.03 + attempt * 0.01))
            if last_error:
                raise last_error
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_universe(self) -> list[dict[str, Any]]:
        return self._read_json(self.universe_path, [])

    def save_universe(self, fixtures: list[dict[str, Any]]) -> None:
        with _LOCK:
            ordered = sorted(fixtures, key=lambda item: (item.get("kickoff", ""), item.get("fixture_id", "")))
            self._write_json(self.universe_path, ordered)

    def merge_universe(self, fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with _LOCK:
            indexed = {str(item.get("fixture_id")): item for item in self.load_universe() if item.get("fixture_id")}
            for incoming in fixtures:
                fixture_id = str(incoming.get("fixture_id") or "")
                if not fixture_id:
                    continue
                indexed[fixture_id] = _merge_fixture(indexed.get(fixture_id, {}), incoming)
            merged = sorted(indexed.values(), key=lambda item: (item.get("kickoff", ""), item.get("fixture_id", "")))
            self._write_json(self.universe_path, merged)
            return merged

    def load_matches(self) -> list[dict[str, Any]]:
        return self._read_json(self.matches_path, [])

    def save_matches(self, matches: list[dict[str, Any]]) -> None:
        with _LOCK:
            ordered = sorted(matches, key=lambda item: (item.get("kickoff", ""), item.get("fixture_id", "")))
            self._write_json(self.matches_path, ordered)

    def load_backtest(self) -> dict[str, Any]:
        return self._read_json(self.backtest_path, {})

    def save_backtest(self, payload: dict[str, Any]) -> None:
        with _LOCK:
            self._write_json(self.backtest_path, payload)

    def load_sync_state(self) -> dict[str, Any]:
        return self._read_json(self.sync_path, {})

    def save_sync_state(self, payload: dict[str, Any]) -> None:
        with _LOCK:
            self._write_json(self.sync_path, payload)

    def load_model_state(self) -> dict[str, Any]:
        return self._read_json(self.model_path, {"version": 1})

    def save_model_state(self, payload: dict[str, Any]) -> None:
        with _LOCK:
            self._write_json(self.model_path, payload)

    def load_predictions(self) -> list[dict[str, Any]]:
        return self._read_json(self.predictions_path, [])

    def prediction_for_fixture(self, fixture_id: str) -> dict[str, Any] | None:
        for prediction in reversed(self.load_predictions()):
            if str(prediction.get("fixture_id")) == str(fixture_id):
                return prediction
        return None

    def save_prediction(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Save one permanent pre-match snapshot per ESPN fixture."""
        with _LOCK:
            predictions = self.load_predictions()
            fixture_id = str(payload.get("fixture_id") or "")
            existing = next(
                (item for item in reversed(predictions) if str(item.get("fixture_id")) == fixture_id),
                None,
            )
            if existing:
                return dict(existing)
            if fixture_has_started(payload.get("fixture")):
                raise ValueError("Матч уже начался: создать прогноз задним числом нельзя.")

            saved = dict(payload)
            saved.setdefault("prediction_id", uuid.uuid4().hex)
            saved.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            saved.setdefault("status", "pending")
            saved["snapshot_type"] = "pre_match"
            saved["immutable"] = True
            predictions.append(saved)
            self._write_json(self.predictions_path, predictions[-500:])
            return dict(saved)

    def review_prediction(self, fixture_id: str, review: dict[str, Any]) -> dict[str, Any] | None:
        with _LOCK:
            predictions = self.load_predictions()
            updated = None
            for item in predictions:
                if str(item.get("fixture_id")) == str(fixture_id):
                    item["status"] = "reviewed"
                    item["review"] = review
                    item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                    updated = dict(item)
                    break
            self._write_json(self.predictions_path, predictions[-500:])
            return updated


def _merge_fixture(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value not in (None, "", [], {}):
            merged[key] = value
        elif key not in merged:
            merged[key] = value
    # Once a rich protocol has been downloaded, a light scoreboard refresh must
    # not erase its lineups, referee or team statistics.
    if existing.get("details_loaded") and not incoming.get("details_loaded"):
        for key in ("lineups", "home_formation", "away_formation", "referee", "home_stats", "away_stats"):
            if key in existing:
                merged[key] = existing[key]
        merged["details_loaded"] = True
    return merged
