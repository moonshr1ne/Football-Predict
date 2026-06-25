from __future__ import annotations

import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .aliases import parse_matchup
from .data_store import DataStore, project_root
from .autocheck import AutoChecker
from .learning import OnlineLearner
from .predictor import MatchPredictor
from .providers import EspnWorldCupProvider, ProviderError
from .sync import WorldCupDataSync


class PredictorHandler(SimpleHTTPRequestHandler):
    store = DataStore()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(project_root() / "web"), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            query = parse_qs(parsed.query)
            matchup = query.get("matchup", [""])[0]
            home_venue = query.get("home_venue", ["false"])[0] == "true"
            match_date = query.get("date", [None])[0] or None
            remember = query.get("remember", ["true"])[0] != "false"
            try:
                home, away = parse_matchup(matchup, self.store.resolver)
                sync_info = WorldCupDataSync(self.store).sync_all()
                fixture = None
                warnings = []
                if sync_info.get("imported"):
                    recent = "уже свежие" if sync_info.get("skipped_full_sync") else f"{sync_info.get('recent_imported', 0)}"
                    action = "База проверена" if sync_info.get("skipped_full_sync") else "База обновлена"
                    warnings.append(
                        f"{action}: участников {sync_info.get('participants', 0)}, last-10 матчей {recent}, матчей ЧМ {sync_info['imported']}, тактических профилей {sync_info['profiles_updated']}, судей {sync_info.get('referees_updated', 0)}, обучающих матчей {sync_info.get('trained', 0)}."
                    )
                if not match_date:
                    try:
                        fixture = EspnWorldCupProvider().find_fixture(home, away)
                        match_date = fixture.get("date")
                    except ProviderError as exc:
                        warnings.append(f"Автодата: {exc}")
                prediction = MatchPredictor(self.store).predict(
                    home,
                    away,
                    neutral=not home_venue,
                    remember=remember,
                    match_date=match_date,
                    fixture=fixture,
                    extra_warnings=warnings,
                )
                self._json(200, prediction.to_dict())
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/auto-check":
            try:
                summary = AutoChecker(self.store).check_pending()
                self._json(200, summary)
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/train":
            try:
                query = parse_qs(parsed.query)
                epochs = int(query.get("epochs", ["2"])[0] or 2)
                syncer = WorldCupDataSync(self.store)
                sync_summary = syncer.sync_all(force=True)
                summary = syncer.retrain_model_from_history(epochs=epochs)
                summary["sync"] = sync_summary
                self._json(200, summary)
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path != "/api/result":
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            home, away = parse_matchup(body["matchup"], self.store.resolver)
            home_goals, away_goals = _parse_score(body["score"])
            baseline = self.store.latest_prediction(home, away, match_date=body["date"], status="pending")
            review = OnlineLearner(self.store).record_result(
                home_team=home,
                away_team=away,
                date=body["date"],
                home_goals=home_goals,
                away_goals=away_goals,
                corners_total=_optional_float(body.get("corners")),
                neutral=not bool(body.get("home_venue")),
                baseline_prediction=baseline,
            )
            if baseline and baseline.get("prediction_id"):
                self.store.update_prediction(
                    baseline["prediction_id"],
                    {"status": "reviewed", "review": review, "reviewed_at": review["updated_at"]},
                )
            self._json(200, review)
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def _parse_score(value: str) -> tuple[int, int]:
    home, away = value.replace(":", "-").split("-")
    return int(home), int(away)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    WorldCupDataSync(PredictorHandler.store).sync_all()
    _start_auto_check_worker(PredictorHandler.store)
    server = ThreadingHTTPServer((host, port), PredictorHandler)
    print(f"Открывайте: http://{host}:{port}")
    server.serve_forever()


def _start_auto_check_worker(store: DataStore, interval_seconds: int = 3600) -> None:
    def worker() -> None:
        while True:
            try:
                WorldCupDataSync(store).sync_all()
                AutoChecker(store).check_pending()
            except Exception:
                pass
            time.sleep(interval_seconds)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
