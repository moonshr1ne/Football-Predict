from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .aliases import parse_matchup
from .data_store import DataStore, project_root
from .learning import OnlineLearner
from .predictor import MatchPredictor


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
            try:
                home, away = parse_matchup(matchup, self.store.resolver)
                prediction = MatchPredictor(self.store).predict(home, away, neutral=not home_venue)
                self._json(200, prediction.to_dict())
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/result":
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size).decode("utf-8"))
            home, away = parse_matchup(body["matchup"], self.store.resolver)
            home_goals, away_goals = _parse_score(body["score"])
            review = OnlineLearner(self.store).record_result(
                home_team=home,
                away_team=away,
                date=body["date"],
                home_goals=home_goals,
                away_goals=away_goals,
                corners_total=_optional_float(body.get("corners")),
                neutral=not bool(body.get("home_venue")),
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
    server = ThreadingHTTPServer((host, port), PredictorHandler)
    print(f"Открывайте: http://{host}:{port}")
    server.serve_forever()
