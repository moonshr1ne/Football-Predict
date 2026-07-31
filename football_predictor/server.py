from __future__ import annotations

import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .barcelona_service import BarcelonaService
from .data_store import project_root


class PredictorHandler(SimpleHTTPRequestHandler):
    service = BarcelonaService()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(project_root() / "web"), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            query = parse_qs(parsed.query)
            opponent = query.get("opponent", [""])[0]
            remember = query.get("remember", ["true"])[0] != "false"
            try:
                self._json(200, self.service.predict(opponent, remember=remember))
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/opponents":
            try:
                self._json(200, {"opponents": self.service.opponents()})
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/status":
            self._json(200, self.service.status())
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/auto-check":
            self._json(404, {"error": "Not found"})
            return
        try:
            self._json(200, self.service.auto_check())
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def end_headers(self) -> None:
        if self.path.endswith((".html", ".js", ".css")) or self.path == "/":
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), PredictorHandler)
    _start_background_worker(PredictorHandler.service)
    print(f"Barcelona Match Lab: http://{host}:{port}")
    server.serve_forever()


def _start_background_worker(service: BarcelonaService, interval_seconds: int = 3600) -> None:
    def worker() -> None:
        while True:
            try:
                service.background_refresh()
            except Exception:
                pass
            time.sleep(interval_seconds)

    threading.Thread(target=worker, daemon=True, name="barcelona-data-refresh").start()
