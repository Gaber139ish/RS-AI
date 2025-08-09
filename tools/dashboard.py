import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any

from tools.metrics import MetricsRegistry


class ControlBridge:
    def __init__(self):
        self._orchestrator = None

    def set_orchestrator(self, orch: Any):
        self._orchestrator = orch

    def handle_action(self, action: str, params: dict) -> dict:
        if self._orchestrator is None:
            return {"status": "error", "message": "orchestrator not set"}
        if action == "pause":
            self._orchestrator.pause()
            return {"status": "ok", "paused": True}
        if action == "resume":
            self._orchestrator.resume()
            return {"status": "ok", "paused": False}
        if action == "set":
            key = params.get("key")
            value = params.get("value")
            if key is None:
                return {"status": "error", "message": "missing key"}
            try:
                self._orchestrator.set_param(key, value)
                return {"status": "ok", "key": key, "value": value}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"unknown action {action}"}


class Handler(BaseHTTPRequestHandler):
    registry: MetricsRegistry = None  # type: ignore
    bridge: ControlBridge = None  # type: ignore

    def _write_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, code: int, body: str):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/metrics":
            data = self.registry.snapshot()
            self._write_json(200, data)
            return
        if parsed.path == "/metrics_prom":
            snap = self.registry.snapshot()
            lines = []
            for k, v in snap.items():
                if isinstance(v, (int, float)):
                    lines.append(f"{k} {float(v)}")
            self._write_text(200, "\n".join(lines) + "\n")
            return
        if parsed.path == "/control":
            qs = parse_qs(parsed.query)
            action = (qs.get("action", [None])[0] or "").lower()
            params = {k: v[0] for k, v in qs.items()}
            res = self.bridge.handle_action(action, params)
            self._write_json(200, res)
            return
        self._write_json(404, {"status": "not_found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/control":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            action = (payload.get("action") or "").lower()
            res = self.bridge.handle_action(action, payload)
            self._write_json(200, res)
            return
        self._write_json(404, {"status": "not_found"})


def start_dashboard(host: str, port: int, registry: MetricsRegistry, bridge: ControlBridge):
    Handler.registry = registry
    Handler.bridge = bridge
    server = ThreadingHTTPServer((host, port), Handler)
    return server