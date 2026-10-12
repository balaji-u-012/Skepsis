"""
A deliberately tiny HTTP layer built entirely on Python's standard library
(`http.server`, `asyncio`, `json`) so Skepsis needs no compiled packages at
all — no FastAPI, no Uvicorn, no Pydantic. Everything here ships with
Python itself.

Route handlers are `async def handler(path_params: dict, query_params: dict,
body: dict | None) -> Any` and their return value is JSON-serialized. Raise
`ApiError(status_code, detail)` to return a specific error status, the same
way you'd raise FastAPI's HTTPException.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("skepsis.http")

# Reject request bodies larger than this. Protects against trivial DoS
# from a client sending a huge Content-Length.
_MAX_BODY_BYTES = 1 * 1024 * 1024  # 1 MB


class ApiError(Exception):
    """Raise this from a handler to short-circuit with a specific HTTP status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def to_jsonable(value: Any) -> Any:
    """Recursively converts dataclasses (including nested ones inside lists/
    dicts) into plain JSON-serializable structures."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


Handler = Callable[[Dict[str, str], Dict[str, str], Optional[dict]], Any]
Route = Tuple[str, "re.Pattern[str]", Handler]


class Router:
    def __init__(self) -> None:
        self._routes: List[Route] = []

    def add_route(self, method: str, path: str, handler: Handler) -> None:
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", path)
        compiled = re.compile(f"^{pattern}$")
        self._routes.append((method.upper(), compiled, handler))

    def get(self, path: str):
        def deco(fn: Handler) -> Handler:
            self.add_route("GET", path, fn)
            return fn
        return deco

    def post(self, path: str):
        def deco(fn: Handler) -> Handler:
            self.add_route("POST", path, fn)
            return fn
        return deco

    def match(self, method: str, path: str):
        for route_method, pattern, handler in self._routes:
            if route_method != method:
                continue
            match = pattern.match(path)
            if match:
                return handler, match.groupdict()
        return None, None


# ---------------------------------------------------------------------------
# Persistent background event loop
#
# Creating a new event loop per request via asyncio.run() is slow and
# breaks shared async state. Instead we run one loop for the entire
# process lifetime in a dedicated thread, and submit each handler
# coroutine to it via run_coroutine_threadsafe(). Handlers still run
# concurrently with each other (the loop schedules them), but we avoid
# per-request loop setup/teardown.
# ---------------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_run, name="skepsis-asyncio", daemon=True)
        thread.start()
        _loop = loop
        logger.info("Started background asyncio loop in thread %s", thread.name)
        return _loop


def _run_coroutine(coro) -> Any:
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class SkepsisHTTPServer(ThreadingHTTPServer):
    daemon_threads = True       # don't block shutdown on in-flight requests
    allow_reuse_address = True  # avoid "address already in use" on restart


def make_handler_class(router: Router) -> type:
    class SkepsisRequestHandler(BaseHTTPRequestHandler):
        server_version = "Skepsis/0.1"
        protocol_version = "HTTP/1.1"  # enables keep-alive

        def log_message(self, fmt: str, *args) -> None:
            # Route through the logging module instead of printing.
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def do_OPTIONS(self) -> None:  # CORS preflight
            self.send_response(204)
            self._cors_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _write_json(self, status: int, payload: Any) -> None:
            body = json.dumps(to_jsonable(payload), default=str).encode("utf-8")
            self.send_response(status)
            self._cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> Tuple[bool, Optional[dict]]:
            """Returns (ok, body). On error, writes a 4xx response and
            returns (False, None)."""
            raw_length = self.headers.get("Content-Length") or "0"
            try:
                length = int(raw_length)
            except ValueError:
                self._write_json(400, {"detail": "Invalid Content-Length header."})
                return False, None

            if length <= 0:
                return True, None

            if length > _MAX_BODY_BYTES:
                self._write_json(413, {"detail": "Request body too large."})
                return False, None

            raw = self.rfile.read(length)
            if not raw:
                return True, None

            try:
                return True, json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(400, {"detail": "Request body is not valid JSON."})
                return False, None

        def _dispatch(self, method: str) -> None:
            parsed = urlparse(self.path)
            handler, path_params = router.match(method, parsed.path)
            if handler is None:
                self._write_json(404, {"detail": f"No route for {method} {parsed.path}"})
                return

            query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

            ok, body = self._read_body()
            if not ok:
                return

            try:
                result = _run_coroutine(handler(path_params, query_params, body))
                self._write_json(200, result)
            except ApiError as exc:
                self._write_json(exc.status_code, {"detail": exc.detail})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error in handler %s %s", method, parsed.path)
                self._write_json(500, {"detail": f"Internal server error: {exc}"})

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return SkepsisRequestHandler


def run_server(router: Router, host: str = "127.0.0.1", port: int = 8000) -> None:
    # Make sure the background loop is alive before we start serving.
    _get_background_loop()

    handler_cls = make_handler_class(router)
    httpd = SkepsisHTTPServer((host, port), handler_cls)
    logger.info("Skepsis backend listening on http://%s:%d", host, port)
    print(f"Skepsis backend listening on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()