"""Local inference UI using only Python's standard library."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
import math
from pathlib import Path

from .engine import solve
from .tasks import Maze, generate

PAGE = Path(__file__).with_name('ui.html').read_bytes()

def make_server(model, port=7860, attempts=4, temperature=.8):
    if not 1 <= attempts <= 32 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Invalid attempts or temperature")

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, value, content_type="application/json"):
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/example":
                self.send(200, {"rows": generate(random.randrange(2**32)).rows})
            elif self.path == "/api/model":
                self.send(200, {**model.info(), "supports_search":hasattr(model,"search"), "training_steps":getattr(model,"training_steps",None)})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/solve":
                return self.send(404, {"error": "Not found"})
            # Reject cross-origin browser requests to this local service.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                return self.send(403, {"error": "Cross-origin requests are disabled"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Expected a small JSON request")
                payload = json.loads(self.rfile.read(length))
                rows = payload["rows"]
                if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
                    raise ValueError("rows must be a list of strings")
                mode = payload.get("mode","model")
                maze = Maze(tuple(rows))
                if mode == "guided" and hasattr(model,"search"):
                    result = model.search(maze,budget=64)
                elif mode == "model":
                    result = solve(model,maze,attempts,temperature)
                    result["mode"] = "neural_only"
                else:
                    raise ValueError("Unsupported solve mode")
                self.send(200,result)
            except (ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})

        def log_message(self, *_):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)

def serve(model, port=7860, attempts=4, temperature=.8):
    server = make_server(model, port, attempts, temperature)
    print(f"Local inference UI listening on port {server.server_port}. Ctrl-C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
