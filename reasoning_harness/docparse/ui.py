"""Local browser UI: upload a document, see Markdown, JSON and the blocks (standard library HTTP server)."""
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

PAGE = Path(__file__).with_name("ui.html")
MAX_BYTES = 40 * 2 ** 20
LOCK = Lock()  # one extraction at a time keeps memory and CPU bounded


def serve(port=7862, host="127.0.0.1", model=None):
    from .extract import extract, to_markdown

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            # ignore the query string: Hugging Face embeds the page as /?__theme=dark etc.
            if self.path.split("?", 1)[0] in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if not self.path.startswith("/api/extract"):
                return self._send(404, json.dumps({"error": "not found"}))
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BYTES:
                return self._send(413, json.dumps({"error": f"file must be 1 byte to {MAX_BYTES // 2 ** 20} MB"}))
            name = Path(self.headers.get("X-Filename", "upload.bin")).name
            ocr = self.headers.get("X-OCR", "auto")
            if ocr not in ("auto", "always", "never"):
                ocr = "auto"
            data = self.rfile.read(n)
            try:
                with LOCK, tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / name
                    path.write_bytes(data)
                    doc = extract(path, ocr=ocr, ocr_images=self.headers.get("X-OCR-Images") == "1", model=model)
                self._send(200, json.dumps({"markdown": to_markdown(doc), "document": doc}, default=str))
            except Exception as exc:  # report parse errors to the page instead of dropping the connection
                self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"docparse UI on http://{host}:{port}", flush=True)
    srv.serve_forever()
