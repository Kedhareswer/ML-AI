"""Local browser UI: upload a document, see Markdown, JSON and the blocks (standard library HTTP server)."""
import base64
import io
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

PAGE = Path(__file__).with_name("ui.html")
MAX_BYTES = 40 * 2 ** 20
LOCK = Lock()  # one extraction at a time keeps memory and CPU bounded
PREVIEW_PAGES = 40  # page images sent back for the box overlay
PREVIEW_WIDTH = 1300  # px: sharp enough to read at 2x zoom; JPEGs keep memory and the response small
THUMB_WIDTH = 160  # px: page strip for jumping between pages


def _jpeg(img, width):
    img = img.convert("RGB")
    img.thumbnail((width, width * 2))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def page_previews(path, doc):
    """Render pages one at a time (each image is freed before the next) for PDFs and images."""
    try:
        doc["previews"], doc["thumbs"] = _render_previews(Path(path))
    except Exception as exc:  # a preview problem must not lose the extraction
        doc["previews"], doc["thumbs"] = [], []
        doc["warnings"].append(f"page images unavailable: {type(exc).__name__}")


def _render_previews(path):
    out, thumbs = [], []

    def add(img):
        out.append(_jpeg(img, PREVIEW_WIDTH))
        thumbs.append(_jpeg(img, THUMB_WIDTH))
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in range(min(len(pdf), PREVIEW_PAGES)):
                page = pdf[i]
                add(page.render(scale=PREVIEW_WIDTH / page.get_width()).to_pil())
                page.close()
        finally:
            pdf.close()
    elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as img:
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                if i >= PREVIEW_PAGES:
                    break
                add(frame.copy())
    return out, thumbs


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
                    doc = extract(path, ocr=ocr, ocr_images=self.headers.get("X-OCR-Images") == "1", model=model,
                                  on_parsed=page_previews)
                previews, thumbs = doc.pop("previews", []), doc.pop("thumbs", [])
                self._send(200, json.dumps({"markdown": to_markdown(doc), "document": doc, "previews": previews, "thumbs": thumbs}, default=str))
            except Exception as exc:  # report parse errors to the page instead of dropping the connection
                self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"docparse UI on http://{host}:{port}", flush=True)
    srv.serve_forever()
