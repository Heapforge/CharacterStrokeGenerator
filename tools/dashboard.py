#!/usr/bin/env python3
"""Review dashboard: pick glyph SVGs, run strokegen.predict on them, check, reorder and
merge the strokes in hanzi-writer, then save the result to the training data.

    uv run python tools/dashboard.py            # http://127.0.0.1:8417
    uv run python tools/dashboard.py --port 9000 --device cpu

Saving writes training_data/hanzi-writer-data-youyin/data/{char}{variant}.json and, for a plain
file whose glyph exists in training_data/font_glyphs, makes sure the manifest lists it so
build_dataset picks it up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GLYPHS = os.path.join(ROOT, "training_data", "font_glyphs")
HWDATA = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
MANIFEST = os.path.join(GLYPHS, "manifest.json")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
VARIANTS = ("", "-jp", "-ko")
MAX_BODY = 8 * 1024 * 1024


class Model:
    """Loads StrokeGenerator in the background so the page is usable right away."""

    def __init__(self, device: str | None):
        self.gen = None
        self.error = None
        self.device = device
        self.lock = threading.Lock()
        self.ready = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from strokegen.predict import StrokeGenerator
            self.gen = StrokeGenerator(device=self.device)
            self.device = str(self.gen.dev)
        except Exception as e:  # shown in the page header
            self.error = f"{type(e).__name__}: {e}"
        self.ready.set()

    def predict(self, svg_text: str) -> dict:
        self.ready.wait()
        if self.gen is None:
            raise RuntimeError(f"model failed to load: {self.error}")
        from strokegen.svg_io import load_svg_glyph
        with self.lock:
            r = self.gen.predict_contours(load_svg_glyph(svg_text))
        return {"strokes": r["strokes"], "medians": r["medians"], "segStroke": r["_seg_stroke"].tolist()}

    def status(self) -> dict:
        if not self.ready.is_set():
            return {"state": "loading"}
        if self.gen is None:
            return {"state": "error", "error": self.error}
        return {"state": "ready", "device": self.device}


def font_names() -> list[str]:
    return sorted(d for d in os.listdir(GLYPHS) if os.path.isdir(os.path.join(GLYPHS, d)))


def safe_char(c) -> str:
    """A character name usable as a file stem (normally one code point)."""
    if not isinstance(c, str) or not c or len(c) > 8 or c.startswith(".") or any(
            ch in c for ch in "/\\\0") or c.strip() != c:
        raise ValueError(f"invalid character name: {c!r}")
    return c


def data_path(char: str, variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return os.path.join(HWDATA, f"{safe_char(char)}{variant}.json")


def glyph_path(font: str, char: str) -> str:
    if font not in font_names():
        raise ValueError(f"unknown font: {font!r}")
    return os.path.join(GLYPHS, font, f"{safe_char(char)}.svg")


def is_median(m) -> bool:
    return isinstance(m, list) and len(m) >= 1 and all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in m)


def merge(svg_text: str, req: dict) -> dict:
    """Join strokes into one; see strokegen.predict.merge_strokes."""
    from strokegen.predict import merge_strokes
    from strokegen.svg_io import load_svg_glyph
    seg, parts = req.get("segStroke"), req.get("parts")
    if not isinstance(seg, list) or not all(isinstance(v, int) for v in seg):
        raise ValueError("segStroke must be a list of integers")
    if not isinstance(parts, list) or len(parts) < 2 or not all(isinstance(v, int) for v in parts):
        raise ValueError("parts must list at least two stroke ids")
    if not is_median(req.get("first")) or not is_median(req.get("second")):
        raise ValueError("first and second must be medians ([[x, y], ...])")
    return merge_strokes(load_svg_glyph(svg_text), seg, parts, req["first"], req["second"])


def validate_hw(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    strokes, medians = data.get("strokes"), data.get("medians")
    if not isinstance(strokes, list) or not strokes or not all(isinstance(s, str) and s for s in strokes):
        raise ValueError("strokes must be a non-empty list of path strings")
    if not isinstance(medians, list) or len(medians) != len(strokes):
        raise ValueError(f"need one median per stroke ({len(strokes)} strokes)")
    for m in medians:
        if not is_median(m) or len(m) < 2:
            raise ValueError("each median must be a list of at least two [x, y] points")
    return {"strokes": strokes, "medians": [[[int(round(x)), int(round(y))] for x, y in m] for m in medians]}


_manifest_lock = threading.Lock()


def register_in_manifest(char: str) -> str:
    """Make build_dataset see {char}.json. Returns a short note for the page."""
    fonts = [f for f in json.load(open(MANIFEST, encoding="utf-8"))["fonts"]
             if os.path.exists(os.path.join(GLYPHS, f, f"{char}.svg"))]
    if not fonts:
        return "no font glyph for this character, so build_dataset will skip it"
    with _manifest_lock:
        m = json.load(open(MANIFEST, encoding="utf-8"))
        e = m["characters"].setdefault(char, {"data_files": [], "fonts": []})
        changed = False
        if f"{char}.json" not in e["data_files"]:
            e["data_files"] = sorted(e["data_files"] + [f"{char}.json"])
            changed = True
        if e["fonts"] != fonts:
            e["fonts"] = fonts
            changed = True
        if char in m.get("missing", []):
            m["missing"].remove(char)
            changed = True
        if not changed:
            return "already in the training manifest"
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        os.replace(tmp, MANIFEST)
    return "added to the training manifest"


class Handler(BaseHTTPRequestHandler):
    model: Model

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/svg"):
            sys.stderr.write(f"{self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

    def send(self, status: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, status: int = 200):
        self.send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def fail(self, status: int, msg: str):
        self.json({"error": msg}, status)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:  # http.server reads the request line as latin-1; recover raw UTF-8 characters
            path = self.path.encode("latin-1").decode("utf-8")
        except UnicodeError:
            path = self.path
        url = urlparse(path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
            if url.path == "/api/status":
                return self.json({"model": self.model.status(), "fonts": font_names(),
                                  "dataDir": os.path.relpath(HWDATA, ROOT)})
            if url.path == "/api/glyphs":
                font = q.get("font", "")
                if font not in font_names():
                    return self.fail(404, f"unknown font: {font!r}")
                have = {n[:-5] for n in os.listdir(HWDATA) if n.endswith(".json")}
                chars = sorted(n[:-4] for n in os.listdir(os.path.join(GLYPHS, font)) if n.endswith(".svg"))
                return self.json({"font": font, "chars": chars, "hasData": [c in have for c in chars]})
            if url.path == "/api/svg":
                p = glyph_path(q.get("font", ""), q.get("char", ""))
                if not os.path.exists(p):
                    return self.fail(404, "no such glyph")
                return self.send(200, open(p, "rb").read(), "image/svg+xml", cache=True)
            if url.path == "/api/data":
                p = data_path(q.get("char", ""), q.get("variant", ""))
                if not os.path.exists(p):
                    return self.fail(404, "no training data for this character")
                return self.json(json.load(open(p, encoding="utf-8")))
            return self.fail(404, "not found")
        except ValueError as e:
            return self.fail(400, str(e))

    @staticmethod
    def source_svg(req: dict) -> str:
        """The glyph a request refers to: uploaded SVG text, or font + char."""
        if "svg" in req:
            svg = req["svg"]
            if not isinstance(svg, str) or "<path" not in svg:
                raise ValueError("svg must be SVG text with at least one <path>")
            return svg
        p = glyph_path(req.get("font", ""), req.get("char", ""))
        if not os.path.exists(p):
            raise FileNotFoundError("no such glyph")
        return open(p, encoding="utf-8").read()

    def do_POST(self):
        url = urlparse(self.path)
        try:
            req = self.body()
            if url.path == "/api/merge":
                return self.json(merge(self.source_svg(req), req))
            if url.path == "/api/predict":
                svg = self.source_svg(req)
                t = time.perf_counter()
                res = self.model.predict(svg)
                if not res["strokes"]:
                    return self.fail(422, "no outline found in this SVG")
                return self.json({**res, "ms": round((time.perf_counter() - t) * 1000)})
            if url.path == "/api/commit":
                char, variant = safe_char(req.get("char")), req.get("variant", "")
                p = data_path(char, variant)
                data = validate_hw(req.get("data"))
                existed = os.path.exists(p)
                if existed and not req.get("overwrite"):
                    return self.fail(409, f"{os.path.basename(p)} already exists")
                tmp = p + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp, p)
                note = register_in_manifest(char) if variant == "" else "variant files are not used for training"
                return self.json({"path": os.path.relpath(p, ROOT), "overwrote": existed, "note": note})
            return self.fail(404, "not found")
        except FileNotFoundError as e:
            return self.fail(404, str(e))
        except (ValueError, json.JSONDecodeError) as e:
            return self.fail(400, str(e))
        except RuntimeError as e:
            return self.fail(503, str(e))
        except Exception as e:
            return self.fail(500, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8417)
    ap.add_argument("--device", help="torch device, e.g. cpu or cuda (default: cuda if available)")
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = ap.parse_args()
    Handler.model = Model(args.device)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard on {url} (model loading in the background)")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
