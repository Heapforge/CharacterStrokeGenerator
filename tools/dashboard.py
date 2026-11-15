#!/usr/bin/env python3
"""Review dashboard: pick glyph SVGs, run strokegen.predict on them, check, reorder and
merge the strokes in hanzi-writer, then save the result to the training data.

    uv run python tools/dashboard.py            # http://127.0.0.1:8417
    uv run python tools/dashboard.py --port 9000 --device cpu

The server has no authentication: it only answers requests addressed to its own loopback
address and refuses cross-site POSTs. Binding it to another interface (--host 0.0.0.0) lets
anyone on that network overwrite the training data.

Saving writes training_data/hanzi-writer-data-youyin/data/{char}{variant}.json and, for a plain
file whose glyph exists in training_data/font_glyphs, makes sure the manifest lists it so
build_dataset picks it up.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strokegen.fonts import FONT_PRIORITY, font_spec  # noqa: E402  (no torch: the page loads fast)
from strokegen.paths import GLYPHS, HWDATA, TRAINING_DATA  # noqa: E402

EXPORT_SCRIPT = os.path.join(TRAINING_DATA, "scripts", "export_glyphs.py")
MANIFEST = os.path.join(GLYPHS, "manifest.json")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
VARIANTS = ("", "-jp", "-ko")
MAX_BODY = 16 * 1024 * 1024
LOOPBACK = ("127.0.0.1", "localhost", "::1")

FONT_FILES: dict[str, str] = {name: font_spec(name) for name in FONT_PRIORITY}


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
        """Prediction for one SVG; raises ValueError (strokegen.predict.GlyphError) for a bad glyph."""
        self.ready.wait()
        if self.gen is None:
            raise RuntimeError(f"model failed to load: {self.error}")
        from strokegen.svg_io import load_svg_glyph
        contours = load_svg_glyph(svg_text)
        with self.lock:
            r = self.gen.predict_contours(contours)
        return {"strokes": r["strokes"], "medians": r["medians"], "segStroke": r["_seg_stroke"].tolist()}

    def predict_files_batch(self, paths: list[str | None]) -> list[dict | None]:
        self.ready.wait()
        if self.gen is None:
            raise RuntimeError(f"model failed to load: {self.error}")
        with self.lock:
            raw = self.gen.predict_files_batch(paths)
        return [
            None if r is None else {"strokes": r["strokes"], "medians": r["medians"],
                                    "segStroke": r["_seg_stroke"].tolist(), "error": r.get("_error")}
            for r in raw
        ]

    def status(self) -> dict:
        if not self.ready.is_set():
            return {"state": "loading"}
        if self.gen is None:
            return {"state": "error", "error": self.error}
        return {"state": "ready", "device": self.device}


def _font_binary_exists(spec: str) -> bool:
    return os.path.exists(spec.split("(", 1)[0])


def font_names() -> list[str]:
    dirs = {d for d in os.listdir(GLYPHS) if os.path.isdir(os.path.join(GLYPHS, d))} if os.path.isdir(GLYPHS) else set()
    srcs = {f for f, spec in FONT_FILES.items() if _font_binary_exists(spec)}
    return sorted(dirs | srcs)


_unpack_guard = threading.Lock()
_unpack_locks: dict[str, threading.Lock] = {}
_absent_chars: dict[tuple[str, str], set[str]] = {}
_ff_python: str | None = None


def _fontforge_python() -> str:
    global _ff_python
    if _ff_python is not None:
        return _ff_python
    for cand in ("/usr/bin/python3", shutil.which("python3"), sys.executable):
        if cand and os.path.exists(cand):
            r = subprocess.run([cand, "-I", "-c", "import fontforge"], capture_output=True, check=False)
            if r.returncode == 0:
                _ff_python = cand
                return cand
    raise RuntimeError("FontForge Python bindings are required to unpack font binaries")


def unpack_font_chars(font: str, chars: list[str] | None = None) -> str:
    """Ensure requested glyph SVGs for `font` are unpacked once into training_data/font_glyphs/<font>.

    chars None exports the whole font (slow: use unpack_font_background from request handlers).
    Only single code points can be glyphs; longer names are ignored.
    """
    if font not in font_names():
        raise ValueError(f"unknown font: {font!r}")
    if chars is not None:
        chars = [c for c in chars if len(c) == 1]
    with _unpack_guard:
        lock = _unpack_locks.setdefault(font, threading.Lock())
    out_dir = os.path.join(GLYPHS, font)
    spec = FONT_FILES.get(font)
    with lock:
        present = {n for n in os.listdir(out_dir) if n.endswith(".svg")} if os.path.isdir(out_dir) else set()
        if not spec or not _font_binary_exists(spec) or not os.path.exists(EXPORT_SCRIPT):
            return out_dir
        absent = _absent_chars.setdefault((out_dir, font), set())
        if chars is None:
            if present:
                return out_dir
            py = _fontforge_python()
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run([py, "-I", EXPORT_SCRIPT, spec, "--all", out_dir], check=True)
            return out_dir
        missing = sorted({c for c in chars if f"{c}.svg" not in present and c not in absent})
        if not missing:
            return out_dir
        py = _fontforge_python()
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f"chars_{font}_", suffix=".txt")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(missing) + "\n")
            subprocess.run([py, "-I", EXPORT_SCRIPT, spec, tmp_path, out_dir], check=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        after = {n for n in os.listdir(out_dir) if n.endswith(".svg")}
        for c in missing:
            if f"{c}.svg" not in after:
                absent.add(c)
    return out_dir


_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def unpack_font_background(font: str) -> str:
    """Export every glyph of `font` in a background thread.

    Returns "running" while the export runs, "done" once it finished (or was not needed), or
    raises RuntimeError with the export's error (a later call starts it again).
    """
    with _jobs_lock:
        job = _jobs.get(font)
        if job is None:
            job = {"done": False, "error": None}

            def run():
                try:
                    unpack_font_chars(font)
                except Exception as e:  # noqa: BLE001  (reported to the page)
                    job["error"] = f"{type(e).__name__}: {e}"
                finally:
                    job["done"] = True

            _jobs[font] = job
            threading.Thread(target=run, daemon=True).start()
        if not job["done"]:
            return "running"
        del _jobs[font]
    if job["error"]:
        raise RuntimeError(f"could not unpack {font}: {job['error']}")
    return "done"


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
_manifest_cache: dict = {}


def _manifest() -> dict:
    """The parsed manifest, re-read only when the file changed. Call with _manifest_lock held."""
    mtime = os.path.getmtime(MANIFEST)
    if _manifest_cache.get("mtime") != mtime:
        with open(MANIFEST, encoding="utf-8") as f:
            _manifest_cache.update(mtime=mtime, data=json.load(f))
    return _manifest_cache["data"]


def register_in_manifest(char: str) -> str:
    """Make build_dataset see {char}.json. Returns a short note for the page."""
    with _manifest_lock:
        try:
            return _register(char)
        except BaseException:
            _manifest_cache.clear()  # it may have been changed in memory only
            raise


def _register(char: str) -> str:
    m = _manifest()
    fonts = [f for f in m["fonts"] if os.path.exists(os.path.join(GLYPHS, f, f"{char}.svg"))]
    if not fonts:
        return "no font glyph for this character, so build_dataset will skip it"
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
    _manifest_cache["mtime"] = os.path.getmtime(MANIFEST)
    return "added to the training manifest"


class Handler(BaseHTTPRequestHandler):
    model: Model
    # Host headers this server answers to (None: no check). Pinning them to the loopback address
    # stops DNS rebinding: another site's page cannot reach the server through a name of its own.
    allowed_hosts: set[str] | None = None

    def refused(self, post: bool) -> str | None:
        """Why this request must be refused, or None."""
        host = self.headers.get("Host", "")
        if self.allowed_hosts is not None and host.lower() not in self.allowed_hosts:
            return f"unexpected Host header {host!r}"
        if not post:
            return None
        # Writes must come from this page: browsers send Origin on every POST, and a cross-site
        # page cannot send application/json without a CORS preflight, which this server never allows.
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc.lower() != host.lower():
            return "cross-origin request refused"
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return "cross-site request refused"
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return "requests must be sent as application/json"
        return None

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
        why = self.refused(post=False)
        if why:
            return self.fail(403, why)
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
                gdir = os.path.join(GLYPHS, font)
                spec = FONT_FILES.get(font)
                unpacked = os.path.isdir(gdir) and any(n.endswith(".svg") for n in os.listdir(gdir))
                if not unpacked and spec and _font_binary_exists(spec):
                    if unpack_font_background(font) == "running":
                        return self.json({"font": font, "unpacking": True}, 202)
                have = {n[:-5] for n in os.listdir(HWDATA) if n.endswith(".json")}
                chars = sorted(n[:-4] for n in os.listdir(gdir) if n.endswith(".svg")) if os.path.isdir(gdir) else []
                return self.json({"font": font, "chars": chars, "hasData": [c in have for c in chars]})
            if url.path == "/api/svg":
                font, char = q.get("font", ""), q.get("char", "")
                p = glyph_path(font, char)
                if not os.path.exists(p):
                    unpack_font_chars(font, [safe_char(char)])
                if not os.path.exists(p):
                    return self.fail(404, "no such glyph")
                return self.send(200, open(p, "rb").read(), "image/svg+xml", cache=True)
            if url.path == "/api/data":
                p = data_path(q.get("char", ""), q.get("variant", ""))
                if not os.path.exists(p):
                    return self.fail(404, "no training data for this character")
                return self.json(json.load(open(p, encoding="utf-8")))
            return self.fail(404, "not found")
        except FileNotFoundError as e:
            return self.fail(404, str(e))
        except ValueError as e:
            return self.fail(400, str(e))
        except (RuntimeError, subprocess.CalledProcessError) as e:
            return self.fail(503, str(e))
        except Exception as e:  # noqa: BLE001  (an answer beats a dropped connection)
            return self.fail(500, f"{type(e).__name__}: {e}")

    @staticmethod
    def source_svg(req: dict) -> str:
        """The glyph a request refers to: uploaded SVG text, or font + char."""
        if "svg" in req:
            svg = req["svg"]
            if not isinstance(svg, str) or "<path" not in svg:
                raise ValueError("svg must be SVG text with at least one <path>")
            return svg
        font, char = req.get("font", ""), req.get("char", "")
        p = glyph_path(font, char)
        if not os.path.exists(p):
            unpack_font_chars(font, [safe_char(char)])
        if not os.path.exists(p):
            raise FileNotFoundError("no such glyph")
        return open(p, encoding="utf-8").read()

    def do_POST(self):
        url = urlparse(self.path)
        why = self.refused(post=True)
        if why:
            return self.fail(403, why)
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
            if url.path == "/api/predict_batch":
                font = req.get("font", "")
                if font not in font_names():
                    raise FileNotFoundError(f"unknown font: {font!r}")
                raw_chars = req.get("chars")
                if not isinstance(raw_chars, list) or not raw_chars:
                    raise ValueError("chars must be a non-empty list")
                chars = [safe_char(c) for c in raw_chars]
                t = time.perf_counter()
                gdir = unpack_font_chars(font, chars)
                present = set(os.listdir(gdir)) if os.path.isdir(gdir) else set()
                unique_chars = list(dict.fromkeys(chars))
                paths = [os.path.join(gdir, f"{c}.svg") if f"{c}.svg" in present else None for c in unique_chars]
                preds_by_char = dict(zip(unique_chars, self.model.predict_files_batch(paths)))
                total_ms = round((time.perf_counter() - t) * 1000)
                n_ok = sum(1 for r in preds_by_char.values() if r and r["strokes"])
                avg_ms = round(total_ms / max(1, n_ok))
                have_data = set(os.listdir(HWDATA)) if os.path.isdir(HWDATA) else set()
                existing_by_char: dict[str, dict | None] = {}
                for c in unique_chars:
                    fname = f"{c}.json"
                    if fname in have_data:
                        with open(os.path.join(HWDATA, fname), encoding="utf-8") as fh:
                            existing_by_char[c] = json.load(fh)
                    else:
                        existing_by_char[c] = None
                results = []
                for c in chars:
                    if f"{c}.svg" not in present:
                        results.append({"error": "no such glyph"})
                    else:
                        r = preds_by_char.get(c)
                        if not r or not r["strokes"]:
                            results.append({"error": (r or {}).get("error") or "no outline found in this SVG"})
                        else:
                            r = {k: v for k, v in r.items() if k != "error"}
                            results.append({**r, "ms": avg_ms, "existing": existing_by_char[c]})
                return self.json({"results": results, "ms": total_ms})
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
        except (RuntimeError, subprocess.CalledProcessError) as e:
            return self.fail(503, str(e))
        except Exception as e:  # noqa: BLE001
            return self.fail(500, f"{type(e).__name__}: {e}")


def host_header_names(hosts: list[str], port: int) -> set[str]:
    """Host header values that address this server: 'name:port' (and bare 'name' on port 80)."""
    out = set()
    for h in hosts:
        h = h.strip().lower()
        if h.startswith("["):  # [::1] or [::1]:8417
            names = [h] if "]:" in h else [f"{h}:{port}", h]
        elif h.count(":") == 1:  # name:port
            names = [h]
        elif ":" in h:  # bare IPv6 address
            names = [f"[{h}]:{port}", f"[{h}]"]
        else:
            names = [f"{h}:{port}", h]
        out.update(n for n in names if ":" in n.rsplit("]", 1)[-1] or port == 80)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8417)
    ap.add_argument("--device", help="torch device, e.g. cpu or cuda (default: cuda if available)")
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME[:PORT]",
                    help="also answer requests addressed to this host name (repeatable)")
    args = ap.parse_args()
    if args.host in LOOPBACK or args.allow_host:
        Handler.allowed_hosts = host_header_names([*LOOPBACK, args.host, *args.allow_host], args.port)
    else:
        print(f"WARNING: listening on {args.host} without authentication: anyone who can reach this port can "
              "overwrite the training data. Use --allow-host to pin the accepted host names.", file=sys.stderr)
    Handler.model = Model(args.device)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'[' + args.host + ']' if ':' in args.host else args.host}:{args.port}/"
    print(f"dashboard on {url} (model loading in the background)")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
