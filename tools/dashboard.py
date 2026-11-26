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

Unless started with --no-attempts, every prediction is also recorded in the attempt store
(training_data/attempts, see strokegen.attempts) and checked in the background, which keeps the
staging view of not-yet-approved glyphs current; Save approves the saved version there (recording
it as a human edit when it differs from every stored version) and Reject records why a version is
wrong. A store error never fails a Save: the training file is written first, exactly as without it.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
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
UPLOAD_FONT = "upload"  # attempt-store font name for uploaded SVGs


def _pred_view(r: dict) -> dict:
    """A StrokeGenerator result as the page gets it, plus the private keys the attempt store needs
    (_vetoed, _rounds of the refinement loop); public() drops those before a reply."""
    out = {"strokes": r["strokes"], "medians": r["medians"], "segStroke": r["_seg_stroke"].tolist()}
    if "_vetoed" in r:
        out["_vetoed"] = r["_vetoed"].tolist()
    if "_rounds" in r:
        out["_rounds"] = [{"round": rd["round"], "strokes": rd["strokes"], "medians": rd["medians"],
                           "seg_stroke": [int(v) for v in rd["seg_stroke"]], "checker": rd["checker"],
                           "vetoed": None if rd["vetoed"] is None else [int(v) for v in rd["vetoed"]]}
                          for rd in r["_rounds"]]
        out["refine"] = r["_refine"]
    return out


def public(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


class Model:
    """Loads StrokeGenerator in the background so the page is usable right away."""

    def __init__(self, device: str | None, refine_iters: int = 0, refine_ckpt: str | None = None):
        self.gen = None
        self.error = None
        self.device = device
        self.refine = dict(refine_iters=refine_iters, refine_ckpt=refine_ckpt) if refine_iters else {}
        self.lock = threading.Lock()
        self.ready = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from strokegen.predict import StrokeGenerator
            self.gen = StrokeGenerator(device=self.device, **self.refine)
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
        return {**_pred_view(r), "_contours": contours}

    def predict_files_batch(self, paths: list[str | None]) -> list[dict | None]:
        self.ready.wait()
        if self.gen is None:
            raise RuntimeError(f"model failed to load: {self.error}")
        with self.lock:
            raw = self.gen.predict_files_batch(paths)
        return [None if r is None else {**_pred_view(r), "error": r.get("_error")} for r in raw]

    def source_ref(self) -> dict:
        """The loaded checkpoints and decoding parameters, for the attempt store ({} if unknown)."""
        fn = getattr(self.gen, "source_ref", None)
        return fn() if fn else {}

    def checkpoint_info(self) -> dict:
        fn = getattr(self.gen, "checkpoint_info", None)
        return fn() if fn else {}

    def refine_ref(self, round_: int) -> dict:
        return self.gen.refine_ref(round_)

    def status(self) -> dict:
        if not self.ready.is_set():
            return {"state": "loading"}
        if self.gen is None:
            return {"state": "error", "error": self.error}
        return {"state": "ready", "device": self.device}


class Recorder:
    """Writes predictions to the attempt store and runs the checker, on one background thread.

    Recording a glyph (parsing, mapping it onto segment labels) and checking it take longer than
    predicting it, so a bulk prediction answers right away; flush(font, char) waits until a glyph's
    jobs are done, before a Save or a history lookup reads the store.
    """

    def __init__(self, store, model: Model):
        self.store = store
        self.model = model
        self.jobs: queue.Queue = queue.Queue()
        self.cond = threading.Condition()
        self.pending: dict[tuple[str, str], int] = {}
        self.errors: list[str] = []
        self._noted = False
        threading.Thread(target=self._run, daemon=True).start()

    def _ref(self) -> dict:
        ref = getattr(self.model, "source_ref", dict)()
        if not self._noted:
            self._noted = True
            for sha, info in getattr(self.model, "checkpoint_info", dict)().items():
                self.store.note_checkpoint(sha, info)
        return ref

    def _store(self, font: str, char: str, contours, pred: dict) -> tuple[str, bool]:
        """Record a prediction: (attempt id of the version shown, whether it still needs a check).

        With the refinement loop on, every round is a version: round 0 as `model`, later rounds as
        `refiner`, each the parent of the next, with the checker result the loop already computed."""
        rounds = pred.get("_rounds")
        if not rounds:
            aid, created = self.store.record(font, char, contours, pred, "model", self._ref(),
                                             seg_stroke=pred["segStroke"], vetoed=pred.get("_vetoed"))
            return aid, created or self.store.get(aid)["checker"] is None
        aid = None
        for rd in rounds:
            first = rd["round"] == 0
            aid, _ = self.store.record(
                font, char, contours, rd, "model" if first else "refiner",
                self._ref() if first else self.model.refine_ref(rd["round"]), parent=aid,
                seg_stroke=rd["seg_stroke"], vetoed=rd["vetoed"], checker=rd["checker"])
        return aid, self.store.get(aid)["checker"] is None

    def record_now(self, font: str, char: str, contours, pred: dict) -> str:
        """Record a prediction synchronously (one glyph) and queue its check. Returns the attempt id."""
        aid, needs_check = self._store(font, char, contours, pred)
        if needs_check:
            self._submit(font, char, ("check", aid))
        return aid

    def record_later(self, font: str, char: str, svg_path: str, pred: dict) -> None:
        self._submit(font, char, ("record", svg_path, pred))

    def _submit(self, font: str, char: str, job: tuple) -> None:
        with self.cond:
            self.pending[(font, char)] = self.pending.get((font, char), 0) + 1
        self.jobs.put((font, char, job))

    def flush(self, font: str, char: str, timeout: float = 120.0) -> None:
        with self.cond:
            self.cond.wait_for(lambda: not self.pending.get((font, char)), timeout)

    def _run(self):
        while True:
            font, char, job = self.jobs.get()
            try:
                if job[0] == "record":
                    from strokegen.svg_io import load_svg_glyph
                    with open(job[1], encoding="utf-8") as f:
                        contours = load_svg_glyph(f.read())
                    aid, needs_check = self._store(font, char, contours, job[2])
                    if needs_check:
                        self.store.check(aid)
                else:
                    self.store.check(job[1])
            except Exception as e:  # noqa: BLE001  (a bad glyph must not stop the recorder)
                self.errors.append(f"{font}/{char}: {type(e).__name__}: {e}")
                sys.stderr.write(f"attempt store: {self.errors[-1]}\n")
            finally:
                with self.cond:
                    self.pending[(font, char)] -= 1
                    if not self.pending[(font, char)]:
                        del self.pending[(font, char)]
                    self.cond.notify_all()


def resolve_attempt(store, font: str, char: str, data: dict | None, attempt_id: str | None) -> dict | None:
    """The stored version whose content is `data` (preferring attempt_id among equals), or None when no
    stored version has that content (an edit). Without data: the attempt attempt_id."""
    from strokegen.attempts.labels import content_hash
    hist = store.history(font, char)
    if data is not None:
        h = content_hash(validate_hw(data))
        same = [a for a in hist if a["content_hash"] == h]
        return next((a for a in same if a["attempt_id"] == attempt_id), same[-1]) if same else None
    if attempt_id:
        return next((a for a in hist if a["attempt_id"] == attempt_id), None)
    return None


def approve_saved(store, font: str, char: str, data: dict, attempt_id: str | None, edits, reviewer: str) -> str:
    """Approve the saved version in the store; a version no producer made is recorded as a human edit."""
    att = resolve_attempt(store, font, char, data, attempt_id)
    if att is None:
        hist = store.history(font, char, sync=False)
        base = (next((a for a in hist if a["attempt_id"] == attempt_id), None) if attempt_id else None) or \
            store.staged(font, char) or (hist[-1] if hist else None)
        if base is None:
            return "attempt store: no recorded version of this glyph to approve"
        contours = store.contours(font, char, base["glyph_key"])
        aid, _ = store.record(font, char, contours, data, "human_edit", {"reviewer": reviewer, "edits": edits},
                              parent=base["attempt_id"])
        store.check(aid)
        att = store.get(aid)
        how = "recorded your edit and approved it"
    else:
        how = f"approved the {att['source_type']} version"
    if att["verdict"] != "approved":
        store.approve(att["attempt_id"], by=reviewer)
    return f"attempt store: {how}"


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
    """A character name usable as a file stem (normally one code point).

    Any single code point but a path separator, NUL or "." is a glyph name: fonts export glyphs for
    spaces, line breaks and the backslash too. Longer names may not have surrounding whitespace.
    """
    if not isinstance(c, str) or not c or len(c) > 8 or c.startswith(".") or any(
            ch in c for ch in ("/", "\0", os.sep, os.altsep) if ch) or (len(c) > 1 and c.strip() != c):
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


def attempt_view(a: dict, full: bool = False) -> dict:
    """What the page needs of a stored version (full: with its strokes, to load it into the queue)."""
    c = a["checker"]
    out = {"attemptId": a["attempt_id"], "parent": a["parent_attempt_id"], "source": a["source_type"],
           "ref": a["source_ref"], "createdAt": a["created_at"], "verdict": a["verdict"], "reason": a["reason"],
           "tags": a["verdict_tags"], "nStrokes": len(a["hw"]["strokes"]),
           "checker": None if c is None else {"passed": c.get("passed"), "iou": c.get("iou"),
                                              "failures": c.get("failures", []), "error": c.get("error"),
                                              "strokes": [r.get("failures", []) for r in c.get("strokes", [])]}}
    if full:
        out.update(strokes=a["hw"]["strokes"], medians=a["hw"]["medians"], segStroke=a["labels"]["seg_stroke"],
                   font=a["font"], char=a["char"])
    return out


class Handler(BaseHTTPRequestHandler):
    model: Model
    store = None  # strokegen.attempts.AttemptStore, or None with --no-attempts
    recorder: Recorder | None = None
    reviewer = "reviewer"
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
            if url.path == "/api/staging":
                if self.store is None:
                    return self.json({"enabled": False, "items": []})
                font = q.get("font", "")
                return self.json({"enabled": True, "font": font, "items": self.store.staged_list(font)})
            if url.path == "/api/attempt":
                if self.store is None:
                    return self.fail(404, "the attempt store is off (--no-attempts)")
                try:
                    a = self.store.get(q.get("id", ""))
                except KeyError as e:
                    return self.fail(404, str(e.args[0]))
                return self.json(attempt_view(a, full=True))
            if url.path == "/api/history":
                if self.store is None:
                    return self.json({"enabled": False, "versions": []})
                font, char = q.get("font", ""), safe_char(q.get("char", ""))
                if self.recorder:
                    self.recorder.flush(font, char, timeout=30)
                hist = self.store.history(font, char)
                st, cur = self.store.status(hist)
                return self.json({"enabled": True, "status": st, "staged": cur["attempt_id"] if cur else None,
                                  "versions": [attempt_view(a) for a in hist]})
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
                contours = res.pop("_contours", None)
                if not res["strokes"]:
                    return self.fail(422, "no outline found in this SVG")
                ms = round((time.perf_counter() - t) * 1000)
                if self.recorder and contours is not None:
                    try:
                        font = req.get("font") if "svg" not in req else UPLOAD_FONT
                        res["attemptId"] = self.recorder.record_now(font, safe_char(req.get("char")), contours, res)
                    except Exception as e:  # noqa: BLE001  (the prediction is still good)
                        res["storeError"] = f"{type(e).__name__}: {e}"
                return self.json({**public(res), "ms": ms})
            if url.path == "/api/predict_batch":
                font = req.get("font", "")
                if font not in font_names():
                    raise FileNotFoundError(f"unknown font: {font!r}")
                raw_chars = req.get("chars")
                if not isinstance(raw_chars, list) or not raw_chars:
                    raise ValueError("chars must be a non-empty list")
                # an invalid name fails only its own item, not the whole batch
                invalid: dict[int, str] = {}
                chars = []
                for i, c in enumerate(raw_chars):
                    try:
                        chars.append(safe_char(c))
                    except ValueError as e:
                        invalid[i] = str(e)
                        chars.append(None)
                t = time.perf_counter()
                valid = [c for c in chars if c is not None]
                gdir = unpack_font_chars(font, valid) if valid else os.path.join(GLYPHS, font)
                present = set(os.listdir(gdir)) if os.path.isdir(gdir) else set()
                unique_chars = list(dict.fromkeys(valid))
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
                for i, c in enumerate(chars):
                    if c is None:
                        results.append({"error": invalid[i]})
                    elif f"{c}.svg" not in present:
                        results.append({"error": "no such glyph"})
                    else:
                        r = preds_by_char.get(c)
                        if not r or not r["strokes"]:
                            results.append({"error": (r or {}).get("error") or "no outline found in this SVG"})
                        else:
                            r = {k: v for k, v in r.items() if k != "error"}
                            if self.recorder:
                                self.recorder.record_later(font, c, os.path.join(gdir, f"{c}.svg"), r)
                            results.append({**public(r), "ms": avg_ms, "existing": existing_by_char[c]})
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
                out = {"path": os.path.relpath(p, ROOT), "overwrote": existed, "note": note}
                font = req.get("font")
                if self.store is not None and variant == "" and isinstance(font, str) and font:
                    try:  # the training file is written: the store only records that it was approved
                        if self.recorder:
                            self.recorder.flush(font, char)
                        out["store"] = approve_saved(self.store, font, char, data, req.get("attemptId"),
                                                     req.get("edits"), self.reviewer)
                    except Exception as e:  # noqa: BLE001
                        out["store"] = f"attempt store failed: {type(e).__name__}: {e}"
                    out["note"] += " · " + out["store"]
                return self.json(out)
            if url.path == "/api/reject":
                if self.store is None:
                    return self.fail(409, "the attempt store is off (--no-attempts)")
                font, char = req.get("font"), safe_char(req.get("char"))
                if not isinstance(font, str) or not font:
                    raise ValueError("font is required")
                if self.recorder:
                    self.recorder.flush(font, char)
                att = resolve_attempt(self.store, font, char, req.get("data"), req.get("attemptId"))
                if att is None:
                    return self.fail(404, "this version is not in the attempt store (edited versions cannot be "
                                          "rejected: reject the version you started from, or save your fix)")
                tags = req.get("tags") or []
                if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
                    raise ValueError("tags must be a list of strings")
                verdict = "unresolved" if req.get("unresolved") else "rejected"
                self.store.set_verdict(att["attempt_id"], verdict, req.get("reason"), tags, by=self.reviewer)
                return self.json({"attemptId": att["attempt_id"], "verdict": verdict})
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
    ap.add_argument("--no-attempts", action="store_true",
                    help="do not record predictions, verdicts and approvals in the attempt store")
    ap.add_argument("--attempts-root", help="attempt store directory (default: training_data/attempts)")
    ap.add_argument("--reviewer", help="name recorded with approvals and rejections (default: your login)")
    ap.add_argument("--refine-iters", type=int, default=0, metavar="K",
                    help="refine glyphs the checker fails, up to K rounds (default 0: single pass)")
    ap.add_argument("--refine-ckpt", help="refiner checkpoint (default checkpoints/refine.pt)")
    args = ap.parse_args()
    if args.host in LOOPBACK or args.allow_host:
        Handler.allowed_hosts = host_header_names([*LOOPBACK, args.host, *args.allow_host], args.port)
    else:
        print(f"WARNING: listening on {args.host} without authentication: anyone who can reach this port can "
              "overwrite the training data. Use --allow-host to pin the accepted host names.", file=sys.stderr)
    Handler.model = Model(args.device, args.refine_iters, args.refine_ckpt)
    if not args.no_attempts:
        import getpass

        from strokegen.attempts import AttemptStore
        from strokegen.paths import ATTEMPTS
        Handler.store = AttemptStore(args.attempts_root or ATTEMPTS)
        Handler.recorder = Recorder(Handler.store, Handler.model)
        Handler.reviewer = args.reviewer or getpass.getuser()
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
