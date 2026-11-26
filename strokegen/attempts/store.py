"""Append-only attempt store: one JSONL event log per glyph, a derived SQLite index, a derived staging view.

Layout under the store root (training_data/attempts by default; see its README.md):

    log/<font>/<U+XXXX>.jsonl   events for one glyph (font + character), append-only, tracked in git
    log/_checkpoints.jsonl      model checkpoints seen, in first-seen order (recency weighting)
    index.sqlite                derived from log/ (gitignored); rebuilt for any log file that changed
    staging/<font>/<U+XXXX>.json   newest live version of every glyph without an approved one (gitignored)

Events (one JSON object per line, never rewritten):

    glyph       the font outline a glyph's attempts refer to (exact segments), once per outline
    attempt     one version: labels in segment space, hanzi-writer data, producer and parent
    check       a checker result for an attempt (the last one counts)
    verdict     pending | approved | rejected | superseded | unresolved (the last one counts)
    checkpoint  a model checkpoint id, first time it produced an attempt

Ordering is by the event's "at" timestamp, then its line, so logs merged from several machines
(git keeps both sides' lines) still read consistently.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import UTC, datetime

import numpy as np

from ..paths import ATTEMPTS, HWDATA
from ..svg_io import Seg
from .labels import content_hash, glyph_key, labels_from_hw, script_of

SOURCE_TYPES = ("model", "fixup_script", "llm_repair", "human_edit", "refiner", "synthetic", "imported")
VERDICTS = ("pending", "approved", "rejected", "superseded", "unresolved")
LIVE = ("pending", "unresolved", "rejected")  # a glyph whose newest attempt is one of these still needs work
CHECKPOINTS_LOG = "_checkpoints.jsonl"
_FONT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER);
CREATE TABLE IF NOT EXISTS glyphs (file TEXT, font TEXT, char TEXT, glyph_key TEXT, outline TEXT, at TEXT,
    PRIMARY KEY (font, char, glyph_key));
CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT PRIMARY KEY, file TEXT, line INTEGER, font TEXT, char TEXT,
    glyph_key TEXT, parent TEXT, source_type TEXT, source_ref TEXT, script TEXT, content_hash TEXT,
    created_at TEXT, tags TEXT, labels TEXT, hw TEXT);
CREATE INDEX IF NOT EXISTS attempts_glyph ON attempts (font, char);
CREATE TABLE IF NOT EXISTS checks (attempt_id TEXT, file TEXT, line INTEGER, at TEXT, passed INTEGER, checker TEXT);
CREATE INDEX IF NOT EXISTS checks_attempt ON checks (attempt_id);
CREATE TABLE IF NOT EXISTS verdicts (attempt_id TEXT, file TEXT, line INTEGER, at TEXT, verdict TEXT, reason TEXT,
    tags TEXT, by TEXT);
CREATE INDEX IF NOT EXISTS verdicts_attempt ON verdicts (attempt_id);
CREATE TABLE IF NOT EXISTS checkpoints (sha TEXT PRIMARY KEY, line INTEGER, first_seen TEXT, info TEXT);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def new_id() -> str:
    """Time-ordered and random, so ids made on different machines never collide."""
    return f"{time.time_ns() // 1_000_000:012x}-{secrets.token_hex(4)}"


def glyph_stem(char: str) -> str:
    """File stem for a character: its code points, so any glyph name is a safe file name."""
    if not isinstance(char, str) or not char or len(char) > 8:
        raise ValueError(f"invalid character name: {char!r}")
    return "_".join(f"U+{ord(c):04X}" for c in char)


def check_font(font: str) -> str:
    if not isinstance(font, str) or not _FONT_RE.match(font):
        raise ValueError(f"invalid font name: {font!r}")
    return font


def outline_to_json(contours) -> list:
    return [[[s.kind, np.round(s.pts, 3).tolist()] for s in c] for c in contours]


def outline_from_json(outline) -> list[list[Seg]]:
    return [[Seg(k, np.asarray(p, float)) for k, p in c] for c in outline]


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=_plain)


class AttemptStore:
    """Read and append glyph versions. Safe to share between threads of one process; several
    processes (dashboard + CLI) may append at the same time, each keeps its own index in sync."""

    def __init__(self, root: str = ATTEMPTS, hwdata: str | None = HWDATA):
        self.root = os.path.abspath(root)
        self.log_dir = os.path.join(self.root, "log")
        self.staging_dir = os.path.join(self.root, "staging")
        self.hwdata = hwdata
        self._lock = threading.RLock()
        os.makedirs(self.log_dir, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(self.root, "index.sqlite"), timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.executescript(SCHEMA)
        self.sync()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ files and index
    def log_path(self, font: str, char: str) -> str:
        return os.path.join(self.log_dir, check_font(font), glyph_stem(char) + ".jsonl")

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self.log_dir)

    def _append(self, path: str, events: list[dict]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = "".join(_dumps(e) + "\n" for e in events).encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:  # one write per call: concurrent appenders never interleave inside a line
            os.write(fd, data)
        finally:
            os.close(fd)
        self._index_file(path, force=True)

    def sync(self, font: str | None = None) -> int:
        """Re-read every log file (of one font) that changed since it was indexed. Returns how many."""
        base = self.log_dir if font is None else os.path.join(self.log_dir, check_font(font))
        n = 0
        with self._lock:
            seen = set()
            for d, _, names in os.walk(base):
                for name in names:
                    if name.endswith(".jsonl"):
                        p = os.path.join(d, name)
                        seen.add(self._rel(p))
                        n += self._index_file(p)
            prefix = "" if font is None else font + os.sep
            gone = [r["path"] for r in self._db.execute("SELECT path FROM files")
                    if r["path"].startswith(prefix) and r["path"] not in seen]
            for rel in gone:
                self._drop_file(rel)
        return n

    def reindex(self) -> None:
        """Forget the index and rebuild it from the logs."""
        with self._lock, self._db:
            for t in ("files", "glyphs", "attempts", "checks", "verdicts", "checkpoints"):
                self._db.execute(f"DELETE FROM {t}")
        self.sync()

    def _drop_file(self, rel: str) -> None:
        with self._db:
            for t in ("glyphs", "attempts", "checks", "verdicts"):
                self._db.execute(f"DELETE FROM {t} WHERE file = ?", (rel,))
            self._db.execute("DELETE FROM files WHERE path = ?", (rel,))

    def _index_file(self, path: str, force: bool = False) -> int:
        rel = self._rel(path)
        st = os.stat(path)
        with self._lock:
            row = self._db.execute("SELECT size, mtime_ns FROM files WHERE path = ?", (rel,)).fetchone()
            if not force and row and row["size"] == st.st_size and row["mtime_ns"] == st.st_mtime_ns:
                return 0
            self._drop_file(rel)
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
            with self._db:
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:  # a half-resolved merge conflict marker, say: skip, don't lose the rest
                        continue
                    self._index_event(rel, i, e)
                self._db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (rel, st.st_size, st.st_mtime_ns))
        return 1

    def _index_event(self, rel: str, line: int, e: dict) -> None:
        t = e.get("type")
        q = self._db.execute
        if t == "glyph":
            q("INSERT OR IGNORE INTO glyphs VALUES (?, ?, ?, ?, ?, ?)",
              (rel, e["font"], e["char"], e["glyph_key"], _dumps(e["outline"]), e["at"]))
        elif t == "attempt":
            q("INSERT OR IGNORE INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (e["attempt_id"], rel, line, e["font"], e["char"], e["glyph_key"], e.get("parent_attempt_id"),
               e["source_type"], _dumps(e.get("source_ref") or {}), e.get("script"), e["content_hash"],
               e["created_at"], _dumps(e.get("tags") or []), _dumps(e["labels"]), _dumps(e["hw"])))
        elif t == "check":
            c = e.get("checker") or {}
            q("INSERT INTO checks VALUES (?, ?, ?, ?, ?, ?)",
              (e["attempt_id"], rel, line, e["at"], int(bool(c.get("passed"))), _dumps(c)))
        elif t == "verdict":
            q("INSERT INTO verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (e["attempt_id"], rel, line, e["at"], e["verdict"], e.get("reason"), _dumps(e.get("tags") or []),
               e.get("by")))
        elif t == "checkpoint":
            q("INSERT OR IGNORE INTO checkpoints VALUES (?, ?, ?, ?)",
              (e["sha"], line, e["at"], _dumps({k: v for k, v in e.items() if k not in ("type", "sha", "at")})))

    # ------------------------------------------------------------------ reading
    def _attempt(self, row: sqlite3.Row) -> dict:
        a = dict(attempt_id=row["attempt_id"], parent_attempt_id=row["parent"], font=row["font"], char=row["char"],
                 glyph_key=row["glyph_key"], source_type=row["source_type"], source_ref=json.loads(row["source_ref"]),
                 script=row["script"], content_hash=row["content_hash"], created_at=row["created_at"],
                 tags=json.loads(row["tags"]), labels=json.loads(row["labels"]), hw=json.loads(row["hw"]))
        v = self._db.execute("SELECT * FROM verdicts WHERE attempt_id = ? ORDER BY at DESC, line DESC LIMIT 1",
                             (a["attempt_id"],)).fetchone()
        a.update(verdict=v["verdict"] if v else "pending", reason=v["reason"] if v else None,
                 verdict_tags=json.loads(v["tags"]) if v else [], verdict_by=v["by"] if v else None,
                 verdict_at=v["at"] if v else None)
        c = self._db.execute("SELECT checker FROM checks WHERE attempt_id = ? ORDER BY at DESC, line DESC LIMIT 1",
                             (a["attempt_id"],)).fetchone()
        a["checker"] = json.loads(c["checker"]) if c else None
        return a

    def get(self, attempt_id: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
            if row is None:
                raise KeyError(f"no attempt {attempt_id!r}")
            return self._attempt(row)

    def history(self, font: str, char: str, sync: bool = True) -> list[dict]:
        """Every version of one glyph, oldest first."""
        with self._lock:
            p = self.log_path(font, char)
            if sync and os.path.exists(p):
                self._index_file(p)
            rows = self._db.execute("SELECT * FROM attempts WHERE font = ? AND char = ? ORDER BY created_at, line",
                                    (font, char)).fetchall()
            return [self._attempt(r) for r in rows]

    def glyphs(self, font: str | None = None) -> list[tuple[str, str]]:
        with self._lock:
            if font is None:
                rows = self._db.execute("SELECT DISTINCT font, char FROM attempts ORDER BY font, char")
            else:
                rows = self._db.execute("SELECT DISTINCT font, char FROM attempts WHERE font = ? ORDER BY char",
                                        (font,))
            return [(r["font"], r["char"]) for r in rows]

    def contours(self, font: str, char: str, key: str) -> list[list[Seg]]:
        with self._lock:
            row = self._db.execute("SELECT outline FROM glyphs WHERE font = ? AND char = ? AND glyph_key = ?",
                                   (font, char, key)).fetchone()
        if row is None:
            raise KeyError(f"no outline for {font}/{char} ({key})")
        return outline_from_json(json.loads(row["outline"]))

    def checkpoints(self) -> list[dict]:
        """Checkpoints in the order they first produced attempts (oldest first)."""
        with self._lock:
            p = os.path.join(self.log_dir, CHECKPOINTS_LOG)
            if os.path.exists(p):
                self._index_file(p)
            rows = self._db.execute("SELECT * FROM checkpoints ORDER BY first_seen, line").fetchall()
        return [dict(sha=r["sha"], first_seen=r["first_seen"], **json.loads(r["info"])) for r in rows]

    @staticmethod
    def status(history: list[dict]) -> tuple[str, dict | None]:
        """(glyph status, the attempt it refers to) for a glyph's history.

        approved: an approved version exists (the newest one is returned). pending / unresolved: no
        approved version, and the newest live one (pending, unresolved or rejected) is returned; its
        verdict "unresolved" makes the glyph unresolved. replaced: every version was superseded.
        """
        approved = [a for a in history if a["verdict"] == "approved"]
        if approved:
            return "approved", approved[-1]
        live = [a for a in history if a["verdict"] in LIVE]
        if not live:
            return ("replaced" if history else "new"), None
        top = live[-1]
        return ("unresolved" if top["verdict"] == "unresolved" else "pending"), top

    def staged(self, font: str, char: str) -> dict | None:
        """The version a repair pass or reviewer should work from, or None when the glyph is approved."""
        st, a = self.status(self.history(font, char))
        return a if st in ("pending", "unresolved") else None

    # ------------------------------------------------------------------ writing
    def _ensure_glyph(self, path: str, font: str, char: str, contours, key: str) -> list[dict]:
        row = self._db.execute("SELECT 1 FROM glyphs WHERE font = ? AND char = ? AND glyph_key = ?",
                               (font, char, key)).fetchone()
        if row:
            return []
        return [dict(type="glyph", font=font, char=char, glyph_key=key, outline=outline_to_json(contours), at=now())]

    def record(self, font: str, char: str, contours, hw: dict, source_type: str, source_ref: dict | None = None,
               parent: str | None = None, seg_stroke=None, tags: list[str] | None = None,
               checker: dict | None = None, created_at: str | None = None, vetoed=None) -> tuple[str, bool]:
        """Append one version of a glyph. Returns (attempt_id, created).

        hw: {strokes, medians} in writing order. seg_stroke: the producer's own segment labels when it
        has them. vetoed: segments of merges the decoder's bridge validator refused (model and refiner
        versions). A version identical to an earlier one from the same producer is not stored twice:
        that attempt's id comes back with created=False. created_at: when the version was really made
        (backfill from git history); default now.
        """
        if source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SOURCE_TYPES}")
        hw = {"strokes": list(hw["strokes"]), "medians": [[[int(round(x)), int(round(y))] for x, y in m]
                                                          for m in hw["medians"]]}
        if len(hw["strokes"]) != len(hw["medians"]):
            raise ValueError("need one median per stroke")
        key = glyph_key(contours)
        ref = source_ref or {}
        ch = content_hash(hw)
        path = self.log_path(font, char)
        with self._lock:
            if os.path.exists(path):
                self._index_file(path)
            for r in self._db.execute("SELECT attempt_id, source_ref FROM attempts WHERE font = ? AND char = ? AND "
                                      "glyph_key = ? AND content_hash = ? AND source_type = ?",
                                      (font, char, key, ch, source_type)):
                if r["source_ref"] == _dumps(ref):
                    return r["attempt_id"], False
            if parent is not None and self._db.execute("SELECT 1 FROM attempts WHERE attempt_id = ?",
                                                       (parent,)).fetchone() is None:
                raise KeyError(f"parent attempt {parent!r} does not exist")
            events = self._ensure_glyph(path, font, char, contours, key)
            aid = new_id()
            events.append(dict(
                type="attempt", attempt_id=aid, parent_attempt_id=parent, font=font, char=char, glyph_key=key,
                codepoint=" ".join(f"U+{ord(c):04X}" for c in char), script=script_of(char),
                source_type=source_type, source_ref=ref, labels=labels_from_hw(contours, hw, seg_stroke, vetoed), hw=hw,
                content_hash=ch, tags=list(tags or []), created_at=created_at or now()))
            if checker is not None:
                events.append(dict(type="check", attempt_id=aid, checker=checker, at=now()))
            self._append(path, events)
            self._restage(font, char)
        return aid, True

    def add_check(self, attempt_id: str, checker: dict) -> None:
        with self._lock:
            a = self.get(attempt_id)
            self._append(self.log_path(a["font"], a["char"]),
                         [dict(type="check", attempt_id=attempt_id, checker=checker, at=now())])
            self._restage(a["font"], a["char"])

    def check(self, attempt_id: str) -> dict:
        """Run the checker on a stored version and record the result."""
        from .check import run_check
        a = self.get(attempt_id)
        res = run_check(self.contours(a["font"], a["char"], a["glyph_key"]), a["hw"])
        self.add_check(attempt_id, res)
        return res

    def set_verdict(self, attempt_id: str, verdict: str, reason: str | None = None, tags: list[str] | None = None,
                    by: str | None = None) -> None:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        with self._lock:
            a = self.get(attempt_id)
            known_bad = a["checker"] is not None and not a["checker"]["passed"]
            if verdict == "rejected" and not (reason or "").strip() and not known_bad:
                # a version the checker passed (or never saw) that is still wrong is exactly the case the
                # refiner needs to learn from: say what is wrong with it
                raise ValueError("rejecting a version that passed the checker needs a reason")
            if verdict == "unresolved" and not (reason or "").strip():
                raise ValueError("marking a glyph unresolved needs a reason")
            self._append(self.log_path(a["font"], a["char"]), [dict(
                type="verdict", attempt_id=attempt_id, verdict=verdict, reason=(reason or "").strip() or None,
                tags=list(tags or []), by=by, at=now())])
            self._restage(a["font"], a["char"])

    def approve(self, attempt_id: str, by: str | None = None, reason: str | None = None) -> list[str]:
        """Approve a version: its glyph's other versions (except rejected ones) become superseded, and so
        do approved versions of the same character in other fonts, whose training file this replaces.
        Returns the superseded attempt ids."""
        with self._lock:
            a = self.get(attempt_id)
            at = now()
            events = [dict(type="verdict", attempt_id=attempt_id, verdict="approved", reason=reason, tags=[], by=by,
                           at=at)]
            done = []
            for o in self.history(a["font"], a["char"], sync=False):
                if o["attempt_id"] != attempt_id and o["verdict"] in ("pending", "unresolved", "approved"):
                    events.append(dict(type="verdict", attempt_id=o["attempt_id"], verdict="superseded",
                                       reason=f"superseded by {attempt_id}", tags=[], by=by, at=at))
                    done.append(o["attempt_id"])
            self._append(self.log_path(a["font"], a["char"]), events)
            self._restage(a["font"], a["char"])
            stem = glyph_stem(a["char"]) + ".jsonl"
            for font in sorted(os.listdir(self.log_dir)):
                p = os.path.join(self.log_dir, font, stem)
                if font == a["font"] or not os.path.isfile(p):
                    continue
                others = [o for o in self.history(font, a["char"]) if o["verdict"] == "approved"]
                if others:
                    self._append(p, [dict(type="verdict", attempt_id=o["attempt_id"], verdict="superseded",
                                          reason=f"training file replaced by {a['font']} ({attempt_id})", tags=[],
                                          by=by, at=at) for o in others])
                    done += [o["attempt_id"] for o in others]
                    self._restage(font, a["char"])
        return done

    def note_checkpoint(self, sha: str, info: dict) -> None:
        """Remember a checkpoint the first time it produces an attempt (orders checkpoints for recency)."""
        with self._lock:
            p = os.path.join(self.log_dir, CHECKPOINTS_LOG)
            if os.path.exists(p):
                self._index_file(p)
            if self._db.execute("SELECT 1 FROM checkpoints WHERE sha = ?", (sha,)).fetchone():
                return
            self._append(p, [dict(type="checkpoint", sha=sha, at=now(), **info)])

    # ------------------------------------------------------------------ staging view
    def staging_path(self, font: str, char: str) -> str:
        return os.path.join(self.staging_dir, check_font(font), glyph_stem(char) + ".json")

    def _has_training_file(self, char: str) -> bool:
        return bool(self.hwdata) and os.path.exists(os.path.join(self.hwdata, f"{char}.json"))

    def _restage(self, font: str, char: str) -> None:
        hist = self.history(font, char, sync=False)
        st, a = self.status(hist)
        p = self.staging_path(font, char)
        if st not in ("pending", "unresolved"):
            if os.path.exists(p):
                os.unlink(p)
            return
        rec = dict(
            char=char, font=font, status=st, attempt_id=a["attempt_id"], source_type=a["source_type"],
            verdict=a["verdict"], reason=a["reason"], versions=len(hist),
            checker=None if a["checker"] is None else {k: a["checker"].get(k) for k in
                                                         ("passed", "iou", "failures", "strokes")},
            has_training_file=self._has_training_file(char),
            history=[dict(attempt_id=h["attempt_id"], source_type=h["source_type"], verdict=h["verdict"],
                          reason=h["reason"], passed=None if h["checker"] is None else h["checker"].get("passed"))
                     for h in hist],
            strokes=a["hw"]["strokes"], medians=a["hw"]["medians"], segStroke=a["labels"]["seg_stroke"],
        )
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
        os.replace(tmp, p)

    def stage(self, font: str | None = None) -> int:
        """Rebuild the staging view (of one font) from the logs. Returns how many glyphs are staged."""
        self.sync(font)
        with self._lock:
            fonts = [font] if font else None
            if fonts is None:
                fonts = sorted({f for f, _ in self.glyphs()})
                if os.path.isdir(self.staging_dir):
                    fonts = sorted(set(fonts) | {d for d in os.listdir(self.staging_dir)
                                                 if os.path.isdir(os.path.join(self.staging_dir, d))})
            for f in fonts:
                d = os.path.join(self.staging_dir, f)
                if os.path.isdir(d):
                    for name in os.listdir(d):
                        if name.endswith((".json", ".json.tmp")):
                            os.unlink(os.path.join(d, name))
                for _, c in self.glyphs(f):
                    self._restage(f, c)
            return sum(len([n for n in os.listdir(os.path.join(self.staging_dir, f)) if n.endswith(".json")])
                       for f in fonts if os.path.isdir(os.path.join(self.staging_dir, f)))

    def staged_list(self, font: str) -> list[dict]:
        """Summary rows of the staged glyphs of a font, read from the staging view (synced first)."""
        if self.sync(font):
            self.stage(font)
        d = os.path.join(self.staging_dir, check_font(font))
        out = []
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.endswith(".json"):
                    try:
                        with open(os.path.join(d, name), encoding="utf-8") as f:
                            r = json.load(f)
                    except (OSError, ValueError):
                        continue
                    out.append({k: r.get(k) for k in ("char", "status", "attempt_id", "source_type", "verdict",
                                                      "reason", "versions", "has_training_file")}
                               | {"passed": (r.get("checker") or {}).get("passed")})
        return out
