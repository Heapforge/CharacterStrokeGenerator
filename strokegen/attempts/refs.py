"""Stable references to whatever produced a version: checkpoint hashes and script revisions."""
from __future__ import annotations

import hashlib
import os
import subprocess

from ..paths import ROOT

_sha_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str, n: int = 16) -> str:
    """Hex digest prefix of a file's bytes (cached by path, size and mtime)."""
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if k not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _sha_cache[k] = h.hexdigest()
    return _sha_cache[k][:n]


def _git(*args: str, cwd: str = ROOT) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def script_ref(path: str, args: str | None = None) -> dict:
    """{script, git_sha, blob_sha, dirty[, args]} for a fix-up script.

    blob_sha is the git object id of the file as it is now, so the exact script can be found even
    when it was run uncommitted (dirty=True); commit it before importing to keep it reproducible.
    """
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise FileNotFoundError(f"no such script: {path}")
    rel = os.path.relpath(ap, ROOT)
    blob = _git("hash-object", ap)
    committed = _git("rev-parse", f"HEAD:{rel}")
    ref = dict(script=rel, git_sha=_git("rev-parse", "HEAD"), blob_sha=blob, dirty=blob is None or blob != committed)
    if args:
        ref["args"] = args
    return ref
