"""The dashboard with the attempt store: staging, Save approves, Reject, and --no-attempts."""
import json

from glyph_fixtures import MODEL_REF, PLUS_LAB, hw_from_labels, plus, svg_of

# ---------------------------------------------------------------- dashboard

def _dashboard(monkeypatch, tmp_path, with_store: bool):
    from test_pipeline import _load_tool
    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    c = plus()
    font_dir = tmp_path / "glyphs" / "font-a"
    font_dir.mkdir(parents=True)
    (font_dir / "十.svg").write_text(svg_of(c), encoding="utf-8")
    hwdir = tmp_path / "hw"
    hwdir.mkdir(exist_ok=True)
    (tmp_path / "glyphs" / "manifest.json").write_text(json.dumps({"fonts": ["font-a"], "characters": {}}))
    monkeypatch.setattr(dashboard, "GLYPHS", str(tmp_path / "glyphs"))
    monkeypatch.setattr(dashboard, "MANIFEST", str(tmp_path / "glyphs" / "manifest.json"))
    monkeypatch.setattr(dashboard, "HWDATA", str(hwdir))
    monkeypatch.setattr(dashboard, "FONT_FILES", {})
    hw = hw_from_labels(c, PLUS_LAB)

    class StubModel:
        def predict_files_batch(self, paths):
            return [None if p is None else {"strokes": hw["strokes"], "medians": hw["medians"],
                                            "segStroke": PLUS_LAB.tolist(), "error": None} for p in paths]

        def source_ref(self):
            return dict(MODEL_REF)

        def checkpoint_info(self):
            return {"c" * 16: {"kind": "cut", "epoch": 3}}

    model = StubModel()
    monkeypatch.setattr(dashboard.Handler, "model", model, raising=False)
    store = None
    if with_store:
        from strokegen.attempts import AttemptStore
        store = AttemptStore(str(tmp_path / "attempts"), hwdata=str(hwdir))
        monkeypatch.setattr(dashboard.Handler, "store", store)
        monkeypatch.setattr(dashboard.Handler, "recorder", dashboard.Recorder(store, model))
        monkeypatch.setattr(dashboard.Handler, "reviewer", "tester")
    return dashboard, store, hw


def _post(dashboard, path, body):
    from test_pipeline import _request
    return _request(dashboard.Handler, "POST", path, json.dumps(body).encode())


def _get(dashboard, path):
    from test_pipeline import _request
    return _request(dashboard.Handler, "GET", path, headers={"Content-Type": None})


def test_dashboard_without_store_is_unchanged(monkeypatch, tmp_path):
    dashboard, _, hw = _dashboard(monkeypatch, tmp_path, with_store=False)
    status, res = _post(dashboard, "/api/predict_batch", {"font": "font-a", "chars": ["十"]})
    assert status == 200 and res["results"][0]["strokes"] == hw["strokes"]
    status, res = _post(dashboard, "/api/commit", {"char": "十", "font": "font-a", "data": hw})
    assert status == 200 and "store" not in res
    assert not (tmp_path / "attempts").exists()
    assert _get(dashboard, "/api/staging?font=font-a") == (200, {"enabled": False, "items": []})


def test_dashboard_stages_batches_and_save_approves(monkeypatch, tmp_path):
    dashboard, store, hw = _dashboard(monkeypatch, tmp_path, with_store=True)
    status, res = _post(dashboard, "/api/predict_batch", {"font": "font-a", "chars": ["十"]})
    assert status == 200 and res["results"][0]["strokes"] == hw["strokes"]
    dashboard.Handler.recorder.flush("font-a", "十")
    status, res = _get(dashboard, "/api/staging?font=font-a")
    assert status == 200 and [r["char"] for r in res["items"]] == ["十"]
    item = res["items"][0]
    assert item["source_type"] == "model" and item["passed"] is True
    assert store.checkpoints()[0]["sha"] == "c" * 16

    # the training file is byte-identical to what the dashboard writes without the store
    edited = {"strokes": hw["strokes"][::-1], "medians": hw["medians"][::-1]}
    status, res = _post(dashboard, "/api/commit", {"char": "十", "font": "font-a", "data": edited,
                                                   "attemptId": item["attempt_id"], "edits": {"order": [1, 0]}})
    assert status == 200, res
    assert "recorded your edit" in res["store"] and res["store"] in res["note"]
    saved = (tmp_path / "hw" / "十.json").read_bytes()
    assert saved == json.dumps(dashboard.validate_hw(edited), ensure_ascii=False).encode()
    hist = store.history("font-a", "十")
    assert [(h["source_type"], h["verdict"]) for h in hist] == [("model", "superseded"), ("human_edit", "approved")]
    assert hist[1]["parent_attempt_id"] == item["attempt_id"] and hist[1]["source_ref"]["reviewer"] == "tester"
    assert _get(dashboard, "/api/staging?font=font-a")[1]["items"] == []
    status, res = _get(dashboard, "/api/history?font=font-a&char=%E5%8D%81")
    assert status == 200 and res["status"] == "approved" and len(res["versions"]) == 2

    # a fresh prediction of the same glyph dedups onto the stored version; saving it unedited approves it
    _post(dashboard, "/api/predict_batch", {"font": "font-a", "chars": ["十"]})
    dashboard.Handler.recorder.flush("font-a", "十")
    status, res = _post(dashboard, "/api/commit", {"char": "十", "font": "font-a", "data": hw, "overwrite": True})
    assert status == 200 and "approved the model version" in res["store"]
    assert [h["verdict"] for h in store.history("font-a", "十")] == ["approved", "superseded"]


def test_dashboard_reject_needs_a_reason_and_a_failing_store_still_saves(monkeypatch, tmp_path):
    dashboard, store, hw = _dashboard(monkeypatch, tmp_path, with_store=True)
    _post(dashboard, "/api/predict_batch", {"font": "font-a", "chars": ["十"]})
    status, res = _post(dashboard, "/api/reject", {"font": "font-a", "char": "十", "data": hw})
    assert status == 400 and "reason" in res["error"]
    status, res = _post(dashboard, "/api/reject", {"font": "font-a", "char": "十", "data": hw,
                                                   "reason": "the vertical bar comes first", "tags": ["order"]})
    assert status == 200 and res["verdict"] == "rejected"
    assert store.get(res["attemptId"])["reason"] == "the vertical bar comes first"
    # rejected glyphs stay staged for the next repair pass
    assert _get(dashboard, "/api/staging?font=font-a")[1]["items"][0]["verdict"] == "rejected"

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "approve", boom)
    status, res = _post(dashboard, "/api/commit", {"char": "十", "font": "font-a", "data": hw})
    assert status == 200 and "attempt store failed" in res["note"]
    assert (tmp_path / "hw" / "十.json").exists()


