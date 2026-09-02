"""C13: result_query compact-verbs surface over ResultStore (Gap B+A).

Design refs (tool-output-compaction-design.md):
- L57: ResultStore SQLite schema + retention (keep last 200 / 72h).
- L62: query(result_id, verb, pattern, n) compact verbs:
  grep | errors | head | tail | count | lines x-y | json | summary.
- L69: ``id="last"`` aliases the most recent result.
- L72: output of result_query is itself budget-capped and never re-spilled
  into the store (it is already the filtered view).
"""

import json

import pytest

from tools.result_cache import (
    RESULT_QUERY_BUDGET_CHARS,
    ResultStore,
    compact,
    query,
)


@pytest.fixture()
def store(tmp_path):
    return ResultStore(str(tmp_path / "result_cache.sqlite"))


def _ingest(store, text, tool="terminal"):
    return store.ingest(full_text=text, tool=tool, args_json={"cmd": "x"},
                        session_id="s1")


class TestStore:
    def test_ingest_returns_id_and_fetch_roundtrip(self, store):
        rid = _ingest(store, "hello world\nsecond line")
        fetched = store.fetch(rid)
        assert "hello world" in fetched
        assert "second line" in fetched

    def test_ingest_writes_spill_file(self, store, tmp_path):
        rid = _ingest(store, "persist me")
        rec = store.get_record(rid)
        assert rec is not None
        assert rec["full_path"]
        from pathlib import Path
        assert Path(rec["full_path"]).read_text(encoding="utf-8") == "persist me"

    def test_fetch_last_alias(self, store):
        _ingest(store, "first")
        rid2 = _ingest(store, "second")
        assert store.fetch("last") == "second"
        assert store.fetch("last") == store.fetch(rid2)

    def test_fetch_missing_returns_none(self, store):
        assert store.fetch(99999) is None

    def test_retention_keeps_last_200(self, store):
        for i in range(210):
            _ingest(store, f"result-{i}")
        rec = store.get_record("last")
        assert rec is not None
        # The most recent 200 survive; the oldest are vacuumed.
        assert "result-209" == store.fetch("last")
        assert store.fetch(1) is None

    def test_older_than_72h_vacuumed(self, store):
        import sqlite3
        rid = _ingest(store, "old result")
        conn = sqlite3.connect(store.db_path)
        conn.execute("UPDATE results SET created_at = created_at - 73*3600")
        conn.commit()
        conn.close()
        assert store.fetch(rid) is None

    def test_sha256_recorded(self, store):
        import hashlib
        rid = _ingest(store, "checksum me")
        rec = store.get_record(rid)
        assert rec["sha256"] == hashlib.sha256(b"checksum me").hexdigest()


class TestVerbs:
    TEXT = "\n".join(
        [
            "build started",
            "ERROR: missing dep foo",
            "warning: deprecated",
            "line four",
            "ERROR: tests failed in bar.py",
            "tail line",
        ]
    )

    def test_grep(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "grep", pattern="ERROR")
        assert "missing dep foo" in out
        assert "tests failed in bar.py" in out
        assert "build started" not in out

    def test_grep_no_match(self, store):
        rid = _ingest(store, self.TEXT)
        assert "no match" in query(store, rid, "grep", pattern="zzz-not-there").lower()

    def test_errors(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "errors")
        assert "missing dep foo" in out
        assert "warning: deprecated" not in out
        assert "line four" not in out

    def test_head(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "head", n=2)
        assert out.splitlines()[0] == "build started"
        assert len(out.strip().splitlines()) == 2

    def test_tail(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "tail", n=2)
        assert out.strip().splitlines()[-1] == "tail line"

    def test_count(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "count")
        assert "6 lines" in out

    def test_count_with_pattern(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "count", pattern="ERROR")
        assert "2" in out

    def test_lines_range(self, store):
        rid = _ingest(store, self.TEXT)
        out = query(store, rid, "lines", n=(2, 4))
        assert "line four" in out
        assert "build started" not in out

    def test_json_projection(self, store):
        data = {"a": 1, "b": {"c": 2}, "items": [1, 2]}
        rid = _ingest(store, json.dumps(data))
        out = query(store, rid, "json", pattern="b.c")
        assert json.loads(out.split(":", 1)[1].strip()) if False else json.loads(out[out.index(":")+1:].strip()) == 2

    def test_json_invalid_shows_error(self, store):
        rid = _ingest(store, "not json at all")
        out = query(store, rid, "json")
        assert "not valid json" in out.lower()

    def test_summary(self, store):
        text = "\n".join(
            ["ok tests passed"] * 5
            + ["ERROR: one bad thing", "Traceback (most recent call last):"]
        )
        rid = _ingest(store, text)
        out = query(store, rid, "summary")
        assert "7 lines" in out
        assert "one bad thing" in out

    def test_unknown_verb(self, store):
        rid = _ingest(store, "x")
        with pytest.raises(ValueError):
            query(store, rid, "rm-rf")


class TestBudgetCap:
    def test_output_budget_capped_never_respilled(self, store, tmp_path):
        """Design L72: result_query output is budget-capped and never
        re-spilled into the store."""
        big = "\n".join(f"ERROR: line {i} " + "x" * 100 for i in range(500))
        spill_dir = tmp_path / "spillover"
        n_before = len(list(spill_dir.iterdir())) if spill_dir.exists() else 0
        rid = _ingest(store, big)
        assert len(big) > RESULT_QUERY_BUDGET_CHARS
        out = query(store, rid, "errors")
        assert len(out) <= RESULT_QUERY_BUDGET_CHARS + 200  # marker overhead
        # No new spill files written by the query path.
        if spill_dir.exists():
            assert len(list(spill_dir.iterdir())) == n_before
        rec = store.get_record(rid)
        # The store still holds the original full text, untouched.
        assert rec["size"] == len(big)

    def test_compact_caps(self):
        big = "\n".join(f"line {i}" for i in range(10000))
        out, truncated = compact(big, budget_chars=1000)
        assert len(out) <= 1000 + 200
        assert truncated

    def test_under_budget_not_truncated(self):
        out, truncated = compact("short", budget_chars=1000)
        assert out == "short"
        assert not truncated


class TestSessionScoping:
    """C14: id lookups are session-scoped, not store-global.

    "last" and numeric ids resolve within the calling session first; a
    result owned by another session is never returned. Legacy rows with an
    empty session_id (CLI ingests) remain visible as the explicit
    store-global escape; passing session_id=None preserves old behavior.
    """

    def test_last_resolves_within_calling_session(self, store):
        store.ingest(full_text="s1 result", tool="terminal", args_json={}, session_id="s1")
        store.ingest(full_text="s2 result", tool="terminal", args_json={}, session_id="s2")
        assert store.fetch("last", session_id="s1") == "s1 result"
        assert store.fetch("last", session_id="s2") == "s2 result"

    def test_last_missing_session_falls_back_empty(self, store):
        store.ingest(full_text="s1 result", tool="terminal", args_json={}, session_id="s1")
        assert store.fetch("last", session_id="s9") is None

    def test_numeric_id_owned_by_other_session_not_returned(self, store):
        rid_other = store.ingest(full_text="other session", tool="terminal",
                                 args_json={}, session_id="s2")
        assert store.fetch(rid_other, session_id="s1") is None
        assert store.get_record(rid_other, session_id="s1") is None

    def test_numeric_id_same_session_returns(self, store):
        rid = store.ingest(full_text="mine", tool="terminal", args_json={}, session_id="s1")
        assert store.fetch(rid, session_id="s1") == "mine"

    def test_unowned_legacy_row_still_visible(self, store):
        rid = store.ingest(full_text="cli ingest", tool="terminal", args_json={})
        assert store.fetch(rid, session_id="s1") == "cli ingest"

    def test_none_session_id_preserves_global_behavior(self, store):
        store.ingest(full_text="s1 result", tool="terminal", args_json={}, session_id="s1")
        store.ingest(full_text="s2 result", tool="terminal", args_json={}, session_id="s2")
        assert store.fetch("last") == "s2 result"

    def test_query_verb_scoped(self, store):
        store.ingest(full_text="ERROR from s1", tool="terminal", args_json={}, session_id="s1")
        store.ingest(full_text="ERROR from s2", tool="terminal", args_json={}, session_id="s2")
        out = query(store, "last", "errors", session_id="s1")
        assert "s1" in out and "s2" not in out

    def test_tool_handler_uses_session_context(self, tmp_path, monkeypatch):
        import tools.result_query_tool as rqm
        from tools.approval import (
            reset_current_observability_context,
            set_current_observability_context,
        )
        store = ResultStore(str(tmp_path / "r.sqlite"))
        monkeypatch.setattr(rqm, "_store", store)
        store.ingest(full_text="mine\nERROR a", tool="terminal",
                     args_json={}, session_id="sess-a")
        store.ingest(full_text="theirs\nERROR b", tool="terminal",
                     args_json={}, session_id="sess-b")
        tokens = set_current_observability_context(session_id="sess-a")
        try:
            out = rqm.result_query_tool(id="last", verb="grep", pattern="ERROR")
            assert "ERROR a" in out and "ERROR b" not in out
        finally:
            reset_current_observability_context(tokens)


class TestToolHandler:
    """The registered model-tool surface (tools.result_query_tool)."""

    def test_module_imports_and_registers(self):
        import tools.result_query_tool as m
        from tools.registry import registry
        assert registry.get_definitions(["result_query"]), "result_query must be registered"

    def test_handler_grep_last(self, tmp_path, monkeypatch):
        import tools.result_query_tool as m
        store = ResultStore(str(tmp_path / "r.sqlite"))
        monkeypatch.setattr(m, "_store", store)
        store.ingest(full_text="alpha\nERROR beta\ngamma", tool="terminal",
                     args_json={}, session_id="s1")
        out = m.result_query_tool(id="last", verb="grep", pattern="beta")
        assert "ERROR beta" in out

    def test_schema_teaches_verbs(self):
        import tools.result_query_tool as m
        assert set(m.VERBS) == {
            "grep", "errors", "head", "tail", "count", "lines", "json", "summary"
        }
