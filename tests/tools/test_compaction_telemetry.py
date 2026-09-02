"""Tests for tools/compaction_telemetry.py -- retention telemetry for
compacted (spilled) tool outputs that are later re-read from disk.

Design source: tool-output-compaction-design.md (Gap B: "spilled files are
write-only"; this module adds the re-read measurement half).
"""

import json

import pytest

from tools import compaction_telemetry as ct


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and reset module state per test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ct._reset_for_tests()
    yield
    ct._reset_for_tests()


class TestRetentionEvent:
    def test_event_appended_with_required_fields(self):
        spill_path = r"C:\tmp\spill\abc.txt"
        ct.record_persisted(
            spill_path, tool="terminal", original_chars=50_000,
            compacted_chars=1_200, session="sess-1",
        )
        ct.record_retention_event(spill_path)

        jsonl = ct.get_jsonl_path()
        assert jsonl.exists()
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["tool"] == "terminal"
        assert event["original_chars"] == 50_000
        assert event["compacted_chars"] == 1_200
        assert event["session"] == "sess-1"
        assert isinstance(event["age_s"], (int, float))
        assert event["age_s"] >= 0
        assert "ts" in event

    def test_unknown_path_is_noop(self, tmp_path):
        ct.record_retention_event(str(tmp_path / "never-spilled.txt"))
        assert not ct.get_jsonl_path().exists()

    def test_never_raises(self, monkeypatch):
        # Corrupt internal registry; append path must still not raise.
        monkeypatch.setattr(ct, "_registry", None)
        ct.record_retention_event("whatever")  # must not raise
        ct.record_persisted("p", tool="t", original_chars=1,
                            compacted_chars=1, session="s")  # must not raise

    def test_registry_capped(self):
        for i in range(ct.MAX_REGISTRY_ENTRIES + 100):
            ct.record_persisted(
                f"/spill/{i}.txt", tool="terminal", original_chars=100,
                compacted_chars=10, session="s",
            )
        assert len(ct._registry) <= ct.MAX_REGISTRY_ENTRIES


class TestCompactionStats:
    def test_empty_when_no_events(self):
        stats = ct.compaction_stats()
        assert stats["retention_events"] == 0

    def test_totals_across_events(self):
        for i, (orig, comp) in enumerate([(40_000, 800), (20_000, 400)]):
            spill = f"/spill/{i}.txt"
            ct.record_persisted(spill, tool="terminal", original_chars=orig,
                                compacted_chars=comp, session="s")
            ct.record_retention_event(spill)
        stats = ct.compaction_stats()
        assert stats["retention_events"] == 2
        assert stats["total_original_chars"] == 60_000
        assert stats["total_compacted_chars"] == 1_200
        # Aggregate compression ratio: 1200/60000 = 0.02
        assert stats["avg_compaction_ratio"] == pytest.approx(0.02)


class TestSetCurrentSession:
    def test_session_threaded_from_context(self):
        ct.set_current_session("sess-9")
        ct.record_persisted("/spill/x.txt", tool="read_file",
                            original_chars=500, compacted_chars=100,
                            session=None)
        ct.record_retention_event("/spill/x.txt")
        event = json.loads(
            ct.get_jsonl_path().read_text(encoding="utf-8").strip()
        )
        assert event["session"] == "sess-9"
