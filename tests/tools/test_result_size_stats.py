"""Tests for per-tool p95-bytes-per-result telemetry in compaction_telemetry.

Design source: tool-output-compaction-design.md L87 — "directly measurable
via per-tool observation-size telemetry (add p95-bytes-per-result to Hermes
performance diagnostics before/after)". "Before" = original_chars at the
compaction point; "after" = compacted_chars that entered context.

Contract:
- ``result_size_stats()`` aggregates the retention JSONL log per tool:
  count, p95 and max of original (before) and compacted (after) bytes.
- Computed lazily on read — nothing runs on the compaction hot path beyond
  the existing O(1) record_persisted registry insert.
- Best-effort: missing/unreadable log yields an empty dict.
- Small samples (< MIN_SAMPLE_FOR_P95 = 2) report max only, no p95 —
  avoids implying a percentile over n=1.
"""

import json

import pytest

from tools import compaction_telemetry as ct


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    ct._reset_for_tests()
    yield
    ct._reset_for_tests()


def _spill_and_replay(tool, sizes):
    """Record spills of *sizes* (original, compacted) for *tool*, then a
    retention re-read for each so they land in the JSONL log."""
    for i, (orig, comp) in enumerate(sizes):
        spill = f"/spill/{tool}-{i}.txt"
        ct.record_persisted(spill, tool=tool, original_chars=orig,
                            compacted_chars=comp, session="s")
        ct.record_retention_event(spill)


class TestResultSizeStats:
    def test_empty_when_no_events(self):
        assert ct.result_size_stats() == {}

    def test_per_tool_before_and_after(self):
        _spill_and_replay("terminal", [(40_000, 800), (20_000, 400)])
        _spill_and_replay("web_extract", [(10_000, 2_000)])
        stats = ct.result_size_stats()
        assert set(stats) == {"terminal", "web_extract"}
        term = stats["terminal"]
        assert term["count"] == 2
        assert term["before"]["max"] == 40_000
        assert term["after"]["max"] == 800
        web = stats["web_extract"]
        assert web["count"] == 1
        # single sample: no percentile claim, max only
        assert "p95" not in web["before"]

    def test_p95_matches_nearest_rank(self):
        # 20 sorted before-samples: nearest-rank p95 = 19th of 20
        sizes = [(i * 1_000, i * 100) for i in range(1, 21)]
        _spill_and_replay("terminal", sizes)
        stats = ct.result_size_stats()["terminal"]
        # before: sorted 1k..20k -> 19th = 19000
        assert stats["before"]["p95"] == 19_000
        # after: sorted 100..2000 -> 19th = 1900
        assert stats["after"]["p95"] == 1_900

    def test_lazy_read_does_not_mutate_log(self):
        _spill_and_replay("terminal", [(5_000, 500)])
        before = ct.get_jsonl_path().read_text(encoding="utf-8")
        ct.result_size_stats()
        after = ct.get_jsonl_path().read_text(encoding="utf-8")
        assert before == after

    def test_corrupt_lines_skipped(self, tmp_path):
        p = ct.get_jsonl_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"tool": "terminal", "original_chars": 100,
                        "compacted_chars": 10}) + "\n"
            + "not json\n", encoding="utf-8")
        stats = ct.result_size_stats()
        assert stats["terminal"]["count"] == 1
