"""Unit tests for citation grounding in ``rag_generator``.

Covers both halves of the feature:

* ``_extract_cited_timestamps`` — the regex that pulls ``[M:SS]`` / ``(M:SS)``
  references out of a free-form LLM answer.
* ``apply_citation_grounding`` — filter retrieved citations against the
  parsed references, with a configurable tolerance and an honest handling of
  the "hallucinated timestamp" case.
"""

from __future__ import annotations

import pytest

from quadrag.generation.rag_generator import (
    _extract_cited_timestamps,
    apply_citation_grounding,
)
from quadrag.models import IndexType, RetrievalResult


def _mk(timestamp: float, similarity: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        content=f"content@{timestamp}",
        timestamp=timestamp,
        similarity=similarity,
        source=IndexType.AUDIO,
    )


class TestExtractCitedTimestamps:
    def test_empty_answer_returns_empty(self):
        assert _extract_cited_timestamps("") == []
        assert _extract_cited_timestamps(None) == []

    def test_bracketed_M_SS_is_extracted(self):
        assert _extract_cited_timestamps("See [0:12].") == [12.0]

    def test_parenthesised_M_SS_is_extracted(self):
        assert _extract_cited_timestamps("See (1:23).") == [83.0]

    def test_multiple_references_preserved_in_order(self):
        got = _extract_cited_timestamps("First at [0:05], then [2:30], finally [10:00].")
        assert got == [5.0, 150.0, 600.0]

    def test_fractional_seconds_are_dropped_but_base_kept(self):
        # The model sometimes includes a fractional part we don't care about.
        assert _extract_cited_timestamps("At [0:12.5] something happens.") == [12.0]

    def test_unbracketed_timestamps_are_ignored(self):
        # "meet at 3:00" is conversational, not a video reference.
        assert _extract_cited_timestamps("Let's meet at 3:00 tomorrow") == []

    def test_malformed_timestamp_with_seconds_over_60_is_skipped(self):
        # "1:99" is nonsense; drop it rather than coercing.
        assert _extract_cited_timestamps("Watch [1:99] for details") == []

    def test_garbage_inside_brackets_does_not_match(self):
        assert _extract_cited_timestamps("[NaN:??]") == []
        assert _extract_cited_timestamps("[12]") == []

    def test_long_video_minutes_are_supported(self):
        # A 90-minute video mention must still parse.
        assert _extract_cited_timestamps("See [90:15]") == [5415.0]


class TestApplyCitationGrounding:
    def test_no_timestamps_means_ungrounded_returns_original(self):
        retrieved = [_mk(10.0), _mk(20.0)]
        citations, grounded = apply_citation_grounding("Just plain text", retrieved)
        assert grounded is False
        assert citations == retrieved

    def test_matching_timestamp_produces_grounded_true(self):
        retrieved = [_mk(12.0), _mk(30.0), _mk(60.0)]
        citations, grounded = apply_citation_grounding("See [0:12].", retrieved, tolerance_sec=3.0)
        assert grounded is True
        assert len(citations) == 1
        assert citations[0].timestamp == 12.0

    def test_tolerance_window_is_inclusive(self):
        retrieved = [_mk(9.0), _mk(12.0), _mk(15.0)]
        # Cite [0:12] with 3s tolerance → 9-15s inclusive match.
        citations, grounded = apply_citation_grounding("At [0:12].", retrieved, tolerance_sec=3.0)
        assert grounded is True
        assert {c.timestamp for c in citations} == {9.0, 12.0, 15.0}

    def test_outside_tolerance_is_filtered(self):
        retrieved = [_mk(5.0), _mk(12.0), _mk(20.0)]
        # Cite [0:12] with 3s tolerance → only 12.0 matches.
        citations, grounded = apply_citation_grounding("At [0:12]", retrieved, tolerance_sec=3.0)
        assert grounded is True
        assert [c.timestamp for c in citations] == [12.0]

    def test_multiple_cited_timestamps_all_match(self):
        retrieved = [_mk(5.0), _mk(12.0), _mk(60.0), _mk(90.0)]
        citations, grounded = apply_citation_grounding(
            "At [0:12] and [1:00].", retrieved, tolerance_sec=1.0
        )
        assert grounded is True
        assert sorted(c.timestamp for c in citations) == [12.0, 60.0]

    def test_hallucinated_timestamp_with_no_matches_is_ungrounded_with_full_list(self):
        """Answer cites a timestamp but nothing retrieved is near it.

        Returning empty citations with grounded=True would lie — the LLM
        anchored its answer to something we can't corroborate. Be honest:
        grounded=False and hand back the whole retrieved set so the UI can
        show what we actually found.
        """
        retrieved = [_mk(10.0), _mk(20.0)]
        citations, grounded = apply_citation_grounding("See [5:00]", retrieved, tolerance_sec=3.0)
        assert grounded is False
        assert citations == retrieved

    def test_duplicate_citation_is_not_double_counted(self):
        retrieved = [_mk(12.0)]
        citations, _ = apply_citation_grounding("At [0:12] and again at [0:13]", retrieved, tolerance_sec=3.0)
        assert len(citations) == 1  # the one retrieved chunk appears once

    def test_tolerance_defaults_to_settings_value(self):
        # Not passing tolerance_sec → falls back to settings.CITATION_TIMESTAMP_TOLERANCE_SEC (=3.0).
        retrieved = [_mk(10.0)]
        citations, grounded = apply_citation_grounding("At [0:12]", retrieved)  # diff = 2s, inside default 3s
        assert grounded is True
        assert citations == retrieved

    def test_empty_retrieved_returns_empty_and_ungrounded(self):
        citations, grounded = apply_citation_grounding("See [0:12]", [], tolerance_sec=3.0)
        # No citations to ground against → honest ungrounded signal.
        assert grounded is False
        assert citations == []

    def test_returns_independent_list_not_input_reference(self):
        """Callers mutating the returned list must not corrupt the input."""
        retrieved = [_mk(10.0)]
        citations, _ = apply_citation_grounding("plain", retrieved)
        citations.clear()
        assert retrieved == [_mk(10.0)]


class TestDifferentSourceTypes:
    """The filter is source-agnostic — a domain hit at 12s matches [0:12] just like audio does."""

    @pytest.mark.parametrize("source", list(IndexType))
    def test_filter_works_across_index_types(self, source):
        retrieved = [
            RetrievalResult(content="x", timestamp=12.0, similarity=0.5, source=source),
        ]
        citations, grounded = apply_citation_grounding("At [0:12]", retrieved, tolerance_sec=1.0)
        assert grounded is True
        assert len(citations) == 1
