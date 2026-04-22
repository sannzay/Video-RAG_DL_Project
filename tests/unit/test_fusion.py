"""Unit tests for ``quadrag.retrieval.fusion.ResultFusion``."""

from __future__ import annotations

import pytest

from quadrag.config import get_settings
from quadrag.models import IndexType, RetrievalResult
from quadrag.retrieval.fusion import ResultFusion


def _mk(source: IndexType, timestamp: float, similarity: float, content: str = "") -> RetrievalResult:
    return RetrievalResult(
        content=content or f"{source.value}@{timestamp}",
        timestamp=timestamp,
        similarity=similarity,
        source=source,
    )


class TestNormalizeScores:
    def test_empty_input_returns_empty(self):
        fusion = ResultFusion()
        assert fusion.normalize_scores({}) == {}

    def test_no_results_across_indexes_returns_unchanged(self):
        fusion = ResultFusion()
        payload = {IndexType.AUDIO: [], IndexType.IMAGE: []}
        assert fusion.normalize_scores(payload) == payload

    def test_all_equal_scores_returns_input_unchanged(self):
        # score_range == 0 triggers the passthrough branch.
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [_mk(IndexType.AUDIO, 1.0, 0.5)],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 2.0, 0.5)],
        }
        out = fusion.normalize_scores(payload)
        assert out is payload

    def test_min_max_normalization_across_indexes(self):
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [
                _mk(IndexType.AUDIO, 0.0, 0.1),
                _mk(IndexType.AUDIO, 1.0, 0.9),
            ],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 2.0, 0.5)],
        }
        out = fusion.normalize_scores(payload)
        # Global min=0.1, max=0.9 → 0.1→0, 0.5→0.5, 0.9→1.
        assert out[IndexType.AUDIO][0].similarity == pytest.approx(0.0)
        assert out[IndexType.AUDIO][1].similarity == pytest.approx(1.0)
        assert out[IndexType.IMAGE][0].similarity == pytest.approx(0.5)

    def test_normalize_preserves_content_and_timestamp(self):
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [_mk(IndexType.AUDIO, 12.5, 0.2, content="hello")],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 30.0, 0.8, content="world")],
        }
        out = fusion.normalize_scores(payload)
        assert out[IndexType.AUDIO][0].content == "hello"
        assert out[IndexType.AUDIO][0].timestamp == 12.5
        assert out[IndexType.IMAGE][0].content == "world"
        assert out[IndexType.IMAGE][0].timestamp == 30.0


class TestApplyWeights:
    def test_each_index_applies_its_configured_weight(self):
        fusion = ResultFusion()
        settings = get_settings()
        payload = {
            IndexType.AUDIO: [_mk(IndexType.AUDIO, 0.0, 1.0)],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 0.0, 1.0)],
            IndexType.DESCRIPTION: [_mk(IndexType.DESCRIPTION, 0.0, 1.0)],
            IndexType.DOMAIN: [_mk(IndexType.DOMAIN, 0.0, 1.0)],
        }
        out = fusion.apply_weights(payload)

        by_src = {r.source: r.similarity for r in out}
        assert by_src[IndexType.AUDIO] == pytest.approx(settings.WEIGHT_AUDIO)
        assert by_src[IndexType.IMAGE] == pytest.approx(settings.WEIGHT_IMAGE)
        assert by_src[IndexType.DESCRIPTION] == pytest.approx(settings.WEIGHT_DESCRIPTION)
        assert by_src[IndexType.DOMAIN] == pytest.approx(settings.WEIGHT_DOMAIN)

    def test_apply_weights_flattens_output(self):
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [
                _mk(IndexType.AUDIO, 0.0, 1.0),
                _mk(IndexType.AUDIO, 1.0, 0.5),
            ],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 2.0, 0.8)],
        }
        out = fusion.apply_weights(payload)
        assert len(out) == 3


class TestDeduplicateByTimestamp:
    def test_empty_returns_empty(self):
        assert ResultFusion().deduplicate_by_timestamp([]) == []

    def test_close_timestamps_keep_highest_score(self):
        fusion = ResultFusion()
        results = [
            _mk(IndexType.AUDIO, 10.0, 0.3),
            _mk(IndexType.AUDIO, 10.5, 0.9),  # within default 2.0s window → drop the weaker
            _mk(IndexType.AUDIO, 20.0, 0.5),
        ]
        out = fusion.deduplicate_by_timestamp(results)
        scores = sorted(r.similarity for r in out)
        assert scores == [0.5, 0.9]

    def test_explicit_time_window_override(self):
        fusion = ResultFusion()
        results = [
            _mk(IndexType.AUDIO, 10.0, 0.9),
            _mk(IndexType.AUDIO, 10.5, 0.3),
        ]
        # Narrow window keeps both.
        out = fusion.deduplicate_by_timestamp(results, time_window=0.1)
        assert len(out) == 2
        # Wide window collapses to one.
        out = fusion.deduplicate_by_timestamp(results, time_window=5.0)
        assert len(out) == 1
        assert out[0].similarity == pytest.approx(0.9)

    def test_far_apart_timestamps_all_kept(self):
        fusion = ResultFusion()
        results = [
            _mk(IndexType.AUDIO, 0.0, 0.5),
            _mk(IndexType.AUDIO, 10.0, 0.5),
            _mk(IndexType.AUDIO, 20.0, 0.5),
        ]
        out = fusion.deduplicate_by_timestamp(results)
        assert len(out) == 3


class TestFuseResults:
    def test_end_to_end_returns_sorted_top_k(self):
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [
                _mk(IndexType.AUDIO, 0.0, 0.9),
                _mk(IndexType.AUDIO, 30.0, 0.1),
            ],
            IndexType.IMAGE: [_mk(IndexType.IMAGE, 60.0, 0.5)],
            IndexType.DESCRIPTION: [_mk(IndexType.DESCRIPTION, 90.0, 0.7)],
            IndexType.DOMAIN: [_mk(IndexType.DOMAIN, 120.0, 0.3)],
        }
        out = fusion.fuse_results(payload, top_k=3, deduplicate=False)
        assert len(out) == 3
        # Sorted descending by weighted similarity.
        assert out == sorted(out, key=lambda r: r.similarity, reverse=True)

    def test_default_top_k_from_settings(self):
        fusion = ResultFusion()
        settings = get_settings()
        payload = {
            IndexType.AUDIO: [_mk(IndexType.AUDIO, float(i), 0.1 * i) for i in range(1, 21)],
        }
        out = fusion.fuse_results(payload, deduplicate=False)
        assert len(out) == settings.FUSION_TOP_K

    def test_deduplicate_flag_is_respected(self):
        fusion = ResultFusion()
        payload = {
            IndexType.AUDIO: [
                _mk(IndexType.AUDIO, 10.0, 0.9),
                _mk(IndexType.AUDIO, 10.5, 0.8),  # inside default 2s window
            ],
        }
        assert len(fusion.fuse_results(payload, deduplicate=False, top_k=10)) == 2
        assert len(fusion.fuse_results(payload, deduplicate=True, top_k=10)) == 1
