"""Unit tests for ``quadrag.utils.calculate_frame_count``.

Covers every bucket boundary in the adaptive frame-sampling schedule so an
off-by-one in the thresholds fails loudly. Schedule calibrated against the
60 s Railway edge-proxy cap on lazy domain-view builds — see
``calculate_frame_count``'s docstring for the math.
"""

from __future__ import annotations

import pytest

from quadrag.utils import calculate_frame_count


class TestEdgeCases:
    def test_zero_duration_returns_default(self):
        assert calculate_frame_count(0) == 40

    def test_negative_duration_returns_default(self):
        assert calculate_frame_count(-1.0) == 40

    def test_tiny_duration_maps_to_short_bucket(self):
        assert calculate_frame_count(0.5) == 40


class TestBucketBoundaries:
    @pytest.mark.parametrize(
        "duration,expected",
        [
            (1.0, 40),       # just over zero → short bucket
            (299.0, 40),     # just below 5 min boundary
            (300.0, 60),     # 5 min exactly → next bucket
            (1799.0, 60),    # just below 30 min boundary
            (1800.0, 80),    # 30 min exactly → next bucket
            (3599.0, 80),    # just below 1 h boundary
            (3600.0, 100),   # 1 h exactly → next bucket
            (7199.0, 100),   # just below 2 h boundary
        ],
    )
    def test_short_and_mid_length_buckets(self, duration, expected):
        assert calculate_frame_count(duration) == expected


class TestVeryLongVideos:
    """For videos >= 7200s the formula is ``min(150, max(100, int(duration // 90)))``."""

    def test_exactly_two_hours_hits_lower_floor(self):
        # 7200 // 90 == 80 → max(100, 80) == 100 → min(150, 100) == 100.
        assert calculate_frame_count(7200.0) == 100

    def test_three_hours_scales_linearly(self):
        # 10800 // 90 == 120 → max(100, 120) == 120 → min(150, 120) == 120.
        assert calculate_frame_count(10800.0) == 120

    def test_four_hours_near_cap(self):
        # 14400 // 90 == 160 → max(100, 160) == 160 → min(150, 160) == 150.
        assert calculate_frame_count(14400.0) == 150

    def test_long_video_hits_upper_cap_of_150(self):
        # 36000 // 90 == 400 → min(150, 400) == 150.
        assert calculate_frame_count(36000.0) == 150

    def test_never_exceeds_150(self):
        # 24-hour stress test — upper cap is 150.
        assert calculate_frame_count(24 * 3600.0) == 150

    def test_returns_int(self):
        assert isinstance(calculate_frame_count(600.0), int)
        assert isinstance(calculate_frame_count(36000.0), int)
