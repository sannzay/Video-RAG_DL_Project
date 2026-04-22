"""Unit tests for ``quadrag.utils.calculate_frame_count``.

Covers every bucket boundary in the adaptive frame-sampling schedule so an
off-by-one in the thresholds fails loudly. Schedule was tightened after the
initial prod deploy to stay under OpenAI Tier-1's 200 k TPM limit when both
description and domain indexes run on the same video.
"""

from __future__ import annotations

import pytest

from quadrag.utils import calculate_frame_count


class TestEdgeCases:
    def test_zero_duration_returns_default(self):
        assert calculate_frame_count(0) == 20

    def test_negative_duration_returns_default(self):
        assert calculate_frame_count(-1.0) == 20

    def test_tiny_duration_maps_to_short_bucket(self):
        assert calculate_frame_count(0.5) == 20


class TestBucketBoundaries:
    @pytest.mark.parametrize(
        "duration,expected",
        [
            (1.0, 20),      # just over zero → short bucket
            (299.0, 20),    # just below 5 min boundary
            (300.0, 30),    # 5 min exactly → next bucket
            (1799.0, 30),   # just below 30 min boundary
            (1800.0, 45),   # 30 min exactly → next bucket
            (3599.0, 45),   # just below 1 h boundary
            (3600.0, 60),   # 1 h exactly → next bucket
            (7199.0, 60),   # just below 2 h boundary
        ],
    )
    def test_short_and_mid_length_buckets(self, duration, expected):
        assert calculate_frame_count(duration) == expected


class TestVeryLongVideos:
    """For videos >= 7200s the formula is ``min(90, max(60, int(duration // 120)))``."""

    def test_exactly_two_hours_uses_long_bucket_formula(self):
        # 7200 // 120 == 60 → max(60, 60) == 60 → min(90, 60) == 60.
        assert calculate_frame_count(7200.0) == 60

    def test_three_hours(self):
        # 10800 // 120 == 90 → max(60, 90) == 90 → min(90, 90) == 90.
        assert calculate_frame_count(10800.0) == 90

    def test_long_video_hits_upper_cap_of_90(self):
        # 36000 // 120 == 300 → min(90, 300) == 90.
        assert calculate_frame_count(36000.0) == 90

    def test_never_exceeds_90(self):
        # 24-hour stress test — upper cap is 90.
        assert calculate_frame_count(24 * 3600.0) == 90

    def test_returns_int(self):
        assert isinstance(calculate_frame_count(600.0), int)
        assert isinstance(calculate_frame_count(36000.0), int)
