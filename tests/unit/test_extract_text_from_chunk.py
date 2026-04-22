"""Unit tests for the transcript-text extractor used by the audio index."""

from __future__ import annotations

from quadrag.video.functions import _extract_text_from_chunk


def test_extracts_text_from_whisper_dict():
    # Whisper returns a rich dict; we only want the ``text`` field.
    whisper_output = {
        "text": "hello world",
        "segments": [{"start": 0.0, "end": 1.5, "text": "hello world"}],
        "language": "en",
    }
    assert _extract_text_from_chunk(whisper_output) == "hello world"


def test_missing_text_key_returns_empty_string():
    assert _extract_text_from_chunk({"segments": [], "language": "en"}) == ""


def test_empty_dict_returns_empty_string():
    assert _extract_text_from_chunk({}) == ""


def test_none_returns_empty_string():
    # None is coerced to empty, not the literal string "None".
    assert _extract_text_from_chunk(None) == ""


def test_plain_string_is_returned_verbatim():
    assert _extract_text_from_chunk("already plain text") == "already plain text"


def test_numeric_input_is_stringified():
    # Unusual but should not crash.
    assert _extract_text_from_chunk(42) == "42"


def test_non_string_text_field_is_stringified():
    # Whisper always returns str, but if a mock returns something else, don't crash.
    assert _extract_text_from_chunk({"text": 123}) == "123"
