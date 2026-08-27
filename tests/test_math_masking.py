"""Tests for Phase 2: Math masking & restoration.

Tests cover:
- mask_math() / unmask_math() roundtrip
- No-op when no math spans
- Tokenizer placeholder survival (using a mock tokenizer)
- Builder _subtract_bboxes carves out correct regions
- MathSpan dataclass creation
"""
from __future__ import annotations

import fitz

from pdf_translator_ru_uz.builder import (
    InPlaceBuilder,
    _subtract_bboxes,
)
from pdf_translator_ru_uz.engine import (
    check_placeholder_survival,
    mask_math,
    unmask_math,
)
from pdf_translator_ru_uz.parser import MathSpan
from pdf_translator_ru_uz.placeholders import PlaceholderRegistry


# ── MathSpan dataclass ──────────────────────────────────────────────


def test_mathspan_creation():
    bbox = fitz.Rect(100, 100, 120, 115)
    span = MathSpan(placeholder="\uE000", original_text="x²", bbox=bbox)
    assert span.placeholder == "\uE000"
    assert span.original_text == "x²"
    assert span.bbox == bbox


def test_next_placeholder_returns_unique_chars():
    registry = PlaceholderRegistry()
    p1 = registry.mask_math()
    p2 = registry.mask_math()
    assert p1 != p2
    assert ord(p1) >= 0xE000
    assert ord(p2) >= 0xE000
    assert ord(p1) <= 0xE7FF
    assert ord(p2) <= 0xE7FF


# ── mask_math / unmask_math ─────────────────────────────────────────


def test_mask_math_no_spans_is_noop():
    text = "Tenglamani yeching."
    masked, mapping = mask_math(text, [])
    assert masked == text
    assert mapping == {}


def test_mask_math_replaces_placeholder_in_text():
    """MathSpan with a placeholder that appears in the text gets masked."""
    span = MathSpan(
        placeholder="\uE000",
        original_text="",
        bbox=fitz.Rect(0, 0, 10, 10),
    )
    text = "Tenglamani yeching: \uE000 = 0"
    masked, mapping = mask_math(text, [span])
    assert "\uE000" not in masked  # placeholder replaced with ▨
    assert "▨" in masked
    assert "\uE000" in mapping


def test_unmask_math_restores_placeholders():
    """When original_text is empty, the ▨ marker is removed after unmask."""
    span = MathSpan(
        placeholder="\uE000",
        original_text="",
        bbox=fitz.Rect(0, 0, 10, 10),
    )
    text = "Tenglamani yeching: \uE000 = 0"
    masked, mapping = mask_math(text, [span])
    # Simulate translation (▨ passes through NLLB unchanged)
    translated = masked.replace(
        "Tenglamani yeching", "Решить уравнение"
    )
    restored = unmask_math(translated, mapping)
    assert "Решить уравнение" in restored
    # ▨ is removed because original_text is empty (unreadable math glyph)
    assert "▨" not in restored


def test_mask_unmask_roundtrip():
    """Roundtrip preserves the placeholder when original_text is empty."""
    span = MathSpan(
        placeholder="\uE000",
        original_text="",
        bbox=fitz.Rect(0, 0, 10, 10),
    )
    text = "Tenglamani yeching: \uE000 = 0"
    masked, mapping = mask_math(text, [span])
    restored = unmask_math(masked, mapping)
    # Full roundtrip restores the placeholder back
    assert restored == text


def test_mask_math_multiple_spans():
    spans = [
        MathSpan(
            placeholder="\uE000",
            original_text="",
            bbox=fitz.Rect(0, 0, 10, 10),
        ),
        MathSpan(
            placeholder="\uE001",
            original_text="",
            bbox=fitz.Rect(20, 0, 30, 10),
        ),
    ]
    text = "Tenglamani yeching: \uE000 \uE001 = 0"
    masked, mapping = mask_math(text, spans)
    assert "\uE000" not in masked
    assert "\uE001" not in masked
    assert masked.count("▨") == 2
    assert len(mapping) == 2
    # Restore all placeholders
    restored = unmask_math(masked, mapping)
    assert "▨" not in restored  # both removed since original_text is empty


# ── check_placeholder_survival ──────────────────────────────────────


class MockTokenizer:
    """A mock tokenizer that preserves ▨ through encode/decode."""

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(
            chr(t) if t < 0x110000 else "?" for t in tokens
        )


class BrokenTokenizer:
    """A mock tokenizer that drops ▨ during decode."""

    def encode(self, text: str):
        return [1, 2, 3]  # arbitrary tokens

    def decode(self, tokens: list[int]) -> str:
        return "garbled output without the marker"


def test_check_placeholder_survival_passes():
    tokenizer = MockTokenizer()
    assert check_placeholder_survival(tokenizer, "▨") is True


def test_check_placeholder_survival_fails():
    tokenizer = BrokenTokenizer()
    assert check_placeholder_survival(tokenizer, "▨") is False


# ── _subtract_bboxes (module-level) ─────────────────────────────────


def test_subtract_bboxes_no_exclude_returns_outer():
    outer = fitz.Rect(0, 0, 100, 100)
    result = _subtract_bboxes(outer, [])
    assert result == [outer]


def test_subtract_bboxes_carves_top_strip():
    outer = fitz.Rect(0, 0, 100, 100)
    exclude = [fitz.Rect(10, 40, 90, 60)]
    result = _subtract_bboxes(outer, exclude)
    # Should have top strip above exclude
    assert any(r.y0 == 0 and r.y1 == 40 for r in result)
    # Should NOT have anything covering the excluded rect
    for r in result:
        overlap = r.intersect(exclude[0])
        assert overlap.width <= 0 or overlap.height <= 0


def test_subtract_bboxes_multiple_excludes():
    outer = fitz.Rect(0, 0, 200, 200)
    exclude = [
        fitz.Rect(10, 30, 80, 50),
        fitz.Rect(10, 80, 80, 100),
    ]
    result = _subtract_bboxes(outer, exclude)
    assert len(result) >= 3


# ── Builder replace_mixed_paragraph → replace_paragraph with exclude_bboxes ──


def test_replace_paragraph_with_excludes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Text before math", fontsize=12)
    page.insert_text((72, 130), "math glyph", fontsize=12)
    page.insert_text((72, 160), "Text after math", fontsize=12)
    builder = InPlaceBuilder()

    para_bbox = fitz.Rect(72, 90, 300, 180)
    math_bbox = fitz.Rect(72, 125, 180, 142)

    result = builder.replace_paragraph(
        page=page,
        bbox=para_bbox,
        translated_text="Translated text",
        original_font="DejaVuSans",
        original_fontsize=12,
        original_color=(0.0, 0.0, 0.0),
        exclude_bboxes=[math_bbox],
    )
    assert result.fit is True or result.bled is False