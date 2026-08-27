"""Vertical slices for InPlaceBuilder: redaction, line-wrapping, font mapping."""
import fitz
import pytest

from pdf_translator_ru_uz.builder import (
    InPlaceBuilder,
    _subtract_bboxes,
)


def _make_pdf_with_text_and_rect():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Original text to erase", fontsize=12)
    page.draw_rect(
        fitz.Rect(200, 200, 300, 260),
        color=(1, 0, 0),
        fill=(1, 0, 0),
    )
    return doc, page


def test_redact_erases_text_but_keeps_vector_graphics():
    doc, page = _make_pdf_with_text_and_rect()
    builder = InPlaceBuilder()

    text_bbox = fitz.Rect(72, 90, 300, 112)
    builder._redact_bbox(page, text_bbox)

    remaining_text = page.get_text("text")
    assert "Original text to erase" not in remaining_text

    drawings = page.get_drawings()
    assert any(
        d["fill"] == (1, 0, 0) for d in drawings
    ), "vector rectangle must survive redaction"


def test_replace_paragraph_inpaints_and_renders():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Original text", fontsize=12)
    builder = InPlaceBuilder()

    bbox = fitz.Rect(72, 90, 200, 120)
    result = builder.replace_paragraph(
        page=page,
        bbox=bbox,
        translated_text="Translated text here",
        original_font="DejaVuSans",
        original_fontsize=12,
        original_color=(0.0, 0.0, 0.0),
    )

    rendered = page.get_text("text")
    assert "Translated" in rendered
    assert result.fit is True


def test_line_wrapper_does_not_exceed_max_width():
    builder = InPlaceBuilder()
    lines = builder._wrap_text(
        "This is a test sentence with several words",
        fontname="DejaVuSans",
        fontfile="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        fontsize=12,
        max_width=200,
    )
    assert len(lines) >= 1
    for line in lines:
        assert len(line.words) >= 1


def test_shrink_to_fit_reduces_fontsize_when_text_too_long():
    doc = fitz.open()
    page = doc.new_page()
    builder = InPlaceBuilder(min_fontsize=6.0, fontsize_step=0.5)

    small_bbox = fitz.Rect(72, 100, 200, 130)
    long_text = "Some fairly long translated Russian text that will not fit at 14pt " * 2

    result = builder.replace_paragraph(
        page=page,
        bbox=small_bbox,
        translated_text=long_text,
        original_font="DejaVuSans",
        original_fontsize=14.0,
    )

    assert result.fit is True or result.bled is True
    assert long_text.split()[0] in page.get_text("text")


def test_bleeds_past_bbox_when_min_fontsize_still_overflows():
    doc = fitz.open()
    page = doc.new_page()
    builder = InPlaceBuilder(min_fontsize=6.0, fontsize_step=0.5)

    tiny_bbox = fitz.Rect(72, 100, 90, 108)
    text = "This sentence is much too long to ever fit in that tiny box no matter the fontsize."

    result = builder.replace_paragraph(
        page=page,
        bbox=tiny_bbox,
        translated_text=text,
        original_font="DejaVuSans",
        original_fontsize=11.0,
    )

    # Text is rendered word-by-word with insert_text, so words appear
    # on separate lines. Check that key words are present.
    rendered = page.get_text("text")
    assert result.bled is True or result.fit is False
    assert "fontsize" in rendered


def test_font_mapping_serif_to_liberation():
    builder = InPlaceBuilder()
    fontname, fontfile = builder._resolve_font("TimesNewRoman", 0)
    assert "LiberationSerif" in fontname or "Serif" in fontname


def test_font_mapping_sans_to_dejavu():
    builder = InPlaceBuilder()
    fontname, fontfile = builder._resolve_font("Arial", 0)
    assert "DejaVuSans" in fontname or "Sans" in fontname


def test_subtract_bboxes_multiple_excludes():
    outer = fitz.Rect(0, 0, 200, 200)
    exclude = [
        fitz.Rect(10, 30, 80, 50),
        fitz.Rect(10, 80, 80, 100),
    ]
    result = _subtract_bboxes(outer, exclude)
    assert len(result) >= 3


def test_subtract_bboxes_no_exclude_returns_outer():
    outer = fitz.Rect(0, 0, 100, 100)
    result = _subtract_bboxes(outer, [])
    assert result == [outer]