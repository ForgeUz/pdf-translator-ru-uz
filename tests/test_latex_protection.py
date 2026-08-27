"""Tests for Phase 5: LaTeX math block protection.

Tests cover:
- protect_latex_blocks: inline $...$ and display $$...$$ math
- restore_latex_blocks: roundtrip
- No-op for plain text without math
"""
from __future__ import annotations

from pdf_translator_ru_uz.engine import protect_latex_blocks, restore_latex_blocks


def test_protect_inline_math():
    text = "Solve $x^2 + y^2 = z^2$ for x."
    protected, mapping = protect_latex_blocks(text)
    assert "$" not in protected  # all math $...$ replaced
    assert len(mapping) == 1
    # The placeholder is a private-use char
    ph = list(mapping.keys())[0]
    assert ord(ph) >= 0xF000
    assert mapping[ph] == "$x^2 + y^2 = z^2$"


def test_protect_display_math():
    text = "Equation: $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$"
    protected, mapping = protect_latex_blocks(text)
    assert "$$" not in protected
    assert len(mapping) == 1


def test_restore_latex_blocks():
    text = "Solve $x^2 = 4$ for x."
    protected, mapping = protect_latex_blocks(text)
    restored = restore_latex_blocks(protected, mapping)
    assert restored == text


def test_restore_latex_blocks_empty_mapping():
    text = "Plain text without math."
    restored = restore_latex_blocks(text, {})
    assert restored == text


def test_protect_multiple_math_blocks():
    text = "$a^2$ and $b^2$ and $$\\sum_{i=1}^n i$$"
    protected, mapping = protect_latex_blocks(text)
    assert len(mapping) == 3
    restored = restore_latex_blocks(protected, mapping)
    assert restored == text


def test_protect_no_math_is_noop():
    text = "Tenglamani yeching."
    protected, mapping = protect_latex_blocks(text)
    assert protected == text
    assert mapping == {}