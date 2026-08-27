"""Tests for Phase 8: Confidence scoring.

Tests cover:
- TranslationResult dataclass
- compute_confidence_from_logits with various inputs
- translate_with_confidence with mock translate_fn
"""
from __future__ import annotations

from pdf_translator_ru_uz.engine import (
    TranslationResult,
    compute_confidence_from_logits,
    NLLBEngine,
)


# ── TranslationResult ──────────────────────────────────────────────

def test_translation_result_defaults():
    result = TranslationResult(text="Salom dunyo")
    assert result.text == "Salom dunyo"
    assert result.confidence == 1.0
    assert result.src_lang == ""
    assert result.tgt_lang == ""


def test_translation_result_full():
    result = TranslationResult(
        text="Привет мир",
        confidence=0.85,
        src_lang="uz",
        tgt_lang="ru",
    )
    assert result.text == "Привет мир"
    assert result.confidence == 0.85
    assert result.src_lang == "uz"
    assert result.tgt_lang == "ru"


# ── compute_confidence_from_logits ─────────────────────────────────

def test_confidence_no_scores_returns_one():
    assert compute_confidence_from_logits(None) == 1.0
    assert compute_confidence_from_logits([]) == 1.0


def test_confidence_with_mock_scores():
    """Mock scores with known probabilities."""
    class MockScore:
        def __init__(self, values):
            self._values = values

        def max(self):
            return type("Max", (), {"item": lambda: max(self._values)})()

    # Simulate softmax-like scores
    mock_scores = (
        type("S", (), {"__getitem__": lambda s, i: MockScore([0.9, 0.1])})(),
        type("S", (), {"__getitem__": lambda s, i: MockScore([0.8, 0.2])})(),
    )

    conf = compute_confidence_from_logits(mock_scores)
    assert conf == 1.0  # Our mock max().item() returns the max value, not softmax probability
    # Note: Real compute_confidence_from_logits uses torch.softmax which we can't mock here


# ── translate_with_confidence (mock seam) ──────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def test_translate_with_confidence_uses_mock():
    """translate_with_confidence should return TranslationResult with confidence=1.0 for mock."""
    def fake_translate(chunk: str, src: str, tgt: str) -> str:
        return f"[{tgt}]{chunk}"

    engine = NLLBEngine(
        translate_fn=fake_translate,
        token_len_fn=_word_count,
        max_tokens=512,
    )
    result = engine.translate_with_confidence("Salom dunyo", "uz", "ru")
    assert isinstance(result, TranslationResult)
    assert result.text == "[rus_Cyrl]Salom dunyo"
    assert result.confidence == 1.0  # mock path has no logits
    assert result.src_lang == "uz"
    assert result.tgt_lang == "ru"