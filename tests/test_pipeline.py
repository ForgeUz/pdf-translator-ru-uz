"""End-to-end pipeline tests: parse → translate → redraw cycle + cache reuse.

These tests inject a fake translate_fn so they never load a real NLLB model.
No torch/transformers import needed.
"""
from __future__ import annotations

import fitz

from pdf_translator_ru_uz.cache import TranslationCache
from pdf_translator_ru_uz.engine import NLLBEngine, CachedEngine
from pdf_translator_ru_uz.pipeline import run_pipeline


def _make_test_pdf(path: str) -> None:
    """Create a minimal 2-page PDF with known text for testing."""
    doc = fitz.open()
    # Page 1
    page = doc.new_page()
    page.insert_text((72, 100), "Birinchi sahifadagi matn.", fontsize=11)
    page.insert_text((72, 150), "Ikkinchi paragraf.", fontsize=11)
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((72, 100), "Ikkinchi sahifadagi matn.", fontsize=11)
    doc.save(path)
    doc.close()


def _fake_translate(chunk: str, src: str, tgt: str) -> str:
    """Deterministic fake translation: wrap text with target language marker."""
    return f"[{tgt}]{chunk}"


def _word_count(text: str) -> int:
    return len(text.split())


def test_end_to_end_pipeline_parse_translate_redraw(tmp_path):
    """Verify the full pipeline: PDF → parse → translate → redact → redraw → PDF.

    The output PDF should:
    - Contain the translated text (not the original)
    - Have the same page count as the input
    - Not crash on any paragraph
    """
    input_pdf = str(tmp_path / "input.pdf")
    output_pdf = str(tmp_path / "output.pdf")
    _make_test_pdf(input_pdf)

    engine = NLLBEngine(
        translate_fn=_fake_translate,
        token_len_fn=_word_count,
        max_tokens=512,
    )

    result = run_pipeline(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        target_lang="ru",
        src_lang="uz",
        engine=engine,
        cache_db=str(tmp_path / "cache.db"),
    )

    assert result.exists()
    assert result.name == "output.pdf"

    # Verify output PDF contents
    out_doc = fitz.open(str(result))
    assert len(out_doc) == 2  # same page count as input

    # Verify output PDF contains the translated text
    full_text = ""
    for page in out_doc:
        full_text += page.get_text("text")

    # The pipeline applies restore_case() which may title-case the output.
    # Check that translation markers appear (proving text was "translated" and redrawn).
    # After restore_case with title-case source, "[rus_Cyrl]" becomes "[Rus_Cyrl]".
    assert "rus_Cyrl" in full_text or "Rus_Cyrl" in full_text
    assert "Birinchi" in full_text or "sahifadagi" in full_text

    out_doc.close()


def test_cache_reuse_across_pipeline_runs(tmp_path):
    """Two pipeline runs with the same input should reuse cached translations.

    The fake_translate function increments a counter on each call.
    Second run should have zero calls (all cache hits).
    """
    input_pdf = str(tmp_path / "input.pdf")
    output_pdf = str(tmp_path / "output.pdf")
    _make_test_pdf(input_pdf)

    call_counter = {"count": 0}

    def counting_translate(chunk: str, src: str, tgt: str) -> str:
        call_counter["count"] += 1
        return f"[{tgt}]{chunk}"

    engine = NLLBEngine(
        translate_fn=counting_translate,
        token_len_fn=_word_count,
        max_tokens=512,
    )

    cache_db = str(tmp_path / "cache.db")

    # First run — should call translate for each paragraph
    run_pipeline(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        target_lang="ru",
        src_lang="uz",
        engine=engine,
        cache_db=cache_db,
    )
    first_run_calls = call_counter["count"]
    assert first_run_calls > 0, "First run should call translate at least once"

    # Second run with same input — should hit cache
    call_counter["count"] = 0
    output_pdf2 = str(tmp_path / "output2.pdf")
    run_pipeline(
        input_pdf=input_pdf,
        output_pdf=output_pdf2,
        target_lang="ru",
        src_lang="uz",
        engine=engine,
        cache_db=cache_db,
    )
    assert call_counter["count"] == 0, "Second run should have zero translate calls (all cached)"