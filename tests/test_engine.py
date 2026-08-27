import pytest

from pdf_translator_ru_uz.cache import TranslationCache
from pdf_translator_ru_uz.engine import (
    BatchingEngine,
    CachedEngine,
    EngineError,
    ModelLoadError,
    NLLBEngine,
    TokenAwareChunker,
    apply_russian_typography,
    restore_case,
)


def _word_count(text: str) -> int:
    return len(text.split())


def test_chunker_never_exceeds_max_tokens():
    text = " ".join(f"word{i}" for i in range(50))  # 50 "tokens" by word count
    chunker = TokenAwareChunker(token_len_fn=_word_count, max_tokens=10)
    chunks = chunker.split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _word_count(chunk) <= 10


def test_chunker_keeps_whole_text_when_under_limit():
    text = "short paragraph under the limit"
    chunker = TokenAwareChunker(token_len_fn=_word_count, max_tokens=512)
    assert chunker.split(text) == [text]


def test_nllb_engine_uses_injected_translate_fn_and_respects_chunking():
    calls = []

    def fake_translate(chunk: str, src: str, tgt: str) -> str:
        calls.append(chunk)
        return f"[{tgt}]{chunk}"

    engine = NLLBEngine(
        translate_fn=fake_translate,
        token_len_fn=_word_count,
        max_tokens=5,
    )

    # Use sentence punctuation with uppercase (required by sentence splitter)
    text = "W0 w1. W2 w3. W4 w5. W6 w7. W8 w9. W10 w11."
    result = engine.translate(text, "uz", "ru")

    assert len(calls) >= 2  # multiple sentences → multiple chunks
    assert result.startswith("[rus_Cyrl]") or result.startswith("[Rus_Cyrl]")


def test_batching_engine_queues_and_flushes_in_batches():
    """BatchingEngine with batch_size=3 should group 9 items into 3 flushes."""
    call_count = {"count": 0}
    batch_call_args = []

    def batch_translate(texts: list[str], src: str, tgt: str) -> list[str]:
        call_count["count"] += 1
        batch_call_args.append(len(texts))
        return [f"[{tgt}]{t}" for t in texts]

    engine = NLLBEngine(
        translate_fn=lambda c, s, t: f"[{t}]{c}",
        token_len_fn=lambda t: len(t.split()),
        max_tokens=512,
    )
    batcher = BatchingEngine(engine, batch_translate_fn=batch_translate, batch_size=3)

    texts = [f"text{i}" for i in range(9)]
    for t in texts:
        batcher.queue(t, "uz", "ru")
    batcher.flush()

    assert call_count["count"] == 3  # 9 items / 3 per batch = 3 flushes
    assert batch_call_args == [3, 3, 3]
    assert len(batcher.results) == 9


def test_batching_engine_serial_fallback():
    """Without batch_translate_fn, BatchingEngine falls back to serial engine.translate()."""
    call_log = []

    def fake_translate(chunk: str, src: str, tgt: str) -> str:
        call_log.append(chunk)
        return f"[{tgt}]{chunk}"

    engine = NLLBEngine(
        translate_fn=fake_translate,
        token_len_fn=lambda t: len(t.split()),
        max_tokens=512,
    )
    batcher = BatchingEngine(engine, batch_size=4)

    for i in range(5):
        batcher.queue(f"text{i}", "uz", "ru")
    batcher.flush()

    assert len(call_log) == 5
    assert len(batcher.results) == 5
    # restore_case("text0", "[rus_Cyrl]text0") → "[rus_Cyrl]text0" (lowercase t preserved)
    expected = [
        "[rus_Cyrl]text0",
        "[rus_Cyrl]text1",
        "[rus_Cyrl]text2",
        "[rus_Cyrl]text3",
        "[rus_Cyrl]text4",
    ]
    assert batcher.results == expected


def test_nllb_engine_backend_switching():
    """Verify backend='ctranslate2' sets the backend attribute correctly."""
    engine = NLLBEngine(
        backend="ctranslate2",
        model_name="facebook/nllb-200-distilled-600M",
    )
    assert engine.backend == "ctranslate2"
    # CTranslate2 path requires model_path; without it, translate() should fail
    engine._token_len_fn = lambda t: len(t.split())
    with pytest.raises(EngineError):
        engine.translate("Salom", "uz", "ru")


def test_nllb_engine_default_backend_is_transformers():
    """Default backend is 'transformers', calling _real_translate path."""
    engine = NLLBEngine(
        translate_fn=lambda c, s, t: f"[{t}]{c}",
        token_len_fn=lambda t: len(t.split()),
    )
    assert engine.backend == "transformers"
    result = engine.translate("Salom", "uz", "ru")
    # restore_case("Salom", "[rus_Cyrl]Salom") → "[Rus_Cyrl]Salom" (first letter uppercase)
    assert result == "[Rus_Cyrl]Salom"


def test_cached_engine_skips_underlying_engine_on_hit(tmp_path):
    calls = []

    def fake_translate(chunk: str, src: str, tgt: str) -> str:
        calls.append(chunk)
        return "translated"

    engine = NLLBEngine(
        translate_fn=fake_translate,
        token_len_fn=_word_count,
        max_tokens=512,
    )
    cache = TranslationCache(str(tmp_path / "cache.db"))
    cached = CachedEngine(engine, cache)

    first = cached.translate("Salom", "uz", "ru")
    second = cached.translate("Salom", "uz", "ru")

    # restore_case("Salom", "translated") → "Translated"
    assert first == "Translated"
    assert second == "Translated"
    assert len(calls) == 1  # second call hit the cache


def test_restore_case_uppercase():
    assert restore_case("HELLO", "translated") == "TRANSLATED"


def test_restore_case_title():
    assert restore_case("Hello World", "translated text") == "Translated Text"


def test_restore_case_first_word():
    assert restore_case("Salom", "privet") == "Privet"


def test_restore_case_lowercase():
    assert restore_case("hello", "TRANSLATED") == "TRANSLATED"


def test_apply_russian_typography_quotes():
    result = apply_russian_typography('"Hello"')
    assert "«" in result
    assert "»" in result


def test_apply_russian_typography_prepositions():
    result = apply_russian_typography("в доме")
    assert "в\xa0доме" in result
