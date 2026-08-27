from pdf_translator_ru_uz.cache import TranslationCache


def test_cache_roundtrip(tmp_path):
    cache = TranslationCache(str(tmp_path / "cache.db"))
    assert cache.get("Salom dunyo", "uz", "ru") is None

    cache.set("Salom dunyo", "uz", "ru", "Привет мир")
    assert cache.get("Salom dunyo", "uz", "ru") == "Привет мир"


def test_cache_distinguishes_language_direction(tmp_path):
    cache = TranslationCache(str(tmp_path / "cache.db"))
    cache.set("text", "uz", "ru", "russian version")
    assert cache.get("text", "ru", "uz") is None  # different direction, no collision


def test_cache_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "cache.db")
    TranslationCache(db_path).set("hello", "uz", "ru", "привет")
    reopened = TranslationCache(db_path)
    assert reopened.get("hello", "uz", "ru") == "привет"
