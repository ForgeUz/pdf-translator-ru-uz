# pdf_translator_ru_uz/cache.py

"""Module: offline SQLite translation cache.

Keyed on MD5(src_lang|tgt_lang|normalized_source_text|glossary_hash) so identical
paragraphs skip the model entirely.

The cache key is metadata-free regarding layout, but tracks glossary state
to prevent stale translations when term locks change (F6 fix).
"""
from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional


class TranslationCache:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # Intent: Add schema_version and glossary_hash for cache invalidation (F6).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    key TEXT PRIMARY KEY,
                    translated_text TEXT NOT NULL,
                    schema_version INTEGER DEFAULT 1,
                    glossary_hash TEXT DEFAULT NULL
                )
                """
            )

    @staticmethod
    def _make_key(source_text: str, src_lang: str, tgt_lang: str, glossary_hash: Optional[str] = None) -> str:
        """Generate a versioned cache key.
        
        Dependencies: NFKC normalized text + language pair + glossary state.
        """
        normalized = unicodedata.normalize("NFKC", source_text.strip())
        g_hash = glossary_hash or "null"
        raw = f"{src_lang}|{tgt_lang}|{normalized}|{g_hash}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def get(
        self, 
        source_text: str, 
        src_lang: str, 
        tgt_lang: str, 
        glossary_hash: Optional[str] = None
    ) -> str | None:
        key = self._make_key(source_text, src_lang, tgt_lang, glossary_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT translated_text FROM translations WHERE key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def set(
        self,
        source_text: str,
        src_lang: str,
        tgt_lang: str,
        translated_text: str,
        glossary_hash: Optional[str] = None,
    ) -> None:
        key = self._make_key(source_text, src_lang, tgt_lang, glossary_hash)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO translations (key, translated_text, schema_version, glossary_hash) VALUES (?, ?, ?, ?)",
                (key, translated_text, 1, glossary_hash),
            )
