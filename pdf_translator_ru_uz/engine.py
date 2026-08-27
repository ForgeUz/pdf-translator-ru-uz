# pdf_translator_ru_uz/engine.py

"""Module: translation engine with advanced pre/post-processing.

Stages implemented:
  2. Pre-processing: NFKC normalisation, fasttext LID, NER masking,
     semantic sentence chunking, length-sorted batching, CT2 tuning.
  3. Post-processing: case restoration, Russian micro-typography.
  8. Confidence scoring (preserved from original).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

from pdf_translator_ru_uz.placeholders import PlaceholderRegistry

logger = logging.getLogger(__name__)

NLLB_LANG_CODES: dict[str, str] = {
    "ru": "rus_Cyrl",
    "uz": "uzn_Latn",
}
NLLB_LANG_CODES_CYRL: dict[str, str] = {
    "ru": "rus_Cyrl",
    "uz": "uzn_Cyrl",
}

# Russian short prepositions for micro-typography
_RUSSIAN_SHORT_WORDS = (
    "в", "на", "с", "к", "под", "о", "об", "от", "до", "по",
    "за", "из", "у", "или", "и", "а", "но", "да", "не", "ни",
    "без", "для", "над", "перед", "при", "про", "через", "сквозь",
)

# Intent: Enforce deterministic generation to ensure cache validity.
# State Transition: Stochastic sampling -> Greedy/beam search determinism.
DETERMINISM_CONTRACT = {
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "num_beams": 1,
    "seed": 42
}

# Global placeholder registry shared across the pipeline.
# Reset per-document via registry.reset() — called from parser.extract_paragraphs().
_REGISTRY = PlaceholderRegistry()


def get_placeholder_registry() -> PlaceholderRegistry:
    """Return the global placeholder registry instance."""
    return _REGISTRY


# ═══════════════════════════════════════════════════════════════════
# Stage 2: Pre-Processing
# ═══════════════════════════════════════════════════════════════════

# ── 2.1 Unicode Normalisation & LID ─────────────────────────────────


def normalize_uzbek(text: str) -> str:
    """Canonicalize Uzbek Latin apostrophe-like characters.

    Uzbek Latin script uses U+02BB (MODIFIER LETTER TURNED COMMA) for
    ``oʻ`` and ``gʻ``.  However, PDFs often contain U+2019 (RIGHT SINGLE
    QUOTATION MARK) or U+0027 (APOSTROPHE) instead.  NFKC does NOT
    normalize between these, so explicit mapping is required.

    Maps {U+2019, U+0027, U+02BC} → U+02BB.
    """
    table = str.maketrans({
        "\u2019": "\u02bb",  # ’ → ʻ
        "'": "\u02bb",       # ' → ʻ
        "\u02bc": "\u02bb",  # ʼ → ʻ (MODIFIER LETTER APOSTROPHE)
    })
    return text.translate(table)


def normalize_text(text: str, src_lang: str = "uz") -> str:
    """Apply NFKC Unicode normalisation, with Uzbek-specific pre-pass."""
    if src_lang == "uz":
        text = normalize_uzbek(text)
    return unicodedata.normalize("NFKC", text)


_lid_warned = False


def _load_lid_model():
    """Lazily load fasttext language ID model."""
    global _lid_warned
    try:
        import fasttext

        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".cache", "fasttext", "lid.176.ftz"),
            os.path.join(home, ".cache", "fasttext", "lid.176", "lid.176.ftz"),
            "/usr/share/fasttext/lid.176.ftz",
        ]
        for path in candidates:
            if os.path.exists(path):
                return fasttext.load_model(path)

        if not _lid_warned:
            logger.warning(
                "fasttext LID model (lid.176.ftz) not found in %s. "
                "Download with: "
                "wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz -P ~/.cache/fasttext/",
                candidates[0],
            )
            _lid_warned = True
        return None
    except ImportError:
        if not _lid_warned:
            logger.warning("fasttext not installed — skipping LID check")
            _lid_warned = True
        return None
    except Exception as exc:
        if not _lid_warned:
            logger.warning(
                "fasttext LID load failed (%s) — skipping LID check", exc
            )
            _lid_warned = True
        return None


_lid_model = None
_lid_lock = threading.Lock()


def _get_lid_model():
    global _lid_model
    if _lid_model is None:
        with _lid_lock:
            if _lid_model is None:
                _lid_model = _load_lid_model()
    return _lid_model


def is_target_language(
    text: str, target_lang: str, confidence_threshold: float = 0.8
) -> bool:
    """Use fasttext LID to check if *text* is already in *target_lang*."""
    model = _get_lid_model()
    if model is None:
        return False

    try:
        predictions = model.predict(
            text.replace("\n", " ").strip(), k=1
        )
        label = predictions[0][0]
        confidence = predictions[1][0]

        predicted_lang = label.replace("__label__", "")
        expected_label = target_lang

        if predicted_lang == expected_label and confidence >= confidence_threshold:
            return True
        return False
    except Exception as exc:
        logger.debug("LID prediction failed: %s", exc)
        return False


# ── 2.2 NER Masking ────────────────────────────────────────────────

_DATE_PATTERNS = [
    re.compile(r"\b\d{2,4}[-/\.]\d{1,2}[-/\.]\d{2,4}\b"),
    re.compile(
        r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|"
        r"июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
]

_NUMBER_PATTERNS = [
    re.compile(r"\b\d+[.,]?\d*%\b"),
    re.compile(r"\$\d+[.,]?\d*\b"),
    re.compile(r"\€\d+[.,]?\d*\b"),
    re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\b"),
    re.compile(r"\b\d+[.,]\d+\b"),
]

_LATIN_WORD_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]{1,}\b")

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
_URL_PATTERN = re.compile(
    r"\b(?:https?://|ftp://|www\.)[A-Za-z0-9./?=&%#_~-]+\b"
)


def mask_ner(
    text: str,
    src_lang: str = "uz",
) -> tuple[str, dict[str, str]]:
    """Replace NER entities with private-use Unicode placeholders."""
    mapping: dict[str, str] = {}
    masked = text

    for pattern in _DATE_PATTERNS + _NUMBER_PATTERNS + [
        _EMAIL_PATTERN,
        _URL_PATTERN,
    ]:
        masked = _replace_with_placeholders(
            masked, pattern, mapping
        )

    if src_lang == "uz":
        latin_pattern = re.compile(r"\b[A-Z]{2,}\b")
    else:
        latin_pattern = _LATIN_WORD_PATTERN

    masked = _replace_with_placeholders(
        masked, latin_pattern, mapping
    )

    return masked, mapping


def _replace_with_placeholders(
    text: str, pattern: re.Pattern, mapping: dict[str, str]
) -> str:
    result_parts: list[str] = []
    last_end = 0

    for m in pattern.finditer(text):
        start, end = m.start(), m.end()
        original = m.group(0)

        if any(
            ord(c) >= 0xE000 and ord(c) <= 0xEFFF
            for c in text[start:end]
        ):
            continue

        if original in mapping:
            placeholder = mapping[original]
        else:
            placeholder = _REGISTRY.mask_ner(original)
            mapping[original] = placeholder

        result_parts.append(text[last_end:start])
        result_parts.append(placeholder)
        last_end = end

    result_parts.append(text[last_end:])
    return "".join(result_parts)


def restore_ner(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    restored = text
    for placeholder, original in sorted(
        mapping.items(), key=lambda x: -len(x[0])
    ):
        restored = restored.replace(placeholder, original)
    return restored


# ── 2.3 Semantic Sentence Chunking ─────────────────────────────────

_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9\"'`(«])"
)

_ABBREVIATIONS = {
    "г.", "ул.", "др.", "пр.", "т.", "стр.", "с.", "п.",
    "доц.", "проф.", "акад.", "канд.", "док.", "тел.", "факс",
    "e.g.", "i.e.", "etc.", "vs.", "p.", "pp.", "vol.", "ed.",
    "dept.", "st.", "ave.", "blvd.", "Dr.", "Mr.", "Mrs.", "Ms.",
    "Jr.", "Sr.", "Inc.", "Ltd.", "Co.",
}


class SentenceChunker:
    def __init__(
        self,
        token_len_fn: Callable[[str], int],
        max_tokens: int = 512,
        budget_tokens: Optional[int] = None,
    ):
        self.token_len_fn = token_len_fn
        self.max_tokens = max_tokens
        self.budget_tokens = budget_tokens or int(max_tokens * 0.78)

    def split(self, text: str) -> list[str]:
        if self.token_len_fn(text) <= self.budget_tokens:
            return [text]

        sentences = self._split_sentences(text)
        if not sentences:
            return [text]

        chunks: list[str] = []
        current: list[str] = []

        for sentence in sentences:
            if self.token_len_fn(sentence) > self.budget_tokens:
                if current:
                    chunks.append(" ".join(current))
                    current = []
                chunks.append(sentence)
                continue

            candidate = " ".join(current + [sentence])
            if current and self.token_len_fn(candidate) > self.budget_tokens:
                chunks.append(" ".join(current))
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        protected = text
        abbr_map: dict[str, str] = {}
        for i, abbr in enumerate(_ABBREVIATIONS):
            if abbr in protected:
                ph = f"\x00ABBR{i}\x00"
                abbr_map[ph] = abbr
                protected = protected.replace(abbr, ph)

        parts = _SENTENCE_SPLIT_PATTERN.split(protected)

        result = []
        for part in parts:
            restored = part
            for ph, abbr in abbr_map.items():
                restored = restored.replace(ph, abbr)
            restored = restored.strip()
            if restored:
                result.append(restored)

        return result


# ── 2.4 Length-Sorted Batching ──────────────────────────────────────


class LengthSortedBatcher:
    def __init__(
        self,
        token_len_fn: Callable[[str], int],
        batch_size: int = 8,
    ):
        self.token_len_fn = token_len_fn
        self.batch_size = batch_size

    def batch(
        self, chunks: list[str]
    ) -> list[tuple[list[str], list[int]]]:
        indexed = [
            (i, chunk, self.token_len_fn(chunk))
            for i, chunk in enumerate(chunks)
        ]
        indexed.sort(key=lambda x: x[2])

        batches: list[tuple[list[str], list[int]]] = []
        current_texts: list[str] = []
        current_indices: list[int] = []

        for i, text, _tokens in indexed:
            current_texts.append(text)
            current_indices.append(i)
            if len(current_texts) >= self.batch_size:
                batches.append(
                    (current_texts[:], current_indices[:])
                )
                current_texts = []
                current_indices = []

        if current_texts:
            batches.append(
                (current_texts[:], current_indices[:])
            )

        return batches

    @staticmethod
    def reorder(
        results: list[list[str]], batches: list[tuple[list[str], list[int]]]
    ) -> list[str]:
        total = sum(len(b[0]) for b in batches)
        ordered: list[tuple[int, str]] = []

        for batch_results, (_, indices) in zip(results, batches):
            for result, orig_idx in zip(batch_results, indices):
                ordered.append((orig_idx, result))

        ordered.sort(key=lambda x: x[0])
        return [t[1] for t in ordered]


# ═══════════════════════════════════════════════════════════════════
# Stage 2.5: Math Masking (from original — preserved)
# ═══════════════════════════════════════════════════════════════════


def mask_math(text: str, math_spans: list) -> tuple[str, dict[str, str]]:
    if not math_spans:
        return text, {}

    mapping: dict[str, str] = {}
    sorted_spans = sorted(
        math_spans,
        key=lambda s: len(getattr(s, "original_text", "")),
        reverse=True,
    )
    masked = text
    for span in sorted_spans:
        placeholder = getattr(span, "placeholder", "")
        orig = getattr(span, "original_text", "")
        if placeholder and placeholder in masked:
            mapping[placeholder] = orig if orig else placeholder
            masked = masked.replace(placeholder, "▨", 1)
    return masked, mapping


def unmask_math(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    placeholders = list(mapping.keys())
    restored = text
    for ph in placeholders:
        replacement = mapping[ph] if mapping[ph] else "▨"
        restored = restored.replace("▨", replacement, 1)
    return restored


def protect_latex_blocks(
    text: str,
    registry: Optional[PlaceholderRegistry] = None,
) -> tuple[str, dict[str, str]]:
    reg = registry or _REGISTRY
    mapping: dict[str, str] = {}

    def _replace_math(match: re.Match) -> str:
        ph = reg.mask_latex()
        mapping[ph] = match.group(0)
        return ph

    text = re.sub(
        r"\$\$(.*?)\$\$", _replace_math, text, flags=re.DOTALL
    )
    text = re.sub(r"\$(.*?)\$", _replace_math, text)
    return text, mapping


def restore_latex_blocks(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text
    restored = text
    for ph, original in sorted(
        mapping.items(), key=lambda x: -len(x[0])
    ):
        restored = restored.replace(ph, original)
    return restored


def check_placeholder_survival(
    tokenizer, placeholder: str = "▨"
) -> bool:
    tokens = tokenizer.encode(placeholder)
    decoded = tokenizer.decode(tokens).strip()
    return placeholder in decoded


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Post-Processing
# ═══════════════════════════════════════════════════════════════════


def restore_case(original: str, translated: str) -> str:
    if not original or not translated:
        return translated

    if original.isupper():
        return translated.upper()
    if original.istitle():
        return translated.title()
    if original[0].isupper():
        return translated[0].upper() + translated[1:]
    return translated


def apply_russian_typography(text: str) -> str:
    result = re.sub(
        r'"([^"]*)"', r'«\1»', text
    )

    left_quote = "\u2018"
    right_quote = "\u2019"
    result = re.sub(
        r"'([^']*)'", lambda m: left_quote + m.group(1) + right_quote, result
    )

    short_words_pattern = r"\b(" + "|".join(
        re.escape(w) for w in _RUSSIAN_SHORT_WORDS
    ) + r")\s+(?=\S)"

    result = re.sub(
        short_words_pattern,
        lambda m: m.group(1) + "\xa0",
        result,
    )

    return result


@dataclass
class TranslationResult:
    text: str
    confidence: float = 1.0
    src_lang: str = ""
    tgt_lang: str = ""


def compute_confidence_from_logits(scores: tuple | None) -> float:
    if not scores:
        return 1.0
    try:
        import torch
        total = 0.0
        for step_scores in scores:
            probs = torch.softmax(step_scores, dim=-1)
            total += probs.max().item()
        return total / len(scores)
    except Exception:
        return 1.0


class EngineError(RuntimeError):
    pass


class ModelLoadError(EngineError):
    pass


class TokenAwareChunker:
    def __init__(
        self,
        token_len_fn: Callable[[str], int],
        max_tokens: int = 512,
    ):
        self.token_len_fn = token_len_fn
        self.max_tokens = max_tokens

    def split(self, text: str) -> list[str]:
        if self.token_len_fn(text) <= self.max_tokens:
            return [text]
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if (
                current
                and self.token_len_fn(candidate) > self.max_tokens
            ):
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
        return chunks


class NLLBEngine:
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        model_path: Optional[str] = None,
        backend: str = "transformers",
        max_tokens: int = 512,
        translate_fn: Optional[Callable[[str, str, str], str]] = None,
        token_len_fn: Optional[Callable[[str], int]] = None,
        batch_translate_fn: Optional[Callable[[list[str], str, str], list[str]]] = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.backend = backend
        self.max_tokens = max_tokens
        self._translate_fn = translate_fn
        self._token_len_fn = token_len_fn
        self._batch_translate_fn = batch_translate_fn
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._ct2_model = None
        self._chunker: Optional[SentenceChunker] = None
        self._last_scores = None

    def _ensure_loaded(self) -> None:
        if self._translate_fn is not None and self._token_len_fn is not None:
            self._chunker = SentenceChunker(
                self._token_len_fn, self.max_tokens
            )
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import (
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                logger.info(
                    "Loading NLLB model %s...", self.model_name
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name
                )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype="auto",
                )
                self._token_len_fn = lambda t: len(
                    self._tokenizer(t)["input_ids"]
                )
                self._chunker = SentenceChunker(
                    self._token_len_fn, self.max_tokens
                )
                logger.info("NLLB model loaded.")
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load '{self.model_name}': {exc}"
                ) from exc

    @staticmethod
    def _resolve_lang_code(
        lang: str, prefer_cyrl: bool = False
    ) -> str:
        table = NLLB_LANG_CODES_CYRL if prefer_cyrl else NLLB_LANG_CODES
        if lang not in table:
            raise EngineError(
                f"Unsupported lang '{lang}'. Use {list(table)}."
            )
        return table[lang]

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self._ensure_loaded()
        src_code = self._resolve_lang_code(src_lang)
        tgt_code = self._resolve_lang_code(tgt_lang)

        text = normalize_text(text, src_lang=src_lang)

        if is_target_language(text, tgt_lang):
            logger.debug("LID: text already in %s, skipping translation", tgt_lang)
            return text

        chunks = self._chunker.split(text)

        if self._translate_fn is not None:
            translate_fn = self._translate_fn
        elif self._batch_translate_fn is not None and self.batch_size > 1:
            translate_fn = lambda t, s, tg: self._batch_translate_fn(t, s, tg)
        elif self.backend == "ctranslate2":
            translate_fn = self._real_translate_ct2
        else:
            translate_fn = self._real_translate

        if (
            self._batch_translate_fn is not None
            and self.batch_size > 1
        ):
            return self._translate_with_batching(
                chunks, src_code, tgt_code
            )

        translated = []
        for chunk in chunks:
            try:
                out = translate_fn(chunk, src_code, tgt_code)
            except Exception as exc:
                raise EngineError(
                    f"NLLB generation failed: {exc}"
                ) from exc
            if not out or not out.strip():
                raise EngineError("Empty NLLB output for chunk.")
            out = restore_case(chunk, out)
            out = apply_russian_typography(out)
            translated.append(out.strip())

        return " ".join(translated)

    def _translate_with_batching(
        self,
        chunks: list[str],
        src_code: str,
        tgt_code: str,
    ) -> str:
        from pdf_translator_ru_uz.engine import (
            LengthSortedBatcher,
        )

        batcher = LengthSortedBatcher(
            self._token_len_fn, self.batch_size
        )
        batches = batcher.batch(chunks)
        all_results: list[str] = [""] * len(chunks)

        for batch_texts, indices in batches:
            if self._batch_translate_fn:
                translated_batch = self._batch_translate_fn(
                    batch_texts, src_code, tgt_code
                )
            else:
                translated_batch = []
                for bt in batch_texts:
                    translated_batch.append(
                        self._real_translate(bt, src_code, tgt_code)
                    )

            for idx, translated_text in zip(
                indices, translated_batch
            ):
                original_chunk = chunks[idx]
                translated_text = restore_case(
                    original_chunk, translated_text
                )
                translated_text = apply_russian_typography(
                    translated_text
                )
                all_results[idx] = translated_text.strip()

        return " ".join(all_results)

    def _real_translate(
        self, text: str, src_code: str, tgt_code: str
    ) -> str:
        import torch

        self._tokenizer.src_lang = src_code
        self._tokenizer.tgt_lang = tgt_code
        self._model.generation_config.max_length = None

        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self._model.device)
        forced_bos_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=512,
                **DETERMINISM_CONTRACT,  # Applied determinism contract
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = self._tokenizer.decode(
            output_ids["sequences"][0], skip_special_tokens=True
        ).strip()
        self._last_scores = output_ids.get("scores")
        return decoded

    def translate_with_confidence(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> TranslationResult:
        self._ensure_loaded()
        src_code = self._resolve_lang_code(src_lang)
        tgt_code = self._resolve_lang_code(tgt_lang)

        text = normalize_text(text, src_lang=src_lang)
        chunks = self._chunker.split(text)

        if self._translate_fn is not None:
            translate_fn = self._translate_fn
        elif self.backend == "ctranslate2":
            translate_fn = self._real_translate_ct2
        else:
            translate_fn = self._real_translate

        translated = []
        all_confidences: list[float] = []

        for chunk in chunks:
            try:
                out = translate_fn(chunk, src_code, tgt_code)
            except Exception as exc:
                raise EngineError(
                    f"NLLB generation failed: {exc}"
                ) from exc
            if not out or not out.strip():
                raise EngineError("Empty NLLB output for chunk.")
            out = restore_case(chunk, out)
            out = apply_russian_typography(out)
            translated.append(out.strip())

            if (
                hasattr(self, "_last_scores")
                and self._last_scores is not None
            ):
                all_confidences.append(
                    compute_confidence_from_logits(
                        self._last_scores
                    )
                )
                self._last_scores = None

        avg_conf = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else 1.0
        )
        return TranslationResult(
            text=" ".join(translated),
            confidence=avg_conf,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

    def _ensure_ct2_loaded(self) -> None:
        if self._ct2_model is not None:
            return
        with self._lock:
            if self._ct2_model is not None:
                return
            try:
                import ctranslate2

                if not self.model_path:
                    raise EngineError(
                        "CTranslate2 backend requires model_path. "
                        f"Convert NLLB first: ct2-transformers-converter "
                        f"--model {self.model_name} --output_dir ./nllb_ct2_model"
                    )

                logger.info(
                    "Loading CTranslate2 model from %s...",
                    self.model_path,
                )
                physical_cores = os.cpu_count() or 4
                self._ct2_model = ctranslate2.Translator(
                    self.model_path,
                    device="cpu",
                    intra_threads=physical_cores,
                )
                self._tokenizer = self._load_tokenizer()
                self._token_len_fn = lambda t: len(
                    self._tokenizer(t)["input_ids"]
                )
                self._chunker = SentenceChunker(
                    self._token_len_fn, self.max_tokens
                )
                logger.info("CTranslate2 model loaded.")
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load CTranslate2 model "
                    f"from '{self.model_path}': {exc}"
                ) from exc

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)

    def _real_translate_ct2(
        self, text: str, src_code: str, tgt_code: str
    ) -> str:
        self._ensure_ct2_loaded()

        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(text)
        token_ids = encoded["input_ids"]
        input_tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
        tgt_token = tgt_code

        input_length = len(input_tokens)
        max_decoding = int(input_length * 1.5 + 10)

        # CT2 determinism: beam_size=1 is greedy. No sampling temp.
        results = self._ct2_model.translate_batch(
            [input_tokens],
            target_prefix=[[tgt_token]],
            beam_size=DETERMINISM_CONTRACT["num_beams"],
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_decoding_length=max_decoding,
        )
        output_tokens = results[0].hypotheses[0]
        output_ids = self._tokenizer.convert_tokens_to_ids(
            output_tokens
        )
        output_ids = [
            i
            for i in output_ids
            if i is not None
            and i != self._tokenizer.unk_token_id
        ]
        return self._tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()

    def translate_batch(
        self, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        self._ensure_loaded()
        if self.backend == "ctranslate2":
            self._ensure_ct2_loaded()

        src_code = self._resolve_lang_code(src_lang)
        tgt_code = self._resolve_lang_code(tgt_lang)

        batcher = LengthSortedBatcher(
            self._token_len_fn, self.batch_size
        )
        normalized = [normalize_text(t, src_lang=src_lang) for t in texts]

        skip_mask = [
            is_target_language(t, tgt_lang) for t in normalized
        ]

        to_translate: list[int] = []
        translation_inputs: list[str] = []
        for i, (skip, t) in enumerate(zip(skip_mask, normalized)):
            if not skip:
                to_translate.append(i)
                translation_inputs.append(t)

        if not translation_inputs:
            return texts

        all_chunks: list[tuple[int, str]] = []
        for idx in to_translate:
            chunks = self._chunker.split(normalized[idx])
            all_chunks.append((idx, " ".join(chunks)))

        batches = batcher.batch(
            [c[1] for c in all_chunks]
        )

        total_batches = len(batches)
        logger.info(
            "Translating %d text(s) in %d batch(es)...",
            len(translation_inputs), total_batches,
        )
        translated_all: list[str] = [""] * len(texts)
        for batch_idx, (batch_texts, indices) in enumerate(batches):
            logger.info(
                "  Batch %d/%d (%d texts, ~%d tokens each)",
                batch_idx + 1, total_batches,
                len(batch_texts),
                max(self._token_len_fn(t) for t in batch_texts),
            )
            if self.backend == "ctranslate2":
                batch_results = self._batch_translate_ct2(
                    batch_texts, src_code, tgt_code
                )
            else:
                batch_results = [
                    self._real_translate(t, src_code, tgt_code)
                    for t in batch_texts
                ]

            for batch_idx, result_text in zip(
                indices, batch_results
            ):
                orig_idx = all_chunks[batch_idx][0]
                original = normalized[orig_idx]
                result_text = restore_case(original, result_text)
                result_text = apply_russian_typography(result_text)
                translated_all[orig_idx] = result_text.strip()

        for i in range(len(texts)):
            if skip_mask[i]:
                translated_all[i] = texts[i]

        return translated_all

    def _batch_translate_ct2(
        self,
        texts: list[str],
        src_code: str,
        tgt_code: str,
    ) -> list[str]:
        self._ensure_ct2_loaded()
        self._tokenizer.src_lang = src_code
        tgt_token = tgt_code

        batch_tokens: list[list[str]] = []
        for text in texts:
            encoded = self._tokenizer(text)
            tokens = self._tokenizer.convert_ids_to_tokens(
                encoded["input_ids"]
            )
            batch_tokens.append(tokens)

        max_input_len = max(len(t) for t in batch_tokens)
        max_decoding = int(max_input_len * 1.5 + 10)

        results = self._ct2_model.translate_batch(
            batch_tokens,
            target_prefix=[[tgt_token]] * len(batch_tokens),
            beam_size=DETERMINISM_CONTRACT["num_beams"],
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_decoding_length=max_decoding,
        )

        output_texts: list[str] = []
        for result in results:
            output_tokens = result.hypotheses[0]
            output_ids = self._tokenizer.convert_tokens_to_ids(
                output_tokens
            )
            output_ids = [
                i
                for i in output_ids
                if i is not None
                and i != self._tokenizer.unk_token_id
            ]
            decoded = self._tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            output_texts.append(decoded)

        return output_texts


class CachedEngine:
    def __init__(self, engine, cache):
        self.engine = engine
        self.cache = cache

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        cached = self.cache.get(text, src_lang, tgt_lang)
        if cached is not None:
            return cached
        result = self.engine.translate(text, src_lang, tgt_lang)
        self.cache.set(text, src_lang, tgt_lang, result)
        return result


class BatchingEngine:
    def __init__(
        self,
        engine,
        batch_translate_fn=None,
        batch_size: int = 8,
    ):
        self.engine = engine
        self.batch_translate_fn = batch_translate_fn
        self.batch_size = batch_size
        self.queue_items: list[tuple[str, str, str]] = []
        self.results: list[str] = []

    def queue(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> None:
        self.queue_items.append((text, src_lang, tgt_lang))
        if len(self.queue_items) >= self.batch_size:
            self.flush()

    def flush(self) -> list[str]:
        if not self.queue_items:
            return self.results
        if self.batch_translate_fn:
            _, src_lang, tgt_lang = self.queue_items[0]
            texts = [item[0] for item in self.queue_items]
            batch_results = self.batch_translate_fn(
                texts, src_lang, tgt_lang
            )
            self.results.extend(batch_results)
        else:
            for text, src_lang, tgt_lang in self.queue_items:
                result = self.engine.translate(
                    text, src_lang, tgt_lang
                )
                self.results.append(result)
        self.queue_items = []
        return self.results
