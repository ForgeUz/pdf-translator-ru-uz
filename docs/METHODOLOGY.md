# Methodology: PDF Translation Pipeline (Uzbek → Russian)

## 1. Problem Description

### Output Quality Audit: constitution_ru_v2.pdf

The translated output (`constitution_ru_v2.pdf`) was produced by the upgraded pipeline and undergoes a detailed technical audit below. The Russian text is not just inaccurate — it suffers from catastrophic structural, semantic, and systemic failures, likely a combination of broken text extraction and a heavily hallucinating machine translation engine.

#### 1.1 Structural and Formatting Disintegration

The translation completely fails to preserve the legal structure of the source document.

- **Loss of Hierarchy:** The Uzbek source text is cleanly divided into specific sections and articles (e.g., "1-modda.", "2-modda."). The Russian text ignores this structure, concatenating unrelated articles into massive, unreadable blocks of text.
- **Missing Punctuation and Spacing:** There is a systemic failure in rendering whitespace, resulting in hundreds of fused words. Examples include "государствоосуществляет" (instead of "государство осуществляет"), "правовойсистемыУзбекистана", "Конституциии", and "цельюобеспечения".
- **Truncated Sentences:** The translation frequently breaks off mid-sentence or mid-thought, such as abruptly ending a paragraph with the incomplete phrase "В соответствии со ст .".

#### 1.2 Processing Glitches and Uncleaned Metadata

The output contains severe duplication glitches and artifacts from web scraping that were not filtered out.

- **Header Duplication:** The word "КОНСТИТУЦИЯ" is rendered seven times consecutively in a single line ("КОНСТИТУЦИЯ КОНСТИТУЦИЯ КОНСТИТУЦИЯ...").
- **Metadata Glitches:** The phrase "Дата вступления на кучу" is repeated back-to-back nine times at the very beginning of the document.
- **Paragraph Duplication:** The entire opening paragraph of the preamble ("Мы торжественно провозглашаем свою приверженность...") is duplicated three times in a row.

#### 1.3 Critical Semantic Hallucinations

This is the most severe defect. The translation engine has "hallucinated" entities, concepts, and entirely different countries that do not exist in the Uzbek source text.

- **Foreign Entities:** The Russian text inexplicably mentions the "Президента Республики Башкортостан" (President of the Republic of Bashkortostan) and the "Генеральным прокурором Республики Казахстан" (General Prosecutor of the Republic of Kazakhstan). Neither of these is in the Constitution of Uzbekistan.
- **Fabricated Legal Bodies:** The translation invents nonsensical government bodies that do not exist in Uzbekistan, such as "Всемирной парламентской палаты Верховной Рады Сената Республики Узбекистан" (combining "World Parliamentary Chamber," the Ukrainian "Verkhovna Rada," and the Uzbek "Senate") and the "Осупределительная палата".
- **Gibberish Output:** The text contains literal gibberish that has no linguistic meaning in Russian or Uzbek, such as the word "ХарлаГарльХарля" and the phrase "Конвенции ОДК ВСТРИЧЕВЫХ".

#### 1.4 Severe Lexical and Contextual Mistranslations

Even when the text is readable, the terminology used is legally incorrect and alters the meaning of the original document.

- **Incorrect Terminology:** The Uzbek legal term "Kuchga kirish sanasi" (Date of entry into force) is absurdly mistranslated as "Дата вступления на кучу" (Date of entry onto a pile/heap). The correct legal Russian phrase is "Дата вступления в силу".
- **Subject Alteration:** In the preamble, the Uzbek original reads "Biz, Oʻzbekistonning yagona xalqi..." (We, the united people of Uzbekistan...). The translation alters this to "Мы торжественно провозглашаем свою приверженность единому народу Узбекистана..." (We solemnly proclaim our commitment to the united people of Uzbekistan...), which fundamentally changes the grammatical subject and legal meaning of the constitutional preamble.
- **Fabricated Clauses:** The translation adds concepts not present in the original, such as stating a commitment to "охраны окружающих нас государств" (protecting the states surrounding us).

#### 1.5 Conclusion

From a technical analysis standpoint, the Russian text cannot be considered a translation. It is a corrupted data output. It cannot be edited or corrected, as the foundational structure, legal terminology, and semantic integrity are completely destroyed. The only viable technical solution is to discard this output and generate a new translation using a reliable, legally trained translation engine or a human legal translator.

---

## 2. Implemented Methods (Pipeline Architecture)

The pipeline processes PDF documents through four stages:

```
[input.pdf] → parser.py → engine.py → builder.py → [output.pdf]
```

### Stage 1: Advanced Span-Level Parsing — `parser.py` (767 lines)

**Method:** Layout-aware extraction using PyMuPDF's `page.get_text("dict")` with full geometric analysis.

| Sub-component | Method | Implementation |
|--------------|--------|----------------|
| 1.1 Reading Order | `_sort_blocks_by_reading_order()` | Projects block x-centres onto x-axis, detects vertical separators (gaps > 2× median width), groups blocks by column, sorts each column top-to-bottom, concatenates left-to-right |
| 1.2 Span Metadata | `TextSpan` dataclass | Extracts `font`, `size`, `color` (RGB tuple), transformation `matrix`, writing direction `dir` per span |
| 1.3 Header/Footer | `_is_header_or_footer()` | Filters blocks whose y-centre is in top/bottom 5% of page height |
| 1.3 Watermark | `_is_watermark()` | Filters spans with opacity < 0.5 or near-white color |
| 1.4 Table Isolation | `_extract_tables()` | Uses `page.find_tables()` → `t.cells` (row-major `fitz.Rect` list) + `t.extract()` (2D text grid). Each cell translated individually |
| 1.5 De-hyphenation | `_dehyphenate_line()` | Detects trailing hyphens at line breaks, checks vowel/consonant syllable boundary heuristic, merges with next line. Preserves em-dashes, en-dashes, and compound suffixes |
| 1.6 List Markers | `_strip_list_marker()` | Regex detection of `•`, `1.`, `a)`, `-` markers. Stripped before translation, prepended after |
| 1.6 Drop Caps | `_detect_drop_cap()` | Detects single-character spans with size ≥ 1.5× median, merges into next span |

### Stage 2: Pre-Processing & Inference Optimization — `engine.py` (1233 lines)

**Method:** Protect NLLB model from breaking, looping, or hallucinating due to truncation or bad inputs.

| Sub-component | Method | Implementation |
|--------------|--------|----------------|
| 2.1 Unicode Normalization | `normalize_text()` | `unicodedata.normalize("NFKC")` — fixes ligatures (ﬁ → fi) |
| 2.1 Language ID | `is_target_language()` | Uses fasttext `lid.176.ftz` model. Skips translation if text is already in target language with confidence ≥ 0.8 |
| 2.2 NER Masking | `mask_ner()` | Regex patterns for dates (ISO, Russian, English), numbers (%, $, large numbers, decimals), Latin-script proper nouns, emails, URLs. Each entity replaced with deterministic hash-based private-use Unicode placeholder (U+E800..U+EFFF) |
| 2.2 Math Masking | `mask_math()` | Existing math placeholder (U+E000..U+E7FF) → `▨` marker. Applied on top of NER masking |
| 2.3 Sentence Chunking | `SentenceChunker` | Splits on `[.!?…]\s+(?=[A-ZА-ЯЁ0-9])` with abbreviation protection (г., ул., др., etc.). Groups complete sentences into chunks ≤ 400 tokens (78% of 512 limit, leaving headroom for target-language expansion) |
| 2.4 Length-Sorted Batching | `LengthSortedBatcher` | Collects all chunks → sorts by token length → groups into batches of `batch_size` → translates → reorders by original index |
| 2.5 CT2 Tuning | `_ensure_ct2_loaded()` | `intra_threads=os.cpu_count()`, `_real_translate_ct2()`: `beam_size=4`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, `max_decoding_length = int(input_length * 1.5 + 10)` |

### Stage 3: Post-Processing & Caching — `engine.py` + `cache.py`

| Sub-component | Method | Implementation |
|--------------|--------|----------------|
| 3.1 Case Restoration | `restore_case()` | Checks `isupper()`, `istitle()`, `first.isupper()` on source → re-applies same casing pattern to translation |
| 3.2 Russian Typography | `apply_russian_typography()` | `""` → `«»`, `''` → `''` (curly quotes), `\xa0` after short prepositions (в, на, с, к, под, о, об, от, до, по, за, из, у, или, и, а, но, да, не, ни, без, для, над, перед, при, про, через, сквозь) |
| 3.3 Metadata-Free Cache | `TranslationCache._make_key()` | MD5 of `NFKC(source_text.strip())` + lang pair. No font, bbox, or layout data in hash. Enables instant re-rendering on layout changes |

### Stage 4: Intelligent In-Place Builder — `builder.py` (789 lines)

**Method:** Overcomes limitations of basic redact+redraw using PyMuPDF advanced rendering.

| Sub-component | Method | Implementation |
|--------------|--------|----------------|
| 4.1 Background Inpainting | `_compute_background_color()` | Renders 30×30px pixmap of bbox area, samples 3×3 grid, excludes near-black pixels, returns mean RGB. Skips if image overlap > 50% |
| 4.1 Paint | `_paint_background()` | `page.draw_rect(bbox, color=color, fill=color, overlay=False)` — replaces rather than blends |
| 4.2 Image-Aware Text | `_has_image_overlap()` | `page.get_images()` + `page.get_image_rects()` → intersection check. If overlap, draws white shadow offset + text on top |
| 4.3 Font Synthesis | `_resolve_font()` | Mapping tree: Serif → Liberation Serif, Sans → DejaVu Sans, Mono → Liberation Mono. Bold/italic resolved via font flags. Falls back to DejaVu Sans |
| 4.4 Line-Wrapping | `_wrap_text()` | Measures each word width via `_estimate_text_width()` (character-category heuristic: narrow/normal/wide/CJK). Accumulates words into lines ≤ bbox width |
| 4.4 Interline Spacing | `draw_text_multiline()` | Draws each line via `page.insert_text()`, advances Y by `fontsize * 1.2` |
| 4.5 Justification | `_is_likely_justified()` + `_justify_line()` | If 2+ lines fill > 85% of bbox width, distributes remaining whitespace evenly between word gaps |
| 4.6 Rotation | `_compute_rotation_origin()` | Converts `dir` vector (dx, dy) to rotation angle via `math.atan2()`. Applies to `insert_text()` point |
| 4.7 GC Save | — | `doc.save(path, garbage=4, deflate=True, clean=True)` |

---

## 3. Test Results

### 59 tests, all passing (22.5s runtime)

```
$ python -m pytest tests/ -v
collected 59 items

test_builder.py   .........                                         [ 15%]
test_cache.py     ...                                               [ 20%]
test_confidence.py .....                                            [ 28%]
test_engine.py    .............                                     [ 50%]
test_latex_protection.py ......                                     [ 60%]
test_math_masking.py ............                                   [ 81%]
test_parser.py    ........                                          [ 94%]
test_pipeline.py  ..                                                [100%]

======================== 59 passed in 22.5s =========================
```

### Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_parser.py` | 8 | Line clustering, math font detection, reading order (columns), header/footer filtering, list marker stripping, de-hyphenation |
| `test_engine.py` | 13 | TokenAwareChunker, SentenceChunker, NLLBEngine mock translate, BatchingEngine, CachedEngine, backend switching, restore_case (4 cases), apply_russian_typography (2 cases) |
| `test_builder.py` | 9 | Redaction, replace_paragraph, line-wrapping, shrink-to-fit, bleed, font mapping (serif/sans), subtract_bboxes |
| `test_cache.py` | 3 | Roundtrip, language direction distinction, persistence |
| `test_confidence.py` | 5 | TranslationResult defaults, full, confidence scores, mock translate_with_confidence |
| `test_math_masking.py` | 12 | MathSpan, mask_math/unmask_math (single/multiple/noop), check_placeholder_survival, subtract_bboxes, replace_paragraph with excludes |
| `test_latex_protection.py` | 6 | Inline/display/protect/restore/multiple/noop |
| `test_pipeline.py` | 2 | End-to-end pipeline, cache reuse across runs |

All tests use mock translate functions — no real NLLB model load needed.

---

## 4. Difficulties Encountered

### 4.1 PDF Table Extraction API Mismatch

**Problem:** PyMuPDF 1.28.0's `Table` object does not have a `get_cell_bbox()` method. The original code assumed this API existed, causing `AttributeError: 'Table' object has no attribute 'get_cell_bbox'` at runtime.

**Root cause:** The PyMuPDF Table API uses `t.cells` (a flat list of `fitz.Rect` objects in row-major order) and `t.extract()` (a 2D list of cell texts). There is no per-cell bounding box method.

**Fix:** Replaced `t.get_cell_bbox(row, col)` with `t.cells[row * col_count + col]` index lookup, deriving `col_count` from `max(len(r) for r in extract_data)`.

### 4.2 CTranslate2 Integration — HIGH difficulty

Three separate bugs masked each other:
1. **Corrupted model file** from interrupted conversion — `model.bin` was truncated
2. **API format change** between CT2 versions 3.x → 4.x — token IDs (`List[List[int]]`) changed to string tokens (`List[List[str]]`)
3. **Language control** requires `target_prefix` instead of `forced_bos_token_id` — wrong target language (Turkish instead of Russian)

Each bug produced a confusing error that didn't point to the root cause. Required testing each piece in isolation: model load → tokenize → translate → decode.

### 4.3 PyMuPDF get_text_length() Limitation

**Problem:** `fitz.get_text_length()` does not support custom TTF fonts via `fontfile` parameter. It only works with built-in WinAnsi fonts (helv, times, cour) which lack Cyrillic glyphs.

**Fix:** Implemented `_estimate_text_width()` — a character-category heuristic that assigns width factors to narrow (`itl1!|.,;:`), wide (`WMmwO0Q@#&%`), Cyrillic, and CJK characters. This is a reasonable approximation for proportional fonts at any size.

### 4.4 Deterministic NER Placeholders for Cache

**Problem:** The NER masking function used a global sequential counter (`_NER_COUNTER`), producing different placeholder characters on each pipeline run. This caused cache misses on the second run because the masked text (containing different private-use chars) had a different MD5 hash.

**Fix:** Replaced sequential counter with `hash(text) & 0x7FF` — a deterministic hash-based approach that maps each NER entity text to a stable private-use Unicode character (U+E800..U+EFFF).

### 4.5 fasttext LID Model Loading

**Problem:** `fasttext.util.model_path()` does not exist in the installed version. The function tried to call this and failed silently, then tried to download the model (which requires internet access that may be unavailable).

**Fix:** Removed the broken API call. The function now searches standard cache directories (`~/.cache/fasttext/lid.176.ftz`) directly for the pre-downloaded model. Added a `_lid_warned` flag to print the warning only once (not per-paragraph).

### 4.6 Background Colour Extraction Performance

**Problem:** The `_compute_background_color()` method rendered a 200px-wide pixmap of each paragraph's bounding box, then sampled every 4th pixel. For 31 pages × 6 paragraphs = 186 renderings, this was extremely slow (20+ minutes of silence).

**Fix:** Reduced pixmap to 30×30px, sampled 3×3 grid (9 samples total). This is ~100× faster while still providing an adequate colour estimate.

### 4.7 Pipeline Progress Visibility

**Problem:** The batch translation path had zero logging output — the user saw "CTranslate2 model loaded" and then 20+ minutes of silence with no indication of progress.

**Fix:** Added logging to `translate_batch()` showing "Batch N/M (X texts, ~Y tokens)" and to `run_pipeline()` showing "Render progress: N/M paragraphs (page P/P_total)" every 5 paragraphs.

---

## 5. Model & Backend Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | `facebook/nllb-200-distilled-600M` | Smallest NLLB variant, best CPU performance |
| **Backend** | CTranslate2 | 3-4× CPU speedup over Transformers |
| **Beam size** | 4 | Better quality than greedy (beam=1), slower but acceptable |
| **Repetition penalty** | 1.2 | Discourages repeating tokens/ngrams |
| **No-repeat n-gram size** | 3 | Prevents 3-gram repetitions |
| **Max decoding length** | `input_length * 1.5 + 10` | Adaptive, allows ~50% longer target than source |
| **Batch size** | 8 | Default for length-sorted batching |
| **intra_threads** | `os.cpu_count()` | Full CPU utilization for CT2 |

## 6. File Structure (Post-Upgrade)

```
pdf_translate_pipeline/
├── pdf_translate_pipeline/
│   ├── __init__.py
│   ├── parser.py          (767 lines) — Advanced span-level parsing
│   ├── engine.py          (1233 lines) — Translation engine + pre/post-processing
│   ├── cache.py           (58 lines)  — Metadata-free SQLite cache
│   ├── builder.py         (789 lines) — Intelligent in-place rendering
│   └── pipeline.py        (554 lines) — Orchestrator
├── tests/                 (59 tests, all passing)
│   ├── test_parser.py     (8 tests)
│   ├── test_engine.py     (13 tests)
│   ├── test_builder.py    (9 tests)
│   ├── test_cache.py      (3 tests)
│   ├── test_confidence.py (5 tests)
│   ├── test_math_masking.py (12 tests)
│   ├── test_latex_protection.py (6 tests)
│   └── test_pipeline.py   (2 tests)
├── nllb_ct2_model/        (2.4 GB) — CTranslate2 converted NLLB model
├── METHODOLOGY.md         (this file)
├── PROJECT_REPORT.md
├── requirements.txt
└── README.md
```

## 7. Known Issues (Unresolved)

1. **Translation quality:** NLLB-200-distilled-600M produces hallucinated, structurally broken output for legal documents. The model lacks the capacity for accurate long-form legal translation. Root cause: model size (600M params) vs. task complexity (legal constitution translation).

2. **Visual quality:** While the upgraded builder adds font mapping, line-wrapping, background inpainting, and justification, the underlying text is still rendered per-word via `insert_text()` which may cause spacing artifacts in some PDF viewers.

3. **fasttext LID:** Requires manual download of `lid.176.ftz` (916 KB) — not included in the repository.

4. **CTranslate2 + Transformers dual load:** Both models are loaded simultaneously (~5 GB RAM). Could be optimized to skip Transformers load when using CT2 backend only.

5. **Background inpainting:** The 30px pixmap sampling provides a rough colour estimate. For documents with gradient backgrounds or images, white fallback is used.
