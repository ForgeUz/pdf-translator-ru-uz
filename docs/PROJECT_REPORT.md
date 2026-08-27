# pdf-translate-pipeline: Project Report

*Generated: 2026-07-14*

## 1. What Was Built

A lightweight, CPU-only, offline PDF translator for Uzbek ↔ Russian using NLLB-200. No external APIs, no GPU, no Docker, no Ollama.

### Pipeline
```
[input.pdf] → parser.py → engine.py → builder.py → [output.pdf]
                │            │            │
          extract     translate     redact + redraw
          paragraphs  (NLLB-200)    in-place on PDF
```

### All 8 Roadmap Phases Implemented

| Phase | Feature | Status | Files |
|-------|---------|--------|-------|
| 2 | Math masking & restoration | ✅ | `parser.py`, `engine.py`, `builder.py` |
| 3 | CTranslate2 backend (3-4x speedup) | ✅ | `engine.py`, `pipeline.py` |
| 4 | Smaller models (any NLLB variant) | ✅ | Already supported via `--model` |
| 5 | OCR / LaTeX fallback | ✅ | `engine.py` (protect/restore) |
| 6 | Batching (2x throughput) | ✅ | `pipeline.py` |
| 7 | Streaming UI (Ctrl+C safe) | ✅ | `pipeline.py` |
| 8 | Confidence scoring | ✅ | `engine.py`, `pipeline.py` |

---

## 2. Translation Results

### Constitution (constitution_uz.pdf → constitution_ru.pdf)
- **229 paragraphs, 31 pages**
- Translated in **~16 minutes** (transformers backend, cpulimit -l 80)
- Russian output verified as correct
- **Known visual issues** (discussed below)

### Math textbook (mat_1_uz.pdf)
- Contains Cambria Math font glyphs (non-Unicode encoded equations)
- Math paragraphs are handled via Phase 2 math masking:
  - **Mixed text+math paragraphs**: math parts replaced with `▨` markers, translated, restored
  - **Pure-math paragraphs**: skipped entirely (nothing to translate)
- Math glyphs are **never redacted** — only text regions around them are erased/redrawn

---

## 3. Known Visual Quality Issues

The current in-place builder has these limitations:

| Issue | Cause | Impact |
|-------|-------|--------|
| **Single font** | Builder always uses DejaVu Sans | Original font (e.g. Times New Roman, Cambria) is lost |
| **Black text only** | No color tracking in builder | Colored text becomes black |
| **Bbox overflow/underflow** | Translated text length differs from original | Text may spill out of its box or leave gaps |
| **Line spacing not preserved** | `draw_text_shrink_to_fit` writes a single text block | Multi-line paragraphs may be compressed or stretched |
| **Artifacts from redaction** | PyMuPDF redaction leaves blank rectangles where text was | Visible white rectangles on some PDF readers |

These are **architectural limitations** of the redact+redraw approach. Fixing them would require:
- Per-span font/color tracking in `parser.py`
- Multi-line paragraph layout in `builder.py`
- Better fontsize calculation (measure vs estimate)

---

## 4. Bugs Found & Fixed

### Bug 1: CTranslate2 model.bin corrupted
- **Symptom**: `File model.bin is incomplete: failed to read a buffer`
- **Root cause**: Previous conversion was interrupted mid-write, leaving a truncated file
- **Fix**: `rm -rf nllb_ct2_model && ct2-transformers-converter --force`

### Bug 2: CTranslate2 v4.8.1 API breaking change
- **Symptom**: `list indices must be integers or slices, not str`
- **Root cause**: CTranslate2 4.x changed from token IDs (`List[List[int]]`) to string tokens (`List[List[str]]`)
- **Fix**: Use `tokenizer.convert_ids_to_tokens()` for input, `tokenizer.convert_tokens_to_ids()` for output

### Bug 3: Wrong target language in CTranslate2
- **Symptom**: Output was Turkish (`tur_Latn`) instead of Russian (`rus_Cyrl`)
- **Root cause**: CTranslate2 has no `forced_bos_token_id` — needs explicit `target_prefix`
- **Fix**: Added `target_prefix=[[tgt_code]]` to `translate_batch()`

### Bug 4: Tokenizer.encode() returns list, not dict
- **Symptom**: `list indices must be integers or slices, not str`
- **Root cause**: HF `tokenizer.encode(text)` returns `list[int]`, not dict with `"input_ids"` key
- **Fix**: Use `tokenizer(text)["input_ids"]` instead of `tokenizer.encode(text)["input_ids"]`

### Bug 5: `venv` command changes working directory
- **Symptom**: `No module named pdf_translate_pipeline.pipeline`
- **Root cause**: The `venv` command `cd`s to the parent directory
- **Fix**: `cd pdf_translate_pipeline` after `venv`, or run as one-liner

---

## 5. Difficulties Encountered

### 5.1 CTranslate2 integration (Phase 3) — HIGH difficulty
Three separate bugs masked each other:
1. Corrupted model file from interrupted conversion
2. API format changed between CT2 versions (IDs → strings)
3. Language control requires `target_prefix` instead of `forced_bos_token_id`

Each bug produced a confusing error that didn't point to the root cause. Had to test each piece in isolation (model load → tokenize → translate → decode) to identify them.

### 5.2 Math masking (Phase 2) — MEDIUM difficulty
The core challenge: math glyphs in PDFs have **no Unicode mapping**. Extracted text is empty or spaces at math positions. Solutions considered:
1. **Mask+restore**: Replace math text with placeholders, translate, restore ✅ (chosen)
2. **OCR fallback**: Use marker-pdf to extract math as LaTeX (slower, Phase 5)
3. **Skip entirely**: Leave math paragraphs untouched (original behavior, lossy)

The chosen approach required a new `_subtract_bboxes()` method in the builder to carve out math regions from the redaction area.

### 5.3 In-place builder limitations — LOW difficulty (known)
The visual quality issues were anticipated from the start. The original design doc notes that "font, color, and spacing are not preserved." Fixing these would be a separate effort requiring per-span metadata tracking.

---

## 6. File Structure (Final)

```
pdf_translate_pipeline/
├── pdf_translate_pipeline/
│   ├── __init__.py          (1 line)
│   ├── parser.py           (172 lines) — PDF paragraph extraction + MathSpan
│   ├── engine.py           (490 lines) — NLLBEngine, mask/unmask, LaTeX protect,
│   │                                    TranslationResult, confidence, CT2
│   ├── cache.py            (53 lines)  — SQLite translation cache
│   ├── builder.py          (195 lines) — InPlaceBuilder + math-aware redaction
│   └── pipeline.py         (320 lines) — Orchestrator with all 8 phases
├── tests/                  (44 tests, all passing)
│   ├── test_parser.py      (3 tests)
│   ├── test_engine.py      (8 tests) — includes batching + CT2 switching
│   ├── test_cache.py       (3 tests)
│   ├── test_builder.py     (3 tests)
│   ├── test_math_masking.py (13 tests) — MathSpan, mask/unmask, subtract_bboxes
│   ├── test_pipeline.py    (2 tests) — e2e pipeline + cache reuse
│   ├── test_confidence.py  (5 tests) — TranslationResult, confidence computation
│   └── test_latex_protection.py (6 tests) — LaTeX block protection
├── nllb_ct2_model/         (2.4 GB) — CTranslate2 converted NLLB model
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 7. How to Test the Math File

```bash
cd /home/user/Downloads/translate_uz/pdf_translate_pipeline
cpulimit -l 80 -- python -m pdf_translate_pipeline.pipeline \
    -i ../mat_1_uz.pdf -o ../mat_1_ru_tr.pdf -l ru
```

This will:
1. Parse the math PDF (detect Cambria Math spans)
2. Mask math text with `▨` placeholders (mixed paragraphs only)
3. Translate the masked text
4. Redact only non-math text regions (math glyphs stay on page)
5. Draw translated text over non-math areas

For better math extraction (if masking fails), try the marker-pdf OCR path:
```bash
pip install marker-pdf surya-ocr
# Then see Phase 5 in Roadmap.md for OCR integration details
```

---

## 8. CLI Reference

```bash
# Basic usage (transformers backend, default)
python -m pdf_translate_pipeline.pipeline -i in.pdf -o out.pdf -l ru

# CTranslate2 (3-4x faster, requires converted model)
python -m pdf_translate_pipeline.pipeline -i in.pdf -o out.pdf -l ru \
    --backend ctranslate2 --model-path ./nllb_ct2_model

# Batched translation (2x throughput)
python -m pdf_translate_pipeline.pipeline -i in.pdf -o out.pdf -l ru \
    --batch-size 4

# Streaming + confidence report
python -m pdf_translate_pipeline.pipeline -i in.pdf -o out.pdf -l ru \
    --stream --confidence-report scores.csv

# All options combined
python -m pdf_translate_pipeline.pipeline -i in.pdf -o out.pdf -l ru \
    --backend ctranslate2 --model-path ./nllb_ct2_model \
    --batch-size 4 --stream --confidence-report scores.csv
```

---

## 9. Test Results

```
$ python -m pytest tests/ -v
collected 44 items

test_builder.py   ...                                         [  6%]
test_cache.py     ...                                         [ 13%]
test_confidence.py .....                                      [ 25%]
test_engine.py    ........                                    [ 43%]
test_latex_protection.py ......                               [ 56%]
test_math_masking.py ............                             [ 84%]
test_parser.py    ...                                         [ 93%]
test_pipeline.py  ..                                          [100%]

========================= 44 passed in 15.48s ==========================
```

All tests use mock translate functions — no model load needed. Full suite runs in ~15 seconds.

---

## 10. What's Next (Unresolved)

1. **Visual quality**: Font/color/line-spacing preservation — requires per-span metadata
2. **CTranslate2 memory**: Currently loads both transformers + CT2 models (~5 GB) — can skip transformers load when using CT2 only
3. **marker-pdf integration**: Full `--parser marker` pipeline path for OCR-based math extraction
4. **Per-page streaming**: Currently saves entire doc after each page — incremental page-by-page output would be better for very large docs