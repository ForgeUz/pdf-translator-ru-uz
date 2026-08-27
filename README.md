# pdf-translator-ru-uz

> **⚠️ Experimental / early-stage.** This is a research prototype where
> different translation and layout-preservation methods are being tested.
> Output quality is **not** production-grade and may contain errors,
> hallucinations, or structural corruption. See
> [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for a detailed quality audit.

Offline, CPU-only, **in-place** PDF translator for **Uzbek ↔ Russian** using
the **NLLB-200** sequence-to-sequence model. It preserves the original PDF
layout by redacting and redrawing text directly on the page — no Markdown/HTML
roundtrip, no external APIs, no GPU, no Docker, no Ollama.

## Features

- **In-place translation** — PDF → parse paragraphs → translate → redact + redraw → PDF.
- **No hallucination** — uses a seq2seq model (NLLB-200), not a generative LLM.
- **Fully offline** — no API calls; models are downloaded once and cached locally.
- **Math-aware** — detects math fonts (Cambria Math, STIX, LaTeX, etc.) and
  preserves math glyphs intact via private-use-Unicode masking.
- **Token-aware chunking** — splits text by *tokenized* length, not character
  count, to avoid truncating mid-sentence.
- **Caching** — SQLite MD5-keyed paragraph cache for fast re-runs.
- **Layout preservation** — span-level font/color/size tracking, background
  inpainting, line-wrapping, justification, and rotation preservation.
- **Verification & reporting** — mechanical checks (numeric preservation,
  length ratio, language bleed, n-gram repetition, LID) with CSV reports and
  PDF highlighting.

## Status

This is an **early-stage** project. The pipeline runs end-to-end, but
translation quality on real legal/technical documents is currently poor and
under active development. Different backends and methods are being evaluated
(see [`docs/ROADMAP.md`](docs/ROADMAP.md)).

## Installation

### Requirements

- **Python 3.10+**
- **~15 GB RAM** (NLLB-200-distilled-600M: ~2.5 GB model + working memory)
- **Linux** (font paths are hard-coded to `/usr/share/fonts/...`)

### Install

```bash
git clone https://github.com/ForgeUz/pdf-translator-ru-uz.git
cd pdf-translator-ru-uz
pip install -r requirements.txt
```

### System libraries (Debian/Ubuntu)

```bash
sudo apt-get install -y fonts-dejavu-core fonts-noto-core
```

## Usage

### CLI

```bash
python -m pdf_translator_ru_uz.pipeline \
  --input source.pdf \
  --output translated.pdf \
  --lang ru
```

Or via the installed console script:

```bash
pdf-translator-ru-uz -i source.pdf -o translated.pdf -l ru
```

### Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--input` | `-i` | Path to source PDF (required) |
| `--output` | `-o` | Path to write translated PDF (required) |
| `--lang` | `-l` | Target language: `ru` or `uz` (required) |
| `--src-lang` | | Source language; defaults to opposite of `--lang` |
| `--model` | | NLLB model id (default `facebook/nllb-200-distilled-600M`) |
| `--backend` | | `transformers` (default) or `ctranslate2` |
| `--model-path` | | Path to CTranslate2 converted model directory |
| `--batch-size` | | Batch size for length-sorted batching (default 8) |
| `--cache-db` | | SQLite cache path (default `{output}.cache.db`) |
| `--stream` | | Streaming mode: save intermediate PDF page-by-page |
| `--confidence-report` | | Write per-paragraph confidence scores to CSV |
| `-v` | | Debug logging |

### Python API

```python
from pdf_translator_ru_uz.pipeline import run_pipeline

run_pipeline(
    input_pdf="source.pdf",
    output_pdf="translated.pdf",
    target_lang="ru",
    src_lang="uz",
)
```

## Architecture

```
[input.pdf]
     │  PyMuPDF (fitz) dict-level extraction
     ▼
[parser.py]  → Paragraph(bbox, text, fontsize, is_math, spans) + tables
     │  reading-order sort, column detection, de-hyphenation,
     │  list-marker strip, drop-cap detection, math-font detection
     ▼
[engine.py]  → NLLB-200 translation
     │  NFKC normalization, fasttext LID, NER masking, math masking,
     │  sentence chunking, length-sorted batching, CT2 tuning,
     │  case restoration, Russian micro-typography
     ▼
[builder.py] → InPlaceBuilder
     │  background inpainting, font synthesis, line-wrapping,
     │  justification, rotation, image-aware rendering
     ▼
[output.pdf]  (same layout, translated text)
```

### Module responsibilities

| Module | Responsibility |
|--------|----------------|
| [`parser.py`](pdf_translator_ru_uz/parser.py) | PDF → `Paragraph` list with span metadata, tables, math spans |
| [`engine.py`](pdf_translator_ru_uz/engine.py) | Text → Text via NLLB, chunking, masking, batching, post-processing |
| [`cache.py`](pdf_translator_ru_uz/cache.py) | SQLite MD5-keyed translation cache |
| [`builder.py`](pdf_translator_ru_uz/builder.py) | In-place redact + redraw with layout preservation |
| [`pipeline.py`](pdf_translator_ru_uz/pipeline.py) | Orchestrator + CLI entry point |
| [`placeholders.py`](pdf_translator_ru_uz/placeholders.py) | Centralized private-use-Unicode placeholder registry |
| [`article_segmenter.py`](pdf_translator_ru_uz/article_segmenter.py) | Groups paragraphs into articles for sibling context |
| [`verification.py`](pdf_translator_ru_uz/verification.py) | Automated translation integrity checks |
| [`reporting.py`](pdf_translator_ru_uz/reporting.py) | CSV report + PDF highlighting of flagged segments |

## Backends

| Backend | Speed (CPU) | Quality | Notes |
|---------|-------------|---------|-------|
| `transformers` (default) | Slow | Good | Pure PyTorch, flexible |
| `ctranslate2` | 3–4× faster | Same | Requires converted model |

To use CTranslate2, convert the model once:

```bash
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir ./nllb_ct2_model
```

Then run with `--backend ctranslate2 --model-path ./nllb_ct2_model`.

## Testing

All tests use dependency injection (`translate_fn`) so no real model is
downloaded or loaded.

```bash
pytest -v
```

## Documentation

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — detailed methodology and quality audit
- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — project report and known issues
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — roadmap and optimization plans

## License

This project is licensed under the **GNU Affero General Public License v3**
(AGPLv3). See [`LICENSE`](LICENSE) and [`EULA.md`](EULA.md).

> **Important:** This project depends on **PyMuPDF**, which is also AGPLv3.
> If you need to distribute this software as **closed-source / proprietary**
> (e.g. to government or enterprise clients), you must obtain a commercial
> license from the PyMuPDF authors (Artifex). See [`EULA.md`](EULA.md) for
> details and permissive alternatives.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Run tests before submitting: `pytest -v --tb=short`.
4. Open a pull request.

## References

- [NLLB-200 (Meta)](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)