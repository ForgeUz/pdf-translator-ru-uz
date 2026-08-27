# Roadmap v1 — Addendum (post code review: parser.py, pipeline.py, builder.py)

Confirms/denies A.3 hypothesis, adds new P0 findings from full source review. Read
with `ROADMAP_v1.md` — this does not repeat items already listed there.

---

## P0 — new critical findings

### E.1 Cache is completely bypassed in the default (batch) execution path
`pipeline.py`, `run_pipeline()`:
```python
cached_engine = CachedEngine(engine, cache)
...
if batch_size > 1 and hasattr(engine, "translate_batch"):
    batch_results = engine.translate_batch(all_texts, src_lang, target_lang)
    translated_all = batch_results
else:
    for text in all_texts:
        translated = cached_engine.translate(text, src_lang, target_lang)
```
CLI default is `--batch-size 8`. Batch mode calls `engine.translate_batch()` directly
— `cached_engine` is built and then never used. No `cache.get()`, no `cache.set()`
anywhere in the batch branch. Every pipeline run, on every document, re-translates
100% of content from scratch through NLLB, regardless of prior runs. This directly
contradicts the cache's own docstring purpose ("enables instant re-rendering on
layout changes") and is the single largest CPU/time waste in the whole project —
worse than any model-size choice, since it means *every* iteration during
development (tuning `builder.py`, re-testing rendering) re-pays the full translation
cost. This is the top-priority fix for "maximal effective use of CPU."

**Fix:** route the batch path through the cache too — split `all_texts` into
cached/uncached before batching, only send uncached texts to `translate_batch()`,
write results back with `cache.set()`, merge. Cache must be consulted regardless
of `batch_size`.

### E.2 Math and NER placeholder ranges overlap (documented as safe, actually aren't)
- `parser.py`: `_PLACEHOLDER_COUNTER = iter(range(0xE000, 0xF000))` — math
  placeholders span **U+E000–U+EFFF** (4096 codepoints).
- `engine.py` comment claims: *"U+E000..U+E7FF: math placeholders ... U+E800..U+EFFF:
  NER placeholders — no overlap."* But NER placeholders (`_ner_placeholder_for`) are
  computed as `0xE800 + (hash(text) & 0x7FF)` → also land in U+E800–U+EFFF.
  Math's actual range (E000–EFFF) fully **contains** NER's range. Any document with
  more than ~2048 math spans (a real math textbook will hit this) starts allocating
  math placeholders inside the exact codepoint space NER also uses — on top of the
  A.1 NER-internal collision bug already found. Two independent, uncoordinated
  placeholder allocators sharing one codepoint space is a correctness bug by
  construction, not just under adversarial conditions.

**Fix:** one single placeholder manager for the whole pipeline (math + NER + LaTeX,
which already separately uses **U+F000–U+FFFF** in `protect_latex_blocks` — a third
uncoordinated allocator). Partition the Private Use Area explicitly and centrally:
e.g. math = E000–E7FF (2048), NER = E800–EFFF (2048, needs collision-safe hashing
per A.1), LaTeX = F000–F7FF. Enforce this in one module, not scattered across three.

### E.3 Math placeholder counter is a module-level global — will crash on real books
```python
_PLACEHOLDER_COUNTER = iter(range(0xE000, 0xF000))  # module level, never reset
```
This is a plain Python iterator created once at import time. It is never reset
per-document, per-page, or even per `PDFParser` instance. Two consequences:
1. **Hard cap of 4096 math spans per process lifetime**, not per document. A single
   math-heavy textbook (your literal stated goal — "translate math books") can
   plausibly contain thousands of inline formulas; once exhausted, `next()` raises
   `StopIteration` and the entire pipeline crashes with no graceful degradation.
2. Processing multiple PDFs in one Python process (batch-converting a folder,
   running tests, a notebook session) shares the same counter across documents —
   document 2 starts wherever document 1 left off, silently shrinking its available
   placeholder budget.

**Fix:** placeholder allocation must be instance-scoped (reset at the start of
`extract_paragraphs()`) and must raise a clear, catchable error *before* silently
corrupting output if a document's math-span count exceeds the allocated range —
never let it hit a bare `StopIteration` deep in paragraph building.

### E.4 Math detection = one specific font name. This is the actual blocker for "math books"
```python
MATH_FONT_MARKERS = ("cambriamath",)
```
`is_math_font()` only recognizes Cambria Math. Any PDF using a different math font —
STIX Two Math, Latin Modern Math, Asana Math, XITS Math, or (very common in
Uzbek-authored textbooks and OCR'd PDFs) formulas typeset with a plain italic serif
font and Unicode math symbols instead of a dedicated math font — will not be flagged
as math at all. Every such formula goes straight into NLLB as ordinary text and gets
"translated," i.e. corrupted, exactly the failure mode the whole masking system
exists to prevent. **This is not an edge case for your stated use case — it's the
main path.** You have `mat_1_uz.pdf` sitting in the project folder specifically —
before anything else in this addendum, extract its actual font names
(`page.get_text("dict")` → per-span `font`) and confirm whether Cambria Math is even
present. If not, math masking is currently a no-op on your own test file.

**Fix, in order of robustness vs effort:**
1. Widen `MATH_FONT_MARKERS` to cover the common math font family names (cheap,
   partial).
2. Add a **Unicode-range heuristic** independent of font name: spans containing a
   high density of Mathematical Operators (U+2200–U+22FF), Mathematical Alphanumeric
   Symbols (U+1D400–U+1D7FF), Latin Extended used for variable italics, superscript/
   subscript digits, or fraction slashes — flag as math regardless of font. This
   catches "plain italic + Unicode symbols" formulas that no font-name check ever
   will.
3. For genuinely embedded formula images (many math PDFs render equations as
   images, not text) — these are already untouched by the text pipeline, which is
   correct (see F.1), but confirm this is actually happening on your PDFs and not
   silently dropping equation images during redaction.

### E.5 Table cells carry zero font/size/color metadata
`parser.py` `Table.cells` is `list[list[tuple[str, fitz.Rect]]]` — text and bbox
only. Nothing captures the cell's original font, size, or color the way `Paragraph`
does via `original_spans`. `pipeline.py` then renders every cell with a hardcoded
default:
```python
builder.replace_paragraph(..., original_font="", original_fontsize=8.0,
                           original_color=(0.0, 0.0, 0.0), ...)
```
Every table in every document renders in fixed 8pt black DejaVu Sans, regardless of
the source table's actual styling (header row bold, larger size, colored cells,
etc.). This directly contradicts your stated priority ("visual structure saved").

**Fix:** extend table cell extraction to also pull first-span metadata per cell
(same `TextSpan` extraction already written for paragraphs — reuse it), thread it
through to `replace_paragraph()` instead of the hardcoded defaults.

---

## P1 — efficiency & correctness

### E.6 `--confidence-report` is silently dead in the default path
```python
use_confidence = confidence_report is not None   # computed, never read again
...
if confidence_report:
    logger.info("Confidence report omitted for batched pipeline.")
```
The flag exists, is documented in `--help`, computes nothing, and always no-ops
under the default `batch_size=8` path. This matters specifically for your
"no hallucination" goal: `compute_confidence_from_logits` + `TranslationResult`
already exist and are unit-tested — they are the one built-in mechanism for
*automatically flagging* low-confidence (likely hallucinated) spans for review, and
it is currently unreachable in production use. Either wire it into the batch path
(CT2's `translate_batch` can return score info per hypothesis — check API) or
remove the flag so it doesn't imply a capability that doesn't exist.

### E.7 Streaming mode saves+cleans the whole PDF after every single paragraph
```python
if stream:
    doc.save(str(output_path), garbage=4, deflate=True, clean=True)
```
This is inside the per-paragraph render loop. `garbage=4` (full garbage collection)
+ `deflate` + `clean` is the most expensive save mode PyMuPDF offers, and it runs
once per paragraph — for a 300-paragraph document that's 300 full recompressions
of the entire (growing) PDF, dominating total runtime for no benefit beyond crash
recovery. **Fix:** save every N paragraphs or once per page, and use a cheap save
(`garbage=0`) for intermediate checkpoints, reserving the expensive `garbage=4`
pass for the final save only.

### E.8 Whole-document monolithic translate-then-render is the wrong shape for "lightweight"
Current structure: parse *all* pages → translate *all* paragraphs across the *entire*
document in one batch call → only then render anything. For a multi-hundred-page
book this means zero output until 100% of translation finishes, peak memory holding
every paragraph and its translation simultaneously, and — combined with E.1 (no
cache in batch mode) — a crash on page 250 discards all prior translation work with
nothing salvageable. **Restructure to page-level or chunk-level pipelining:**
parse page → translate page (through cache) → render page → commit cache + cheap
save → next page. This is both more CPU/memory-efficient and makes partial
progress durable, which matters more as document size grows toward "math book"
scale.

### E.9 Justification bug: last line of a paragraph gets stretched
`replace_paragraph` always passes `justify=True`; `draw_text_multiline` then applies
`_justify_line` to **every** wrapped line including the last one, with no
last-line exclusion:
```python
lines = [self._justify_line(ln, bbox.width, ...) for ln in lines]
```
Standard typography never justifies a paragraph's final line (it's usually short;
justifying it produces visibly huge word gaps). Cheap, visible fix: skip
`_justify_line` on `lines[-1]`.

---

## P2 — worth doing, lower urgency

- **`_estimate_text_width` heuristic vs real glyph metrics.** Known Issues §4.3
  claims `fitz.get_text_length()` can't use custom TTF fonts — true for that
  specific function, but PyMuPDF's `fitz.Font(fontfile=...).text_length(text,
  fontsize)` (a different API, available in reasonably recent PyMuPDF) measures
  actual glyph widths for arbitrary embedded fonts. If available in your installed
  version, this removes an entire class of wrap/fit heuristic error (including the
  Cyrillic-width approximation, which currently treats all Cyrillic letters as one
  width regardless of е vs ш/щ/ю) — worth a quick spike before trusting `_wrap_text`
  on anything visually important.
- **Dead/redundant code, low risk but worth cleaning while in this file:**
  `_sort_blocks_by_reading_order`'s `cols_in_group` filter is always true (column
  membership already guaranteed by construction); `_compute_background_color`'s
  internal 0.5 image-overlap threshold is unreachable because the caller's 0.1
  threshold always gates first.
- **Multi-column spanning-block reclassification** (`avg_col_w` computed from
  possibly single-block columns) can misclassify ordinary blocks as "spanning" on
  genuinely multi-column pages with sparse columns — not urgent since your primary
  test document (constitution) is very likely single-column and this path never
  fires (`if len(separators) < 1: return sorted(...)` early return), but flag for
  when you test a two-column source.

---

## Structural recommendation: one placeholder authority

Right now there are **three independent, uncoordinated private-use-Unicode
allocators**: math (`parser.py`, sequential, module-global), NER (`engine.py`,
hash-based, collision-prone), LaTeX (`engine.py` `protect_latex_blocks`, sequential,
also module-global, also never reset). This is the common thread behind E.2, E.3,
and A.1 — three separate ad-hoc systems solving the same problem (protect a span
from translation) independently, each with its own bug. Before fixing them
individually, consider collapsing into one `PlaceholderRegistry` class:
instance-scoped (reset per document), owns the full PUA partition, exposes
`mask(text, kind)` / `unmask(text)`, guarantees no cross-kind collision by
construction, raises a clear error on exhaustion instead of crashing. This is more
work upfront but removes an entire category of future bugs in one place instead of
three, which matters for a project meant to handle math books at scale.

---

## Answering your efficiency ask directly

The two highest-leverage changes for "don't waste CPU/tokens":
1. **E.1 (cache bypass)** — fix this first. It affects every single run, not just
   math-heavy or table-heavy documents. Nothing else in this addendum matters for
   CPU cost until re-runs stop re-translating everything from scratch.
2. **E.8 (page-level pipelining)** — restructure so translation, rendering, and
   cache commits happen incrementally per page, not as two monolithic
   parse-everything / translate-everything / render-everything phases. This bounds
   memory, makes progress durable, and is the correct shape for scaling to
   book-length documents.

Everything else (E.2–E.7, E.9) is correctness/visual-fidelity, not efficiency —
important, but secondary to the two above for the "smart structure" ask.
