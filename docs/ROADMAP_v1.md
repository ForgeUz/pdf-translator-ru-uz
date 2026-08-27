# Roadmap v1 — Uzbek→Russian Offline PDF Translator

Status: current pipeline (parser.py + engine.py + builder.py + pipeline.py) produces
structurally and semantically corrupted output on real legal text (see
`METHODOLOGY.md` §1 audit). Two confirmed root-cause bugs found in `engine.py` (below).
This doc separates: (A) confirmed bugs to fix now, (B) architecture that must change,
(C) test/verification matrix required before any full-document run is trusted,
(D) what code is still needed to finish the audit.

Constraint: 100% local, CPU-only, no API calls, no GPU. All recommendations respect this.

---

## A. Confirmed bugs — fix before anything else (P0)

### A.1 `_ner_placeholder_for()` — collision + non-determinism
File: `engine.py`
```python
code = 0xE800 + (hash(text) & 0x7FF)   # 2048 slots, Python hash() is salted
```
- **Collision:** whole document → hundreds of entities → 2048-slot space guarantees
  collisions. `restore_ner()` replaces the placeholder char *globally*, so every
  occurrence of a colliding placeholder gets the wrong entity. This is the direct
  cause of "Bashkortostan / Kazakhstan" fabrication in the audit.
- **Non-determinism:** `hash(str)` is `PYTHONHASHSEED`-randomized per process by
  default. Methodology §4.4 claims this was fixed with "deterministic hash-based
  approach" — it was not; it's still `hash()`, not a stable digest. Cache keys
  computed on masked text (if any exist) would miss every run.
- **Dead branch:** `if original in mapping:` never true — mapping keys are
  placeholder chars, not original text. Harmless but remove.

**Fix:** stable digest (`hashlib.blake2b(text.encode(), digest_size=2)`), widen
placeholder space to ≥65536 slots, and add explicit collision detection: maintain
`orig→placeholder` dict, on new-text-different-placeholder-already-used bump digest
or linear-probe to next free slot. Add regression test with 3000+ synthetic distinct
entities asserting zero cross-contamination.

### A.2 `_LATIN_WORD_PATTERN` — wrong for Latin-script source
File: `engine.py`
```python
_LATIN_WORD_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]{1,}\b")
```
Source is `uzn_Latn` (Latin-script Uzbek). This regex matches **every** capitalized
word in the source — including ordinary Uzbek words, sentence-initial words, "Biz",
"Oʻzbekiston" — and masks them as NER, so they are never translated and get pasted
back verbatim into the Russian output. Adjacent Cyrillic + un-translated Latin
fragments is a direct contributor to the "fused word" symptom in the audit.

**Fix:** this pattern must not run at all when `src_lang == "uz"`. Replace with either
(a) disable Latin-proper-noun masking for Latin-script source entirely, or (b) require
ALL-CAPS acronym match (`\b[A-Z]{2,}\b`) only, or (c) a real NER model / gazetteer.
Given "no hallucination, no GPU" constraint, prefer (a)+(b): don't mask ordinary
capitalized words, only mask hard acronyms and known entity classes (dates, numbers,
emails, URLs — which are script-independent and already correctly patterned).

### A.3 Duplication / loop bug (header ×7, metadata phrase ×9)
Not yet located — needs `parser.py` `_sort_blocks_by_reading_order()` and
`pipeline.py` orchestrator to confirm. Hypothesis: a block is appended to more than
one column group, or the pipeline loop re-processes the same page/block without a
seen-set. **Send `parser.py` + `pipeline.py` to confirm** (see §D).

### A.4 Uzbek Latin special characters vs NFKC
File: `engine.py`, `normalize_text()`:
```python
return unicodedata.normalize("NFKC", text)
```
Uzbek Latin uses `oʻ`, `gʻ` — with U+02BB (MODIFIER LETTER TURNED COMMA) or
sometimes a right single quote U+2019 as the apostrophe-like character, not ASCII
`'`. NFKC does **not** normalize between U+02BB / U+2019 / U+0027 (they are
compatibility-distinct), so `oʻzbek` typed with one variant vs another will not
match, will break NER regexes that assume plain `\w`, and can break dehyphenation
and cache-key hashing consistency (`oʻzbek` vs `o'zbek` = different cache entries,
different words to the tokenizer/model). **This must be tested explicitly** — it's
invisible until you feed it inconsistent source PDFs.

**Fix:** add an explicit Uzbek-specific normalization step *before* NFKC: map
`{U+2019, U+0027, U+02BC} → U+02BB` (or whatever NLLB's `uzn_Latn` tokenizer was
trained on — verify against tokenizer vocab, don't assume) so the model always sees
one canonical form.

---

## B. Architecture-level changes (not bug fixes — structural)

### B.1 Chunking must follow document structure, not blind token budget
Current `SentenceChunker` splits purely on sentence-boundary regex + token budget,
with zero awareness of legal structure (`1-modda.`, `2-modda.`). This is why the
audit shows articles concatenated into unreadable blocks. **Change:** chunk at
structural boundaries first (article/section markers detected in `parser.py`, already
extracted as list markers per §1.6), then sentence-split only within an article if it
exceeds token budget. One article = one translation unit whenever it fits. This is a
correctness requirement for any legal/structured document, not an optimization.

### B.2 Terminology must be glossary-constrained, not left to the model
NLLB-200-distilled-600M has no domain adaptation. "Kuchga kirish sanasi" → "Дата
вступления на кучу" is not a hallucination in the scary sense — it's an
out-of-domain lexical failure a 600M general model will always have on legal
terminology. **A bigger model will reduce but not eliminate this.** The fix that
actually guarantees correctness for known terms, at zero GPU/latency cost:

- Build a **bilingual legal-term glossary** (uz→ru) for recurring fixed phrases
  ("kuchga kirish sanasi" → "дата вступления в силу", article/chapter headers,
  institution names — "Oliy Majlis" → "Олий Мажлис", not invented). Match glossary
  terms in `mask_ner`-style pre-pass (same placeholder mechanism, already built,
  just needs a second dictionary-driven mask type), substitute directly, skip the
  model for that span entirely. This is the single highest-leverage change for
  "no hallucination" on legal/institutional nouns and dates — it makes those spans
  deterministic instead of probabilistic.
- For everything else, the model translates; glossary terms never touch it.

### B.3 Model choice — evaluate, don't assume
`facebook/nllb-200-distilled-600M` is documented in Known Issues as the likely
capacity bottleneck for the genuine (non-bug) hallucinations (gibberish tokens).
Options within the no-GPU constraint, all CPU-CT2-int8 feasible:
- `nllb-200-distilled-1.3B` — ~2x the params, still runs int8 on CPU via CT2,
  slower (~2-3x latency) but meaningfully better in NLLB's own reported BLEU
  tables for lower-resource directions. **Must be benchmarked**, not assumed.
- Keep 600M + glossary constraint (B.2) + structural chunking (B.1) — cheapest,
  may already remove most of what the audit called "hallucination" since most of
  it was actually masking/chunking bugs, not the model inventing content.

**Required before deciding:** build a small (~30-50 sentence) uz→ru gold-reference
set from a *non-legal* and a *legal* Uzbek source, run both model sizes through the
fixed pipeline (bugs from §A resolved), score with chrF or manual review. Don't
scale to full documents until this benchmark exists — you don't currently know how
much of the audit's damage was model capacity vs pipeline bugs. My hypothesis:
majority was pipeline bugs (A.1–A.4). This needs to be verified, not assumed.

### B.4 Visual rendering — `insert_text()` per-word is the wrong tool
Known Issue #2 already flags this. Manual word-by-word `insert_text()` +
hand-rolled `_wrap_text()` / `_justify_line()` / `_estimate_text_width()` heuristics
is fragile and is exactly the kind of thing that causes "spacing artifacts."
**Change:** use PyMuPDF's built-in rich-text layout (`page.insert_htmlbox()` /
`fitz.Story` API, available in PyMuPDF ≥1.23) which does real text shaping, line
breaking, and justification internally — instead of reimplementing a text layout
engine by hand. This removes `_wrap_text`, `_justify_line`, `_estimate_text_width`
as a maintenance burden and removes an entire class of visual bugs at once. This is
a bigger rewrite of `builder.py` but is the correct fix for "visual structure" being
the stated priority.

### B.5 Table translation — unverified end-to-end
`_extract_tables()` translates each cell individually via `t.cells` / `t.extract()`.
No evidence in the audit or test list that a *rendered* (not just extracted) table
was ever visually verified post-translation — text expansion (Russian ~15-25%
longer than Uzbek on average) inside a fixed-width cell is a classic overflow
source. **Must test:** a real table-containing PDF (do you have one beyond
`mat_1_uz.pdf`? if that's a math doc, tables may be untested entirely) end-to-end,
visually inspect output.

### B.6 Math — validate the assumption, not just unit-test it
`mask_math`/`unmask_math` + `check_placeholder_survival` are unit-tested (12 tests)
against a tokenizer roundtrip, but placeholder survival through the *actual*
NLLB/CT2 beam search (repetition_penalty=1.2, no_repeat_ngram_size=3) on a
**full real math document** (`mat_1_uz.pdf` exists in your folder — use it) is not
confirmed. Repetition penalty in particular can suppress a placeholder char if it
appears multiple times close together (common in math-heavy text) — worth an
explicit test with a paragraph containing 3+ math placeholders.

---

## C. Test/verification matrix required before trusting full-document output

| # | Target | Test | Priority |
|---|--------|------|----------|
| 1 | `_ner_placeholder_for` | 3000+ distinct entities, assert zero collision cross-contamination | P0 |
| 2 | `_ner_placeholder_for` | same process run twice, same input → identical placeholders (determinism) | P0 |
| 3 | `mask_ner` w/ `src_lang=uz` | ordinary capitalized Uzbek words are NOT masked | P0 |
| 4 | `normalize_text` | `oʻzbek` (U+02BB) vs `o'zbek` (U+0027) vs `o'zbek` (U+2019) → same canonical form | P0 |
| 5 | parser + pipeline | full constitution run → no block/line duplicated in output | P0 |
| 6 | `SentenceChunker` | article marker (`N-modda.`) never split across two chunks | P1 |
| 7 | glossary mask (new) | fixed legal term always translates identically, model never touches it | P1 |
| 8 | `mask_math`/`unmask_math` | full run on `mat_1_uz.pdf`, all placeholders survive, none garbled | P1 |
| 9 | `_extract_tables` | real table PDF, rendered output visually inspected, no cell overflow | P1 |
| 10 | model benchmark | 600M vs 1.3B on gold set (legal + general), chrF score, decide | P1 |
| 11 | builder rewrite (B.4) | `insert_htmlbox` output vs old `insert_text` on same page, visual diff | P2 |
| 12 | end-to-end | `constitution_uz.pdf` full run post-fixes, manual read-through of entire output | P0 (final gate) |

No full-document production run should be treated as representative until #1–5 and #12 pass.

---

## D. What code is still needed to close the audit

Already reviewed: `engine.py`, `cache.py` (full).

Still needed, in priority order:
1. **`pipeline.py`** (554 lines) — orchestrator; needed to confirm/deny the A.3
   duplication-loop hypothesis (most likely location).
2. **`parser.py`** (767 lines) — specifically `_sort_blocks_by_reading_order()`,
   `_extract_tables()`, `_dehyphenate_line()` — needed to confirm A.3 alternate
   location, verify B.5 table cell mapping, and check dehyphenation doesn't
   interact badly with the apostrophe issue in A.4.
3. **`builder.py`** (789 lines) — needed before any B.4 rewrite decision; want to
   see current `_wrap_text`/`insert_text` call sites exactly.
4. A **sample of the raw extracted text** (pre-translation, post-parser) for one
   page of `constitution_uz.pdf` that contains a duplicated header — fastest way
   to confirm whether duplication originates in extraction or in the translate/build
   loop.

Not needed right now: test files (test_*.py) — useful later when writing new tests
per §C, not for bug-hunting the current failures.
