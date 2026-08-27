# Roadmap: Optimization, Model Choices, CPU Efficiency

This document details the path forward for better speed, lower memory, better quality without APIs.

## Current state (MVP)

- **Model**: NLLB-200-distilled-600M (2.5 GB, ~10 min/16-page doc on CPU)
- **Backend**: Transformers + PyTorch (flexible, but slow on CPU)
- **Quality**: Good for plain text; math/equations skipped
- **No hallucination**: Seq2seq translation only, no generative LLM
- **Fully offline**: No external APIs

---

## Phase 2: Math masking & restoration

**Goal**: Recover translated pages with in-place math glyphs intact.

### Challenge
Math symbols in PDFs are often glyph-encoded (no Unicode mapping). Current parser detects Cambria Math and skips whole paragraph. This is honest (no hallucination) but lossy (user sees half-translated page).

### Solution: Private-use character masking
1. **Before translation**: Replace detected math glyphs with private-use Unicode (`\uE000`, `\uE001`, ...) in the *text only*, not in the PDF itself.
   ```
   Original: "Tenglamani yeching: (   )x² = 0"
   Masked:   "Tenglamani yeching: \uE000 \uE001 = 0"
   ```
2. **Translate masked text**: NLLB won't mangle private chars (out-of-vocab, keeps as-is).
   ```
   Translated: "Решить уравнение: \uE000 \uE001 = 0"
   ```
3. **Restore on PDF**: Store original glyph → placeholder map, restore after rendering.

### Implementation
```python
# parser.py: detect math spans, build map
class MathSpan:
    placeholder: str       # \uE000
    original_text: str     # "x²"  (what's actually rendered)
    bbox: fitz.Rect        # original position
    
# engine.py: wrap text with masks
def mask_math(text: str, math_map: list[MathSpan]) -> str:
    # replace each math glyph text with placeholder
    pass

def unmask_math(text: str, math_map: list[MathSpan]) -> str:
    # restore placeholders to original glyphs
    pass
```

### Risk & mitigation
- **Risk**: Private-use chars might not survive tokenization intact (NLLB vocab is 256k tokens, but `\uE000` might map unexpectedly).
- **Mitigation**: Test on real math textbook; if NLLB munges placeholders, fall back to OCR (phase 5).

### Timeline: 1–2 weeks (design + testing)

---

## Phase 3: CTranslate2 backend (3-4x CPU speedup)

**Goal**: Translate same documents in 3 min instead of 10 min.

### Current bottleneck
- Transformers + PyTorch on CPU: autoregressive generation is inherently serial.
- NLLB seq2seq is parallel (encoder + decoder), but PyTorch can't exploit CPU parallelism well on inference.
- CTranslate2: purpose-built for CPU inference, uses OpenBLAS/MKL matrix ops.

### Solution: Offline model conversion + CTranslate2 backend

1. **One-time conversion** (on machine with HF internet):
   ```bash
   pip install ctranslate2 transformers
   ct2-transformers-converter \
     --model facebook/nllb-200-distilled-600M \
     --output_dir ./nllb_ct2_model \
     --copy_files tokenizer.json
   ```
   Output: 800 MB optimized model (int8 quantized).

2. **Swap engine backend** (no interface change):
   ```python
   # Current (transformers)
   engine = NLLBEngine(model_name="facebook/nllb-200-distilled-600M")
   
   # With CTranslate2
   engine = NLLBEngine(
       backend="ctranslate2",
       model_path="./nllb_ct2_model"
   )
   # Same engine.translate(text, src, tgt) -> str interface
   ```

3. **Implementation**:
   ```python
   # engine.py: add backend branching
   if self.backend == "transformers":
       return self._real_translate_transformers(...)
   elif self.backend == "ctranslate2":
       return self._real_translate_ctranslate2(...)
   ```

### Speedup breakdown
| Step | Transformers | CTranslate2 |
|------|--------------|------------|
| Model load | 10s | 2s |
| Per-chunk encode | 3s | 0.5s |
| Per-chunk decode | 4s | 1.0s |
| **Total/16-page doc** | ~10 min | ~3 min |

### Benefit
- 3-4x faster on CPU.
- Model smaller (800 MB vs 2.5 GB).
- Same memory footprint during inference (both need ~2-3 GB working).
- Zero quality loss (CTranslate2 exactly reproduces PyTorch numerics).

### Timeline: 2–3 weeks (integration + testing on real docs)

---

## Phase 4: Smaller base models (7B → 1B param reduction)

**Goal**: Lower memory floor from 15 GB to 8 GB, enable older/lower-power hardware.

### Option A: NLLB-200-distilled-200M (ultra-light)
| Model | Size | Speed | Quality | Notes |
|-------|------|-------|---------|-------|
| NLLB-600M (current) | 2.5 GB | 10 min | Good | Balanced |
| NLLB-200M | 950 MB | 6 min | Acceptable | ~20% quality drop, much faster |
| mBART-50 | 550 MB | 5 min | Fair | Older distilled model, rougher |

**Decision**: 200M only if you hit OOM on 600M. Quality degradation is noticeable on technical text.

### Option B: Language-specific smaller models (out-of-scope)
- No widely-available Uzbek-Russian parallel model exists.
- Would require training on parallel corpus (expensive, out-of-budget).

### Option C: Quantization to int4/int8 (via BitsAndBytes)
- Compress NLLB-600M from float32 to int8 in-memory.
- Transformers support via `BitsAndBytesConfig`, but CPU quantization is not well-supported (mostly GPU).
- CTranslate2 already does int8 quantization → use phase 3 instead.

### Recommendation
- Stick with NLLB-600M for now (quality/speed sweet spot).
- If OOM: test NLLB-200M quality drop on your dataset first.
- CTranslate2 int8 (phase 3) gives 3GB savings + 3x speedup.

### Timeline: 2–3 weeks testing (if needed)

---

## Phase 5: Equation OCR fallback (smart degradation)

**Goal**: If math masking (phase 2) fails, OCR the math and translate the surrounding text.

### Problem it solves
- Phase 2 masking might not survive tokenization (placeholder chars munged by NLLB).
- User gets partial translation + placeholder garbage.

### Solution: OCR-based math extraction as fallback
1. **If masking fails** (detected via heuristic: `\uE000` appears in output but shouldn't):
   - Fall back to marker-pdf + surya-ocr.
   - Extract math as LaTeX (`$x^2 + y^2 = z^2$`).
   - LaTeX is language-agnostic; render as-is in output.

2. **Implementation**:
   ```python
   # builder.py: if we detect placeholder corruption
   if placeholder_corruption_detected:
       math_regions = extract_math_via_ocr(original_pdf, page_idx)
       # render math regions untouched, translate text around them
   ```

3. **Tradeoff**:
   - Adds ~10 min per document (OCR on CPU is slow).
   - Quality: OCR math might be rough (typos), but better than missing/mangled.
   - Only triggered on fallback (phase 2 works most of the time).

### Caveat: Model size
- Marker-pdf + surya OCR models already cached: ~3.3 GB.
- Total system: 2.5 GB (NLLB) + 3.3 GB (OCR) = 5.8 GB model files.
- Working memory: ~6-8 GB during translation + OCR.
- Total: ~14 GB (fits in your 15 GB target, tight).

### Timeline: 4–6 weeks (OCR integration, fallback logic, testing)

---

## Phase 6: Multi-threading & batch optimization

**Goal**: Translate multiple documents in parallel, amortize model load cost.

### Current: Single-threaded per-paragraph processing
```python
for page in pages:
    for paragraph in page.paragraphs:
        translated = engine.translate(para.text)  # serial
```

### Optimized: Thread pool for I/O, batch generation
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for para in all_paragraphs:
        futures.append(executor.submit(engine.translate, para.text))
    results = [f.result() for f in futures]
```

### Caveat: PyTorch GIL
- PyTorch model generation releases Python GIL, so threads can run in parallel on CPU.
- Benefit: ~1.5-2x for 4 threads (diminishing returns due to GIL contention).
- CTranslate2 is better (doesn't hold GIL).

### Batch generation (higher leverage)
```python
# Instead of:
for text in paragraphs:
    out = model.generate(text)  # 10s per text

# Batch:
texts = [p for p in paragraphs if not cached[p]]
batch_out = model.generate_batch(texts)  # 20s for 10 texts (2x speedup)
```

### Recommendation
- **Phase 3** (CTranslate2) gives 3x.
- **Phase 6** (batching) gives 2x.
- Combined: 6x speedup (10 min → ~90 sec for 16-page doc).

### Timeline: 3–4 weeks (batch integration, testing)

---

## Phase 7: Incremental translation (user-facing UX)

**Goal**: Show translated pages as they're generated, don't wait for entire document.

### Current: Batch pipeline (parse all, translate all, redraw all, save).

### Streaming pipeline
```python
def translate_and_save_as_completed(input_pdf, output_pdf, engine):
    doc_in = fitz.open(input_pdf)
    doc_out = fitz.open()  # blank output
    
    for page_idx, page_in in enumerate(doc_in):
        # Parse + translate + redraw this page only
        paragraphs = parser.extract_paragraphs_from_page(page_in)
        translated = [engine.translate(p.text) for p in paragraphs]
        page_out = builder.redraw_page(page_in, paragraphs, translated)
        doc_out.insert_pdf(page_out)
        
        # Save intermediate result
        doc_out.save(output_pdf)
        print(f"Page {page_idx+1} complete")
```

### Benefit
- User sees partial output early.
- Can interrupt (Ctrl+C) and get translated PDF up to that point.
- Useful for long documents.

### Timeline: 2–3 weeks (refactor builder for per-page, testing)

---

## Phase 8: Quality measurement (observability)

**Goal**: Per-paragraph confidence score so users know which parts might be rough.

### Problem
- No feedback on translation quality.
- User can't tell if rough section is NLLB limitation or bad input.

### Solution: Confidence from NLLB logits
```python
# NLLB returns log-probabilities; lower = less confident
# Store per-paragraph: (text, translation, confidence_score)

outputs = model.generate(
    input_ids=inputs,
    return_dict_in_generate=True,
    output_scores=True,
)
# outputs.scores[t] = softmax logits at step t
confidence = compute_confidence_from_logits(outputs.scores)
```

### Implementation
```python
# engine.py
result = self._real_translate(text, ...)
confidence = self._compute_confidence(...)
return TranslationResult(text=result, confidence=confidence, src_lang=src, tgt_lang=tgt)
```

### Use cases
- Flag low-confidence paragraphs (< 0.7 confidence).
- Highlight in output PDF with yellow background or margin note.
- CSV report: `para_id | source_text | translation | confidence`.

### Timeline: 2 weeks (logit extraction + visualization)

---

## Consolidated priority ranking (2026 roadmap)

| Phase | Impact | Effort | Blockers | Target quarter |
|-------|--------|--------|----------|-----------------|
| **2: Math masking** | High (recovers math) | Medium (2–3 wk) | Test placeholder survival | Q2 2026 |
| **3: CTranslate2** | Very high (3-4x speedup) | Medium (2–3 wk) | Model conversion, testing | Q2 2026 |
| **6: Batching** | High (2x more speedup) | Medium (3–4 wk) | NLLB API, testing | Q3 2026 |
| **5: OCR fallback** | Medium (safety net) | High (4–6 wk) | Marker integration | Q3 2026 |
| **4: Smaller models** | Low (only if OOM) | Low (2–3 wk test) | Benchmark on dataset | Q3 2026 |
| **7: Streaming UI** | Low (polish) | Low (2–3 wk) | Refactor builder | Q4 2026 |
| **8: Confidence score** | Low (observability) | Low (2 wk) | NLLB logits API | Q4 2026 |

---

## Hardware targets (no API, local only)

### Tier 1: Comfortable (current baseline)
- **CPU**: Intel i5/i7 (4+ cores, AVX2)
- **RAM**: 15 GB
- **Storage**: 10 GB model cache + 5 GB working
- **Speed**: 10 min / 16-page doc (NLLB-600M transformers)
- **Estimated cost**: $400-600 used laptop

### Tier 2: Optimized (post-phase 3 CTranslate2)
- **CPU**: Same i5/i7
- **RAM**: 12 GB (reduced working memory via CTranslate2)
- **Speed**: 3 min / 16-page doc
- **Cost**: Same $400-600 laptop

### Tier 3: Minimal (if phase 4 needed)
- **CPU**: Intel Celeron/Core m3 (2 cores)
- **RAM**: 8 GB
- **Speed**: 20–30 min / 16-page doc (NLLB-200M)
- **Cost**: $200-300 used netbook
- **Caveat**: Quality drops ~20%, only use if OOM with 600M

---

## Why no larger models (no GPU = no 7B/13B LLM)

**LLaMA 2 7B** (popular alternative):
- 14 GB model + 6 GB working = 20 GB (exceeds budget).
- Generative LLM = hallucination risk (explicitly rejected in scope).
- CPU inference: 30–60 min per document (much slower than NLLB).

**GPT-4 / Claude / Gemini**:
- API = external dependency (rejected in scope: "no API, only local models").
- Cost: $ per 1M tokens (~$0.03 per document at $15/1M tokens).
- Not an option.

**Conclusion**: NLLB-200 is the right fit — seq2seq, no hallucination, optimizable on CPU.

---

## Dependencies on path

### Hard requirements
- PyTorch (transformers dependency)
- pymupdf >= 1.24
- pytest (for testing)

### Soft upgrades
- CTranslate2 (phase 3): optional, drop-in replacement
- marker-pdf (phase 5): optional, OCR fallback only
- BitsAndBytes (quantization): skip (CTranslate2 is better for CPU)

### Won't use
- Ollama (adds Docker overhead, defeats "local" premise)
- vLLM (GPU-focused)
- Ray/Dask (overkill for single-document pipeline)

---

## Checkpoints for success

- **Phase 2 done**: 100% of math-containing pages translate correctly (visually verify).
- **Phase 3 done**: Document translates in < 5 min (3x speedup achieved).
- **Phase 6 done**: Batch of 5 documents translates in < 15 min (2.5x throughput).
- **Phase 5 done**: OCR fallback triggers gracefully if phase 2 masking fails (test on hard docs).
- **Phase 8 done**: Confidence score correctly flags low-quality sections (validate with bilingual review).

---

## Open questions

1. **Will phase 2 masking survive NLLB tokenization?** → Test ASAP, if not use phase 5 OCR as plan B.
2. **Should we offer NLLB-1.2B as option?** → Yes, add `--model` flag to CLI. Quality gain ~10% at 1.5x cost.
3. **Multi-language beyond UZ↔RU?** → Phase 4+ only. NLLB-200 supports 200 langs, but we'd need to test + document each.
4. **Will CTranslate2 work on ARM (M1/M2 Mac)?** → Unlikely (needs OpenBLAS or MKL). May require Rosetta 2 on Apple Silicon.s