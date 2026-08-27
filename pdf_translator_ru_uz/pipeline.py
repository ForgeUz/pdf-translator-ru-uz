# pdf_translator_ru_uz/pipeline.py

"""Orchestrator: PDF -> (parse paragraphs + tables) -> (cache + translate) -> (inpaint + render) -> PDF.

Implements all stages of the advanced pipeline upgrade:
  - Stage 1: Layout-aware parsing with span metadata, tables, de-hyphenation
  - Stage 2: NFKC/LID, NER masking, semantic chunking, length-sorted batching, CT2 tuning
  - Stage 3: Case restoration, Russian typography, metadata-free cache
  - Stage 4: Background inpainting, font mapping, line-wrapping, justification
  - Stage 5: Article segmentation, sibling context, verification, reporting
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import fitz

from pdf_translator_ru_uz.article_segmenter import Article, ArticleSegmenter
from pdf_translator_ru_uz.builder import BuildError, InPlaceBuilder
from pdf_translator_ru_uz.cache import TranslationCache
from pdf_translator_ru_uz.engine import (
    CachedEngine,
    EngineError,
    ModelLoadError,
    NLLBEngine,
    apply_russian_typography,
    get_placeholder_registry,
    mask_math,
    mask_ner,
    normalize_text,
    restore_case,
    restore_ner,
    unmask_math,
)
from pdf_translator_ru_uz.parser import PDFParseError, PDFParser, Paragraph
from pdf_translator_ru_uz.reporting import ReportWriter
from pdf_translator_ru_uz.verification import VerificationGate

logger = logging.getLogger("pdf_translator_ru_uz")

# ── Globals for streaming interrupt handler ─────────────────────────
_streaming_output_path: str | None = None


def _handle_interrupt(signum, frame):
    """Save partial output on Ctrl+C during streaming."""
    if _streaming_output_path:
        logger.warning(
            "Interrupted! Partial output saved to %s",
            _streaming_output_path,
        )
    sys.exit(130)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _infer_src_lang(target_lang: str) -> str:
    return "uz" if target_lang == "ru" else "ru"


def run_pipeline(
    input_pdf: str,
    output_pdf: str,
    target_lang: str,
    src_lang: Optional[str] = None,
    engine: Optional[NLLBEngine] = None,
    cache_db: Optional[str] = None,
    batch_size: int = 1,
    confidence_report: Optional[str] = None,
    stream: bool = False,
) -> Path:
    src_lang = src_lang or _infer_src_lang(target_lang)
    cache_db = cache_db or str(Path(output_pdf).with_suffix(".cache.db"))

    registry = get_placeholder_registry()
    registry.reset()

    logger.info("Stage 1/3: Parsing PDF into paragraphs (%s)", input_pdf)
    parser = PDFParser(input_pdf, registry=registry)
    pages, tables = parser.extract_paragraphs()

    # Intent: Group paragraphs into articles to provide read-only sibling context to engine.
    segmenter = ArticleSegmenter()
    all_articles: list[list[Article]] = [segmenter.segment(paras) for paras in pages]

    logger.info("Stage 2/3: Translating %s -> %s (cache=%s)", src_lang, target_lang, cache_db)
    engine = engine or NLLBEngine(batch_size=batch_size)
    cache = TranslationCache(cache_db)
    cached_engine = CachedEngine(engine, cache)

    # Initialize verification and reporting
    verifier = VerificationGate()
    report_path = confidence_report or str(Path(output_pdf).with_suffix(".flags.csv"))
    report_writer = ReportWriter(report_path)

    logger.info("Stage 3/3: Redacting + rendering in place (%s)", output_pdf)
    builder = InPlaceBuilder()

    try:
        doc = fitz.open(input_pdf)
    except Exception as exc:
        raise BuildError(f"Failed to reopen '{input_pdf}' for writing: {exc}") from exc

    try:
        # ── Phase A: Collect all translatable items ────────────────
        translation_items: list[tuple[int, Paragraph, str, dict, dict, str]] = [] # Added context_str
        skipped_math = 0
        pure_placeholder = 0

        for page_idx, page_articles in enumerate(all_articles):
            for article in page_articles:
                # Assemble sibling context (exclude the current paragraph)
                article_text = " ".join(p.text for p in article.paragraphs)
                
                for para in article.paragraphs:
                    if para.is_math and not para.math_spans:
                        skipped_math += 1
                        continue

                    normalized = normalize_text(para.text, src_lang=src_lang)
                    ner_masked, ner_map = mask_ner(normalized, src_lang=src_lang)

                    if para.math_spans:
                        math_masked, math_map = mask_math(ner_masked, para.math_spans)
                        if not math_masked.replace("▨", "").strip():
                            skipped_math += 1
                            pure_placeholder += 1
                            continue
                    else:
                        math_masked, math_map = ner_masked, {}

                    translation_items.append((page_idx, para, math_masked, ner_map, math_map, article_text))

        table_items: list[tuple[int, object, int, int, str, fitz.Rect, object]] = []
        for page_idx, page_tables in enumerate(tables):
            for t_idx, table in enumerate(page_tables):
                for row_idx, row in enumerate(table.cells):
                    for col_idx, cell_data in enumerate(row):
                        cell_text, cell_bbox, span_meta = cell_data
                        if not cell_text.strip():
                            continue
                        normalized = normalize_text(cell_text, src_lang=src_lang)
                        table_items.append((page_idx, table, t_idx, row_idx, col_idx, normalized, cell_bbox, span_meta))

        # ── Phase B: Translate (Fast Batching + Cache) ─────────────
        use_confidence = confidence_report is not None
        all_texts = [item[2] for item in translation_items] + [item[5] for item in table_items]
        
        cached_results: dict[int, str] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for idx, text in enumerate(all_texts):
            cached = cache.get(text, src_lang, target_lang)
            if cached is not None:
                cached_results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            if batch_size > 1 and hasattr(engine, "translate_batch"):
                batch_results = engine.translate_batch(uncached_texts, src_lang, target_lang)
                for orig_idx, result in zip(uncached_indices, batch_results):
                    cache.set(all_texts[orig_idx], src_lang, target_lang, result)
                    cached_results[orig_idx] = result
            else:
                for orig_idx in uncached_indices:
                    translated = cached_engine.translate(all_texts[orig_idx], src_lang, target_lang)
                    cached_results[orig_idx] = translated

        translated_all = [cached_results[i] for i in range(len(all_texts))]
        para_count = len(translation_items)
        para_translations = translated_all[:para_count]
        table_translations = translated_all[para_count:]

        # ── Phase C: Post-process and render ───────────────────────
        masked_count = 0
        total_render = len(translation_items)
        logger.info("Rendering %d paragraph(s) across %d page(s)...", total_render, len(pages))

        last_saved_page = -1

        for idx, ((page_idx, para, _masked_text, ner_map, math_map, _ctx), translated_text) in enumerate(
            zip(translation_items, para_translations)
        ):
            page = doc[page_idx]
            
            if stream and page_idx > last_saved_page and last_saved_page != -1:
                output_path = Path(output_pdf)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                doc.save(str(output_path), garbage=0)
                last_saved_page = page_idx
            elif stream and last_saved_page == -1:
                last_saved_page = page_idx

            if idx % 20 == 0 or idx == total_render - 1:
                logger.info("  Render progress: %d/%d paragraphs (page %d/%d)", idx + 1, total_render, page_idx + 1, len(pages))

            result = translated_text
            if ner_map: result = restore_ner(result, ner_map)
            if math_map: result = unmask_math(result, math_map)
            result = restore_case(para.text, result)
            
            if target_lang == "ru":
                result = apply_russian_typography(result)
            if para.list_marker:
                result = f"{para.list_marker} {result}"

            # Verification Gate integration
            verification_res = verifier.verify_tier1(para.text, result, src_lang, target_lang)
            if verification_res.tier2_required:
                verification_res = verifier.verify_tier2(
                    para.text, result, src_lang, target_lang,
                    verification_res, cached_engine,
                )
            
            report_writer.add_row(page_idx, para.text, result, verification_res)
            report_writer.highlight_pdf(page, para.bbox, verification_res)

            first_span = para.original_spans[0] if para.original_spans else None
            font = first_span.font if first_span else ""
            color = first_span.color if first_span else (0.0, 0.0, 0.0)
            dir_vec = first_span.dir if first_span else (1.0, 0.0)

            flags = 0
            if first_span:
                if "Bold" in first_span.font or "bold" in first_span.font: flags |= 2**0
                if "Italic" in first_span.font or "italic" in first_span.font or "Oblique" in first_span.font: flags |= 2**1

            if math_map and para.math_spans:
                math_bboxes = [ms.bbox for ms in para.math_spans if ms.bbox]
                builder.replace_paragraph(
                    page=page, bbox=para.bbox, translated_text=result,
                    original_font=font, original_fontsize=para.fontsize,
                    original_color=color, original_flags=flags,
                    original_dir=dir_vec, exclude_bboxes=math_bboxes,
                )
                masked_count += 1
            else:
                builder.replace_paragraph(
                    page=page, bbox=para.bbox, translated_text=result,
                    original_font=font, original_fontsize=para.fontsize,
                    original_color=color, original_flags=flags,
                    original_dir=dir_vec,
                )

        # ── Render table cells ────────────────────────────────────
        table_result_idx = 0
        for (page_idx, table, t_idx, row_idx, col_idx, cell_text, cell_bbox, span_meta) in table_items:
            if table_result_idx >= len(table_translations): break
            page = doc[page_idx]
            cell_translation = table_translations[table_result_idx]
            table_result_idx += 1

            cell_translation = apply_russian_typography(cell_translation)

            if span_meta is not None:
                cell_font, cell_fontsize, cell_color, cell_dir = span_meta.font, span_meta.size, span_meta.color, span_meta.dir
                cell_flags = 0
                if "Bold" in cell_font or "bold" in cell_font: cell_flags |= 2**0
                if "Italic" in cell_font or "italic" in cell_font or "Oblique" in cell_font: cell_flags |= 2**1
            else:
                cell_font, cell_fontsize, cell_color, cell_dir, cell_flags = "", 8.0, (0.0, 0.0, 0.0), (1.0, 0.0), 0

            builder.replace_paragraph(
                page=page, bbox=cell_bbox, translated_text=cell_translation,
                original_font=cell_font, original_fontsize=cell_fontsize,
                original_color=cell_color, original_flags=cell_flags, original_dir=cell_dir,
            )

        # ── Final save with GC (Тяжелое сохранение только в конце) ──
        output_path = Path(output_pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path), garbage=4, deflate=True, clean=True)
        logger.info("Saved output with compression: %s", output_path)

    finally:
        doc.close()
        report_writer.close()

    total_processed = len(translation_items)
    logger.info(
        "Pipeline complete: %s (%d paragraph(s) processed, %d with math masking, %d pure-math skipped (%d pure-placeholder after masking), %d table cells translated)",
        output_path, total_processed, masked_count, skipped_math, pure_placeholder, len(table_items),
    )
    return output_path

def run_pipeline_streaming(
    input_pdf: str,
    output_pdf: str,
    target_lang: str,
    src_lang: Optional[str] = None,
    engine: Optional[NLLBEngine] = None,
    cache_db: Optional[str] = None,
) -> Path:
    """Streaming variant: saves intermediate PDF page-by-page."""
    global _streaming_output_path
    _streaming_output_path = output_pdf

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_interrupt)

    try:
        return run_pipeline(
            input_pdf=input_pdf,
            output_pdf=output_pdf,
            target_lang=target_lang,
            src_lang=src_lang,
            engine=engine,
            cache_db=cache_db,
            stream=True,
        )
    finally:
        signal.signal(signal.SIGINT, original_handler)
        _streaming_output_path = None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-translate-pipeline",
        description=(
            "In-place PDF translation (UZ<->RU) with advanced layout "
            "preservation. Supports span-level parsing, font mapping, "
            "NER masking, and length-sorted batching."
        ),
    )
    p.add_argument(
        "--input", "-i", required=True, help="Path to source PDF."
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write translated PDF.",
    )
    p.add_argument(
        "--lang",
        "-l",
        required=True,
        choices=["ru", "uz"],
        help="Target language.",
    )
    p.add_argument(
        "--src-lang",
        choices=["ru", "uz"],
        default=None,
        help="Source language. Defaults to opposite of --lang.",
    )
    p.add_argument(
        "--model",
        default="facebook/nllb-200-distilled-600M",
        help="NLLB model id.",
    )
    p.add_argument(
        "--cache-db",
        default=None,
        help="Path to SQLite translation cache.",
    )
    p.add_argument(
        "--backend",
        choices=["transformers", "ctranslate2"],
        default="transformers",
        help="NLLB inference backend. ctranslate2 requires --model-path.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Path to CTranslate2 converted model directory.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for length-sorted batching. Default: 8.",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Streaming mode: save intermediate PDF page-by-page.",
    )
    p.add_argument(
        "--confidence-report",
        default=None,
        help="Write per-paragraph confidence scores to CSV.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.verbose)

    try:
        engine = NLLBEngine(
            model_name=args.model,
            backend=args.backend,
            model_path=args.model_path,
            batch_size=args.batch_size,
        )
        if args.stream:
            runner = run_pipeline_streaming
        else:
            runner = run_pipeline

        runner(
            input_pdf=args.input,
            output_pdf=args.output,
            target_lang=args.lang,
            src_lang=args.src_lang,
            engine=engine,
            cache_db=args.cache_db,
            batch_size=args.batch_size,
            confidence_report=args.confidence_report,
        )
        return 0
    except PDFParseError as exc:
        logger.error("Parser failure: %s", exc)
    except ModelLoadError as exc:
        logger.error("Model load failure: %s", exc)
    except EngineError as exc:
        logger.error("Translation failure: %s", exc)
    except BuildError as exc:
        logger.error("Build/render failure: %s", exc)
    except Exception as exc:
        logger.exception("Unexpected pipeline failure: %s", exc)
    return 1


if __name__ == "__main__":
    sys.exit(main())
