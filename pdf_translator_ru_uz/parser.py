# pdf_translator_ru_uz/parser.py

"""Module: Advanced span-level PDF parser with layout-aware extraction.

Extracts paragraphs, tables, and span metadata from PDF pages using
PyMuPDF's dict-level text extraction. Implements reading-order sorting,
column detection, header/footer/watermark isolation, de-hyphenation,
list-marker stripping, and drop-cap detection.
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz

from pdf_translator_ru_uz.placeholders import (
    PlaceholderRegistry,
    PlaceholderExhaustedError,
)

logger = logging.getLogger(__name__)

# Math font markers — widened beyond Cambria Math per E.4 fix
MATH_FONT_MARKERS = (
    "cambriamath",
    "stix",
    "latinmodernmath",
    "asanamath",
    "xitsmath",
    "cmmi",      # Computer Modern Math Italic (common in LaTeX PDFs)
    "cmex",      # Computer Modern Math Extension
    "cmsy",      # Computer Modern Math Symbols
    "msam",      # AMS math symbols
    "msbm",      # AMS math symbols
    "eufm",      # Euler Fraktur (math)
    "eurb",      # Euler Roman (math)
    "esint",     # ESINT (integral signs)
    "rsfs",      # Ralph Smith's Formal Script (math)
    "math",      # Generic "math" in font name
)

# Unicode ranges with high density of mathematical characters
# Used as a font-name-independent heuristic for math detection.
_MATH_UNICODE_RANGES = (
    (0x2200, 0x22FF),    # Mathematical Operators
    (0x2A00, 0x2AFF),    # Supplemental Mathematical Operators
    (0x27C0, 0x27EF),    # Miscellaneous Math Symbols A
    (0x2980, 0x29FF),    # Miscellaneous Math Symbols B
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    (0x2100, 0x214F),    # Letterlike Symbols (ℕ, ℝ, ℂ, etc.)
    (0x2070, 0x209F),    # Superscripts and Subscripts
    (0x2030, 0x205E),    # General Punctuation (‰, ⁂, etc.)
    (0x00B0, 0x00BF),    # Degree, plus/minus, superscripts
    (0xFB00, 0xFB4F),    # Alphabetic Presentation Forms (ﬁ, ﬂ)
)


class PDFParseError(RuntimeError):
    pass


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class TextSpan:
    """Per-span metadata extracted from a PyMuPDF line dict."""

    text: str
    font: str
    size: float
    color: tuple[float, float, float]
    bbox: fitz.Rect
    matrix: tuple[float, float, float, float, float, float]
    dir: tuple[float, float]  # writing direction
    is_math: bool = False
    opacity: float = 1.0


@dataclass
class MathSpan:
    """A single math glyph span detected on a PDF page.

    ``original_text`` is whatever text PyMuPDF extracted from the glyph
    (often empty, spaces, or garbled chars for non-Unicode math fonts).
    The ``placeholder`` is a private-use Unicode char that replaces it
    in the text sent to the translator.
    """

    placeholder: str
    original_text: str
    bbox: fitz.Rect


@dataclass
class Table:
    """Isolated table extracted via ``page.find_tables()``.

    Each cell is stored as a ``(text, bbox, span_meta)`` tuple where
    ``span_meta`` is an optional ``TextSpan`` with the cell's original
    font, size, and colour metadata.  When ``span_meta`` is ``None``,
    the pipeline falls back to default styling (E.5 fix).
    """

    bbox: fitz.Rect
    cells: list[list[tuple[str, fitz.Rect, TextSpan | None]]]
    page_number: int

    @property
    def num_rows(self) -> int:
        return len(self.cells)

    @property
    def num_cols(self) -> int:
        return max(len(row) for row in self.cells) if self.cells else 0


@dataclass
class Paragraph:
    """A clustered paragraph with full span metadata."""

    bbox: fitz.Rect
    text: str
    fontsize: float
    font: str = ""
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_dir: tuple[float, float] = (1.0, 0.0)
    is_math: bool = False
    page_number: int = 0
    math_spans: list[MathSpan] = field(default_factory=list)
    list_marker: str | None = None
    is_drop_cap: bool = False
    # Per-span original metadata for font/color/rotation reconstruction
    original_spans: list[TextSpan] = field(default_factory=list)


# ── PDFParser ────────────────────────────────────────────────────────


class PDFParser:
    """Layout-aware PDF parser with advanced heuristics.

    Output: ``pages`` (list of Paragraph per page) and ``tables`` (list of
    Table per page).
    """

    LINE_GAP_FACTOR = 1.2

    def __init__(
        self,
        input_path: str | Path,
        registry: Optional[PlaceholderRegistry] = None,
    ):
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise PDFParseError(f"Input PDF not found: {self.input_path}")
        if self.input_path.suffix.lower() != ".pdf":
            raise PDFParseError(f"Not a PDF file: {self.input_path}")
        self._registry = registry or PlaceholderRegistry()

    # ── Public API ──────────────────────────────────────────────────

    def extract_paragraphs(
        self,
    ) -> tuple[list[list[Paragraph]], list[list[Table]]]:
        """Parse the PDF and return (per-page paragraphs, per-page tables).

        Returns:
            A 2-tuple:
            - ``paragraphs[n]``: list of Paragraph on page *n*
            - ``tables[n]``: list of Table on page *n*
        """
        # Reset placeholder registry for this document (E.3 fix)
        self._registry.reset()
        try:
            doc = fitz.open(str(self.input_path))
        except Exception as exc:
            raise PDFParseError(
                f"Failed to open '{self.input_path}': {exc}"
            ) from exc

        try:
            all_pages_paras: list[list[Paragraph]] = []
            all_pages_tables: list[list[Table]] = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_tables = self._extract_tables(page, page_idx)
                table_bboxes = [t.bbox for t in page_tables]

                lines = self._extract_lines(page, table_bboxes)
                paragraphs = self._cluster_lines(lines, page_idx)

                all_pages_paras.append(paragraphs)
                all_pages_tables.append(page_tables)

        finally:
            doc.close()

        if not all_pages_paras or all(
            len(p) == 0 for p in all_pages_paras
        ):
            raise PDFParseError(f"Extracted no text from: {self.input_path}")

        total = sum(len(p) for p in all_pages_paras)
        logger.info(
            "Extracted %d paragraph(s) across %d page(s) from %s",
            total,
            len(all_pages_paras),
            self.input_path,
        )
        return all_pages_paras, all_pages_tables

    # ── 1.1 Reading Order & Column Detection ────────────────────────

    @staticmethod
    def _sort_blocks_by_reading_order(
        blocks: list[dict], page_width: float
    ) -> list[dict]:
        """Sort text blocks in reading order, detecting multi-column layouts.

        Algorithm:
          1. Project all block x-centres onto the x-axis.
          2. Find vertical separators: gaps > 2× the median block width
             between sorted x-centres.
          3. If multiple columns are found, group blocks by column and
             sort each column top-to-bottom, then concatenate left-to-right.
          4. Blocks that span across columns (e.g. titles at the top) are
             placed first.
        """
        if not blocks:
            return []

        # — Step 1: detect columns via x-axis clustering —
        x_centres = sorted(
            (b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks
        )

        # Median block width for gap threshold
        widths = sorted(b["bbox"][2] - b["bbox"][0] for b in blocks)
        median_w = widths[len(widths) // 2] if widths else 50.0

        # Find gaps between consecutive x_centres that are > 2× median
        gap_threshold = max(median_w * 2.0, 30.0)
        separators: list[float] = []
        for i in range(1, len(x_centres)):
            gap = x_centres[i] - x_centres[i - 1]
            if gap > gap_threshold:
                separators.append((x_centres[i - 1] + x_centres[i]) / 2)

        if len(separators) < 1:
            # Single column — simple sort
            return sorted(
                blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])
            )

        # — Step 2: assign blocks to columns —
        def _col_idx(b: dict) -> int:
            cx = (b["bbox"][0] + b["bbox"][2]) / 2
            for i, sep in enumerate(separators):
                if cx < sep:
                    return i
            return len(separators)

        col_groups: dict[int, list[dict]] = {}
        for b in blocks:
            ci = _col_idx(b)
            col_groups.setdefault(ci, []).append(b)

        # — Step 3: detect "spanning" blocks (wider than a single column) —
        col_widths: list[float] = []
        for ci in sorted(col_groups):
            cols_in_group = [
                b for b in col_groups[ci]
                if _col_idx(b) == ci  # truly inside this column
            ]
            if cols_in_group:
                min_x = min(b["bbox"][0] for b in cols_in_group)
                max_x = max(b["bbox"][2] for b in cols_in_group)
                col_widths.append(max_x - min_x)

        avg_col_w = (
            sum(col_widths) / len(col_widths) if col_widths else page_width
        )

        spanning: list[dict] = []
        non_spanning: list[dict] = []
        for b in blocks:
            bw = b["bbox"][2] - b["bbox"][0]
            if bw > avg_col_w * 1.5:
                spanning.append(b)
            else:
                non_spanning.append(b)

        # — Step 4: sort —
        spanning.sort(key=lambda b: b["bbox"][1])  # top-to-bottom
        sorted_by_column: list[dict] = []
        for ci in sorted(col_groups):
            col_groups[ci].sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            sorted_by_column.extend(
                b for b in col_groups[ci] if b in non_spanning
            )

        return spanning + sorted_by_column

    # ── 1.3 Header/Footer & Watermark Isolation ─────────────────────

    @staticmethod
    def _is_header_or_footer(
        bbox: fitz.Rect, page_height: float
    ) -> bool:
        """Return True if the bbox falls in the top or bottom 5% of the page."""
        margin = page_height * 0.05
        return bbox.y0 < margin or bbox.y1 > (page_height - margin)

    @staticmethod
    def _is_watermark(
        span_opacity: float, span_color: tuple[float, float, float]
    ) -> bool:
        """Return True if opacity is low or color is near-white/background."""
        if span_opacity < 0.5:
            return True
        # Near-white (very light grey) — typical for watermarks
        if all(c > 0.85 for c in span_color[:3]):
            return True
        return False

    # ── 1.4 Table Isolation ─────────────────────────────────────────

    @staticmethod
    def _extract_tables(
        page: fitz.Page, page_idx: int
    ) -> list[Table]:
        """Extract tables using PyMuPDF's ``find_tables()``.

        Returns a list of Table objects, one per detected table on the page.

        Uses ``t.cells`` (list of fitz.Rect in row-major order) for cell
        bounding boxes and ``t.extract()`` for cell text.

        E.5 fix: each cell now also carries its first span's font/size/color
        metadata, extracted from the page's raw text dict.
        """
        # Pre-extract span metadata from the page for cell font lookup
        raw = page.get_text("dict")
        # Build a spatial index: (bbox, span) for every text span on the page
        cell_span_index: list[tuple[fitz.Rect, TextSpan]] = []
        for block in raw.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block.get("lines", []):
                for s in line.get("spans", []):
                    sb = s.get("bbox")
                    if sb is None:
                        continue
                    raw_color = s.get("color", (0, 0, 0))
                    if isinstance(raw_color, (int, float)):
                        span_color = (0.0, 0.0, 0.0)
                    else:
                        span_color = tuple(raw_color)[:3]
                    ts = TextSpan(
                        text=s.get("text", ""),
                        font=s.get("font", ""),
                        size=s.get("size", 11.0),
                        color=span_color,
                        bbox=fitz.Rect(sb),
                        matrix=tuple(s.get("transform", (1, 0, 0, 1, 0, 0))),
                        dir=tuple(s.get("dir", (1.0, 0.0))),
                    )
                    cell_span_index.append((ts.bbox, ts))

        found = page.find_tables()
        tables: list[Table] = []
        for t in found:
            extract_data = t.extract()
            if not extract_data:
                continue

            row_count = len(extract_data)
            col_count = max(len(r) for r in extract_data) if extract_data else 0
            cells_rects = list(t.cells)  # flat list of fitz.Rect in row-major order

            rows: list[list[tuple[str, fitz.Rect, TextSpan | None]]] = []
            for row_idx, row_data in enumerate(extract_data):
                cells: list[tuple[str, fitz.Rect, TextSpan | None]] = []
                for col_idx, cell_text in enumerate(row_data):
                    # Look up cell bbox from t.cells (row-major order)
                    cell_index = row_idx * col_count + col_idx
                    if cell_index < len(cells_rects):
                        cell_bbox = cells_rects[cell_index]
                    else:
                        # Fallback: compute from adjacent cells
                        cell_bbox = fitz.Rect(0, 0, 0, 0)
                    text = (cell_text or "").strip()

                    # E.5: find first span whose bbox overlaps this cell
                    span_meta: TextSpan | None = None
                    cell_rect = fitz.Rect(cell_bbox)
                    for span_bbox, span_ts in cell_span_index:
                        if span_bbox.intersects(cell_rect):
                            span_meta = span_ts
                            break

                    cells.append((text, fitz.Rect(cell_bbox), span_meta))
                rows.append(cells)

            if rows:
                all_rects = [c[1] for r in rows for c in r if c[1].width > 0]
                if all_rects:
                    table_bbox = fitz.Rect(
                        min(r.x0 for r in all_rects),
                        min(r.y0 for r in all_rects),
                        max(r.x1 for r in all_rects),
                        max(r.y1 for r in all_rects),
                    )
                else:
                    table_bbox = t.bbox

                tables.append(
                    Table(
                        bbox=table_bbox,
                        cells=rows,
                        page_number=page_idx,
                    )
                )

        if tables:
            logger.debug(
                "Page %d: found %d table(s)", page_idx, len(tables)
            )
        return tables

    # ── 1.5 De-hyphenation ─────────────────────────────────────────

    @staticmethod
    def _dehyphenate_line(
        prev_text: str, next_text: str
    ) -> tuple[str, str]:
        """If ``prev_text`` ends with a line-break hyphen, merge with next.

        Returns (modified_prev, modified_next).

        Rules:
          - Only merges if the trailing hyphen is NOT part of an em-dash,
            en-dash, or natural compound hyphen.
          - Heuristic: if char before hyphen is a consonant and char after
            is a vowel (or vice versa), it's likely a syllable-break hyphen.
        """
        # Check for various dash/hyphen characters at end
        HYPHEN_RE = re.compile(r"-\s*$")
        m = HYPHEN_RE.search(prev_text)
        if not m:
            return prev_text, next_text

        # Extract char before the hyphen
        hyphen_start = m.start()
        if hyphen_start == 0:
            return prev_text, next_text

        char_before = prev_text[hyphen_start - 1]
        stripped_next = next_text.lstrip()
        if not stripped_next:
            return prev_text, next_text

        char_after = stripped_next[0]

        # Skip if the "hyphen" is actually an em-dash or en-dash context
        if char_before in ("—", "–", "\u2014", "\u2013"):
            return prev_text, next_text

        # Skip known compound-word suffixes that should stay hyphenated
        if prev_text.rstrip().endswith(
            (
                "-то",
                "-либо",
                "-нибудь",
                "-ка",
                "-таки",
                "ко-",
                "из-за",
                "из-под",
            )
        ):
            return prev_text, next_text

        # Vowel / consonant heuristic for Uzbek & Russian
        vowels = set("аеёиоуыэюяaeiou")
        is_break_hyphen = False
        if char_before.lower() in vowels and char_after.lower() not in vowels:
            is_break_hyphen = True
        elif (
            char_before.lower() not in vowels
            and char_after.lower() in vowels
        ):
            is_break_hyphen = True
        elif char_after.isdigit():  # e.g. "199-\n2000" → keep hyphen
            return prev_text, next_text

        if not is_break_hyphen:
            return prev_text, next_text

        # Merge: remove trailing hyphen and whitespace, join
        merged = prev_text[:hyphen_start] + stripped_next
        return merged, ""

    # ── 1.6 List Markers & Drop Caps ────────────────────────────────

    @staticmethod
    def _strip_list_marker(
        text: str,
    ) -> tuple[str, str | None]:
        """Detect and remove bullet/number list markers.

        Returns (text_without_marker, marker_string_or_None).

        Supports: ``•``, ``-``, ``*``, ``1.``, ``1)``, ``a)``, ``A.``, etc.
        """
        # Order matters: try longer patterns first
        patterns = [
            re.compile(
                r"^(\s*[\u2022\u2023\u25E6\u2043\u2219]\s*)"
            ),  # bullets
            re.compile(
                r"^(\s*(?:\d{1,3}[\.\)]|[a-zA-Z][\.\)])\s*)"
            ),  # numbered / lettered
            re.compile(r"^(\s*[\-\*]\s*)"),  # dash / asterisk
        ]
        for pat in patterns:
            m = pat.match(text)
            if m:
                marker = m.group(1).strip()
                rest = text[m.end() :].strip()
                return rest, marker
        return text, None

    @staticmethod
    def _detect_drop_cap(
        spans: list[TextSpan],
    ) -> tuple[list[TextSpan], str | None]:
        """Detect a drop cap (first char much larger than rest).

        If found, merges the first character into the next span's text
        and marks the paragraph as having a drop cap.

        Returns (modified_spans, merged_text) where merged_text is the
        full text including the drop cap.
        """
        if len(spans) < 2:
            return spans, None

        first = spans[0]
        second = spans[1]

        # Drop cap: single character, at least 1.5× larger than next span
        if (
            len(first.text.strip()) == 1
            and first.size >= second.size * 1.5
            and first.bbox.y1 >= second.bbox.y0  # vertically adjacent
        ):
            # Merge drop cap into second span text
            merged_text = first.text.strip() + second.text
            merged_span = TextSpan(
                text=merged_text,
                font=second.font,
                size=second.size,
                color=second.color,
                bbox=fitz.Rect(
                    first.bbox.x0,
                    first.bbox.y0,
                    second.bbox.x1,
                    second.bbox.y1,
                ),
                matrix=second.matrix,
                dir=second.dir,
                is_math=second.is_math,
                opacity=second.opacity,
            )
            new_spans = [merged_span] + spans[2:]
            return new_spans, merged_text

        return spans, None

    # ── Line extraction (internal) ──────────────────────────────────

    def _extract_lines(
        self,
        page: fitz.Page,
        table_bboxes: list[fitz.Rect],
    ) -> list[dict]:
        """Extract lines from a page, applying filters.

        1. Gets ``page.get_text("dict")`` blocks.
        2. Sorts in reading order (multi-column aware).
        3. Filters headers/footers and watermarks.
        4. Extracts per-span metadata.
        5. Applies de-hyphenation.
        """
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        page_height = page.rect.height
        page_width = page.rect.width

        # Keep only text blocks
        text_blocks = [b for b in blocks if "lines" in b]

        # Filter watermarked / header-footer blocks
        filtered_blocks: list[dict] = []
        for block in text_blocks:
            block_bbox = fitz.Rect(block["bbox"])
            if self._is_header_or_footer(block_bbox, page_height):
                continue
            # Check if block is inside a table — skip it (table cell
            # content is extracted separately)
            if any(
                table_bbox.contains(block_bbox) for table_bbox in table_bboxes
            ):
                continue
            # Check block-level watermark
            block_opacity = block.get("opacity", 1.0)
            block_color = block.get("color", (0, 0, 0))
            if self._is_watermark(block_opacity, block_color):
                continue
            filtered_blocks.append(block)

        # Sort remaining blocks in reading order
        sorted_blocks = self._sort_blocks_by_reading_order(
            filtered_blocks, page_width
        )

        # Extract lines with spans
        lines: list[dict] = []
        for block in sorted_blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Filter watermarked spans
                line_spans: list[TextSpan] = []
                for s in spans:
                    raw_color = s.get("color", (0, 0, 0))
                    # PyMuPDF may return color as int or tuple — handle both
                    if isinstance(raw_color, (int, float)):
                        span_color = (0.0, 0.0, 0.0)
                    else:
                        span_color = tuple(raw_color)[:3]
                    span_opacity = s.get("opacity", 1.0)
                    if self._is_watermark(span_opacity, span_color):
                        continue

                    span_bbox = (
                        fitz.Rect(s["bbox"]) if "bbox" in s else None
                    )
                    if span_bbox is None:
                        continue

                    ts = TextSpan(
                        text=s.get("text", ""),
                        font=s.get("font", ""),
                        size=s.get("size", 11.0),
                        color=span_color,
                        bbox=span_bbox,
                        matrix=tuple(
                            s.get("transform", (1, 0, 0, 1, 0, 0))
                        ),
                        dir=tuple(s.get("dir", (1.0, 0.0))),
                        opacity=span_opacity,
                        is_math=(
                            self.is_math_font(s.get("font", ""))
                            or self._has_math_unicode(
                                s.get("text", "")
                            )
                        ),
                    )
                    line_spans.append(ts)

                if not line_spans:
                    continue

                # Build line text from spans
                line_text = "".join(ts.text for ts in line_spans)
                if not line_text.strip():
                    continue

                line_bbox = fitz.Rect(line["bbox"])
                combined_fontsize = line_spans[0].size
                is_math = any(ts.is_math for ts in line_spans)

                lines.append(
                    {
                        "bbox": line_bbox,
                        "text": line_text,
                        "fontsize": combined_fontsize,
                        "is_math": is_math,
                        "spans": line_spans,
                        "block": block,
                    }
                )

        # Apply de-hyphenation
        dehyphenated: list[dict] = []
        for i, line in enumerate(lines):
            if i == 0:
                dehyphenated.append(line)
                continue

            prev_text = dehyphenated[-1]["text"]
            curr_text = line["text"]
            mod_prev, mod_curr = self._dehyphenate_line(
                prev_text, curr_text
            )
            if mod_curr == "":
                # Merge: update previous line and skip current
                dehyphenated[-1]["text"] = mod_prev
                # Extend previous bbox to include this line
                prev_bbox = dehyphenated[-1]["bbox"]
                dehyphenated[-1]["bbox"] = fitz.Rect(
                    prev_bbox.x0,
                    prev_bbox.y0,
                    max(prev_bbox.x1, line["bbox"].x1),
                    line["bbox"].y1,
                )
                # Merge spans too
                dehyphenated[-1]["spans"].extend(line["spans"])
            else:
                line["text"] = mod_curr
                dehyphenated.append(line)

        return dehyphenated

    # ── Paragraph clustering ────────────────────────────────────────

    def _cluster_lines(
        self, lines: list[dict], page_idx: int
    ) -> list[Paragraph]:
        """Cluster lines into paragraphs by vertical proximity."""
        paragraphs: list[Paragraph] = []
        current: list[dict] = []

        for line in lines:
            if not current:
                current = [line]
                continue
            prev = current[-1]
            gap = line["bbox"].y0 - prev["bbox"].y1
            same_column = (
                abs(line["bbox"].x0 - prev["bbox"].x0)
                < max(prev["fontsize"] * 3, 20)
            )
            if (
                gap < prev["fontsize"] * self.LINE_GAP_FACTOR
                and same_column
            ):
                current.append(line)
            else:
                paragraphs.append(
                    self._build_paragraph(current, page_idx)
                )
                current = [line]

        if current:
            paragraphs.append(
                self._build_paragraph(current, page_idx)
            )

        return paragraphs

    def _build_paragraph(
        self, lines: list[dict], page_idx: int
    ) -> Paragraph:
        """Build a Paragraph from clustered lines."""

        # Collect all spans
        all_spans: list[TextSpan] = []
        for line in lines:
            all_spans.extend(line.get("spans", []))

        # Apply drop-cap detection on the first line's spans
        if all_spans:
            modified_spans, drop_cap_text = self._detect_drop_cap(
                all_spans
            )
            if drop_cap_text is not None:
                all_spans = modified_spans
                lines[0]["spans"] = modified_spans

        # Build text with math placeholder handling
        math_spans: list[MathSpan] = []
        text_parts: list[str] = []

        for line in lines:
            if line["is_math"] and "spans" in line:
                for span in line["spans"]:
                    if span.is_math:
                        placeholder = self._registry.mask_math()
                        math_spans.append(
                            MathSpan(
                                placeholder=placeholder,
                                original_text=span.text,
                                bbox=span.bbox,
                            )
                        )
                        text_parts.append(placeholder)
                    else:
                        text_parts.append(span.text)
            else:
                text_parts.append(line["text"])

        # Strip list markers
        full_text = " ".join(
            t.strip() for t in text_parts if t.strip()
        )
        cleaned_text, list_marker = self._strip_list_marker(full_text)

        # Compute bbox
        x0 = min(ln["bbox"].x0 for ln in lines)
        y0 = min(ln["bbox"].y0 for ln in lines)
        x1 = max(ln["bbox"].x1 for ln in lines)
        y1 = max(ln["bbox"].y1 for ln in lines)

        fontsize = lines[0]["fontsize"]
        is_math = any(ln["is_math"] for ln in lines)

        # Metadata from first line's first span for builder
        first_span = (
            all_spans[0]
            if all_spans
            else TextSpan(
                text="",
                font="",
                size=fontsize,
                color=(0, 0, 0),
                bbox=fitz.Rect(0, 0, 0, 0),
                matrix=(1, 0, 0, 1, 0, 0),
                dir=(1, 0),
            )
        )

        return Paragraph(
            bbox=fitz.Rect(x0, y0, x1, y1),
            text=cleaned_text,
            fontsize=fontsize,
            font=first_span.font,
            color=first_span.color,
            rotation_dir=first_span.dir,
            is_math=is_math,
            page_number=page_idx,
            math_spans=math_spans,
            list_marker=list_marker,
            original_spans=all_spans,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def is_math_font(font_name: str) -> bool:
        name = (font_name or "").lower()
        name = name.split("+")[-1]
        name = name.replace(" ", "").replace("-", "").replace("_", "")
        return any(marker in name for marker in MATH_FONT_MARKERS)

    @staticmethod
    def _has_math_unicode(text: str, threshold: float = 0.15) -> bool:
        """Return True if *text* contains a high density of math Unicode codepoints.

        Works independently of font name — catches formulas typeset with a
        plain italic serif font and Unicode math symbols (common in
        OCR'd PDFs and Uzbek-authored textbooks).
        """
        if not text:
            return False
        math_chars = 0
        for ch in text:
            cp = ord(ch)
            for r_start, r_end in _MATH_UNICODE_RANGES:
                if r_start <= cp <= r_end:
                    math_chars += 1
                    break
        return (math_chars / max(len(text), 1)) >= threshold

    # ── Backward-compat shim ────────────────────────────────────────

    def extract_paragraphs_legacy(self) -> list[list[Paragraph]]:
        """Return only paragraphs (no tables) for backward compatibility."""
        paras, _ = self.extract_paragraphs()
        return paras
