# pdf_translator_ru_uz/builder.py

"""Module: Intelligent in-place PDF builder with advanced rendering.

Replaces the basic redact+redraw approach with:
- Adaptive background inpainting (dominant colour extraction)
- Image-aware rendering (shadow/stroke for text over images)
- Font synthesis tree (Serif→Liberation Serif, Sans→DejaVu Sans)
- Custom line-wrapping with interline spacing
- Justification alignment
- Rotation preservation
- GC-optimised save
- Native font metrics (Phase 0 fix: eliminates fused-word bug)
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz

logger = logging.getLogger(__name__)

# ── Font paths known to exist on this system ─────────────────────────
_FONT_PATHS: dict[str, str] = {
    # DejaVu Sans family (sans-serif fallback)
    "DejaVuSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans-Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSerif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "DejaVuSerif-Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "DejaVuSansMono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono-Bold": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    # Liberation Serif (serif mapping)
    "LiberationSerif-Regular": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "LiberationSerif-Bold": "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "LiberationSerif-Italic": "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "LiberationSerif-BoldItalic": "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
    # Liberation Sans
    "LiberationSans-Regular": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "LiberationSans-Bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "LiberationSans-Italic": "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "LiberationSans-BoldItalic": "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    # Liberation Mono
    "LiberationMono-Regular": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "LiberationMono-Bold": "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "LiberationMono-Italic": "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
    "LiberationMono-BoldItalic": "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
    # Noto Sans & Noto Serif (additional fallbacks)
    "NotoSans": "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "NotoSans-Bold": "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "NotoSerif": "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "NotoSerif-Bold": "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
}

# ── Font synthesis tree ────────────────────────────────────────────

# Map original font family names to target families
_SERIF_FONTS = {
    "times", "timesnewroman", "times new roman", "liberationserif",
    "cambria", "cambriamath", "palatino", "georgia", "garamond",
    "bookman", "book antiqua",
}
_SANS_FONTS = {
    "arial", "helvetica", "dejavusans", "calibri", "tahoma",
    "verdana", "trebuchet", "opensans", "freesans", "notosans",
    "ubuntu", "segoeui", "centurygothic",
}
_MONO_FONTS = {
    "courier", "couriernew", "liberationmono", "freemono",
    "notomono", "consolas", "monaco", "menlo",
}

# Font name → (family, variant) lookup
_FONT_MAP: dict[str, tuple[str, bool, bool]] = {
    # DejaVu Sans (sans-serif primary)
    "dejavusans": ("DejaVuSans", False, False),
    "dejavusans-bold": ("DejaVuSans", True, False),
    # DejaVu Serif
    "dejavuserif": ("DejaVuSerif", False, False),
    "dejavuserif-bold": ("DejaVuSerif", True, False),
    # Liberation Serif
    "liberationserif": ("LiberationSerif", False, False),
    "liberationserif-bold": ("LiberationSerif", True, False),
    "liberationserif-italic": ("LiberationSerif", False, True),
    "liberationserif-bolditalic": ("LiberationSerif", True, True),
    # Liberation Sans
    "liberationsans": ("LiberationSans", False, False),
    "liberationsans-bold": ("LiberationSans", True, False),
    "liberationsans-italic": ("LiberationSans", False, True),
    "liberationsans-bolditalic": ("LiberationSans", True, True),
    # Liberation Mono
    "liberationmono": ("LiberationMono", False, False),
    "liberationmono-bold": ("LiberationMono", True, False),
    "liberationmono-italic": ("LiberationMono", False, True),
    "liberationmono-bolditalic": ("LiberationMono", True, True),
    # Noto Sans
    "notosans": ("NotoSans", False, False),
    "notosans-bold": ("NotoSans", True, False),
    "notoserif": ("NotoSerif", False, False),
    "notoserif-bold": ("NotoSerif", True, False),
    # Free Sans
    "freesans": ("DejaVuSans", False, False),
    "freesans-bold": ("DejaVuSans", True, False),
    "freeserif": ("LiberationSerif", False, False),
    "freeserif-bold": ("LiberationSerif", True, False),
    "freemono": ("LiberationMono", False, False),
    "freemono-bold": ("LiberationMono", True, False),
}


class BuildError(RuntimeError):
    pass


@dataclass
class FitResult:
    """Outcome of rendering translated text into a bbox."""

    fontsize: float
    fit: bool
    bled: bool


@dataclass
class RenderedLine:
    """A single line of text ready for drawing."""

    words: list[str]
    word_spacings: list[float] = field(default_factory=list)
    line_width: float = 0.0


# ── InPlaceBuilder ──────────────────────────────────────────────────


class InPlaceBuilder:
    """Intelligent in-place PDF builder with advanced rendering.

    Replaces original text with translated text while preserving the
    visual appearance of the original document (colours, fonts, layout).
    """

    def __init__(
        self,
        min_fontsize: float = 5.0,
        fontsize_step: float = 0.5,
        line_spacing: float = 1.2,
        fontfile: Optional[str] = None,
    ):
        self.min_fontsize = min_fontsize
        self.fontsize_step = fontsize_step
        self.line_spacing = line_spacing

        # Fallback font (DejaVu Sans always available)
        self._fallback_fontfile = (
            fontfile or _FONT_PATHS.get("DejaVuSans")
        )
        if self._fallback_fontfile is None or not Path(
            self._fallback_fontfile
        ).exists():
            raise BuildError(
                "No Unicode-capable font found. "
                "Install fonts-dejavu-core or fonts-liberation2."
            )
        self._fallback_fontname = "DejaVuSans"
        
        # Intent: Cache fitz.Font objects to prevent TTF re-parsing overhead.
        # State Transition: Repeated disk reads -> In-memory cache lookup.
        self._font_cache: dict[str, fitz.Font] = {}

    def get_text_width(
        self, text: str, fontname: str, fontfile: str, fontsize: float
    ) -> float:
        """Get exact pixel width of text using native font metrics.
        
        Dependencies: Requires valid fontfile path. Falls back to DejaVuSans if missing.
        """
        if not fontfile or not Path(fontfile).exists():
            fontfile = self._fallback_fontfile
            
        if fontfile not in self._font_cache:
            try:
                self._font_cache[fontfile] = fitz.Font(fontfile=fontfile)
            except Exception:
                # Fallback to base fitz font if TTF fails
                return fitz.get_text_length(text, fontname="helv", fontsize=fontsize)
                
        return self._font_cache[fontfile].text_length(text, fontsize)

    # ── 4.1 Adaptive Background Inpainting ─────────────────────────

    def _compute_background_color(
        self, page: fitz.Page, bbox: fitz.Rect
    ) -> Optional[tuple[float, float, float]]:
        """Estimate background colour — fast path.

        Renders a tiny (30px wide) pixmap, samples a few pixels,
        excludes near-black, returns mean RGB.

        Returns None if the area overlaps an image by >50%.
        """
        if self._has_image_overlap(page, bbox, threshold=0.5):
            return None

        try:
            pix = page.get_pixmap(
                clip=bbox,
                width=min(int(bbox.width), 30),
                height=min(int(bbox.height), 30),
            )
            samples: list[tuple[int, int, int]] = []
            for y in range(0, pix.height, max(1, pix.height // 3)):
                for x in range(0, pix.width, max(1, pix.width // 3)):
                    r, g, b = pix.pixel(x, y)
                    if r + g + b > 100:
                        samples.append((r, g, b))

            if not samples:
                return (1.0, 1.0, 1.0)

            n = len(samples)
            return (
                sum(s[0] for s in samples) / n / 255.0,
                sum(s[1] for s in samples) / n / 255.0,
                sum(s[2] for s in samples) / n / 255.0,
            )
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _paint_background(
        page: fitz.Page,
        bbox: fitz.Rect,
        color: tuple[float, float, float],
    ) -> None:
        """Paint a solid filled rectangle over the bbox area.

        Uses ``overlay=False`` so the painted background replaces
        rather than blends.
        """
        page.draw_rect(bbox, color=color, fill=color, overlay=False)

    # ── 4.2 Text Over Images ───────────────────────────────────────

    @staticmethod
    def _has_image_overlap(
        page: fitz.Page,
        bbox: fitz.Rect,
        threshold: float = 0.1,
    ) -> bool:
        """Return True if the bbox significantly overlaps with an image.

        ``threshold`` is the minimum overlap fraction of the bbox area.
        """
        try:
            images = page.get_images(full=True)
            for img in images:
                img_rects = page.get_image_rects(img[0])
                for img_rect in img_rects:
                    intersection = fitz.Rect(bbox).intersect(img_rect)
                    if intersection.is_valid and (
                        intersection.width * intersection.height
                    ) >= (bbox.width * bbox.height * threshold):
                        return True
        except Exception:
            pass
        return False

    # ── 4.3 Font Synthesis Tree ────────────────────────────────────

    def _resolve_font(
        self,
        original_font: str,
        original_flags: int,
    ) -> tuple[str, str]:
        """Map an original font name to a resolved (fontname, fontfile) pair.

        Applies the font synthesis tree: Serif→Liberation Serif,
        Sans→DejaVu Sans, Mono→Liberation Mono, with bold/italic
        resolution from ``original_flags``.
        """
        # Normalise font name
        font_key = (
            (original_font or "")
            .lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .split("+")[-1]
        )

        # Determine family from stripped name
        family_base = font_key.rstrip("0123456789")
        # Check for bold/italic in font name
        has_bold = (
            "bold" in family_base or original_flags & 2**0 != 0
        )
        has_italic = (
            "italic" in family_base
            or "oblique" in family_base
            or original_flags & 2**1 != 0
        )

        # Determine font category
        is_serif = any(f in font_key for f in _SERIF_FONTS)
        is_mono = any(f in font_key for f in _MONO_FONTS)
        is_sans = any(f in font_key for f in _SANS_FONTS)

        if is_mono:
            base = "LiberationMono"
        elif is_serif:
            base = "LiberationSerif"
        else:
            # Default to DejaVu Sans for sans-serif or unknown
            base = "DejaVuSans"

        # Build the variant name
        if has_bold and has_italic:
            variant = f"{base}-BoldItalic" if "Liberation" in base else f"{base}-Bold"
        elif has_bold:
            variant = f"{base}-Bold"
        elif has_italic:
            variant = f"{base}-Italic" if "Liberation" in base else base
        else:
            variant = f"{base}-Regular" if "Liberation" in base else base

        # Look up font file
        fontfile = _FONT_PATHS.get(variant)
        if not fontfile or not Path(fontfile).exists():
            # Fall back to regular variant
            fallback_key = (
                f"{base}-Regular"
                if "Liberation" in base
                else base
            )
            fontfile = _FONT_PATHS.get(fallback_key)
        if not fontfile or not Path(fontfile).exists():
            fontfile = self._fallback_fontfile

        return variant, fontfile

    # ── 4.4 Custom Line-Wrapper ────────────────────────────────────

    def _wrap_text(
        self,
        text: str,
        fontname: str,
        fontfile: str,
        fontsize: float,
        max_width: float,
    ) -> list[RenderedLine]:
        """Wrap text to fit ``max_width``, measuring each word.

        Returns a list of RenderedLine objects, each containing the
        words and computed word spacings.
        """
        words = text.split()
        if not words:
            return []

        lines: list[RenderedLine] = []
        current_words: list[str] = []
        current_width = 0.0

        for word in words:
            word_w = self.get_text_width(word, fontname, fontfile, fontsize)
            # Space width (approx 0.33 em ≈ 0.5 * fontsize * 0.6)
            space_w = fontsize * 0.3

            if current_words and (
                current_width + space_w + word_w > max_width
            ):
                # Flush current line
                lines.append(
                    RenderedLine(
                        words=current_words,
                        line_width=current_width,
                    )
                )
                current_words = [word]
                current_width = word_w
            else:
                if current_words:
                    current_width += space_w
                current_words.append(word)
                current_width += word_w

        if current_words:
            lines.append(
                RenderedLine(
                    words=current_words,
                    line_width=current_width,
                )
            )

        return lines

    # ── 4.5 Justification Alignment ────────────────────────────────

    @staticmethod
    def _is_likely_justified(
        bbox: fitz.Rect,
        lines: list[RenderedLine],
    ) -> bool:
        """Heuristic: check if multiple lines fill close to full bbox width.

        If 3+ lines have width > 85% of bbox width, likely justified.
        """
        if len(lines) < 3:
            return False
        bbox_w = bbox.width
        if bbox_w <= 0:
            return False
        wide_count = sum(
            1 for ln in lines if ln.line_width > bbox_w * 0.85
        )
        return wide_count >= 2

    def _justify_line(
        self,
        line: RenderedLine,
        max_width: float,
        fontname: str,
        fontfile: str,
        fontsize: float,
    ) -> RenderedLine:
        """Compute word spacings to justify the line to ``max_width``.

        Distributes remaining whitespace evenly between words.
        Single-word lines are left-aligned.
        """
        if len(line.words) <= 1:
            line.word_spacings = [0.0]
            return line

        remaining = max_width - line.line_width
        if remaining <= 0:
            line.word_spacings = [0.0] * (len(line.words) - 1) + [0.0]
            return line

        gaps = len(line.words) - 1
        extra_per_gap = remaining / gaps
        space_w = fontsize * 0.3

        spacings = [extra_per_gap] * gaps + [0.0]
        line.word_spacings = spacings
        return line

    # ── 4.6 Rotation Preservation ──────────────────────────────────

    @staticmethod
    def _compute_rotation_origin(
        bbox: fitz.Rect, dir_vec: tuple[float, float]
    ) -> tuple[tuple[float, float], float]:
        """Compute (origin, angle_degrees) from a direction vector.

        ``dir_vec`` is ``(dx, dy)`` from PyMuPDF span metadata.

        Default ``(1, 0)`` → no rotation (angle=0).
        ``(0, 1)`` → 90° clockwise.
        ``(-1, 0)`` → 180°.
        ``(0, -1)`` → 270° (or -90°).
        """
        dx, dy = dir_vec
        if abs(dx) > 0.5 and abs(dy) < 0.5:
            angle = 0.0 if dx > 0 else 180.0
        elif abs(dy) > 0.5 and abs(dx) < 0.5:
            angle = 90.0 if dy > 0 else -90.0
        else:
            angle = math.degrees(math.atan2(dy, dx))
        origin = (bbox.x0, bbox.y0)
        return origin, angle

    # ── Drawing ────────────────────────────────────────────────────

    def draw_text_multiline(
        self,
        page: fitz.Page,
        bbox: fitz.Rect,
        text: str,
        fontname: str,
        fontfile: str,
        fontsize: float,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dir_vec: tuple[float, float] = (1.0, 0.0),
        justify: bool = False,
        shadow: bool = False,
        dry_run: bool = False,
    ) -> FitResult:
        """Draw multi-line text into bbox with proper line spacing."""
        # Wrap text
        lines = self._wrap_text(
            text, fontname, fontfile, fontsize, bbox.width
        )
        if not lines:
            return FitResult(fontsize=fontsize, fit=True, bled=False)

        # Justify if requested (E.9 fix: skip last line — standard typography)
        if justify and self._is_likely_justified(bbox, lines) and len(lines) > 1:
            justified_lines = [
                self._justify_line(
                    ln, bbox.width, fontname, fontfile, fontsize
                )
                for ln in lines[:-1]
            ]
            justified_lines.append(lines[-1])  # last line: left-aligned
            lines = justified_lines

        # Compute rotation
        origin, angle = self._compute_rotation_origin(bbox, dir_vec)

        # Draw each line
        line_height = fontsize * self.line_spacing
        y_pos = bbox.y0 + fontsize  # baseline offset

        bled = False
        for line_idx, line in enumerate(lines):
            y_current = y_pos + line_idx * line_height

            # Check if we've exceeded the bbox
            if y_current > bbox.y1 + line_height:
                bled = True
                break

            x_pos = bbox.x0

            # Draw with word-by-word if justified (to control spacing)
            if justify and len(line.words) > 1 and line.word_spacings:
                space_w = fontsize * 0.3
                for wi, word in enumerate(line.words):
                    pt = fitz.Point(x_pos, y_current)
                    # Apply rotation
                    if abs(angle) > 0.5:
                        rotated = fitz.Point(
                            origin[0]
                            + (pt.x - origin[0]) * math.cos(math.radians(angle))
                            - (pt.y - origin[1])
                            * math.sin(math.radians(angle)),
                            origin[1]
                            + (pt.x - origin[0]) * math.sin(math.radians(angle))
                            + (pt.y - origin[1])
                            * math.cos(math.radians(angle)),
                        )
                        pt = rotated

                    if not dry_run:
                        if shadow and self._has_image_overlap(page, bbox):
                            shadow_pt = fitz.Point(pt.x + 0.3, pt.y + 0.3)
                            page.insert_text(
                                shadow_pt,
                                word,
                                fontname=fontname,
                                fontfile=fontfile,
                                fontsize=fontsize,
                                color=(1, 1, 1),
                            )

                        page.insert_text(
                            pt,
                            word,
                            fontname=fontname,
                            fontfile=fontfile,
                            fontsize=fontsize,
                            color=color,
                        )

                    # Advance x by word width + spacing
                    word_w = self.get_text_width(word, fontname, fontfile, fontsize)
                    extra_spacing = (
                        line.word_spacings[wi]
                        if wi < len(line.word_spacings)
                        else 0.0
                    )
                    x_pos += word_w + space_w + extra_spacing
            else:
                # Simple line drawing (left-aligned)
                line_text = " ".join(line.words)
                pt = fitz.Point(x_pos, y_current)

                if abs(angle) > 0.5:
                    rotated = fitz.Point(
                        origin[0]
                        + (pt.x - origin[0]) * math.cos(math.radians(angle))
                        - (pt.y - origin[1]) * math.sin(math.radians(angle)),
                        origin[1]
                        + (pt.x - origin[0]) * math.sin(math.radians(angle))
                        + (pt.y - origin[1])
                        * math.cos(math.radians(angle)),
                    )
                    pt = rotated

                if not dry_run:
                    if shadow and self._has_image_overlap(page, bbox):
                        shadow_pt = fitz.Point(pt.x + 0.3, pt.y + 0.3)
                        page.insert_text(
                            shadow_pt,
                            line_text,
                            fontname=fontname,
                            fontfile=fontfile,
                            fontsize=fontsize,
                            color=(1, 1, 1),
                        )

                    page.insert_text(
                        pt,
                        line_text,
                        fontname=fontname,
                        fontfile=fontfile,
                        fontsize=fontsize,
                        color=color,
                    )

        return FitResult(
            fontsize=fontsize, fit=not bled, bled=bled
        )

    # ── Combined replace operation ──────────────────────────────────

    def replace_paragraph(
        self,
        page: fitz.Page,
        bbox: fitz.Rect,
        translated_text: str,
        original_font: str = "",
        original_fontsize: float = 11.0,
        original_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        original_flags: int = 0,
        original_dir: tuple[float, float] = (1.0, 0.0),
        exclude_bboxes: Optional[list[fitz.Rect]] = None,
    ) -> FitResult:
        # Step 1: Redact original text (but keep images/vectors)
        if not exclude_bboxes:
            self._redact_bbox(page, bbox)
        else:
            self._redact_with_exclusions(page, bbox, exclude_bboxes)

        # Step 2: Adaptive background inpainting (skip if image overlap)
        has_images = self._has_image_overlap(page, bbox)
        if not has_images:
            bg_color = self._compute_background_color(page, bbox)
            if bg_color:
                self._paint_background(page, bbox, bg_color)

        # Step 3: Resolve font
        fontname, fontfile = self._resolve_font(
            original_font, original_flags
        )

        # Step 4: Determine if justified
        justify = True  

        # Step 5: Try to find fitting fontsize (DRY RUN ONLY)
        fontsize = original_fontsize
        result = None

        while fontsize >= self.min_fontsize:
            result = self.draw_text_multiline(
                page=page,
                bbox=bbox,
                text=translated_text,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=fontsize,
                color=original_color,
                dir_vec=original_dir,
                justify=justify,
                shadow=has_images,
                dry_run=True,
            )
            if result.fit:
                break
            fontsize -= self.fontsize_step

        # Step 6: Finally draw it once
        if result is None or not result.fit:
            # Bleed: expand bbox height and draw with min_fontsize
            bleed_bbox = fitz.Rect(
                bbox.x0, bbox.y0, bbox.x1, bbox.y1 + 5000
            )
            self.draw_text_multiline(
                page=page,
                bbox=bleed_bbox,
                text=translated_text,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=self.min_fontsize,
                color=original_color,
                dir_vec=original_dir,
                justify=False,
                shadow=has_images,
                dry_run=False,
            )
            result = FitResult(
                fontsize=self.min_fontsize, fit=False, bled=True
            )
        else:
            # Draw properly with found fontsize
            self.draw_text_multiline(
                page=page,
                bbox=bbox,
                text=translated_text,
                fontname=fontname,
                fontfile=fontfile,
                fontsize=fontsize,
                color=original_color,
                dir_vec=original_dir,
                justify=justify,
                shadow=has_images,
                dry_run=False,
            )

        return result

    # ── Redaction (internal) ───────────────────────────────────────

    @staticmethod
    def _redact_bbox(page: fitz.Page, bbox: fitz.Rect) -> None:
        """Erase text inside bbox, leaving images and line art intact."""
        page.add_redact_annot(bbox)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )

    @staticmethod
    def _redact_with_exclusions(
        page: fitz.Page,
        bbox: fitz.Rect,
        exclude: list[fitz.Rect],
    ) -> None:
        """Redact bbox but skip areas in ``exclude``."""
        to_redact = _subtract_bboxes(bbox, exclude)
        for r in to_redact:
            if r.width > 0 and r.height > 0:
                page.add_redact_annot(r)
        page.apply_redactions(
            images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        )


# ── Module-level helpers ────────────────────────────────────────────


def _subtract_bboxes(
    outer: fitz.Rect, exclude: list[fitz.Rect]
) -> list[fitz.Rect]:
    """Return a list of rectangles covering *outer* minus *exclude*.

    Produces top, left, right, bottom strips around each excluded rect.
    """
    if not exclude:
        return [outer]

    result: list[fitz.Rect] = []
    sorted_ex = sorted(exclude, key=lambda r: r.y0)
    current_y = outer.y0

    for ex in sorted_ex:
        if ex.y0 > current_y + 1:
            result.append(
                fitz.Rect(outer.x0, current_y, outer.x1, ex.y0)
            )
        if ex.x0 > outer.x0 + 1:
            result.append(
                fitz.Rect(outer.x0, ex.y0, ex.x0, ex.y1)
            )
        if ex.x1 < outer.x1 - 1:
            result.append(
                fitz.Rect(ex.x1, ex.y0, outer.x1, ex.y1)
            )
        current_y = max(current_y, ex.y1)

    if current_y < outer.y1 - 1:
        result.append(
            fitz.Rect(outer.x0, current_y, outer.x1, outer.y1)
        )

    return result
