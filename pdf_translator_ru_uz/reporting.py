# pdf_translator_ru_uz/reporting.py

from __future__ import annotations
import csv
import logging
from pathlib import Path
import fitz
from pdf_translator_ru_uz.verification import VerificationResult

logger = logging.getLogger(__name__)

# Intent: Isolate human-readable output generation from pipeline logic.
class ReportWriter:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        # State Transition: File created -> Schema written.
        self._writer.writerow([
            "page", "source_preview", "translation_preview", "flagged", "reasons", "tier2_executed", "tier2_metrics"
        ])

    def add_row(self, page_idx: int, src: str, tgt: str, result: VerificationResult) -> None:
        self._writer.writerow([
            page_idx,
            src[:80],
            tgt[:80],
            result.flagged,
            "|".join(result.reasons),
            result.tier2_executed,
            str(result.tier2_metrics)
        ])

    def highlight_pdf(self, page: fitz.Page, bbox: fitz.Rect, result: VerificationResult) -> None:
        """Draw visual highlight on PDF for flagged segments."""
        if not result.flagged:
            return
            
        # Intent: Red highlight for T1 flags, Yellow if T2 passed (false positive).
        color = (1, 0.8, 0.8) if result.tier2_required and not result.tier2_executed else (1, 1, 0.6)
        if result.tier2_executed and not result.flagged:
             color = (0.8, 1, 0.8) # Green if T2 cleared it
             
        annot = page.add_highlight_annot(bbox)
        annot.set_colors(stroke=color)
        annot.update()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
