# pdf_translator_ru_uz/validation_runner.py

from __future__ import annotations
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Intent: Enforce v1 exit criteria. Block release if mechanical verification fails.
@dataclass
class ValidationResult:
    passed: bool
    summary: str
    checklist: str

class ValidationRunner:
    def __init__(self, flags_csv_path: str):
        self.csv_path = Path(flags_csv_path)
        
    def run(self) -> ValidationResult:
        if not self.csv_path.exists():
            return ValidationResult(False, f"Report not found: {self.csv_path}", "")
            
        unresolved_count = 0
        total_segments = 0
        
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_segments += 1
                if row.get("flagged") == "True":
                    unresolved_count += 1
                    
        if unresolved_count > 0:
            summary = f"FAIL: Unresolved flags: {unresolved_count} / {total_segments}"
            return ValidationResult(False, summary, "")
            
        summary = f"PASS: 0 unresolved flags across {total_segments} segments."
        checklist = self._generate_hitl_checklist()
        return ValidationResult(True, summary, checklist)
        
    @staticmethod
    def _generate_hitl_checklist() -> str:
        """Emit manual verification steps for v1 sign-off."""
        return """
        [v1 EXIT VALIDATION - MANUAL CHECKLIST]
        1. Visual Layout: Compare output.pdf to input.pdf. Verify tables, images, and bboxes match 1:1.
        2. Spot Check: Randomly select 10-15% of unflagged segments. Verify no false negatives.
        3. Typography: Check Russian micro-typography (guillemets, non-breaking spaces).
        4. Sign off if all above pass.
        """
