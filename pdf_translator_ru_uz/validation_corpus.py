# pdf_translator_ru_uz/validation_corpus.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# Intent: Provide a strict data contract for benchmark validation.
# State Transition: Unverified text -> Human-verified ground truth pair.
@dataclass
class ValidationPair:
    source_text: str
    reference_translation: str
    source_lang: str = "uz"
    target_lang: str = "ru"
    domain: str = "legal"  # e.g., constitution, civil_code
    verified_by: str = "manual"

# Intent: Ground truth for hallucination/fluency scoring.
# Dependencies: Requires manual collection of 10-20 pairs.
@dataclass
class ValidationCorpus:
    pairs: List[ValidationPair] = field(default_factory=list)
    
    def add_pair(self, source: str, reference: str, domain: str = "legal") -> None:
        self.pairs.append(
            ValidationPair(
                source_text=source, 
                reference_translation=reference, 
                domain=domain
            )
        )
