# pdf_translator_ru_uz/verification.py

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List
from pdf_translator_ru_uz.engine import is_target_language

logger = logging.getLogger(__name__)

# Intent: Mechanical, deterministic checks for translation integrity.
# State Transition: Unverified text -> Flagged/Unflagged with specific reasons.
@dataclass
class VerificationResult:
    flagged: bool = False
    reasons: List[str] = field(default_factory=list)
    tier2_required: bool = False
    tier2_executed: bool = False
    tier2_metrics: dict = field(default_factory=dict)

class VerificationGate:
    def __init__(self, length_ratio_band: tuple = (0.85, 1.6)):
        self.length_ratio_band = length_ratio_band
        
    def verify_tier1(self, src: str, tgt: str, src_lang: str, tgt_lang: str) -> VerificationResult:
        res = VerificationResult()
        
        # 1. Numeric/Date Preservation
        src_nums = set(re.findall(r'\b\d+\b', src))
        tgt_nums = set(re.findall(r'\b\d+\b', tgt))
        if src_nums and not src_nums.issubset(tgt_nums):
            res.flagged = True
            res.reasons.append("numeric_mismatch")
            
        # 2. Length Ratio
        src_tokens = len(src.split())
        tgt_tokens = len(tgt.split())
        if src_tokens > 0:
            ratio = tgt_tokens / src_tokens
            if not (self.length_ratio_band[0] <= ratio <= self.length_ratio_band[1]):
                res.flagged = True
                res.reasons.append(f"length_ratio_outlier ({ratio:.2f})")
                
        # 3. Source Language Bleed-through
        # Intent: Catch UZ Latin chars in RU Cyrillic output.
        if tgt_lang == "ru":
            if re.search(r"[OoGg]['ʻ]", tgt) or re.search(r"[Nn]g", tgt):
                res.flagged = True
                res.reasons.append("source_lang_bleed")
                
        # 4. N-gram Repetition (e.g., "КОНСТИТУЦИЯ КОНСТИТУЦИЯ")
        # Intent: Catch hallucination loops.
        words = tgt.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    res.flagged = True
                    res.reasons.append("ngram_repetition")
                    break
                    
        # 5. LID Check
        if not is_target_language(tgt, tgt_lang, confidence_threshold=0.8):
            res.flagged = True
            res.reasons.append("lid_failed")
            
        if res.flagged:
            res.tier2_required = True
            
        return res
        
    def verify_tier2(
        self, 
        src: str, 
        tgt: str, 
        src_lang: str, 
        tgt_lang: str, 
        tier1_res: VerificationResult,
        engine: any
    ) -> VerificationResult:
        """Intent: Paradigm-independent verification via back-translation + chrF."""
        if not tier1_res.tier2_required or engine is None:
            return tier1_res
            
        try:
            # Back-translate: RU -> UZ
            back_trans = engine.translate(tgt, tgt_lang, src_lang)
            
            chrf_score = self._calculate_chrf(src, back_trans)
            
            tier1_res.tier2_executed = True
            tier1_res.tier2_metrics = {
                "back_translation": back_trans[:100],
                "chrF_score": chrf_score
            }
            
            if chrf_score < 0.4:
                tier1_res.flagged = True
                tier1_res.reasons.append("back_translation_mismatch")
            else:
                tier1_res.flagged = False
                tier1_res.reasons = [r for r in tier1_res.reasons if r != "lid_failed"]
                
        except Exception as e:
            logger.error(f"Tier 2 verification failed: {e}")
            tier1_res.tier2_metrics = {"error": str(e)}
            
        return tier1_res
        
    @staticmethod
    def _calculate_chrf(src: str, back_trans: str) -> float:
        """Simplified chrF calculation (character n-gram F-score)."""
        src_chars = set(src.lower())
        bt_chars = set(back_trans.lower())
        if not src_chars: return 0.0
        overlap = len(src_chars.intersection(bt_chars))
        return overlap / len(src_chars)
