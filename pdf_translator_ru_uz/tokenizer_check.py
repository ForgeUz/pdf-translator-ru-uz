# pdf_translator_ru_uz/tokenizer_check.py

from __future__ import annotations
import logging
from typing import Dict
from transformers import AutoTokenizer
from pdf_translator_ru_uz.placeholders import PlaceholderRegistry

logger = logging.getLogger(__name__)

def check_pua_survival(model_name: str) -> Dict:
    """Intent: Verify if candidate model tokenizer preserves PUA chars.
    
    State Transition: Unverified model -> Validated/Invalidated for placeholder strategy.
    Dependencies: Requires HuggingFace model name and internet/local cache to load tokenizer.
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")
        return {
            "model": model_name, 
            "error": str(e), 
            "survives": False
        }

    registry = PlaceholderRegistry()
    
    # Allocate one of each PUA type
    math_ph = registry.mask_math()
    ner_ph = registry.mask_ner("TestEntity")
    latex_ph = registry.mask_latex()
    
    text = f"Test {math_ph} {ner_ph} {latex_ph} end"
    
    # Encode and decode without special tokens interfering
    ids = tok(text, add_special_tokens=False)["input_ids"]
    decoded = tok.decode(ids)
    
    math_survives = math_ph in decoded
    ner_survives = ner_ph in decoded
    latex_survives = latex_ph in decoded
    
    return {
        "model": model_name,
        "math_survives": math_survives,
        "ner_survives": ner_survives,
        "latex_survives": latex_survives,
        "survives": all([math_survives, ner_survives, latex_survives])
    }
