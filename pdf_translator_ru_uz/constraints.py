# pdf_translator_ru_uz/constraints.py

from __future__ import annotations
from typing import Dict, Optional

# Intent: Mechanical enforcement of glossary terms.
# State Transition: Unconstrained generation -> Forced term alignment.
class ConstrainedDecoder:
    def __init__(self):
        self._locks: Dict[str, str] = {}
        
    def lock_term(self, source: str, target: str) -> None:
        self._locks[source] = target
        
    def is_locked(self, source: str) -> bool:
        return source in self._locks
        
    def get_target(self, source: str) -> Optional[str]:
        return self._locks.get(source)
        
    def enforce_in_output(self, source_text: str, output_text: str) -> bool:
        """Post-hoc check for LLMs: did locked terms appear in output?"""
        for src, tgt in self._locks.items():
            if src in source_text and tgt not in output_text:
                return False
        return True
