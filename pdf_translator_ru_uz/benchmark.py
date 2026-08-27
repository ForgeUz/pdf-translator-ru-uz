# pdf_translator_ru_uz/benchmark.py

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pdf_translator_ru_uz.validation_corpus import ValidationCorpus

logger = logging.getLogger(__name__)

# Intent: Abstract engine details so benchmark evaluates capability, not plumbing.
# State Transition: Disparate model APIs -> Unified CandidateEngine interface.

@dataclass
class EngineConfig:
    name: str
    backend: str  # "mock", "transformers", "ctranslate2", "llama.cpp"
    model_path: str
    quantization: Optional[str] = None  # e.g., "Q4_K_M"
    context_window: int = 512

class CandidateEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        
    def load(self) -> None:
        if self.config.backend == "mock":
            return
        raise NotImplementedError(f"Backend {self.config.backend} not implemented in base class")
        
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if self.config.backend == "mock":
            return f"[{self.config.name}] {text}"
        raise NotImplementedError

    def unload(self) -> None:
        self._model = None

# Intent: Provide baseline NLLB-3.3B adapter to prove capacity vs architecture failure (ADR-1).
# Dependencies: Existing NLLBEngine plumbing in engine.py.
class NLLBCandidateAdapter(CandidateEngine):
    def load(self):
        from pdf_translator_ru_uz.engine import NLLBEngine
        self._model = NLLBEngine(
            model_name="facebook/nllb-200-3.3B",
            model_path=self.config.model_path,
            backend="ctranslate2"
        )
        self._model._ensure_ct2_loaded()
        
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return self._model.translate(text, src_lang, tgt_lang)

@dataclass
class BenchmarkResult:
    engine_name: str
    translations: List[str]
    # Human review fields (populated manually after generation)
    hallucination_detected: bool = False
    fluency_score: int = 0  # 1-5
    notes: str = ""

@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)

class BenchmarkHarness:
    def __init__(self, corpus: ValidationCorpus):
        self.corpus = corpus
        
    def run(self, engines: List[CandidateEngine]) -> BenchmarkReport:
        report = BenchmarkReport()
        for engine in engines:
            logger.info(f"Loading engine: {engine.config.name}")
            engine.load()
            
            translations = []
            for p in self.corpus.pairs:
                try:
                    out = engine.translate(p.source_text, p.source_lang, p.target_lang)
                except Exception as e:
                    out = f[ERROR: {e}]
                translations.append(out)
                
            report.results.append(
                BenchmarkResult(
                    engine_name=engine.config.name,
                    translations=translations
                )
            )
            engine.unload()
        return report
