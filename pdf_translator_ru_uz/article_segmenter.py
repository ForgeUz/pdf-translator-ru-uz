# pdf_translator_ru_uz/article_segmenter.py

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from pdf_translator_ru_uz.parser import Paragraph

logger = logging.getLogger(__name__)

# Intent: Define hierarchical structure for sibling-context chunking (ADR-2).
@dataclass
class Article:
    article_id: str
    heading_text: Optional[str]
    paragraphs: List[Paragraph]
    detection_method: str  # "regex" | "heading_heuristic" | "undetected"

# Intent: High-precision textual markers for Uzbek/Cyrillic legal docs (ADR-4).
# Matches: "1-modda", "1-модда", "Modda 1", "Модда 1"
_REGEX_PATTERNS = [
    re.compile(r"^(\d+)[-]?\s*(?:modda|модда)[\.\s]", re.IGNORECASE),
    re.compile(r"^(?:modda|модда)\s+(\d+)[\.\s]", re.IGNORECASE),
]

class ArticleSegmenter:
    def __init__(self, heading_size_ratio: float = 1.2):
        self.heading_size_ratio = heading_size_ratio
        
    def segment(self, paragraphs: List[Paragraph]) -> List[Article]:
        if not paragraphs:
            return []
            
        # Try regex primary path
        articles = self._segment_by_regex(paragraphs)
        if articles:
            return articles
            
        # Fallback: heading heuristic based on font size
        return self._segment_by_heading_heuristic(paragraphs)
        
    def _segment_by_regex(self, paragraphs: List[Paragraph]) -> List[Article]:
        articles: List[Article] = []
        current_article: Optional[Article] = None
        block_counter = 0
        
        # Pre-scan: if NO regex matches, return [] to trigger fallback
        has_match = any(self._match_regex(p.text) for p in paragraphs)
        if not has_match:
            return []
            
        for p in paragraphs:
            match = self._match_regex(p.text)
            if match:
                if current_article:
                    articles.append(current_article)
                    
                article_id = match.group(1) if match.lastindex else str(block_counter)
                if not article_id.startswith(("modda", "модда", "block")):
                     article_id = f"{article_id}-modda"
                     
                current_article = Article(
                    article_id=article_id,
                    heading_text=p.text,
                    paragraphs=[p],
                    detection_method="regex"
                )
            else:
                if current_article is None:
                    # Orphan text before first article
                    current_article = Article(
                        article_id=f"block_{block_counter}",
                        heading_text=None,
                        paragraphs=[p],
                        detection_method="undetected"
                    )
                    block_counter += 1
                else:
                    current_article.paragraphs.append(p)
                    
        if current_article:
            articles.append(current_article)
            
        return articles
        
    def _segment_by_heading_heuristic(self, paragraphs: List[Paragraph]) -> List[Article]:
        if not paragraphs:
            return []
            
        # Calculate median font size to detect outliers
        sizes = sorted([p.fontsize for p in paragraphs])
        median_size = sizes[len(sizes) // 2] if sizes else 11.0
        
        articles: List[Article] = []
        current_article = Article(
            article_id="block_0",
            heading_text=None,
            paragraphs=[],
            detection_method="heading_heuristic"
        )
        
        for p in paragraphs:
            is_heading = (
                p.fontsize >= median_size * self.heading_size_ratio or
                "bold" in p.font.lower()
            )
            
            if is_heading and current_article.paragraphs:
                articles.append(current_article)
                current_article = Article(
                    article_id=f"block_{len(articles)}",
                    heading_text=p.text,
                    paragraphs=[p],
                    detection_method="heading_heuristic"
                )
            else:
                current_article.paragraphs.append(p)
                
        if current_article.paragraphs:
            articles.append(current_article)
            
        return articles
        
    @staticmethod
    def _match_regex(text: str) -> Optional[re.Match]:
        text = text.strip()
        for pattern in _REGEX_PATTERNS:
            match = pattern.match(text)
            if match:
                return match
        return None
