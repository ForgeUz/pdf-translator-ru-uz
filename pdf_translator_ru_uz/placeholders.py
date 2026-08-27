# pdf_translator_ru_uz/placeholders.py

"""Module: centralised placeholder registry for the entire pipeline.

Replaces three independent, uncoordinated private-use-Unicode allocators
(parser math → U+E000..U+EFFF, engine NER → U+E800..U+EFFF, engine
LaTeX → U+F000..U+FFFF) with a single ``PlaceholderRegistry`` class.

Partitioned Private Use Area (PUA):
    MATH  = U+E000 – U+E7FF   (2048 slots, sequential)
    NER   = U+E800 – U+EFFF   (2048 slots, stable-hash with collision
                               detection + linear probing)
    LATEX = U+F000 – U+F7FF   (2048 slots, sequential)

All allocation is instance-scoped: each document gets its own registry,
reset at the start of ``extract_paragraphs()``, preventing cross-document
exhaustion and making placeholders deterministic within a single run.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── PUA partition boundaries ────────────────────────────────────────

MATH_START = 0xE000
MATH_END = 0xE7FF  # inclusive; 2048 slots
NER_START = 0xE800
NER_END = 0xEFFF  # inclusive; 2048 slots
LATEX_START = 0xF000
LATEX_END = 0xF7FF  # inclusive; 2048 slots


class PlaceholderExhaustedError(RuntimeError):
    """Raised when all slots in a PUA partition are exhausted."""


# ── Sequential allocator (Math / LaTeX) ─────────────────────────────


class _SequentialAllocator:
    """Allocates codepoints sequentially within [start, end]."""

    __slots__ = ("_start", "_end", "_next", "_lock")

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end
        self._next = start
        self._lock = threading.Lock()

    def allocate(self) -> int:
        """Return the next unused codepoint, or raise."""
        with self._lock:
            if self._next > self._end:
                raise PlaceholderExhaustedError(
                    f"Sequential allocator exhausted range "
                    f"U+{self._start:04X}–U+{self._end:04X} "
                    f"({self._end - self._start + 1} slots)"
                )
            code = self._next
            self._next += 1
        return code

    def reset(self) -> None:
        with self._lock:
            self._next = self._start


# ── Stable-hash allocator (NER) ─────────────────────────────────────


class _StableHashAllocator:
    """Allocates codepoints within [start, end] using a stable digest.

    Uses ``hashlib.blake2b(text.encode(), digest_size=2)`` for a
    reproducible 16-bit hash → mapped into the NER codepoint range.

    Collision resolution: linear probe forward until an unused slot is
    found (wrapping around if needed).  ``orig→placeholder`` is tracked
    so the same text always gets the same placeholder within one
    document.
    """

    __slots__ = (
        "_start", "_end", "_size",
        "_text_to_code", "_code_to_text", "_lock",
    )

    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end
        self._size = end - start + 1  # 2048
        self._text_to_code: dict[str, int] = {}
        self._code_to_text: dict[int, str] = {}
        self._lock = threading.Lock()

    def allocate(self, text: str) -> int:
        """Return a deterministic codepoint for *text*.

        Raises ``PlaceholderExhaustedError`` if the partition is full.
        """
        with self._lock:
            # Already allocated for this text?
            existing = self._text_to_code.get(text)
            if existing is not None:
                return existing

            if len(self._text_to_code) >= self._size:
                raise PlaceholderExhaustedError(
                    f"NER allocator exhausted range "
                    f"U+{self._start:04X}–U+{self._end:04X} "
                    f"({self._size} slots)"
                )

            # Stable 16-bit digest
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=2)
            slot = int.from_bytes(digest.digest(), byteorder="big") % self._size
            code = self._start + slot

            # Linear probe if collision with a *different* text
            while code in self._code_to_text and self._code_to_text[code] != text:
                slot = (slot + 1) % self._size
                code = self._start + slot

            self._text_to_code[text] = code
            self._code_to_text[code] = text
            return code

    def reset(self) -> None:
        with self._lock:
            self._text_to_code.clear()
            self._code_to_text.clear()


# ── Exhaustion info ─────────────────────────────────────────────────


@dataclass
class AllocationInfo:
    """Usage statistics for a single PUA partition."""
    kind: str
    allocated: int
    capacity: int
    exhausted: bool

    @property
    def pct(self) -> float:
        return (self.allocated / self.capacity) * 100 if self.capacity else 0.0


# ── PlaceholderRegistry — the public API ────────────────────────────


class PlaceholderRegistry:
    """One-stop manager for all private-use Unicode placeholders.

    Usage::

        registry = PlaceholderRegistry()
        math_ph = registry.mask_math()         # sequential, U+E000…
        ner_ph  = registry.mask_ner("entity")  # stable-hash, U+E800…
        tex_ph  = registry.mask_latex()        # sequential, U+F000…

        # Per-document reset:
        registry.reset()

        # Check exhaustion before a big document:
        if registry.exhausted("MATH"):
            logger.warning(...)
    """

    def __init__(self):
        self._math = _SequentialAllocator(MATH_START, MATH_END)
        self._ner = _StableHashAllocator(NER_START, NER_END)
        self._latex = _SequentialAllocator(LATEX_START, LATEX_END)

    # ── Public API ────────────────────────────────────────────────

    def mask_math(self) -> str:
        """Return the next private-use char for a math glyph."""
        return chr(self._math.allocate())

    def mask_ner(self, text: str) -> str:
        """Return a deterministic private-use char for an NER entity."""
        return chr(self._ner.allocate(text))

    def mask_latex(self) -> str:
        """Return the next private-use char for a LaTeX block."""
        return chr(self._latex.allocate())

    def reset(self) -> None:
        """Reset all allocators (call at start of each document)."""
        self._math.reset()
        self._ner.reset()
        self._latex.reset()

    def exhaustion_info(self) -> list[AllocationInfo]:
        """Return usage stats for every partition."""
        return [
            AllocationInfo(
                kind="MATH",
                allocated=self._math._next - self._math._start,  # type: ignore[attr-defined]
                capacity=MATH_END - MATH_START + 1,
                exhausted=self._math._next > self._math._end,  # type: ignore[attr-defined]
            ),
            AllocationInfo(
                kind="NER",
                allocated=len(self._ner._text_to_code),  # type: ignore[attr-defined]
                capacity=NER_END - NER_START + 1,
                exhausted=len(self._ner._text_to_code) >= NER_END - NER_START + 1,  # type: ignore[attr-defined]
            ),
            AllocationInfo(
                kind="LATEX",
                allocated=self._latex._next - self._latex._start,  # type: ignore[attr-defined]
                capacity=LATEX_END - LATEX_START + 1,
                exhausted=self._latex._next > self._latex._end,  # type: ignore[attr-defined]
            ),
        ]

    def exhausted(self, kind: str) -> bool:
        """Return True if a partition is fully exhausted."""
        for info in self.exhaustion_info():
            if info.kind == kind:
                return info.exhausted
        return False

    def allocated_count(self, kind: str) -> int:
        """Return how many slots of *kind* have been allocated."""
        for info in self.exhaustion_info():
            if info.kind == kind:
                return info.allocated
        return 0
