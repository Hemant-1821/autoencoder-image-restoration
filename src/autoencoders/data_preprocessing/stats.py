"""Shared counters for the offline cleaning pipeline (see plan.md, Stage 1)."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CleaningStats:
    scanned: int = 0
    kept: int = 0
    dropped: dict = field(default_factory=lambda: defaultdict(int))
    split_counts: dict = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1
