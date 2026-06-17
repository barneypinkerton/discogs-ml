"""User-specified preferences that stack on top of learned profile affinities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UserPreferences:
    preferred_styles: list[str] = field(default_factory=list)
    preferred_countries: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    boost_strength: float = 1.0
    vibe_profile_path: Path | None = None

    def is_empty(self) -> bool:
        return (
            not self.preferred_styles
            and not self.preferred_countries
            and self.year_from is None
            and self.year_to is None
        )
