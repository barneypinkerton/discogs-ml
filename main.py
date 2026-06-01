#!/usr/bin/env python3
"""Entry point for the Discogs recommendation pipeline."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from discogs_recommender.cli import main

if __name__ == "__main__":
    main()
