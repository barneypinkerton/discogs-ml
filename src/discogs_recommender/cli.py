"""Command-line interface for the recommendation pipeline."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Allow running without pip install -e .
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from discogs_recommender.config import load_config
from discogs_recommender.pipeline import STAGES, list_stages, run_stage, run_through
from discogs_recommender.ui.preferences import UserPreferences


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discogs recommendation pipeline",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        help=f"Stage to run. Available: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--through",
        metavar="STAGE",
        help="Run all stages up to and including STAGE",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print pipeline stages and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if cache files exist",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (default: config/default.yaml)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Launch the preference wizard before running. "
            "Prompts for styles, countries, and a year range to boost in scoring."
        ),
    )
    return parser


def _run_wizard(config: "AppConfig") -> UserPreferences:  # type: ignore[name-defined]
    """Connect to the DB, run the interactive wizard, close the connection."""
    from discogs_recommender.ui.wizard import run_wizard

    conn = sqlite3.connect(config.paths.db_path)
    try:
        return run_wizard(conn)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_stages:
        for i, name in enumerate(list_stages(), start=1):
            print(f"{i}. {name}")
        return

    config = load_config(args.config)

    user_prefs: UserPreferences | None = None
    if args.interactive:
        user_prefs = _run_wizard(config)

    if args.through:
        run_through(args.through, config, force=args.force, user_prefs=user_prefs)
    elif args.stage:
        run_stage(args.stage, config, force=args.force, user_prefs=user_prefs)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
