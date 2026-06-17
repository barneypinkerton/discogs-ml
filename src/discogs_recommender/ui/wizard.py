"""Interactive pre-run wizard for explicit recommendation preferences."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import questionary

from discogs_recommender.ui.preferences import UserPreferences

_TOP_STYLES_SQL = """
    SELECT rs.style, COUNT(*) AS n
    FROM release_style rs
    JOIN release_genre rg ON rs.release_id = rg.release_id
    WHERE rg.genre = 'Electronic'
    GROUP BY rs.style
    ORDER BY n DESC
    LIMIT 40
"""

_TOP_COUNTRIES_SQL = """
    SELECT r.country, COUNT(*) AS n
    FROM release r
    JOIN release_genre rg ON r.id = rg.release_id
    WHERE rg.genre = 'Electronic'
      AND r.country IS NOT NULL AND r.country != ''
    GROUP BY r.country
    ORDER BY n DESC
    LIMIT 25
"""

_BOOST_CHOICES = [
    questionary.Choice("Subtle  (+0.5)", value=0.5),
    questionary.Choice("Moderate (+1.0)", value=1.0),
    questionary.Choice("Strong  (+2.0)", value=2.0),
]


def run_wizard(conn: sqlite3.Connection, config=None) -> UserPreferences:
    """Prompt the user for explicit style/country/year preferences.

    Choices are derived from the actual DB so only real options are shown.
    Returns a :class:`UserPreferences` that :func:`compute_final_score` will
    apply on top of the learned profile affinities.

    If *config* is provided, also offers to build a custom vibe-profile NPZ
    from a folder of favourite songs for use in audio ranking.
    """
    print("\n=== Recommendation Preferences ===")
    print("Learned affinities from your collection/wantlist are always applied.")
    print("Use this wizard to add an extra boost to specific styles, countries, or eras.\n")

    styles = [row[0] for row in conn.execute(_TOP_STYLES_SQL).fetchall()]
    selected_styles: list[str] = questionary.checkbox(
        "Boost specific styles? (space to select, enter to skip all)",
        choices=styles,
    ).ask() or []

    countries = [row[0] for row in conn.execute(_TOP_COUNTRIES_SQL).fetchall()]
    selected_countries: list[str] = questionary.checkbox(
        "Preferred release countries? (space to select, enter to skip all)",
        choices=countries,
    ).ask() or []

    year_from: int | None = None
    year_to: int | None = None
    year_input: str = (
        questionary.text(
            "Preferred year range? (e.g. 1993-2005, leave blank to skip)",
            default="",
        ).ask()
        or ""
    ).strip()
    if "-" in year_input:
        parts = year_input.split("-", 1)
        try:
            year_from = int(parts[0].strip())
            year_to = int(parts[1].strip())
        except ValueError:
            print("  Could not parse year range — skipping.")

    boost_strength: float = 1.0
    if selected_styles or selected_countries or year_from is not None:
        result = questionary.select(
            "How strongly should explicit preferences override learned affinities?",
            choices=_BOOST_CHOICES,
        ).ask()
        if result is not None:
            boost_strength = result

    vibe_profile_path: Path | None = None
    if config is not None:
        vibe_profile_path = _ask_vibe_profile(config)

    prefs = UserPreferences(
        preferred_styles=selected_styles,
        preferred_countries=selected_countries,
        year_from=year_from,
        year_to=year_to,
        boost_strength=boost_strength,
        vibe_profile_path=vibe_profile_path,
    )

    if prefs.is_empty():
        print("\nNo explicit preferences set — using learned affinities only.\n")
    else:
        parts = []
        if selected_styles:
            parts.append(f"{len(selected_styles)} style(s): {', '.join(selected_styles[:3])}{'…' if len(selected_styles) > 3 else ''}")
        if selected_countries:
            parts.append(f"{len(selected_countries)} country(ies): {', '.join(selected_countries[:3])}{'…' if len(selected_countries) > 3 else ''}")
        if year_from and year_to:
            parts.append(f"years {year_from}–{year_to}")
        print(f"\nPreferences: {' | '.join(parts)} | boost +{boost_strength}\n")

    if vibe_profile_path:
        print(f"Audio vibe profile: {vibe_profile_path.name}\n")

    return prefs


def _ask_vibe_profile(config) -> Path | None:
    """Optionally build a custom vibe-profile NPZ from a folder of songs."""
    print("\n=== Audio Vibe Profile ===")
    wants_vibe = questionary.confirm(
        "Build a custom audio vibe from specific songs? (uses EffNet — takes a minute)",
        default=False,
    ).ask()

    if not wants_vibe:
        return None

    music_dir_str = questionary.path(
        "Folder containing your favourite songs (MP3/WAV/FLAC):",
    ).ask()
    if not music_dir_str:
        print("  No folder given — skipping vibe profile.")
        return None

    music_dir = Path(music_dir_str.replace("\\ ", " ")).expanduser().resolve()
    if not music_dir.is_dir():
        print(f"  Folder not found: {music_dir} — skipping.")
        return None

    default_name = f"vibe_{date.today().isoformat()}.npz"
    profile_name = (
        questionary.text(
            "Name for this vibe profile (saved to your embeddings folder):",
            default=default_name,
        ).ask()
        or default_name
    ).strip()
    if not profile_name.endswith(".npz"):
        profile_name += ".npz"

    save_path = config.paths.embeddings_dir / profile_name

    print(f"\nBuilding vibe profile from {music_dir} …")
    try:
        from discogs_recommender.audio.embed import build_vibe_profile
        return build_vibe_profile(
            music_dir,
            config.paths.effnet_model,
            save_path,
        )
    except Exception as exc:
        print(f"  ERROR building vibe profile: {exc}")
        return None
