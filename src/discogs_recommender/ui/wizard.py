"""Interactive pre-run wizard for explicit recommendation preferences."""

from __future__ import annotations

import sqlite3

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


def run_wizard(conn: sqlite3.Connection) -> UserPreferences:
    """Prompt the user for explicit style/country/year preferences.

    Choices are derived from the actual DB so only real options are shown.
    Returns a :class:`UserPreferences` that :func:`compute_final_score` will
    apply on top of the learned profile affinities.
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

    prefs = UserPreferences(
        preferred_styles=selected_styles,
        preferred_countries=selected_countries,
        year_from=year_from,
        year_to=year_to,
        boost_strength=boost_strength,
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

    return prefs
