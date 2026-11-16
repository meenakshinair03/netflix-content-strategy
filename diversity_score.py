"""
diversity_score.py
-------------------
THE ORIGINAL PART OF THIS PROJECT.

Defines the "Content Diversity Index" (CDI): a single 0-100 score per country
that answers "how well-rounded and current is Netflix's catalog here?"

CDI is a weighted blend of three sub-scores, each 0-1:

1. EVENNESS (40%)   - Shannon entropy of genre distribution, normalized.
                       A country with titles spread evenly across many genres
                       scores high. A country where 90% of titles are one
                       genre (e.g. only Dramas) scores low, even if it has
                       a lot of titles.

2. COVERAGE (35%)   - What fraction of ALL genres available on Netflix
                       globally are present at all in this country?
                       Captures "how many genres does this region get access
                       to" regardless of the mix.

3. FRESHNESS (25%)  - Share of that country's titles released in the last
                       5 years (relative to the dataset's max release year).
                       Captures whether the region's catalog is being kept
                       current vs. relying on older content.

    CDI = 100 * (0.40*evenness + 0.35*coverage + 0.25*freshness)

Why this design:
- Pure title COUNT is a bad proxy for "good content strategy" — a country
  could have 500 titles that are 90% Dramas, which is not a healthy catalog.
- Entropy rewards spread, not just volume, which is the actual gap Netflix's
  content team would care about when deciding where to invest.
- Freshness catches regions being "back-filled" with catalog titles instead
  of getting new investment.

Any team can swap the weights in WEIGHTS below to match their own priorities.
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "evenness": 0.40,
    "coverage": 0.35,
    "freshness": 0.25,
}

FRESHNESS_WINDOW_YEARS = 5
MIN_TITLES_FOR_SCORE = 10  # countries with fewer titles get excluded / flagged


def shannon_entropy(counts: pd.Series) -> float:
    """Shannon entropy of a distribution, in bits."""
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def compute_diversity_index(long_df: pd.DataFrame, titles_df: pd.DataFrame,
                             min_titles: int = MIN_TITLES_FOR_SCORE) -> pd.DataFrame:
    """
    long_df: tidy dataframe with columns ['title', 'country_single', 'genre_single']
    titles_df: wide dataframe with columns ['title', 'primary_country', 'release_year']

    Returns a dataframe: country, n_titles, evenness, coverage, freshness, CDI
    """
    all_genres = long_df["genre_single"].nunique()
    max_release_year = titles_df["release_year"].max()
    freshness_cutoff = max_release_year - FRESHNESS_WINDOW_YEARS

    rows = []
    for country, grp in long_df.groupby("country_single"):
        if country == "Unknown":
            continue

        n_titles = grp["title"].nunique()
        if n_titles < min_titles:
            continue  # too few titles to score meaningfully

        # --- evenness ---
        genre_counts = grp.groupby("genre_single")["title"].nunique()
        h = shannon_entropy(genre_counts)
        max_h = np.log2(len(genre_counts)) if len(genre_counts) > 1 else 1
        evenness = h / max_h if max_h > 0 else 0

        # --- coverage ---
        n_genres_here = grp["genre_single"].nunique()
        coverage = n_genres_here / all_genres

        # --- freshness ---
        country_titles = titles_df[titles_df["primary_country"] == country]
        if len(country_titles) == 0:
            freshness = 0
        else:
            fresh_count = (country_titles["release_year"] >= freshness_cutoff).sum()
            freshness = fresh_count / len(country_titles)

        cdi = 100 * (
            WEIGHTS["evenness"] * evenness
            + WEIGHTS["coverage"] * coverage
            + WEIGHTS["freshness"] * freshness
        )

        rows.append({
            "country": country,
            "n_titles": n_titles,
            "evenness": round(evenness, 3),
            "coverage": round(coverage, 3),
            "freshness": round(freshness, 3),
            "CDI": round(cdi, 1),
        })

    result = pd.DataFrame(rows).sort_values("CDI", ascending=False).reset_index(drop=True)
    return result


def top_missing_genres(long_df: pd.DataFrame, country: str, top_n: int = 5) -> pd.DataFrame:
    """For a given country, return the genres with the biggest gap vs. the
    global genre share — i.e. what Netflix's content team should invest in
    next for that region."""
    global_share = (
        long_df.groupby("genre_single")["title"].nunique()
        / long_df["title"].nunique()
    )

    country_grp = long_df[long_df["country_single"] == country]
    country_titles = country_grp["title"].nunique()
    country_share = (
        country_grp.groupby("genre_single")["title"].nunique() / country_titles
        if country_titles > 0 else pd.Series(dtype=float)
    )

    gap = (global_share - country_share.reindex(global_share.index).fillna(0))
    gap = gap.sort_values(ascending=False).head(top_n)
    return gap.rename("underrepresentation_gap").reset_index().rename(
        columns={"genre_single": "genre"}
    )
