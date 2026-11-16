"""
01_data_cleaning.py
--------------------
Cleans the raw Kaggle "Netflix Movies and TV Shows" dataset and builds a
SQLite database used by the Streamlit dashboard.

INPUT:  data/netflix_titles.csv   (download from Kaggle, see README)
OUTPUT: data/netflix_clean.csv    (tidy, one genre+country per row)
        data/netflix.db           (SQLite database with 2 tables)

Run:    python 01_data_cleaning.py
"""

import pandas as pd
import numpy as np
import sqlite3
import re
from pathlib import Path

RAW_PATH = Path("data/netflix_titles.csv")
CLEAN_CSV_PATH = Path("data/netflix_clean.csv")
DB_PATH = Path("data/netflix.db")


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download 'netflix_titles.csv' from "
            "https://www.kaggle.com/datasets/shivamb/netflix-shows and place "
            "it in the data/ folder."
        )
    df = pd.read_csv(path)
    return df


def parse_duration(row) -> tuple:
    """Split the messy 'duration' column into numeric value + unit.
    Movies: '90 min' -> (90, 'min')
    TV Shows: '3 Seasons' -> (3, 'Season(s)')
    """
    if pd.isna(row["duration"]):
        return (np.nan, np.nan)
    match = re.match(r"(\d+)\s*(\w+)", str(row["duration"]))
    if not match:
        return (np.nan, np.nan)
    value, unit = match.groups()
    return (int(value), unit)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- basic type / whitespace cleanup ---
    for col in ["director", "cast", "country", "listed_in", "rating", "duration"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan})

    # --- fill missing country with "Unknown" (don't drop rows — country gaps
    # are actually part of the analysis, so we keep them but tag clearly) ---
    df["country"] = df["country"].fillna("Unknown")

    # Netflix often lists multiple countries e.g. "United States, India"
    # -> take a normalized list AND a "primary_country" (first listed)
    df["country_list"] = df["country"].apply(
        lambda x: [c.strip() for c in x.split(",") if c.strip()]
    )
    df["primary_country"] = df["country_list"].apply(
        lambda lst: lst[0] if lst else "Unknown"
    )

    # --- genres: "listed_in" is comma separated, e.g. "Dramas, Comedies" ---
    df["listed_in"] = df["listed_in"].fillna("Not Given")
    df["genre_list"] = df["listed_in"].apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip()]
    )

    # --- release year / date_added ---
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year

    # --- duration split ---
    df[["duration_value", "duration_unit"]] = df.apply(
        parse_duration, axis=1, result_type="expand"
    )

    # --- rating cleanup (a few rows have durations stuck in rating col) ---
    bad_ratings = df["rating"].isin(["74 min", "84 min", "66 min"])
    df.loc[bad_ratings, "rating"] = np.nan
    df["rating"] = df["rating"].fillna("Not Rated")

    # --- drop true duplicates ---
    df = df.drop_duplicates(subset=["title", "type", "release_year"])

    return df


def explode_long(df: pd.DataFrame) -> pd.DataFrame:
    """Create a tidy 'long' table: one row per (title, country, genre).
    This is what makes country x genre gap analysis possible."""
    long_df = df.explode("country_list").explode("genre_list")
    long_df = long_df.rename(columns={"country_list": "country_single",
                                       "genre_list": "genre_single"})
    long_df["country_single"] = long_df["country_single"].fillna("Unknown")
    long_df["genre_single"] = long_df["genre_single"].fillna("Not Given")
    return long_df


def main():
    print("Loading raw data...")
    raw = load_raw(RAW_PATH)
    print(f"  {len(raw)} rows loaded")

    print("Cleaning...")
    clean_df = clean(raw)

    print("Exploding country/genre for gap analysis...")
    long_df = explode_long(clean_df)

    # Save a flat CSV (lists -> joined strings so it opens fine in Excel/Sheets)
    export_df = clean_df.copy()
    export_df["country_list"] = export_df["country_list"].apply(lambda x: "|".join(x))
    export_df["genre_list"] = export_df["genre_list"].apply(lambda x: "|".join(x))
    export_df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Saved cleaned CSV -> {CLEAN_CSV_PATH}")

    # Build SQLite DB with two tables: titles (wide) and titles_long (tidy)
    conn = sqlite3.connect(DB_PATH)
    export_df.to_sql("titles", conn, if_exists="replace", index=False)

    long_export = long_df.drop(columns=["country_list", "genre_list"], errors="ignore").copy()
    # sqlite can't store python lists in remaining columns if any slipped through
    for col in long_export.columns:
        if long_export[col].apply(lambda v: isinstance(v, list)).any():
            long_export[col] = long_export[col].apply(
                lambda x: "|".join(x) if isinstance(x, list) else x
            )
    long_export.to_sql("titles_long", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved SQLite DB -> {DB_PATH} (tables: titles, titles_long)")

    print("\nDone. Quick sanity check:")
    print(f"  Unique titles: {clean_df['title'].nunique()}")
    print(f"  Unique countries (primary): {clean_df['primary_country'].nunique()}")
    print(f"  Date range (release_year): {clean_df['release_year'].min()}–{clean_df['release_year'].max()}")


if __name__ == "__main__":
    main()
