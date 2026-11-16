"""
app.py
------
Netflix Content Strategy Analyzer — Streamlit dashboard.

Answers: "If you were Netflix's content strategy team, where should they
invest next?" via four lenses: country x genre gaps, release-year trends,
duration/runtime patterns, and a custom Content Diversity Index.

Run:  streamlit run app.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from diversity_score import compute_diversity_index, top_missing_genres

DB_PATH = Path("data/netflix.db")

st.set_page_config(
    page_title="Netflix Content Strategy Analyzer",
    page_icon="🎬",
    layout="wide",
)


# ----------------------------- DATA LOADING ------------------------------ #

@st.cache_data
def load_data():
    if not DB_PATH.exists():
        st.error(
            "Database not found. Run `python 01_data_cleaning.py` first "
            "(after placing netflix_titles.csv in the data/ folder)."
        )
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    titles = pd.read_sql("SELECT * FROM titles", conn)
    long_df = pd.read_sql("SELECT * FROM titles_long", conn)
    conn.close()
    return titles, long_df


titles_df, long_df = load_data()

TOP_COUNTRIES = (
    long_df[long_df["country_single"] != "Unknown"]
    .groupby("country_single")["title"].nunique()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)


# --------------------------------- HEADER --------------------------------- #

st.title("🎬 Netflix Content Strategy Analyzer")
st.caption(
    "Where should Netflix invest next? A gap analysis across region, genre, "
    "release-year trends, and runtime — plus a custom Content Diversity Index."
)

with st.sidebar:
    st.header("Filters")
    selected_countries = st.multiselect(
        "Countries to include",
        options=TOP_COUNTRIES,
        default=TOP_COUNTRIES[:8],
    )
    content_type = st.radio("Content type", ["All", "Movie", "TV Show"], horizontal=True)
    st.markdown("---")
    st.markdown(
        "**Dataset:** [Kaggle Netflix Titles]"
        "(https://www.kaggle.com/datasets/shivamb/netflix-shows)"
    )

type_filter = None if content_type == "All" else content_type

# --------------------------- APPLY FILTERS -------------------------------- #
# `active_countries` is what most tabs use to decide which countries to
# show. If the user has explicitly picked some, use exactly those; if the
# multiselect is empty, fall back to the top 15 so the app doesn't go blank.
active_countries = selected_countries if selected_countries else TOP_COUNTRIES

# Type-filtered copies (respect the Movie/TV Show/All radio everywhere)
titles_type_scoped = titles_df if type_filter is None else titles_df[titles_df["type"] == type_filter]
long_type_scoped = long_df if type_filter is None else long_df[long_df["type"] == type_filter]

# Type + country filtered copies (respect both controls together)
titles_scoped = titles_type_scoped[titles_type_scoped["primary_country"].isin(active_countries)]
long_scoped = long_type_scoped[long_type_scoped["country_single"].isin(active_countries)]

if not selected_countries:
    st.sidebar.caption("No countries selected — showing the top 15 by catalog size.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🌍 Content Gaps", "📅 Release Trends",
     "⏱️ Duration Patterns", "🧭 Diversity Index & Recommendations"]
)


# --------------------------------- TAB 1 ---------------------------------- #

with tab1:
    st.caption(f"Showing {len(active_countries)} countries" +
               (f" · {type_filter}s only" if type_filter else " · Movies + TV Shows"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total titles", f"{titles_scoped['title'].nunique():,}")
    col2.metric("Countries shown", f"{long_scoped['country_single'].nunique():,}")
    col3.metric("Genres represented", f"{long_scoped['genre_single'].nunique():,}")
    if titles_scoped["release_year"].notna().any():
        span = f"{int(titles_scoped['release_year'].min())}–{int(titles_scoped['release_year'].max())}"
    else:
        span = "—"
    col4.metric("Release year span", span)

    c1, c2 = st.columns(2)
    with c1:
        if type_filter is None:
            type_counts = titles_scoped["type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
            fig = px.pie(type_counts, names="type", values="count",
                         title="Movies vs TV Shows", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Content type filter is set to **{type_filter}** — "
                    "switch to 'All' in the sidebar to see the Movie/TV split.")
    with c2:
        top_genre = (
            long_scoped.groupby("genre_single")["title"].nunique()
            .sort_values(ascending=False).head(10).reset_index()
        )
        top_genre.columns = ["genre", "count"]
        fig = px.bar(top_genre, x="count", y="genre", orientation="h",
                     title="Top 10 Genres (in selected countries)")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------- TAB 2 ---------------------------------- #

with tab2:
    st.subheader("Which genres are underrepresented in which countries?")
    st.caption(
        "Each cell shows a country's share of its own catalog in that genre, "
        "minus the global share of that genre. Negative (red) = underrepresented "
        "relative to the rest of the world — a potential investment gap."
    )

    # "Global" baseline respects the Movie/TV Show filter but NOT the country
    # filter — a country's mix is compared against the whole world on purpose.
    global_share = (
        long_type_scoped.groupby("genre_single")["title"].nunique()
        / long_type_scoped["title"].nunique()
    )
    top_genres_global = global_share.sort_values(ascending=False).head(12).index.tolist()

    pivot_rows = []
    for country in active_countries:
        c_grp = long_type_scoped[long_type_scoped["country_single"] == country]
        c_total = c_grp["title"].nunique()
        c_share = c_grp.groupby("genre_single")["title"].nunique() / c_total if c_total else pd.Series(dtype=float)
        for genre in top_genres_global:
            gap = c_share.get(genre, 0) - global_share.get(genre, 0)
            pivot_rows.append({"country": country, "genre": genre, "gap": gap})

    gap_df = pd.DataFrame(pivot_rows)
    pivot = gap_df.pivot(index="country", columns="genre", values="gap")

    fig = px.imshow(
        pivot, color_continuous_scale="RdBu", color_continuous_midpoint=0,
        aspect="auto", labels=dict(color="Gap vs global"),
        title="Content Gap Heatmap (country x genre)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Biggest single gaps (opportunities)")
    country_pick = st.selectbox("Pick a country for detail", active_countries)
    gaps = top_missing_genres(long_type_scoped, country_pick, top_n=6)
    fig2 = px.bar(gaps, x="underrepresentation_gap", y="genre", orientation="h",
                  title=f"Top underrepresented genres in {country_pick}")
    fig2.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------- TAB 3 ---------------------------------- #

with tab3:
    st.subheader("Is Netflix aging its catalog or refreshing it?")
    st.caption(f"Showing {len(active_countries)} countries" +
               (f" · {type_filter}s only" if type_filter else ""))

    scope = titles_scoped

    c1, c2 = st.columns(2)
    with c1:
        by_release = scope.dropna(subset=["release_year"]).groupby("release_year")["title"].nunique().reset_index()
        by_release.columns = ["release_year", "titles"]
        fig = px.line(by_release, x="release_year", y="titles",
                      title="Titles by Release Year (when content was made)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        by_added = scope.dropna(subset=["year_added"]).groupby("year_added")["title"].nunique().reset_index()
        by_added.columns = ["year_added", "titles"]
        fig = px.line(by_added, x="year_added", y="titles",
                      title="Titles by Year Added to Netflix")
        st.plotly_chart(fig, use_container_width=True)

    age_df = scope.dropna(subset=["release_year", "year_added"]).copy()
    age_df["content_age_at_addition"] = age_df["year_added"] - age_df["release_year"]
    avg_age = age_df.groupby("year_added")["content_age_at_addition"].mean().reset_index()
    fig3 = px.bar(avg_age, x="year_added", y="content_age_at_addition",
                  title="Average Age of Content When Added (years) — rising line = catalog getting staler")
    st.plotly_chart(fig3, use_container_width=True)

    latest_avg_age = avg_age["content_age_at_addition"].iloc[-1] if len(avg_age) else None
    if latest_avg_age is not None:
        st.info(
            f"In the most recent year of additions in this dataset, content added "
            f"was on average **{latest_avg_age:.1f} years old** at the time it was added."
        )


# --------------------------------- TAB 4 ---------------------------------- #

with tab4:
    st.subheader("Runtime / duration patterns")
    st.caption(f"Showing {len(active_countries)} countries" +
               (f" · {type_filter}s only" if type_filter else ""))

    movies = titles_scoped[(titles_scoped["type"] == "Movie") & titles_scoped["duration_value"].notna()]
    shows = titles_scoped[(titles_scoped["type"] == "TV Show") & titles_scoped["duration_value"].notna()]

    # Only render the column(s) relevant to the current content-type filter
    show_movies = type_filter in (None, "Movie")
    show_tv = type_filter in (None, "TV Show")
    cols = st.columns(int(show_movies) + int(show_tv)) if (show_movies and show_tv) else [st.container()]
    col_iter = iter(cols)

    if show_movies:
        with next(col_iter):
            if movies.empty:
                st.warning("No movie data for the current filters.")
            else:
                avg_runtime = movies.dropna(subset=["year_added"]).groupby("year_added")["duration_value"].mean().reset_index()
                fig = px.line(avg_runtime, x="year_added", y="duration_value",
                              title="Average Movie Runtime (min) by Year Added")
                st.plotly_chart(fig, use_container_width=True)

                fig_hist = px.histogram(movies, x="duration_value", nbins=40,
                                         title="Distribution of Movie Runtimes")
                st.plotly_chart(fig_hist, use_container_width=True)

    if show_tv:
        with next(col_iter):
            if shows.empty:
                st.warning("No TV show data for the current filters.")
            else:
                avg_seasons = shows.dropna(subset=["year_added"]).groupby("year_added")["duration_value"].mean().reset_index()
                fig2 = px.line(avg_seasons, x="year_added", y="duration_value",
                               title="Average Number of Seasons by Year Added")
                st.plotly_chart(fig2, use_container_width=True)

                season_counts = shows["duration_value"].value_counts().sort_index().reset_index()
                season_counts.columns = ["seasons", "count"]
                fig_hist2 = px.bar(season_counts.head(10), x="seasons", y="count",
                                   title="TV Shows by Number of Seasons")
                st.plotly_chart(fig_hist2, use_container_width=True)

    st.caption(
        "Read this alongside Release Trends: if average runtime/seasons is falling "
        "while additions per year are flat or rising, Netflix is shifting toward "
        "shorter-form content rather than just adding more volume."
    )


# --------------------------------- TAB 5 ---------------------------------- #

with tab5:
    st.subheader("Content Diversity Index (CDI) — a custom metric")
    st.markdown(
        """
        **CDI = 100 × (0.40·Evenness + 0.35·Coverage + 0.25·Freshness)**

        - **Evenness** — how spread out a country's catalog is across genres
          (Shannon entropy), vs. being dominated by one or two genres.
        - **Coverage** — what fraction of all genres available on Netflix
          globally are present at all in that country.
        - **Freshness** — share of that country's titles released in the
          last 5 years.

        Countries with fewer than 10 titles are excluded — not enough data
        to score meaningfully.
        """
    )

    st.caption(f"Genre coverage/evenness computed on {type_filter + 's' if type_filter else 'Movies + TV Shows'}; "
               f"table below restricted to your {len(active_countries)} selected countries.")

    # Compute on the full country universe (type-filtered only) so coverage/
    # evenness denominators stay globally correct, then restrict the display
    # to the countries chosen in the sidebar.
    cdi_full = compute_diversity_index(long_type_scoped, titles_type_scoped)
    cdi_df = cdi_full[cdi_full["country"].isin(active_countries)].reset_index(drop=True)

    if cdi_df.empty:
        st.warning("No countries meet the minimum-title threshold for the current filters. "
                   "Try selecting more countries or a wider content-type filter.")
        st.stop()

    fig = px.bar(
        cdi_df.head(20), x="CDI", y="country", orientation="h",
        color="CDI", color_continuous_scale="Viridis",
        title=f"Content Diversity Index — {len(cdi_df)} Selected Countries",
        hover_data=["n_titles", "evenness", "coverage", "freshness"],
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Lowest-scoring countries with meaningful catalog size")
    st.caption("These are the strongest investment-priority candidates: enough titles to matter, but a lopsided or stale catalog.")
    st.dataframe(
        cdi_df.sort_values("CDI").head(10).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("📌 Strategic takeaways")

    lowest = cdi_df.sort_values("CDI").head(3)
    bullets = []
    for _, row in lowest.iterrows():
        gaps = top_missing_genres(long_type_scoped, row["country"], top_n=2)
        genre_list = ", ".join(gaps["genre"].tolist())
        bullets.append(
            f"- **{row['country']}** (CDI {row['CDI']}, {row['n_titles']} titles): "
            f"catalog is skewed / stale. Biggest gaps: **{genre_list}**."
        )
    st.markdown("\n".join(bullets))

    st.markdown(
        "Overall: prioritize markets with a *large enough* catalog to matter "
        "commercially but a *low* CDI — that combination signals real subscriber "
        "reach being underserved by a narrow or aging content mix, which is "
        "exactly where new investment moves the needle most."
    )