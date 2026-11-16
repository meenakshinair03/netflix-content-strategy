-- queries.sql
-- Core analytical queries against data/netflix.db (table: titles_long)
-- Run these in a SQLite browser, or via sqlite3 in Python (see app.py).

-- 1) CONTENT GAPS BY COUNTRY x GENRE
-- Count of titles per country per genre -> pivot this in pandas for a heatmap
SELECT
    country_single AS country,
    genre_single   AS genre,
    COUNT(DISTINCT title) AS title_count
FROM titles_long
WHERE country_single != 'Unknown'
GROUP BY country_single, genre_single
ORDER BY country, title_count DESC;


-- 2) TOP 15 COUNTRIES BY CATALOG SIZE (denominator for gap %)
SELECT
    country_single AS country,
    COUNT(DISTINCT title) AS total_titles
FROM titles_long
WHERE country_single != 'Unknown'
GROUP BY country_single
ORDER BY total_titles DESC
LIMIT 15;


-- 3) GLOBAL GENRE DISTRIBUTION (baseline to compare each country against)
SELECT
    genre_single AS genre,
    COUNT(DISTINCT title) AS title_count
FROM titles_long
GROUP BY genre_single
ORDER BY title_count DESC;


-- 4) RELEASE YEAR TRENDS — is the catalog aging or refreshing?
-- Compares release_year (when content was made) vs year_added (when Netflix added it)
SELECT
    release_year,
    COUNT(DISTINCT title) AS titles_released
FROM titles
WHERE release_year IS NOT NULL
GROUP BY release_year
ORDER BY release_year;

SELECT
    year_added,
    COUNT(DISTINCT title) AS titles_added
FROM titles
WHERE year_added IS NOT NULL
GROUP BY year_added
ORDER BY year_added;

-- "Freshness gap" per year_added: how old was the content Netflix added that year?
SELECT
    year_added,
    AVG(year_added - release_year) AS avg_content_age_at_addition
FROM titles
WHERE year_added IS NOT NULL AND release_year IS NOT NULL
GROUP BY year_added
ORDER BY year_added;


-- 5) DURATION / RUNTIME PATTERNS
-- Movies: distribution of runtime in minutes, by year_added (trend)
SELECT
    year_added,
    AVG(duration_value) AS avg_movie_minutes,
    COUNT(*) AS movie_count
FROM titles
WHERE type = 'Movie' AND duration_unit = 'min' AND year_added IS NOT NULL
GROUP BY year_added
ORDER BY year_added;

-- TV Shows: are shows getting more or fewer seasons over time?
SELECT
    year_added,
    AVG(duration_value) AS avg_seasons,
    COUNT(*) AS show_count
FROM titles
WHERE type = 'TV Show' AND year_added IS NOT NULL
GROUP BY year_added
ORDER BY year_added;


-- 6) RAW INPUTS FOR THE CONTENT DIVERSITY SCORE (computed in Python — see
-- diversity_score.py — because Shannon entropy needs per-country genre
-- proportions, which are easier to compute in pandas than pure SQL)
SELECT
    country_single AS country,
    genre_single   AS genre,
    COUNT(DISTINCT title) AS title_count
FROM titles_long
WHERE country_single != 'Unknown'
GROUP BY country_single, genre_single;
