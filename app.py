from pathlib import Path
import html

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "movie_recommender.joblib"


# ---------------------------------------------------------
# 2. Custom appearance
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #121a2d;
        --panel-soft: #172239;
        --text: #f4f6fb;
        --muted: #a7b1c4;
        --accent: #f5b942;
        --accent-soft: #ffe1a0;
        --line: rgba(255,255,255,0.10);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(245,185,66,0.12), transparent 28%),
            radial-gradient(circle at 92% 18%, rgba(80,115,190,0.13), transparent 30%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(11,16,32,0.72); }
    .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--text); }
    p, label, span { color: inherit; }
    [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stRadio"] > div { gap: 0.5rem; }
    [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.045);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.38rem 0.9rem;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(245,185,66,0.16);
        border-color: rgba(245,185,66,0.55);
    }
    [data-testid="stButton"] button {
        border: 0;
        border-radius: 12px;
        background: linear-gradient(135deg, #f5b942, #e99a2e);
        color: #171717;
        font-weight: 700;
        min-height: 2.8rem;
        padding: 0.55rem 1.15rem;
    }
    [data-testid="stButton"] button:hover { filter: brightness(1.06); border: 0; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSlider"] {
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; }
    hr { border-color: var(--line); }
    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: var(--text);
    }
    .brand span { color: var(--accent); }
    .eyebrow {
        color: var(--accent-soft);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero {
        background: linear-gradient(115deg, rgba(29,42,68,0.98), rgba(17,26,45,0.96));
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 26px;
        padding: clamp(1.5rem, 4vw, 3.25rem);
        position: relative;
        overflow: hidden;
        margin: 1.2rem 0 1.3rem 0;
    }
    .hero:after {
        content: '✦';
        position: absolute;
        right: 5%;
        top: 3%;
        font-size: clamp(7rem, 18vw, 15rem);
        line-height: 1;
        color: rgba(245,185,66,0.10);
        transform: rotate(12deg);
    }
    .hero h1 {
        font-size: clamp(2.1rem, 5vw, 4.3rem) !important;
        line-height: 1.07 !important;
        max-width: 720px;
        margin: 0 0 1rem 0;
    }
    .hero p {
        max-width: 610px;
        color: #b9c4d7 !important;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 0;
    }
    .section-kicker {
        color: var(--accent);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .feature-card, .movie-card, .info-card {
        background: linear-gradient(160deg, rgba(23,34,57,0.96), rgba(17,26,45,0.96));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1.15rem 1.25rem;
        height: 100%;
    }
    .movie-card { margin-bottom: 0.7rem; }
    .movie-rank {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .movie-title {
        font-family: 'Manrope', sans-serif;
        color: var(--text);
        font-weight: 800;
        font-size: 1.12rem;
        margin-bottom: 0.6rem;
    }
    .movie-meta { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
    .match-pill {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.25rem 0.58rem;
        border-radius: 999px;
        background: rgba(245,185,66,0.13);
        border: 1px solid rgba(245,185,66,0.25);
        color: var(--accent-soft);
        font-size: 0.76rem;
        font-weight: 700;
    }
    .footer-note {
        text-align: center;
        color: #77839a;
        font-size: 0.82rem;
        padding-top: 1.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. Load the saved recommendation model
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading the movie collection...")
def load_recommender(model_file: str):
    artifacts = joblib.load(model_file)
    movie_data = artifacts["movies"].reset_index(drop=True)
    matrix = artifacts["tfidf_matrix"]

    if len(movie_data) != matrix.shape[0]:
        raise ValueError(
            "The saved movie list and TF-IDF matrix have different row counts. "
            "Please recreate the saved recommendation file from the notebook."
        )

    required_columns = {"title"}
    missing = required_columns - set(movie_data.columns)
    if missing:
        raise ValueError(f"The saved movie data is missing columns: {sorted(missing)}")

    return movie_data, matrix


if not MODEL_PATH.exists():
    st.error(
        "The recommendation model file was not found. Please place "
        "movie_recommender.joblib inside the project's models folder."
    )
    st.stop()

try:
    movies, tfidf_matrix = load_recommender(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The recommendation model could not be loaded: {exc}")
    st.stop()


# ---------------------------------------------------------
# 4. Small helper functions
# ---------------------------------------------------------
def safe_text(value, fallback="Not available"):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


def get_year(movie):
    release_date = movie.get("release_date", "")
    if pd.notna(release_date) and str(release_date).strip():
        date_text = str(release_date).strip()
        if len(date_text) >= 4 and date_text[:4].isdigit():
            return date_text[:4]

    release_year = movie.get("release_year", "")
    if pd.notna(release_year) and str(release_year).strip():
        try:
            return str(int(float(release_year)))
        except (ValueError, TypeError):
            pass
    return "Year unknown"


def format_movie_label(movie_index):
    movie = movies.iloc[movie_index]
    title = safe_text(movie.get("title"), "Untitled")
    return f"{title} ({get_year(movie)})"


def render_movie_card(movie, rank, score):
    title = html.escape(safe_text(movie.get("title"), "Untitled"))
    genres = html.escape(safe_text(movie.get("genres_text")))
    director = html.escape(safe_text(movie.get("director")))
    year = html.escape(get_year(movie))

    rating = movie.get("vote_average", None)
    rating_text = ""
    if pd.notna(rating):
        try:
            rating_text = f" · TMDB rating {float(rating):.1f}/10"
        except (ValueError, TypeError):
            rating_text = ""

    st.markdown(
        f"""
        <div class="movie-card">
            <div class="movie-rank">RECOMMENDATION {rank:02d}</div>
            <div class="movie-title">{title}</div>
            <div class="movie-meta"><b>Year:</b> {year}</div>
            <div class="movie-meta"><b>Genres:</b> {genres}</div>
            <div class="movie-meta"><b>Director:</b> {director}</div>
            <div class="movie-meta">{html.escape(rating_text.strip(' ·'))}</div>
            <div class="match-pill">Similarity score: {score:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_recommendations(movie_index: int, number_to_return: int):
    scores = cosine_similarity(
        tfidf_matrix[movie_index], tfidf_matrix
    ).ravel()

    # Sort from highest similarity to lowest; exclude the selected movie.
    ordered_indices = scores.argsort()[::-1]
    results = []
    for idx in ordered_indices:
        if idx == movie_index:
            continue
        results.append((int(idx), float(scores[idx])))
        if len(results) == number_to_return:
            break
    return results


def go_to_discover():
    st.session_state["page_choice"] = "Discover Movies"


# ---------------------------------------------------------
# 5. Top navigation
# ---------------------------------------------------------
header_left, header_right = st.columns([2, 5])
with header_left:
    st.markdown('<div class="brand">CINE<span>MATCH</span> 🎬</div>', unsafe_allow_html=True)
with header_right:
    st.markdown(
        '<div style="text-align:right;color:#8793a9;padding-top:0.3rem;">'
        'A smarter way to discover your next movie</div>',
        unsafe_allow_html=True,
    )

if "page_choice" not in st.session_state:
    st.session_state["page_choice"] = "Home"

page = st.radio(
    "Main navigation",
    ["Home", "Discover Movies", "About"],
    horizontal=True,
    label_visibility="collapsed",
    key="page_choice",
)

st.divider()


# ---------------------------------------------------------
# 6. Home page
# ---------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">YOUR PERSONAL MOVIE GUIDE</div>
            <h1>Find your next<br>great movie.</h1>
            <p>Choose a movie you enjoy and CineMatch will discover other films
            with similar genres, themes, cast and creative teams.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_movies = len(movies)
    unique_titles = movies["title"].nunique()
    st.markdown('<div class="section-kicker">THE COLLECTION</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Movies available", f"{total_movies:,}")
    metric_cols[1].metric("Unique titles", f"{unique_titles:,}")
    metric_cols[2].metric("Recommendation method", "Content-based")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-kicker">HOW IT WORKS</div>', unsafe_allow_html=True)
        st.markdown("## Recommendations with a reason")
        st.markdown(
            "The system studies movie information and compares it with the movie "
            "you select. It then ranks other titles by how similar their content is."
        )
        st.markdown(
            "**1. Choose a movie**  \n"
            "Select a title from the collection.  \n\n"
            "**2. Compare movie features**  \n"
            "The system uses TF-IDF and cosine similarity.  \n\n"
            "**3. Explore your matches**  \n"
            "See a ranked list of similar movies."
        )
        st.button("Start discovering movies", on_click=go_to_discover)

    with right:
        st.markdown('<div class="section-kicker">FEATURED FROM THE COLLECTION</div>', unsafe_allow_html=True)
        featured = None
        if "popularity" in movies.columns:
            popularity_values = pd.to_numeric(movies["popularity"], errors="coerce")
            if popularity_values.notna().any():
                featured = movies.loc[popularity_values.idxmax()]
        if featured is None:
            featured = movies.iloc[0]

        featured_title = html.escape(safe_text(featured.get("title"), "Movie"))
        featured_genres = html.escape(safe_text(featured.get("genres_text")))
        featured_director = html.escape(safe_text(featured.get("director")))
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="eyebrow">SPOTLIGHT TITLE</div>
                <div style="font-family:Manrope,sans-serif;font-size:1.65rem;font-weight:800;color:#f4f6fb;margin:0.6rem 0;">
                    {featured_title}
                </div>
                <div class="movie-meta">{html.escape(get_year(featured))}</div>
                <hr>
                <div class="movie-meta"><b>Genres:</b> {featured_genres}</div>
                <div class="movie-meta"><b>Director:</b> {featured_director}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# 7. Movie discovery page
# ---------------------------------------------------------
elif page == "Discover Movies":
    st.markdown('<div class="section-kicker">MOVIE DISCOVERY</div>', unsafe_allow_html=True)
    st.title("What would you like to watch?")
    st.write("Select a movie and choose how many similar titles you want to explore.")

    movie_indices = list(range(len(movies)))
    movie_indices.sort(key=lambda i: safe_text(movies.iloc[i].get("title")).lower())

    selected_index = st.selectbox(
        "Search or select a movie",
        options=movie_indices,
        format_func=format_movie_label,
    )
    number_to_return = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    chosen_movie = movies.iloc[selected_index]
    selected_title = html.escape(safe_text(chosen_movie.get("title"), "Untitled"))
    selected_genres = html.escape(safe_text(chosen_movie.get("genres_text")))
    selected_director = html.escape(safe_text(chosen_movie.get("director")))
    st.markdown(
        f"""
        <div class="feature-card" style="margin:0.6rem 0 1rem 0;">
            <div class="eyebrow">YOUR SELECTED MOVIE</div>
            <div class="movie-title" style="font-size:1.35rem;">{selected_title}</div>
            <div class="movie-meta"><b>Year:</b> {html.escape(get_year(chosen_movie))}</div>
            <div class="movie-meta"><b>Genres:</b> {selected_genres}</div>
            <div class="movie-meta"><b>Director:</b> {selected_director}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Find similar movies", type="primary", use_container_width=True):
        with st.spinner("Finding movies with similar content..."):
            recommendation_results = build_recommendations(
                selected_index, number_to_return
            )

        st.markdown("## Your recommendations")
        st.write(f"Showing up to {len(recommendation_results)} movies similar to **{safe_text(chosen_movie.get('title'))}**.")

        if not recommendation_results:
            st.info("No similar movies were found for this selection.")
        else:
            for start in range(0, len(recommendation_results), 2):
                card_columns = st.columns(2, gap="medium")
                for offset, (movie_index, score) in enumerate(
                    recommendation_results[start:start + 2]
                ):
                    with card_columns[offset]:
                        render_movie_card(
                            movies.iloc[movie_index],
                            start + offset + 1,
                            score,
                        )


# ---------------------------------------------------------
# 8. About page
# ---------------------------------------------------------
elif page == "About":
    st.markdown('<div class="section-kicker">ABOUT THE PROJECT</div>', unsafe_allow_html=True)
    st.title("About CineMatch")
    st.markdown(
        "CineMatch is a movie discovery project built with Python and machine learning. "
        "It uses a content-based filtering approach, which means recommendations are "
        "based on information about the movies rather than a user's viewing history."
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Dataset")
        st.write(
            "The project uses the TMDB 5000 Movies and Credits datasets, which include "
            "movie titles, genres, keywords, cast and crew information."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Recommendation method")
        st.write(
            "TF-IDF converts movie text into numerical vectors. Cosine similarity "
            "compares those vectors and ranks movies with similar content."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Technologies")
    st.write("Python · Pandas · Scikit-learn · TF-IDF · Cosine Similarity · Streamlit")


# ---------------------------------------------------------
# 9. Footer
# ---------------------------------------------------------
st.markdown(
    '<div class="footer-note">CineMatch · Movie Recommendation System · Built with Python and Streamlit</div>',
    unsafe_allow_html=True,
)
