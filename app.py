import streamlit as st
import pandas as pd
import pickle
import difflib
import re
from sklearn.metrics.pairwise import linear_kernel

# Page Configuration
st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        width: 100%;
        height: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    movies = pickle.load(open('artifacts/movies.pkl', 'rb'))
    tfidf = pickle.load(open('artifacts/tfidf.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('artifacts/tfidf_matrix.pkl', 'rb'))
    return movies, tfidf, tfidf_matrix

df, tfidf, tfidf_matrix = load_artifacts()

# Helper Functions
def extract_clean_title(title_str):
    if pd.isna(title_str):
        return ''
    clean = re.sub(r'\(\d{4}\)', '', title_str)
    return clean.strip().lower()

def recommend(movie_name, top_n=10):
    user_input_clean = extract_clean_title(movie_name)
    matches = df[df['clean_title'].str.contains(re.escape(user_input_clean), case=False, na=False)]
    
    if not matches.empty:
        matched_idx = matches.index[0]
    else:
        all_clean_titles = df['clean_title'].tolist()
        close_matches = difflib.get_close_matches(user_input_clean, all_clean_titles, n=1, cutoff=0.4)
        if not close_matches:
            return None, None
        matched_title = close_matches[0]
        matched_idx = df[df['clean_title'] == matched_title].index[0]
        
    matched_movie_full_title = df.loc[matched_idx, 'title']
    
    query_vector = tfidf_matrix[matched_idx]
    sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    scored_indices = list(enumerate(sim_scores))
    scored_indices = sorted(scored_indices, key=lambda x: x[1], reverse=True)
    filtered_indices = [item for item in scored_indices if item[0] != matched_idx][:top_n]
    
    rec_indices = [i[0] for i in filtered_indices]
    rec_scores = [round(float(i[1]), 4) for i in filtered_indices]
    
    results = df.iloc[rec_indices][['title', 'genres']].copy()
    results['Score'] = rec_scores
    return matched_movie_full_title, results

# Header
st.markdown('<p class="main-title">🎬 CineMatch AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Content-Based Movie Recommendation Engine</p>', unsafe_allow_html=True)

# Main UI Layout
movie_titles = df['title'].values
selected_movie = st.selectbox("🔍 Search or Select a Movie:", movie_titles)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button("✨ Get Recommendations")

if submit_button:
    if selected_movie:
        matched_title, recommendations = recommend(selected_movie, top_n=10)
        
        if matched_title is None:
            st.error("❌ Movie not found. Try another search!")
        else:
            st.success(f"Showing top recommendations based on **{matched_title}**:")
            
            # Display Grid
            for idx, row in recommendations.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 4, 2])
                    with col_a:
                        st.markdown(f"### #{idx + 1}")
                    with col_b:
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"Genres: {row['genres']}")
                    with col_c:
                        st.metric("Match Score", f"{int(row['Score'] * 100)}%")
                    st.divider()