
import streamlit as st
import pandas as pd
from src.data_loader import load_and_merge
from src.recommender import BookRecommender

# Set Page Config
st.set_page_config(page_title="Audible Insights", layout="wide")

# Title and Description
st.title("📚 Audible Insights: Intelligent Recommendations")
st.markdown("Discover your next favorite audiobook using AI-powered NLP and Clustering.")

# --- LOAD DATA (Cached for speed) ---
@st.cache_resource
def init_engine():
    df = load_and_merge()
    recommender = BookRecommender(df)
    recommender.prepare_features()
    recommender.apply_clustering(n_clusters=10)
    return df, recommender

df, recommender = init_engine()

# --- SIDEBAR: EDA & Insights ---
st.sidebar.header("Project Insights")
if st.sidebar.checkbox("Show Top Genres"):
    st.sidebar.write(df['Primary_Genre'].value_counts().head(5))

if st.sidebar.checkbox("Show Highest Rated Authors"):
    author_stats = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(5)
    st.sidebar.write(author_stats)

# --- MAIN INTERFACE: Recommendations ---
st.subheader("Get Personalized Recommendations")

# Search Bar / Select Box
book_list = sorted(recommender.df['Book Name'].unique())
selected_book = st.selectbox("Type or select a book you liked:", book_list)

if st.button("Recommend Similar Books"):
    with st.spinner("Analyzing descriptions..."):
        results = recommender.get_recommendations(selected_book)
        
        if isinstance(results, str):
            st.error(results)
        else:
            st.success(f"Because you liked **{selected_book}**, we recommend:")
            
            # Display results in columns
            cols = st.columns(3)
            for i, (index, row) in enumerate(results.head(3).iterrows()):
                with cols[i]:
                    st.info(f"**{row['Book Name']}**")
                    st.write(f"👤 Author: {row['Author']}")
                    st.write(f"⭐ Rating: {row['Rating']}")

# --- SCENARIOS SECTION ---
st.divider()
st.subheader("Scenario Based Discovery")
scenario = st.radio("Choose a scenario:", 
                    ["Science Fiction Starter", "Hidden Gems", "Top Thrillers"])

if scenario == "Science Fiction Starter":
    sci_fi = df[df['Primary_Genre'].str.contains('Science Fiction|Sci-Fi', case=False, na=False)]
    st.table(sci_fi.sort_values(by='Rating', ascending=False)[['Book Name', 'Author', 'Rating']].head(5))

elif scenario == "Hidden Gems":
    gems = df[(df['Rating'] >= 4.5) & (df['Number of Reviews'] < 50) & (df['Number of Reviews'] > 5)]
    st.table(gems[['Book Name', 'Author', 'Rating']].head(5))

elif scenario == "Top Thrillers":
    thrillers = df[df['Primary_Genre'].str.contains('Thriller', case=False, na=False)]
    st.table(thrillers.sort_values(by='Rating', ascending=False)[['Book Name', 'Author', 'Rating']].head(5))