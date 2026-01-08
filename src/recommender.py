from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class BookRecommender: # <--- Make sure this name is exactly like this
    def __init__(self, df):
        # Drop rows without descriptions as they can't be used for NLP
        self.df = df.dropna(subset=['Description']).copy().reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None
        self.model = None

    def prepare_features(self):
        """Convert descriptions into a TF-IDF matrix."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Description'])
        return self.tfidf_matrix

    def apply_clustering(self, n_clusters=10):
        """Group books based on description similarity."""
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['Cluster'] = self.model.fit_predict(self.tfidf_matrix)
        return self.df

    def get_recommendations(self, book_title, top_n=5):
        """Find similar books using Cosine Similarity."""
        if book_title not in self.df['Book Name'].values:
            return f"Book '{book_title}' not found in the description database."

        idx = self.df[self.df['Book Name'] == book_title].index[0]
        
        # Calculate similarity
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)
        
        # Sort and get top results
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

        return self.df.iloc[sim_indices][['Book Name', 'Author', 'Rating']]