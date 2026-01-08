
import pandas as pd
from src.data_loader import load_and_merge
from src.recommender import BookRecommender
from sklearn.metrics import precision_score

def evaluate():
    df = load_and_merge()
    rec = BookRecommender(df)
    rec.prepare_features()
    df_clustered = rec.apply_clustering()

    # Simple Evaluation: Does the recommendation match the genre?
    test_book = "Atomic Habits: An Easy and Proven Way to Build Good Habits and Break Bad Ones"
    actual_genre = df[df['Book Name'] == test_book]['Primary_Genre'].values[0]
    
    recommendations = rec.get_recommendations(test_book)
    
    # Calculate Precision: How many recommendations have the same genre?
    # (In a real company, you'd do this over 100 books)
    hits = recommendations[recommendations['Book Name'].isin(df[df['Primary_Genre'] == actual_genre]['Book Name'])]
    precision = len(hits) / len(recommendations)
    
    print(f"Evaluation for: {test_book}")
    print(f"Model Precision @ 5: {precision * 100}%")
    print("RMSE: Not applicable for Content-Based (usually for Collaborative Filtering)")

if __name__ == "__main__":
    evaluate()