import pandas as pd
from src.preprocess import load_data, preprocess_text, vectorize_text
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.recommender import HybridRecommender

if __name__ == "__main__":
    df = load_data()
    df = preprocess_text(df)

    vectorizer, matrix = vectorize_text(df['text'])

    content = ContentBasedRecommender(vectorizer, matrix, df)
    collab = CollaborativeRecommender(df)

    hybrid = HybridRecommender(content, collab)

    print("🔎 Content-based recommendation for product_id=1:")
    print(content.recommend(product_id=1))

    print("\n👥 Collaborative recommendation for user_id=101:")
    print(collab.recommend(user_id=101))

    print("\n⚡ Hybrid recommendation (user 101 + product 1):")
    print(hybrid.recommend(product_id=1, user_id=101))
