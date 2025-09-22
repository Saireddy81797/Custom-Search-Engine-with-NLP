import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, df):
        self.df = df
        self.user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)
        self.sim_matrix = cosine_similarity(self.user_item_matrix)

    def recommend(self, user_id, top_n=3):
        user_index = list(self.user_item_matrix.index).index(user_id)
        user_scores = self.sim_matrix[user_index]
        similar_users = user_scores.argsort()[::-1][1:]
        
        recommended_items = []
        for u in similar_users:
            top_items = self.user_item_matrix.iloc[u].sort_values(ascending=False).index
            recommended_items.extend(top_items)
            if len(recommended_items) >= top_n:
                break

        return self.df[self.df['product_id'].isin(recommended_items[:top_n])][['product_id','title','category']]
