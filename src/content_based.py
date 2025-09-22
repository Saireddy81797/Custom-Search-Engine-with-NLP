import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, vectorizer, matrix, df):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.df = df

    def recommend(self, product_id, top_n=3):
        idx = self.df[self.df['product_id'] == product_id].index[0]
        cosine_sim = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        return self.df.iloc[similar_indices][['product_id','title','category']]
