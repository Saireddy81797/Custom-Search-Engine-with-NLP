from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, content_recommender, collaborative_recommender):
        self.content = content_recommender
        self.collab = collaborative_recommender

    def recommend(self, product_id=None, user_id=None, top_n=3):
        recs = []
        if product_id:
            recs.append(self.content.recommend(product_id, top_n))
        if user_id:
            recs.append(self.collab.recommend(user_id, top_n))
        return recs
