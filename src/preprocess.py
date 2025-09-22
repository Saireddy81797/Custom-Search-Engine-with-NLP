import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path="../data/sample_products.csv"):
    return pd.read_csv(path)

def preprocess_text(df):
    df['text'] = df['title'] + " " + df['description'] + " " + df['category']
    return df

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix
