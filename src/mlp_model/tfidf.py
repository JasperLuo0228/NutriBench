# tfidf.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class TfidfProcessor:
    def __init__(self, max_features=1000, ngram_range=(1,2), stop_words='english'):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words
        )

    def fit(self, texts):

        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def save_vectorizer(self, path):
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path):
        self.vectorizer = joblib.load(path)
