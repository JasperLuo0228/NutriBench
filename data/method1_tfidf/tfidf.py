# tfidf.py

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class TfidfProcessor:
    def __init__(self, max_features=10000, ngram_range=(1, 2), stop_words='english', min_df=2):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df
        )

    def fit(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def save_vectorizer(self, path):
        joblib.dump(self.vectorizer, path)

    def load_vectorizer(self, path):
        self.vectorizer = joblib.load(path)

    def fit_transform_and_save(self, train_texts, val_texts, test_texts, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        X_train = self.fit_transform(train_texts)
        X_val = self.transform(val_texts)
        X_test = self.transform(test_texts)

        self.save_vectorizer(os.path.join(out_dir, "tfidf_vectorizer.joblib"))
        sparse.save_npz(os.path.join(out_dir, "X_train.npz"), X_train)
        sparse.save_npz(os.path.join(out_dir, "X_val.npz"), X_val)
        sparse.save_npz(os.path.join(out_dir, "X_test.npz"), X_test)

        return X_train, X_val, X_test

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    # Load raw text data
    base_data_dir = os.path.abspath(os.path.join(out_dir, "../../data"))
    train_df = pd.read_csv(os.path.join(base_data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(base_data_dir, "test.csv"))

    processor = TfidfProcessor()
    processor.fit_transform_and_save(
        train_df["query"],
        val_df["query"],
        test_df["query"],
        out_dir=out_dir
    )

    print("TF-IDF files saved to:", out_dir)
