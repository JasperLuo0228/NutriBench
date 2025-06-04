import pandas as pd
import numpy as np

# Load the training data
train_df = pd.read_csv("data/train.csv")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
X_train = model.encode(train_df["query"].tolist())
X_val = model.encode(train_df["query"].tolist())

y_train = train_df["carb"].values
y_val = train_df["carb"].values

# Save the embeddings to a file
np.save("data/method2_sbert/X_train.npy", X_train)
np.save("data/method2_sbert/X_val.npy", X_val)

np.save("data/method2_sbert/y_train.npy", y_train)
np.save("data/method2_sbert/y_val.npy", y_val)

