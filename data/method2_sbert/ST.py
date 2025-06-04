import pandas as pd
import numpy as np

# Load the training data
train_df = pd.read_csv("data/train.csv")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
X_train = model.encode(train_df["query"].tolist())

# Save the embeddings to a file
np.save("data/method2_sbert/X_train.npy", X_train)

y = train_df["carb"].values
np.save("data/method2_sbert/y_train.npy", y)
