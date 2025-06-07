import pandas as pd
from tfidf import TfidfProcessor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import os

# 1. Load data (relative to src/mlp_model/)
train_df = pd.read_csv(os.path.join('../../data/train.csv'))
val_df = pd.read_csv(os.path.join('../../data/val.csv'))
test_df = pd.read_csv(os.path.join('../../data/test.csv'))

# 2. Load the pre-trained TF-IDF vectorizer
tfidf_proc = TfidfProcessor()
tfidf_proc.load_vectorizer("tfidf_vectorizer.joblib")

# 3. Transform text features
X_train = tfidf_proc.transform(train_df["query"])
X_val = tfidf_proc.transform(val_df["query"])
X_test = tfidf_proc.transform(test_df["query"])

y_train = train_df["carb"].values
y_val = val_df["carb"].values

# 4. Build and train the MLP model
mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# 5. Predict and evaluate on the validation set
y_val_pred = mlp.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print("Validation MSE:", mse)

# Save validation predictions and metrics
val_df["carb_pred"] = y_val_pred
val_df.to_csv("../../output/mlp/val_predictions.csv", index=False)
with open("../../output/mlp/metrics.txt", "w") as f:
    f.write(f"Validation MSE: {mse}\n")

# 6. Predict on the test set and save results
y_test_pred = mlp.predict(X_test)
test_df["carb"] = y_test_pred
test_df.to_csv("../../output/mlp/test_predictions.csv", index=False)
print("Test set predictions saved to ../../output/mlp/test_predictions.csv")