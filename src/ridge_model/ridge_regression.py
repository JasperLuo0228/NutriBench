import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
tfidf_path = os.path.join(base_dir, "data/method1_tfidf")
sys.path.append(tfidf_path)

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tfidf import TfidfProcessor

# Initialize the processor and load vectorizer
tfidf_proc = TfidfProcessor()
tfidf_proc.load_vectorizer(os.path.join(base_dir, "data/method1_tfidf/tfidf_vectorizer.joblib"))

# Load training data
df_train = pd.read_csv(os.path.join(base_dir,"data/train.csv"))
X = tfidf_proc.transform(df_train["query"])
y = df_train["carb"]

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MAE: {mae:.3f}")
print(f"Validation MSE: {mse:.3f}")

# (Optional) Save output
import os
os.makedirs("output/ridge", exist_ok=True)

# Save predictions
df_out = pd.DataFrame({
    "y_true": y_val,
    "y_pred": y_pred
})
df_out.to_csv("output/ridge/val_predictions.csv", index=False)

# Save metrics
with open("output/ridge/metrics.txt", "w") as f:
    f.write(f"Validation MAE: {mae:.3f}\n")
    f.write(f"Validation MSE: {mse:.3f}\n")
