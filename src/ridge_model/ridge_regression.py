import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Paths (adjust if needed)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
train_path = os.path.join(base_dir, "data/train.csv")
val_path = os.path.join(base_dir, "data/val.csv")
test_path = os.path.join(base_dir, "data/test.csv")

# Load datasets
df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(df_train["query"])
X_val = vectorizer.transform(df_val["query"])
X_test = vectorizer.transform(df_test["query"])

# Transform target using log1p to reduce outlier effect
y_train_log = np.log1p(df_train["carb"])
y_val = df_val["carb"]

# Train Ridge Regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train_log)

# Predict on validation set
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)  # Inverse transformation

# Evaluate
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MAE: {mae:.3f}")
print(f"Validation MSE: {mse:.3f}")

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_val_pred, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("True Carb")
plt.ylabel("Predicted Carb")
plt.title("TF-IDF + Ridge Regression (log target)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save prediction and metrics
output_dir = os.path.join(base_dir, "output/ridge_log")
os.makedirs(output_dir, exist_ok=True)

# Save validation predictions
val_out = pd.DataFrame({
    "y_true": y_val,
    "y_pred": y_val_pred
})
val_out.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

# Save test predictions
y_test_pred = np.expm1(model.predict(X_test))
test_out = df_test.copy()
test_out["carb"] = y_test_pred
test_out.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"Validation MAE: {mae:.3f}\n")
    f.write(f"Validation MSE: {mse:.3f}\n")
