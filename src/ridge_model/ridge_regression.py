import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add tfidf module path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
tfidf_path = os.path.join(base_dir, "data/method1_tfidf")
sys.path.append(tfidf_path)

from tfidf import TfidfProcessor

# Load Vectorizer
tfidf_proc = TfidfProcessor()
tfidf_proc.load_vectorizer(os.path.join(tfidf_path, "tfidf_vectorizer.joblib"))

# Load and transform training data
df_train = pd.read_csv(os.path.join(base_dir, "data/train.csv"))
X = tfidf_proc.transform(df_train["query"])
y = df_train["carb"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge model
# Define a range of alpha values to search
alphas = np.logspace(-3, 3, 20) 

# Use RidgeCV to automatically find the best alpha
model = RidgeCV(alphas=alphas, store_cv_values=True)
model.fit(X_train, y_train)

# Print best alpha
print(f"Best alpha found: {model.alpha_}")

# Validate
y_val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)

print(f"Validation MAE: {mae:.3f}")
print(f"Validation MSE: {mse:.3f}")

# Save validation output
output_dir = os.path.join(base_dir, "output/ridge")
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({
    "y_true": y_val,
    "y_pred": y_val_pred
}).to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"Validation MAE: {mae:.3f}\n")
    f.write(f"Validation MSE: {mse:.3f}\n")
    f.write(f"Best alpha: {model.alpha_}\n")

# Predict on test set
df_test = pd.read_csv(os.path.join(base_dir, "data/test.csv"))
X_test = tfidf_proc.transform(df_test["query"])
df_test["carb"] = model.predict(X_test)

# Save final test predictions
df_test.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
