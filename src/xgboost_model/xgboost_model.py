import os
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBRegressor

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_dir = os.path.join(base_dir, "data")
tfidf_dir = os.path.join(data_dir, "method1_tfidf")

# Load TF-IDF features and targets
X_train = sparse.load_npz(os.path.join(tfidf_dir, "X_train.npz"))
X_val = sparse.load_npz(os.path.join(tfidf_dir, "X_val.npz"))
X_test = sparse.load_npz(os.path.join(tfidf_dir, "X_test.npz"))

y_train = pd.read_csv(os.path.join(data_dir, "train.csv"))["carb"]
y_val = pd.read_csv(os.path.join(data_dir, "val.csv"))["carb"]


# Combine train and val for cross-validation
X = sparse.vstack([X_train, X_val])
y = pd.concat([y_train, y_val]).reset_index(drop=True)

# Define model and search space
xgb = XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}

search = RandomizedSearchCV(
    xgb,
    param_distributions={
        "max_depth": randint(4, 9),
        "learning_rate": uniform(0.01, 0.1),
        "n_estimators": randint(2000,3000),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
    },
    n_iter=20, 
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=1 
)

# Run search
search.fit(X, y)
best_params = search.best_params_
print("Best Parameters from RandomizedSearchCV:", best_params)



# Best model evaluation
best_model = search.best_estimator_
y_val_pred = best_model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
mse = mean_squared_error(y_val, y_val_pred)

# Check how many predictions are within ±7.5 of ground truth
within_range_mask = (abs(y_val_pred - y_val) <= 7.5)
within_range_count = within_range_mask.sum()
total_count = len(y_val)
percent_within_range = 100 * within_range_count / total_count

# Predict test set
y_test_pred = best_model.predict(X_test)


# Save results
output_dir = os.path.join(base_dir, "output/xgboost")
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({
    "y_true": y_val,
    "y_pred": y_val_pred
}).to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

# Save test predictions
pd.DataFrame({
    "y_pred": y_test_pred
}).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)


# Save metrics
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("Best Params: " + str(search.best_params_) + "\n")
    f.write(f"Validation MAE: {mae:.3f}\n")
    f.write(f"Validation MSE: {mse:.3f}\n")
    f.write(f"Validation predictions within ±7.5: {within_range_count}/{total_count} "
            f"({percent_within_range:.2f}%)\n")

# === Print summary
print("\n Evaluation Summary:")
print("Best Parameters:", search.best_params_)
print(f"Validation MAE: {mae:.3f}")
print(f"Validation MSE: {mse:.3f}")
print(f"Predictions within ±7.5g: {within_range_count}/{total_count} ({percent_within_range:.2f}%)")
print(f"Predictions outside ±7.5g: {total_count - within_range_count} ({100 - percent_within_range:.2f}%)")
print("Test predictions saved to:", os.path.join(output_dir, "test_predictions.csv"))