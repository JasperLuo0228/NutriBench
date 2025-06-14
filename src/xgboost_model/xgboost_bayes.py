import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
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
X = X_train
y = y_train

# Define model
xgb = XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42)

# Define Bayesian search space
search_space = {
    "max_depth": Integer(4, 10),
    "learning_rate": Real(0.01, 0.2, prior='log-uniform'),
    "n_estimators": Integer(2000, 3000),
    "subsample": Real(0.7, 1.0),
    "colsample_bytree": Real(0.7, 1.0),
}

# Bayesian optimization
opt = BayesSearchCV(
    xgb,
    search_spaces=search_space,
    n_iter=50,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=1,
    random_state=42
)

opt.fit(X, y)

best_params = opt.best_params_
print(" Best Parameters from BayesSearchCV:", best_params)

# Evaluate best model
best_model = opt.best_estimator_

# Train metrics
y_train_pred = best_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

train_within_mask   = (np.abs(y_train_pred - y_train) <= 7.5)
train_within_count  = train_within_mask.sum()
train_total         = len(y_train)
train_within_pct    = 100 * train_within_count / train_total


# Validation metrics
y_val_pred = best_model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

val_within_mask   = (np.abs(y_val_pred - y_val) <= 7.5)
val_within_count  = val_within_mask.sum()
val_total         = len(y_val)
val_within_pct    = 100 * val_within_count / val_total

# Predict on test set
y_test_pred = best_model.predict(X_test)

# Save results
output_dir = os.path.join(base_dir, "output/xgboost")
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({
    "y_true": y_val,
    "y_pred": y_val_pred
}).to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)

test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

test_output = pd.DataFrame({
    "id": test_df.iloc[:, 0],
    "query": test_df["query"],
    "y_pred": y_test_pred
})

test_output.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

# Create validation diagnostics
val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
val_df["carb_true"] = y_val.values
val_df["carb_pred"] = y_val_pred
val_df["abs_error"] = np.abs(y_val - y_val_pred)
val_df["within_7.5"] = val_df["abs_error"] <= 7.5

val_df.to_csv(os.path.join(output_dir, "val_diagnostics.csv"), index=False)

# Print top 10 worst predictions
print("\n Top 10 largest prediction errors:")
print(val_df.sort_values("abs_error", ascending=False)[["carb_true", "carb_pred", "abs_error"]].head(10))

with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write("Best Params: " + str(opt.best_params_) + "\n")
    f.write(f"Training MAE: {train_mae:.3f}\n")
    f.write(f"Training MSE: {train_mse:.3f}\n")
    f.write(f"Training predictions within ±7.5: {train_within_count}/{train_total} "
            f"({train_within_pct:.2f}%)\n")
    f.write(f"Validation MAE: {val_mae:.3f}\n")
    f.write(f"Validation MSE: {val_mse:.3f}\n")
    f.write(f"Validation predictions within ±7.5: {val_within_count}/{val_total} "
            f"({val_within_pct:.2f}%)\n")

# Print summary
print("\n Evaluation Summary:")
print("Best Parameters:", opt.best_params_)

print(f"Training MAE: {train_mae:.3f}")
print(f"Training MSE: {train_mse:.3f}")
print(f"Predictions within ±7.5g: {train_within_count}/{train_total} ({train_within_pct:.2f}%)")
print(f"Predictions outside ±7.5g: {train_total - train_within_count} ({100 - train_within_pct:.2f}%)")

print(f"Validation MAE: {val_mae:.3f}")
print(f"Validation MSE: {val_mse:.3f}")
print(f"Predictions within ±7.5g: {val_within_count}/{val_total} ({val_within_pct:.2f}%)")
print(f"Predictions outside ±7.5g: {val_total - val_within_count} ({100 - val_within_pct:.2f}%)")

print("Test predictions saved to:", os.path.join(output_dir, "test_predictions.csv"))