import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Load predictions
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
pred_path = os.path.join(base_dir, "output", "xgboost", "val_predictions.csv")
df = pd.read_csv(pred_path)
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/xgboost"))
diag_path = os.path.join(output_dir, "val_diagnostics.csv")
val_df = pd.read_csv(diag_path)


# Create output directory for plots
plot_dir = os.path.join(base_dir, "output", "xgboost", "plots")
os.makedirs(plot_dir, exist_ok=True)

# True vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="y_true", y="y_pred", data=df, alpha=0.5, edgecolor=None)
plt.plot([df["y_true"].min(), df["y_true"].max()],
         [df["y_true"].min(), df["y_true"].max()],
         color="red", linestyle="--", label="Ideal")
plt.xlabel("True Carbohydrate Value (g)")
plt.ylabel("Predicted Value (g)")
plt.title("True vs. Predicted Carbohydrates")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "true_vs_pred.png"))
plt.close()

# Prediction Error Distribution
df["error"] = df["y_pred"] - df["y_true"]
plt.figure(figsize=(8, 6))
sns.histplot(df["error"], bins=50, kde=True, color="steelblue")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Prediction Error (y_pred - y_true)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Error")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "error_distribution.png"))
plt.close()

# Residuals vs True Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["y_true"], y=df["error"], alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Carbohydrate Value (g)")
plt.ylabel("Residual (y_pred - y_true)")
plt.title("Residuals vs. True Values")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "residuals_vs_true.png"))
plt.close()

# Top-k Largest Errors
topk = val_df.sort_values("abs_error", ascending=False).head(10)
plt.figure()
sns.barplot(x=topk.index, y=topk["abs_error"], color="salmon")
plt.xticks(rotation=45)
plt.ylabel("Absolute Error (g)")
plt.xlabel("Sample Index")
plt.title("Top 10 Largest Prediction Errors")
plt.savefig(os.path.join(plot_dir, "top10_errors.png"))