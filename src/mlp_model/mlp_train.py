import pandas as pd
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Add tfidf module path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../data/method1_tfidf'))
from tfidf import TfidfProcessor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def extract_nutrition_features(text):
    """Extract nutrition-specific features from text"""
    features = {}
    
    # Extract quantities
    quantities = re.findall(r'(\d+(?:\.\d+)?)\s*(g|ml|oz|cup|tbsp|tsp|piece|slice)', text.lower())
    features['has_quantity'] = len(quantities) > 0
    features['quantity_count'] = len(quantities)
    
    # Extract food types
    food_types = ['meat', 'vegetable', 'fruit', 'grain', 'dairy', 'protein']
    for food_type in food_types:
        features[f'has_{food_type}'] = food_type in text.lower()
    
    # Extract cooking methods
    cooking_methods = ['fried', 'baked', 'grilled', 'steamed', 'boiled', 'raw']
    for method in cooking_methods:
        features[f'is_{method}'] = method in text.lower()
    
    return pd.Series(features)

# 1. Load data (relative to src/mlp_model/)
train_df = pd.read_csv(os.path.join('../../data/train.csv'))
val_df = pd.read_csv(os.path.join('../../data/val.csv'))
test_df = pd.read_csv(os.path.join('../../data/test.csv'))

# 2. Feature Engineering
print("Performing feature engineering...")

# Extract nutrition features
train_features = train_df['query'].apply(extract_nutrition_features)
val_features = val_df['query'].apply(extract_nutrition_features)
test_features = test_df['query'].apply(extract_nutrition_features)

# 3. Text Feature Processing
print("Processing text features...")

# Load the pre-trained TF-IDF vectorizer
tfidf_proc = TfidfProcessor()
tfidf_proc.load_vectorizer("tfidf_vectorizer.joblib")

# Transform text features
X_train_tfidf = tfidf_proc.transform(train_df["query"])
X_val_tfidf = tfidf_proc.transform(val_df["query"])
X_test_tfidf = tfidf_proc.transform(test_df["query"])

# Combine TF-IDF features with engineered features
X_train = np.hstack([X_train_tfidf.toarray(), train_features])
X_val = np.hstack([X_val_tfidf.toarray(), val_features])
X_test = np.hstack([X_test_tfidf.toarray(), test_features])

# 4. Target Variable Preprocessing
print("Preprocessing target variable...")

# Log transform and robust scaling for target variable
y_train = train_df["carb"].values
y_val = val_df["carb"].values

# Log transform to handle right-skewed distribution
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Use RobustScaler for better handling of outliers
target_scaler = RobustScaler()
y_train_scaled = target_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val_log.reshape(-1, 1)).flatten()

# 5. Feature Scaling
print("Scaling features...")
feature_scaler = RobustScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

# 6. Enhanced MLP Model Configuration
print("Training enhanced MLP model...")

mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # Balanced network size
    alpha=0.01,  # Stronger regularization
    learning_rate_init=0.001,  # Moderate learning rate
    max_iter=500,  # More iterations
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,  # Standard validation split
    n_iter_no_change=15,  # Standard early stopping patience
    batch_size='auto',
    learning_rate='adaptive',
    power_t=0.5,
    momentum=0.9,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

print("Fitting model...")
mlp.fit(X_train_scaled, y_train_scaled)

print("Best loss achieved:", mlp.best_loss_)
print("Training completed!")

# 7. Training Set Performance Evaluation
y_train_pred_scaled = mlp.predict(X_train_scaled)
y_train_pred_log = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_train_pred = np.expm1(y_train_pred_log)
y_train_pred = np.maximum(y_train_pred, 0)

train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate training set accuracy within different error ranges
train_tolerance_5g = np.abs(y_train_pred - y_train) <= 5.0
train_tolerance_7_5g = np.abs(y_train_pred - y_train) <= 7.5
train_tolerance_10g = np.abs(y_train_pred - y_train) <= 10.0
train_tolerance_15g = np.abs(y_train_pred - y_train) <= 15.0

train_acc_5g = np.mean(train_tolerance_5g) * 100
train_acc_7_5g = np.mean(train_tolerance_7_5g) * 100
train_acc_10g = np.mean(train_tolerance_10g) * 100
train_acc_15g = np.mean(train_tolerance_15g) * 100

# Training set MAPE
train_mape = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-8))) * 100

print("\nTrain Performance:")
print(f"Train MSE: {train_mse:.3f}")
print(f"Train MAE: {train_mae:.3f}")
print(f"Train R² Score: {train_r2:.3f}")
print(f"Train MAPE: {train_mape:.2f}%")
print("\nTraining Set Accuracy Metrics:")
print(f"Accuracy within ±5g: {train_acc_5g:.1f}%")
print(f"Accuracy within ±7.5g: {train_acc_7_5g:.1f}%")
print(f"Accuracy within ±10g: {train_acc_10g:.1f}%")
print(f"Accuracy within ±15g: {train_acc_15g:.1f}%")

# 8. Validation Set Performance Evaluation
y_val_pred_scaled = mlp.predict(X_val_scaled)
y_val_pred_log = target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
y_val_pred = np.expm1(y_val_pred_log)
y_val_pred = np.maximum(y_val_pred, 0)

val_mse = mean_squared_error(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

# Calculate validation set accuracy within different error ranges
val_tolerance_5g = np.abs(y_val_pred - y_val) <= 5.0
val_tolerance_7_5g = np.abs(y_val_pred - y_val) <= 7.5
val_tolerance_10g = np.abs(y_val_pred - y_val) <= 10.0
val_tolerance_15g = np.abs(y_val_pred - y_val) <= 15.0

val_acc_5g = np.mean(val_tolerance_5g) * 100
val_acc_7_5g = np.mean(val_tolerance_7_5g) * 100
val_acc_10g = np.mean(val_tolerance_10g) * 100
val_acc_15g = np.mean(val_tolerance_15g) * 100

# Validation set MAPE
val_mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-8))) * 100

print("\nValidation Performance:")
print(f"Validation MSE: {val_mse:.3f}")
print(f"Validation MAE: {val_mae:.3f}")
print(f"Validation R² Score: {val_r2:.3f}")
print(f"Validation MAPE: {val_mape:.2f}%")
print("\nValidation Set Accuracy Metrics:")
print(f"Accuracy within ±5g: {val_acc_5g:.1f}%")
print(f"Accuracy within ±7.5g: {val_acc_7_5g:.1f}%")
print(f"Accuracy within ±10g: {val_acc_10g:.1f}%")
print(f"Accuracy within ±15g: {val_acc_15g:.1f}%")

# Create output directory
os.makedirs("../../output/mlp", exist_ok=True)

# Save results
val_df_results = val_df.copy()
val_df_results["carb_pred"] = y_val_pred
val_df_results.to_csv("../../output/mlp/val_predictions_fast.csv", index=False)

with open("../../output/mlp/metrics_fast.txt", "w") as f:
    f.write("Train Performance:\n")
    f.write(f"Train MSE: {train_mse:.3f}\n")
    f.write(f"Train MAE: {train_mae:.3f}\n")
    f.write(f"Train R² Score: {train_r2:.3f}\n")
    f.write(f"Train MAPE: {train_mape:.2f}%\n")
    f.write("\nTraining Set Accuracy Metrics:\n")
    f.write(f"Accuracy within ±5g: {train_acc_5g:.1f}%\n")
    f.write(f"Accuracy within ±7.5g: {train_acc_7_5g:.1f}%\n")
    f.write(f"Accuracy within ±10g: {train_acc_10g:.1f}%\n")
    f.write(f"Accuracy within ±15g: {train_acc_15g:.1f}%\n\n")
    
    f.write("Validation Performance:\n")
    f.write(f"Validation MSE: {val_mse:.3f}\n")
    f.write(f"Validation MAE: {val_mae:.3f}\n")
    f.write(f"Validation R² Score: {val_r2:.3f}\n")
    f.write(f"Validation MAPE: {val_mape:.2f}%\n")
    f.write("\nValidation Set Accuracy Metrics:\n")
    f.write(f"Accuracy within ±5g: {val_acc_5g:.1f}%\n")
    f.write(f"Accuracy within ±7.5g: {val_acc_7_5g:.1f}%\n")
    f.write(f"Accuracy within ±10g: {val_acc_10g:.1f}%\n")
    f.write(f"Accuracy within ±15g: {val_acc_15g:.1f}%\n")

# Test set predictions
y_test_pred_scaled = mlp.predict(X_test_scaled)
y_test_pred_log = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.maximum(y_test_pred, 0)

test_df["carb"] = y_test_pred
test_df.to_csv("../../output/mlp/test_predictions_fast.csv", index=False)
print("Fast test predictions saved to ../../output/mlp/test_predictions_fast.csv")