import pandas as pd
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
# 添加tfidf模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../data/method1_tfidf'))
from tfidf import TfidfProcessor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

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

# 4. 改进的目标变量预处理
y_train = train_df["carb"].values
y_val = val_df["carb"].values

# 对数变换处理右偏分布
y_train_log = np.log1p(y_train)  # log(1+x) 避免log(0)
y_val_log = np.log1p(y_val)

# 标准化目标变量
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val_log.reshape(-1, 1)).flatten()

# 5. 改进的MLP模型配置 - 快速版本
print("Training fast MLP model for low MSE...")

# 直接使用经验上好的参数，专门降低MSE
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # 足够的容量
    alpha=0.01,  # L2正则化防止过拟合
    learning_rate_init=0.001,  # 稍小的学习率更稳定
    max_iter=300,  # 减少迭代次数但足够收敛
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,  # 更多验证数据
    n_iter_no_change=15  # 早停条件
)

print("Fitting model...")
mlp.fit(X_train, y_train_scaled)

print("Best loss achieved:", mlp.best_loss_)
print("Training completed!")

# 6. 预测和评估
y_val_pred_scaled = mlp.predict(X_val)

# 逆变换到原始尺度
y_val_pred_log = target_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
y_val_pred = np.expm1(y_val_pred_log)  # exp(x) - 1
y_val_pred = np.maximum(y_val_pred, 0)  # 确保非负

mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)

# 计算多种准确率指标
r2 = r2_score(y_val, y_val_pred)

# 容忍范围内的准确率
tolerance_5g = np.abs(y_val_pred - y_val) <= 5.0
tolerance_10g = np.abs(y_val_pred - y_val) <= 10.0
tolerance_15g = np.abs(y_val_pred - y_val) <= 15.0

acc_5g = np.mean(tolerance_5g) * 100
acc_10g = np.mean(tolerance_10g) * 100
acc_15g = np.mean(tolerance_15g) * 100

# 平均绝对百分比误差 (MAPE)
mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-8))) * 100  # +1e-8避免除零

print("Fast MLP Results:")
print(f"Validation MSE: {mse:.3f}")
print(f"Validation MAE: {mae:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
print("\n准确率指标:")
print(f"±5g范围内准确率: {acc_5g:.1f}%")
print(f"±10g范围内准确率: {acc_10g:.1f}%")
print(f"±15g范围内准确率: {acc_15g:.1f}%")

# 创建输出目录
os.makedirs("../../output/mlp", exist_ok=True)

# 保存结果
val_df_results = val_df.copy()
val_df_results["carb_pred"] = y_val_pred
val_df_results.to_csv("../../output/mlp/val_predictions_fast.csv", index=False)

with open("../../output/mlp/metrics_fast.txt", "w") as f:
    f.write(f"Fast MLP Results:\n")
    f.write(f"Validation MSE: {mse:.3f}\n")
    f.write(f"Validation MAE: {mae:.3f}\n")
    f.write(f"R² Score: {r2:.3f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write("\n准确率指标:\n")
    f.write(f"±5g范围内准确率: {acc_5g:.1f}%\n")
    f.write(f"±10g范围内准确率: {acc_10g:.1f}%\n")
    f.write(f"±15g范围内准确率: {acc_15g:.1f}%\n")

# 测试集预测
y_test_pred_scaled = mlp.predict(X_test)
y_test_pred_log = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = np.expm1(y_test_pred_log)
y_test_pred = np.maximum(y_test_pred, 0)

test_df["carb"] = y_test_pred
test_df.to_csv("../../output/mlp/test_predictions_fast.csv", index=False)
print("Fast test predictions saved to ../../output/mlp/test_predictions_fast.csv")