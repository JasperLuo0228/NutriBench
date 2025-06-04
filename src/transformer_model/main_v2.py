import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import warnings
import re
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NutritionDataset(Dataset):
    def __init__(self, embeddings, targets=None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.embeddings[idx], self.targets[idx]
        else:
            return self.embeddings[idx]

# 改进的神经网络架构 - 针对营养数据的特点
class NutritionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super(NutritionPredictor, self).__init__()
        
        # 特征预处理层
        self.feature_norm = nn.BatchNorm1d(input_dim)
        
        # 主干网络 - 使用更保守的架构
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout * (0.8 ** i))  # 递减的dropout率
            ])
            prev_dim = hidden_dim
        
        # 输出层使用residual connection
        self.main_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 辅助回归头用于多任务学习（预测是否为高碳水食物）
        self.aux_classifier = nn.Linear(prev_dim, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_aux=False):
        x = self.feature_norm(x)
        features = self.main_layers(x)
        
        # 主要输出：碳水化合物含量
        carb_output = self.output_layer(features).squeeze(-1)
        
        if return_aux:
            # 辅助输出：是否为高碳水食物（>20g）
            aux_output = torch.sigmoid(self.aux_classifier(features)).squeeze(-1)
            return carb_output, aux_output
        
        return carb_output

def extract_nutrition_features(texts):
    """
    基于数据样本优化的特征提取
    """
    print("Extracting nutrition-specific features...")
    
    # 基于实际数据观察到的模式
    high_carb_foods = [
        # 主食类
        'bread', 'roll', 'pancake', 'pretzel', 'pizza', 'noodle', 'pasta', 'rice',
        # 甜食类
        'cake', 'candy', 'chocolate', 'pie', 'doughnut', 'eclair', 'pudding', 'syrup',
        # 水果类
        'banana', 'pear', 'persimmon', 'fruit',
        # 蔬菜类（淀粉）
        'potato', 'squash', 'peas', 'carrot', 'corn',
        # 豆类
        'bean', 'refried'
    ]
    
    low_carb_foods = [
        # 肉类
        'beef', 'pork', 'lamb', 'chicken', 'turkey', 'meat', 'steak', 'roast',
        # 海鲜
        'salmon', 'fish', 'perch', 'chinook',
        # 其他蛋白质
        'egg', 'cheese'
    ]
    
    # 数量和单位词汇
    units = ['cup', 'ounce', 'oz', 'pound', 'lb', 'gram', 'g', 'tablespoon', 'tbsp', 
             'teaspoon', 'tsp', 'slice', 'piece', 'serving', 'can', 'jar']
    
    quantities = ['single', 'whole', 'half', 'quarter', 'large', 'medium', 'small', 
                  'personal', 'cubic inch']
    
    # 烹饪方法
    cooking_methods = ['baked', 'broiled', 'grilled', 'cooked', 'roasted', 'braised', 
                      'smoked', 'fried', 'raw', 'frozen', 'canned', 'dried']
    
    features = []
    
    for text in texts:
        text_lower = text.lower()
        feature_vector = []
        
        # 1. 基础文本特征
        words = text_lower.split()
        feature_vector.extend([
            len(text),
            len(words),
            len([w for w in words if len(w) > 6]),  # 长单词数
        ])
        
        # 2. 食物类型特征
        high_carb_count = sum(1 for food in high_carb_foods if food in text_lower)
        low_carb_count = sum(1 for food in low_carb_foods if food in text_lower)
        feature_vector.extend([
            high_carb_count,
            low_carb_count,
            high_carb_count / (high_carb_count + low_carb_count + 1),  # 高碳水比例
        ])
        
        # 3. 数量特征（关键！）
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(n) for n in numbers]
        
        feature_vector.extend([
            len(numbers),
            numbers[0] if numbers else 0,
            max(numbers) if numbers else 0,
            sum(numbers) if numbers else 0,
            np.prod(numbers) if numbers else 0,  # 数字乘积（体积估算）
        ])
        
        # 4. 单位和数量词特征
        feature_vector.extend([
            sum(1 for unit in units if unit in text_lower),
            sum(1 for qty in quantities if qty in text_lower),
        ])
        
        # 5. 特殊模式匹配
        feature_vector.extend([
            1 if re.search(r'\d+\s*(cup|cups)', text_lower) else 0,
            1 if re.search(r'\d+\s*(ounce|ounces|oz)', text_lower) else 0,
            1 if re.search(r'(large|medium|small)', text_lower) else 0,
            1 if 'with' in text_lower else 0,  # 可能表示添加物
            1 if 'topped' in text_lower else 0,
            1 if 'filled' in text_lower else 0,
            1 if 'sauce' in text_lower else 0,
        ])
        
        # 6. 烹饪方法特征
        cooking_count = sum(1 for method in cooking_methods if method in text_lower)
        feature_vector.append(cooking_count)
        
        # 7. 特定高碳水指标词
        high_carb_indicators = ['heavy syrup', 'glazed', 'sugar', 'sweet', 'crust', 'dough']
        feature_vector.append(sum(1 for indicator in high_carb_indicators if indicator in text_lower))
        
        features.append(feature_vector)
    
    features = np.array(features)
    
    # 优化的TF-IDF
    tfidf = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
        max_df=0.8,
        min_df=3,
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # 只保留字母单词
    )
    
    tfidf_features = tfidf.fit_transform(texts).toarray()
    combined_features = np.hstack([features, tfidf_features])
    
    print(f"Feature shape: {combined_features.shape}")
    return combined_features, tfidf

def load_and_preprocess_data_v2():
    """
    改进的数据预处理
    """
    print("Loading and preprocessing data...")
    
    try:
        train_df = pd.read_csv('./dataset/train.csv')
        val_df = pd.read_csv('./dataset/val.csv') 
        test_df = pd.read_csv('./dataset/test.csv')
    except FileNotFoundError:
        print("Error: CSV files not found!")
        return None
    
    print(f"Original sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 数据分析
    print("\nCarb distribution analysis:")
    print(f"Train carb stats: min={train_df['carb'].min():.2f}, max={train_df['carb'].max():.2f}")
    print(f"Mean: {train_df['carb'].mean():.2f}, Median: {train_df['carb'].median():.2f}")
    print(f"Std: {train_df['carb'].std():.2f}")
    
    # 数据分布可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(train_df['carb'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Original Carb Distribution')
    plt.xlabel('Carb (g)')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(train_df['carb']), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Log-transformed Carb Distribution')
    plt.xlabel('Log(Carb + 1)')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(train_df['carb'])
    plt.title('Carb Boxplot')
    plt.ylabel('Carb (g)')
    
    plt.tight_layout()
    plt.show()
    
    # 更智能的异常值处理
    carb_values = train_df['carb'].values
    Q1, Q3 = np.percentile(carb_values, [25, 75])
    IQR = Q3 - Q1
    
    # 使用更宽松的异常值标准
    lower_bound = max(0, Q1 - 2.0 * IQR)  # 碳水不能为负
    upper_bound = Q3 + 2.0 * IQR
    
    print(f"\nOutlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 保留更多数据，只移除极端异常值
    mask = (train_df['carb'] >= lower_bound) & (train_df['carb'] <= upper_bound)
    train_df_clean = train_df[mask].copy()
    
    print(f"Removed {len(train_df) - len(train_df_clean)} outliers")
    print(f"Final train size: {len(train_df_clean)}")
    
    # 特征提取
    train_features, tfidf_vectorizer = extract_nutrition_features(train_df_clean["query"].tolist())
    val_features, _ = extract_nutrition_features(val_df["query"].tolist())
    test_features, _ = extract_nutrition_features(test_df["query"].tolist())
    
    # 特征标准化
    feature_scaler = RobustScaler()  # 对异常值更鲁棒
    train_features_scaled = feature_scaler.fit_transform(train_features)
    val_features_scaled = feature_scaler.transform(val_features)
    test_features_scaled = feature_scaler.transform(test_features)
    
    # 目标变量处理 - 使用对数变换处理偏态分布
    train_targets = train_df_clean["carb"].values
    val_targets = val_df["carb"].values
    
    # 对数变换
    train_targets_log = np.log1p(train_targets)  # log(x+1)避免log(0)
    val_targets_log = np.log1p(val_targets)
    
    # 标准化对数变换后的目标
    target_scaler = StandardScaler()
    train_targets_scaled = target_scaler.fit_transform(train_targets_log.reshape(-1, 1)).flatten()
    val_targets_scaled = target_scaler.transform(val_targets_log.reshape(-1, 1)).flatten()
    
    # 创建辅助标签（高碳水 vs 低碳水）
    train_aux_labels = (train_targets > 20).astype(float)
    val_aux_labels = (val_targets > 20).astype(float)
    
    print(f"Feature dimension: {train_features_scaled.shape[1]}")
    print(f"High-carb ratio in train: {train_aux_labels.mean():.3f}")
    
    return {
        'train_features': train_features_scaled,
        'val_features': val_features_scaled, 
        'test_features': test_features_scaled,
        'train_targets': train_targets_scaled,
        'val_targets': val_targets_scaled,
        'train_aux': train_aux_labels,
        'val_aux': val_aux_labels,
        'target_scaler': target_scaler,
        'original_train_targets': train_targets,
        'original_val_targets': val_targets
    }

def create_data_loaders_v2(data_dict, batch_size=64):
    """创建数据加载器"""
    train_dataset = NutritionDataset(data_dict['train_features'], data_dict['train_targets'])
    val_dataset = NutritionDataset(data_dict['val_features'], data_dict['val_targets'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model_v2(model, train_loader, val_loader, data_dict, num_epochs=150):
    """改进的训练循环"""
    # 使用组合损失函数
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 25
    
    # 获取辅助标签
    train_aux = torch.FloatTensor(data_dict['train_aux'])
    val_aux = torch.FloatTensor(data_dict['val_aux'])
    
    print(f"Training model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_aux_iter = iter(DataLoader(train_aux, batch_size=64, shuffle=True, drop_last=True))
        
        for embeddings, targets in train_loader:
            try:
                aux_targets = next(train_aux_iter)
            except StopIteration:
                train_aux_iter = iter(DataLoader(train_aux, batch_size=64, shuffle=True, drop_last=True))
                aux_targets = next(train_aux_iter)
            
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            aux_targets = aux_targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            carb_pred, aux_pred = model(embeddings, return_aux=True)
            
            # 组合损失
            main_loss = mse_loss(carb_pred, targets)
            aux_loss = bce_loss(aux_pred, aux_targets)
            total_loss = main_loss + 0.1 * aux_loss  # 辅助损失权重较小
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += main_loss.item()
            train_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings, targets = embeddings.to(device), targets.to(device)
                outputs = model(embeddings)
                loss = mse_loss(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        train_loss /= train_batches
        val_loss /= val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 15 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

def evaluate_model_v2(model, val_loader, data_dict):
    """改进的模型评估"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for embeddings, targets in val_loader:
            embeddings, targets = embeddings.to(device), targets.to(device)
            outputs = model(embeddings)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    predictions_scaled = np.array(all_predictions)
    targets_scaled = np.array(all_targets)
    
    # 逆变换到原始尺度
    target_scaler = data_dict['target_scaler']
    
    predictions_log = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    targets_log = target_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()
    
    # 从对数空间转换回原始空间
    predictions_original = np.expm1(predictions_log)  # exp(x) - 1
    targets_original = data_dict['original_val_targets']
    
    # 确保预测值非负
    predictions_original = np.maximum(predictions_original, 0)
    
    # 计算指标
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, predictions_original)
    
    print(f"\nValidation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 分析预测分布
    print(f"\nPrediction Analysis:")
    print(f"Predicted range: [{predictions_original.min():.2f}, {predictions_original.max():.2f}]")
    print(f"Actual range: [{targets_original.min():.2f}, {targets_original.max():.2f}]")
    print(f"Predicted mean: {predictions_original.mean():.2f}")
    print(f"Actual mean: {targets_original.mean():.2f}")
    
    return predictions_original, targets_original

def ensemble_predict(models, test_features, data_dict):
    """集成预测"""
    test_dataset = NutritionDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for embeddings in test_loader:
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
                predictions.extend(outputs.cpu().numpy())
        
        all_predictions.append(np.array(predictions))
    
    # 平均预测
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # 逆变换
    target_scaler = data_dict['target_scaler']
    pred_log = target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
    pred_original = np.expm1(pred_log)
    pred_original = np.maximum(pred_original, 0)
    
    return pred_original

def main_improved():
    """改进的主函数"""
    # 数据预处理
    data_dict = load_and_preprocess_data_v2()
    if data_dict is None:
        return
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders_v2(data_dict, batch_size=64)
    
    # 训练多个模型进行集成
    models = []
    input_dim = data_dict['train_features'].shape[1]
    
    print(f"Input dimension: {input_dim}")
    
    # 训练3个不同配置的模型
    configs = [
        {'hidden_dims': [128, 64], 'dropout': 0.3},
        {'hidden_dims': [96, 48], 'dropout': 0.4},
        {'hidden_dims': [160, 80], 'dropout': 0.25}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n=== Training Model {i+1}/3 ===")
        
        model = NutritionPredictor(
            input_dim=input_dim,
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(device)
        
        print(f"Model {i+1} parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        train_losses, val_losses = train_model_v2(model, train_loader, val_loader, data_dict)
        models.append(model)
        
        # 单个模型评估
        predictions, targets = evaluate_model_v2(model, val_loader, data_dict)
    
    # 集成预测
    print("\n=== Ensemble Prediction ===")
    test_predictions = ensemble_predict(models, data_dict['test_features'], data_dict)
    
    print(f"Test prediction stats:")
    print(f"Mean: {test_predictions.mean():.2f}")
    print(f"Std: {test_predictions.std():.2f}")
    print(f"Range: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    
    # 保存结果
    try:
        test_df = pd.read_csv('./dataset/test.csv')
        test_df['carb'] = test_predictions
        test_df.to_csv('improved_predictions.csv', index=False)
        print("Predictions saved to 'improved_predictions.csv'")
    except:
        pd.DataFrame({'carb': test_predictions}).to_csv('improved_predictions.csv', index=False)
        print("Predictions saved to 'improved_predictions.csv'")
    
    # 保存最佳模型
    torch.save(models[0].state_dict(), 'best_nutrition_model.pth')
    print("Best model saved to 'best_nutrition_model.pth'")

if __name__ == "__main__":
    main_improved()