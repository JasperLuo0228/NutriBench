import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==================== CONFIGURATION ====================
TRAIN_FILE = './dataset/train.csv'
VAL_FILE = './dataset/val.csv'
TEST_FILE = './dataset/test.csv'
MODEL_SAVE = 'high_accuracy_model.pth'
PREDICTION_FILE = 'final_predictions.csv'
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 30
THRESHOLD = 7.5  
TARGET_ACCURACY = 0.95
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== DATASET CLASS ====================
class TextDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx],

# ==================== ENHANCED LOSS FUNCTION ====================
class HybridAccuracyLoss(nn.Module):
    def __init__(self, threshold=THRESHOLD, alpha=0.7):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha
        
    def forward(self, preds, targets):
        abs_errors = torch.abs(preds - targets)
        accuracy_penalty = torch.sigmoid((abs_errors - self.threshold) * 3.0).mean()
        mse_loss = nn.functional.mse_loss(preds, targets)
        return self.alpha * accuracy_penalty + (1-self.alpha) * mse_loss

# ==================== ENHANCED MODEL ARCHITECTURE ====================
class EnhancedTransformerRegressor(nn.Module):
    def __init__(self, input_dim, model_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, 
            dim_feedforward=model_dim*4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        return self.output(x[:, 0, :])

# ==================== DATA AUGMENTATION ====================
def augment_data(queries, targets, augment_factor=0.3):
    num_augment = int(len(queries) * augment_factor)
    indices = np.random.choice(len(queries), num_augment)
    
    augmented_queries = []
    augmented_targets = []
    for idx in indices:
        noise = np.random.normal(0, 0.1, queries[idx].shape)
        augmented_queries.append(queries[idx] + noise)
        target_noise = np.random.uniform(-0.5, 0.5) * targets[idx]
        augmented_targets.append(targets[idx] + target_noise)
    
    return (np.concatenate([queries, augmented_queries]),
            np.concatenate([targets, augmented_targets]))

# ==================== TRAINING LOOP ====================
def train_model():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    train_q = train_df['query'].astype(str).tolist()
    train_y = train_df['carb'].astype(float).values.reshape(-1, 1)
    val_q = val_df['query'].astype(str).tolist()
    val_y = val_df['carb'].astype(float).values.reshape(-1, 1)
    test_q = test_df['query'].astype(str).tolist()
    
    # Text encoding
    print("Encoding text...")
    encoder = SentenceTransformer('all-mpnet-base-v2')
    X_train = encoder.encode(train_q, show_progress_bar=True)
    X_val = encoder.encode(val_q, show_progress_bar=True)
    X_test = encoder.encode(test_q, show_progress_bar=True)
    
    # Data augmentation
    print("Augmenting training data...")
    X_train, train_y = augment_data(X_train, train_y)
    
    # Feature scaling
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)
    X_test = scaler_x.transform(X_test)
    
    # Target scaling
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(train_y)
    y_val = scaler_y.transform(val_y)
    
    # Create DataLoaders
    train_dataset = TextDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TextDataset(torch.tensor(X_val, dtype=torch.float32),
                             torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, 
                           pin_memory=True)
    
    # Initialize model
    print("Initializing model...")
    model = EnhancedTransformerRegressor(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, 
                                 weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-5, epochs=EPOCHS, 
        steps_per_epoch=len(train_loader))
    loss_fn = HybridAccuracyLoss(alpha=0.8)
    
    # Training loop
    print(f"\nTraining for accuracy (error ≤ {THRESHOLD})...")
    best_accuracy = 0.0
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        train_preds, train_trues = [], []
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            train_preds.extend(pred.detach().cpu().numpy())
            train_trues.extend(yb.detach().cpu().numpy())
        
        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(yb.cpu().numpy())
        
        # Calculate metrics
        train_preds = scaler_y.inverse_transform(np.array(train_preds))
        train_trues = scaler_y.inverse_transform(np.array(train_trues))
        val_preds = scaler_y.inverse_transform(np.array(val_preds))
        val_trues = scaler_y.inverse_transform(np.array(val_trues))
        
        train_acc = np.mean(np.abs(train_preds - train_trues) <= THRESHOLD)
        val_acc = np.mean(np.abs(val_preds - val_trues) <= THRESHOLD)
        
        print(f"\nEpoch {epoch}:")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Train Accuracy: {train_acc:.2%} | Val Accuracy: {val_acc:.2%}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"🔥 New Best Accuracy: {best_accuracy:.2%}")
            
            if best_accuracy >= TARGET_ACCURACY:
                print(f"🎯 Target Accuracy of {TARGET_ACCURACY:.0%} Achieved!")
                break
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"🛑 Early Stopping at Epoch {epoch}")
                print(f"Best Accuracy: {best_accuracy:.2%}")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load(MODEL_SAVE))
    model.eval()
    
    # Test prediction
    test_preds = []
    test_loader = DataLoader(
        TextDataset(torch.tensor(X_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE*2)
    
    with torch.no_grad():
        for xb, in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            test_preds.extend(pred.cpu().numpy())
    
    test_preds = scaler_y.inverse_transform(np.array(test_preds))
    pd.DataFrame({'query': test_q, 'predicted_carb': test_preds.flatten()}).to_csv(
        PREDICTION_FILE, index=False)
    print(f"\nPredictions saved to {PREDICTION_FILE}")

if __name__ == "__main__":
    train_model()