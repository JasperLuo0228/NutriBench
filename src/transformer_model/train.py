# train_and_predict.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
from bert_regression import BERTRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class NutriBenchDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

    def __len__(self):
        return len(self.features)

X_train = np.load("data\method2_sbert\X_train.npy")
y_train = np.load("data\method2_sbert\y_train.npy")
X_val = np.load("data\method2_sbert\X_val.npy")
y_val = np.load("data\method2_sbert\y_val.npy")

# Prepare datasets
train_dataset = NutriBenchDataset(X_train, y_train)
val_dataset = NutriBenchDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Init model
model = BERTRegression().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_rmse = mean_squared_error(val_labels, val_preds, squared=False)
    print(f"[Epoch {epoch+1}] Val RMSE: {val_rmse:.4f}")

# Prediction
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        all_preds.extend(outputs.cpu().numpy())

# Save results
test_df["carb"] = all_preds
test_df.to_csv("submission.csv", index=False)
print("✅ Prediction saved to submission.csv")
