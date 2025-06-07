# train_and_predict.py

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from bert_regression import BERTRegression, TextRegressionDataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# ======== Load Data ========
train_df = pd.read_csv("data/train.csv")  # 包含 "query", "carb"
val_df = pd.read_csv("data/val.csv")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TextRegressionDataset(train_df["query"].tolist()[:1000], train_df["carb"].tolist()[:1000], tokenizer)
val_dataset = TextRegressionDataset(val_df["query"].tolist()[:100], val_df["carb"].tolist()[:100], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ======== Training Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTRegression().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = ExponentialLR(optimizer, gamma=0.95)
criterion = nn.MSELoss()

# ======== Training Loop ========
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
        progress_bar.set_postfix({"Train Loss": f"{loss.item():.2f}, Learning Rate = {scheduler.get_last_lr()[0]:.2e}"})

    # avg_loss = total_loss / len(train_loader)
    # print(f"Epoch {epoch+1} Average Train Loss: {avg_loss:.2f}")

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

    mse = mean_squared_error(val_labels, val_preds)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse:.2f}")