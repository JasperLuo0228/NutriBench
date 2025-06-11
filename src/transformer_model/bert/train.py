from bert_regression import BERTRegression, TextRegressionDataset
from plot import plot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import glob
import os

# ======== Load Data ========
train_df = pd.read_csv("data/train.csv")  # 包含 "query", "carb"
val_df = pd.read_csv("data/val.csv")

# ======== Preprocess Data ========
mean = train_df['carb'].mean()
std = train_df['carb'].std()
train_df['carb_norm'] = (train_df['carb'] - mean) / std
# val_df['carb_norm'] = (val_df['carb'] - mean) / std # 保持验证集的归一化一致性

# ======== Tokenization and Dataset Preparation ========
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = TextRegressionDataset(train_df["query"].tolist(), train_df["carb_norm"].tolist(), tokenizer)
val_dataset = TextRegressionDataset(val_df["query"].tolist(), val_df["carb"].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ======== Training Setup ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTRegression().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = ExponentialLR(optimizer, gamma=0.95)
criterion = nn.MSELoss()

# ======== Load Checkpoint ========
ckpt_files = glob.glob("checkpoints/Epoch*_*.pth") # 查找所有 checkpoint 文件
f = 0 # 是否删除原 checkpoint

if ckpt_files:
    # 按文件修改时间排序，取最新一个
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    checkpoint = torch.load(latest_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    latest_ckpt = None
    start_epoch = 1
    print("No checkpoint found, starting from scratch.")

for param_group in optimizer.param_groups:
    param_group['lr'] = 5e-6  # 重设学习率

# ======== Training Loop ========
epochs = 280 # 总训练 epoch 数
log_path = "output/log.csv" # 日志文件路径

for epoch in range(start_epoch, epochs+1):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        total_loss += loss.item()

        avg_loss = total_loss / (step+1) # 累计平均损失
        train_rmse = np.sqrt(avg_loss) * std # 累计rmse, 并反归一化

        preds_real = preds.detach().cpu().numpy() * std + mean  # 反归一化预测值
        labels_real = labels.detach().cpu().numpy() * std + mean  # 反归一化标签值
        errors = np.abs(preds_real - labels_real)

        mask1 = (labels_real < 75) # 处理标签小于100的样本
        correct += np.sum(errors[mask1] < 7.5) # 绝对误差小于7.5的样本计为正确
        total += np.sum(mask1)

        mask2 = (labels_real >= 75) # 处理标签大于等于100的样本
        relative_errors = np.abs(errors[mask2] / (labels_real[mask2] + 1e-6))
        correct += np.sum(relative_errors < 0.1) # 相对误差小于0.1的样本计为正确
        total += np.sum(mask2)

        train_acc = correct / total # average acc for each epoch

        # if len(errors) > 0:
        #     max_idx = np.argmax(errors)
        #     print(f"Max_idx {max_idx} | Prediction: {preds_real[max_idx]:.2f} | Label: {labels_real[max_idx]:.2f} | Error: {errors[max_idx]:.2f}")
        
        progress_bar.set_postfix({
            "Acc": f"{train_acc*100:.2f}%",
            "RMSE": f"{train_rmse:.2f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    scheduler.step() # 更新学习率

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            # loss = nn.MAELoss(outputs, labels)
            # total_loss += loss.item()

            preds = outputs.cpu().numpy() * std + mean  # 反归一化预测值
            labels = labels.cpu().numpy()

            errors = np.abs(preds - labels)
            # correct += np.sum(errors < 7.5)
            # total += len(errors)

            mask1 = (labels < 75) # 处理标签小于100的样本
            correct += np.sum(errors[mask1] < 7.5) # 绝对误差小于7.5的样本计为正确
            total += np.sum(mask1)

            mask2 = (labels >= 75) # 处理标签大于等于100的样本
            relative_errors = np.abs(errors[mask2] / (labels[mask2] + 1e-6))
            correct += np.sum(relative_errors < 0.1) # 相对误差小于0.1的样本计为正确
            total += np.sum(mask2)

            val_preds.extend(preds)
            val_labels.extend(labels)

    # avg_loss = total_loss / len(val_loader)
    val_acc = correct / total
    val_mae = mean_absolute_error(val_labels, val_preds)
    val_mse = mean_squared_error(val_labels, val_preds)
    val_rmse = np.sqrt(val_mse)

    print(f"Val Acc: {val_acc*100:.2f}%, MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")

    # 保存 checkpoint
    if latest_ckpt and f==1: 
        os.remove(latest_ckpt) # 删除之前的 checkpoint
    elif f==0:
        f=1
    latest_ckpt = f"checkpoints/Epoch-{epoch}_Val_Acc-{val_acc*100:.0f}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, latest_ckpt)
    print(f"Saved checkpoint for epoch {epoch}")

    # 保存 log
    df = pd.DataFrame({
        "epoch": [epoch],
        "val_acc": [round(val_acc * 100, 2)],
        "train_acc": [round(train_acc * 100, 2)],
        "val_mae": [round(val_mae, 2)],
        "val_rmse": [round(val_rmse, 2)],
        "train_rmse": [round(train_rmse, 2)],
        "lr": [f"{optimizer.param_groups[0]['lr']:.2e}"],
    })

    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, index=False)

# ======== Plotting ========
plot(log_path)  