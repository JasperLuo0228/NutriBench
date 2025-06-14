import glob
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from bert_regression import BERTRegression, TextRegressionDataset

# train_df = pd.read_csv("data/train.csv")  
# mean = train_df['carb'].mean()
# std = train_df['carb'].std()
# stats_df = pd.DataFrame({
#     "stat": ["mean", "std"],
#     "value": [mean, std]
# })
# stats_df.to_csv("train_mean&std.csv", index=False)

stats_df = pd.read_csv("train_mean&std.csv")
mean, std = stats_df['value'].values
print(f"Mean: {mean}, Std: {std}")

# ======  test_loader ======
test_df = pd.read_csv("data/test.csv")  

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
test_dataset = TextRegressionDataset(
    texts=test_df["query"].tolist(),
    targets=None,
    tokenizer=tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=32)

# ====== load model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTRegression().to(device)

# ==== load checkpoint ====
ckpt_files = glob.glob("checkpoints/Epoch*_*.pth")
if ckpt_files:
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    checkpoint = torch.load(latest_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded the lastest model")
else:
    print("No model found.")
    exit()

# ==== test output ====
model.eval()
test_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        preds = outputs.cpu().numpy() * std + mean  # rescale predictions
        test_preds.extend(preds)

# ====== save result ======
output_df = pd.DataFrame({
    "query": test_df["query"],
    "carb": test_preds
})
output_df.index = range(10000, 10000 + len(output_df))
output_df.to_csv("output/test_preds.csv", index=True)

print("Test predictions saved to test_preds.csv")
