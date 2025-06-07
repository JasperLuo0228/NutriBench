from transformers import BertModel
import torch
import torch.nn as nn

class BERTRegression(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.ReLU()  
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # [CLS] token 表示整句信息
        return self.regressor(cls_output).squeeze(1)
    
from torch.utils.data import Dataset

class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets=None, tokenizer=None, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.targets is not None:
            item["labels"] = self.targets[idx]
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

    

