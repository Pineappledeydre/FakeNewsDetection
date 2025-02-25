import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from database import collection

claims = list(collection.find({}, {"clean_text": 1, "is_fake": 1}))  

if not claims:
    print("❔ No labeled data found in MongoDB! Exiting...")
    exit()

df_train = pd.DataFrame(claims)

class LabeledTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }

class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        return self.sigmoid(self.fc(dropout_output))

model = BertClassifier()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

fine_tuned_model_path = "models/bert_finetuned_model.pth"
initial_model_path = "models/best_model.pth"

# Load existing fine-tuned model if available, otherwise load best_model.pth
try:
    model.load_state_dict(torch.load(fine_tuned_model_path, map_location=torch.device("cpu")))
    print(f"🦖 Loaded fine-tuned model from {fine_tuned_model_path}")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load(initial_model_path, map_location=torch.device("cpu")))
        print(f"🦖 No fine-tuned model found. Loaded initial model from {initial_model_path}")
    except FileNotFoundError:
        print("‼️ No model found! Make sure to add `best_model.pth` to the `models/` folder.")
        exit()

train_size = int(0.85 * len(df_train))  # 85% for training, 15% for validation
train_df = df_train[:train_size]
val_df = df_train[train_size:]

train_dataset = LabeledTextDataset(train_df["clean_text"].tolist(), train_df["is_fake"].tolist(), tokenizer)
val_dataset = LabeledTextDataset(val_df["clean_text"].tolist(), val_df["is_fake"].tolist(), tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Loss & Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Fine-tune Model
epochs = 3
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].unsqueeze(1)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"].unsqueeze(1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save Best Model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), model_path)
        print("🦖 Fine-tuning complete. Best model saved!")

print("🔥 Fine-tuning finished! 🔥")
