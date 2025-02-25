import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from database import collection 

claims = list(collection.find({}, {"clean_text": 1, "is_fake": 1}))  # Only fetch relevant fields

if not claims:
    print("⚠️ No labeled data found in MongoDB! Exiting...")
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

# Initialize Model
model = BertClassifier()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = LabeledTextDataset(df_train["clean_text"].tolist(), df_train["is_fake"].tolist(), tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

epochs = 3
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

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader):.4f}")

torch.save(model.state_dict(), "models/bert_finetuned_model.pth")
print("✅ Fine-tuning complete. Model saved!")
