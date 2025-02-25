import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from database import collection 
from preprocess import preprocess  

# Load Model
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

# Load trained model
model = BertClassifier()
model.load_state_dict(torch.load("models/bert_trained_model.pth", map_location=torch.device("cpu")))
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    """Tokenizes text for BERT"""
    encoding = tokenizer.encode_plus(
        text, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
    )
    return encoding["input_ids"], encoding["attention_mask"]

# Fetch latest claims from MongoDB
claims = list(collection.find({}, {"Claim": 1, "is_fake": 1, "clean_text": 1}))

df = pd.DataFrame(claims)
if df.empty:
    print("⚠️ No data found in MongoDB!")
    exit()

# Run BERT Classification
df["probability_fake"] = df["clean_text"].apply(lambda x: model(*tokenize_text(x))[0].item())
df["probability_real"] = 1 - df["probability_fake"]
df["predicted_label"] = df["probability_fake"].apply(lambda x: 1 if x > 0.5 else 0)  # 1 = Fake, 0 = Real

# Compare with true labels
df["correct"] = df["predicted_label"] == df["is_fake"]
accuracy = df["correct"].mean() * 100

df.to_csv("data/classified_claims.csv", index=False)
print(f"Accuracy: {accuracy:.2f}%")
print("Classified claims saved!")

if accuracy < 85:  # Example: Trigger fine-tuning if accuracy drops below 85%
    print("🚨 Accuracy is low! Fine-tuning required!")
    fine_tune = True
else:
    print("✅ Accuracy is acceptable. No fine-tuning needed.")
    fine_tune = False
