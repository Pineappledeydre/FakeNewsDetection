import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from database import collection  
from preprocess import preprocess  

import sys
import os

# Add the scripts directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from classify_news import model, tokenizer

# ✅ Define BERT Classifier
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

# ✅ Load trained model
model = BertClassifier()
model_path = "models/bert_finetuned_model.pth"  # Ensure the path is correct

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print(f"✅ Loaded model from {model_path}")
except FileNotFoundError:
    print("⚠️ No fine-tuned model found. Exiting...")
    exit()

model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text):
    """Tokenizes text for BERT"""
    encoding = tokenizer.encode_plus(
        text, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
    )
    return encoding["input_ids"], encoding["attention_mask"]

# ✅ Fetch latest claims from MongoDB
claims = list(collection.find({}, {"Claim": 1, "is_fake": 1, "clean_text": 1}))

# ✅ Handle empty database case
if not claims:
    print("⚠️ No data found in MongoDB!")
    exit()

df = pd.DataFrame(claims)

# ✅ Ensure all text is preprocessed before classification
df["clean_text"] = df["clean_text"].apply(lambda x: preprocess(x) if isinstance(x, str) else "")

# ✅ Run BERT Classification
def predict_fake(text):
    """Predicts the probability of fake news using BERT"""
    input_ids, attention_mask = tokenize_text(text)
    
    with torch.no_grad():
        output = model(input_ids, attention_mask)
    
    return output.squeeze().item()

df["probability_fake"] = df["clean_text"].apply(predict_fake)
df["probability_real"] = 1 - df["probability_fake"]
df["predicted_label"] = df["probability_fake"].apply(lambda x: 1 if x > 0.5 else 0)  # 1 = Fake, 0 = Real

# ✅ Compare predictions with true labels
df["correct"] = df["predicted_label"] == df["is_fake"]
accuracy = df["correct"].mean() * 100

# ✅ Save classified claims
df.to_csv("data/classified_claims.csv", index=False)
print(f"✅ Accuracy: {accuracy:.2f}%")
print("✅ Classified claims saved!")

# ✅ Trigger fine-tuning if accuracy drops below threshold
if accuracy < 85:
    print("🚨 Accuracy is low! Fine-tuning required!")
    fine_tune = True
else:
    print("✅ Accuracy is acceptable. No fine-tuning needed.")
    fine_tune = False
