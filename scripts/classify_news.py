import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
import sys
import pandas as pd

# ✅ Add `scripts` directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from database import collection  
from preprocess import preprocess  

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

# ✅ Load Tokenizer FIRST
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    tokenizer = None  # Handle errors gracefully

# ✅ Load Model
model = None

def load_model():
    global model
    model = BertClassifier()
    model_path = "models/bert_finetuned_model.pth"

    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}! Using default weights.")
        return  # Don't exit, just use an untrained model

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        model = None  # Keep model as None if loading fails

    model.eval()

# ✅ Call model loading function
load_model()

# ✅ Debug MongoDB Query
print("🔍 Fetching latest claims from MongoDB...")
claims = list(collection.find({}, {"Claim": 1, "Label": 1}))  # Fetch only required fields

# ✅ Debug: Print sample MongoDB documents
if claims:
    print("🔍 Sample MongoDB Data:", claims[:5])  # Check what fields exist
else:
    print("⚠️ No documents found in MongoDB!")

# ✅ Convert to DataFrame
df = pd.DataFrame(claims)

# ✅ Debug: Print DataFrame columns
print("🔍 DataFrame Columns:", df.columns)

# ✅ Handle Missing "clean_text" Column
df["clean_text"] = df["Claim"].apply(lambda x: preprocess(x) if isinstance(x, str) else "")

# ✅ Convert "Label" to numerical values for comparison
if "Label" in df.columns:
    df["is_fake"] = df["Label"].apply(lambda x: 1 if str(x).lower() == "false" else 0)
else:
    print("⚠️ Warning: 'Label' column is missing! Assigning default values.")
    df["is_fake"] = 0  # Default to not fake if Label is missing

# ✅ Run BERT Classification
def predict_fake(text):
    """Predicts the probability of fake news using BERT"""
    encoding = tokenizer.encode_plus(
        text, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    return output.squeeze().item()

df["probability_fake"] = df["clean_text"].apply(predict_fake)
df["probability_real"] = 1 - df["probability_fake"]
df["predicted_label"] = df["probability_fake"].apply(lambda x: 1 if x > 0.5 else 0)  # 1 = Fake, 0 = Real

# ✅ Compare predictions with true labels
df["correct"] = df["predicted_label"] == df["is_fake"]
accuracy = df["correct"].mean() * 100
print(df.head())
print(df.columns)

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
