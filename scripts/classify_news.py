import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from database import collection  # MongoDB integration
from preprocess import preprocess

# Define the BERT classification model
class BertClassifier(nn.Module):
    """Fake news detection model using BERT."""
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.sigmoid(self.fc(self.dropout(outputs.pooler_output)))

# Load trained model
model = BertClassifier()
model.load_state_dict(torch.load("models/bert_finetuned_model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Process and classify each document
for doc in collection.find({"PredictedLabel": {"$exists": False}}):  # Only classify new claims
    cleaned_text = preprocess(doc["Claim"])
    encoding = tokenizer.encode_plus(cleaned_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

    with torch.no_grad():
        prediction = model(encoding["input_ids"], encoding["attention_mask"]).item()

    # Store prediction results
    doc["ProbabilityFake"] = prediction
    doc["ProbabilityReal"] = 1 - prediction
    doc["PredictedLabel"] = "Fake" if prediction > 0.5 else "Real"

    # Update MongoDB with predictions
    collection.update_one({"_id": doc["_id"]}, {"$set": doc})

print("Classification complete. MongoDB records updated!")
