import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd

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
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        return self.sigmoid(self.fc(dropout_output))

# Load pretrained model
model = BertClassifier()
model.load_state_dict(torch.load("models/bert_trained_model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load scraped data
df = pd.read_csv("data/covid_politifact_claims.csv")

# Tokenize text
def tokenize_text(text, tokenizer):
    encoding = tokenizer.encode_plus(
        text, max_length=128, truncation=True, padding='max_length', return_tensors='pt'
    )
    return encoding["input_ids"], encoding["attention_mask"]

df["probability_fake"] = df["clean_text"].apply(lambda x: model(*tokenize_text(x, tokenizer))[0].item())
df["probability_real"] = 1 - df["probability_fake"]
df["predicted_label"] = df["probability_fake"].apply(lambda x: "Fake" if x > 0.5 else "Real")

df.to_csv("data/new_tweets_predictions.csv", index=False)
