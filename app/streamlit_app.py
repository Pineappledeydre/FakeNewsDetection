import streamlit as st
import pymongo
import os
from dotenv import load_dotenv
from classify_news import model, tokenizer  # Load model from classify_news.py
from preprocess import preprocess  # Load text preprocessing

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

client = pymongo.MongoClient(MONGO_URI)
db = client["FakeNewsDB"]
collection = db["PolitifactClaims"]

st.title("📰 Fake News Detector - Politifact")

st.header("🔍 Latest Fact-Checked Claims")
docs = list(collection.find().sort("_id", -1).limit(10))  # Get latest 10 records

if docs:
    for doc in docs:
        st.subheader(f"📌 {doc.get('Claim', 'Unknown Claim')}")
        st.write(f"🗂 **Label**: {doc.get('PredictedLabel', 'Not classified')}")
        st.write(f"📊 **Fake Probability**: {doc.get('ProbabilityFake', 0):.2%}")
        st.write("---")
else:
    st.write("No classified claims found in MongoDB.")

st.header("📝 Classify Your Own Claim")
user_input = st.text_area("Enter a claim:")

if st.button("🔎 Analyze Claim"):
    if user_input:
        cleaned_text = preprocess(user_input)
        encoding = tokenizer.encode_plus(cleaned_text, return_tensors="pt", 
                                         max_length=128, truncation=True, padding="max_length")

        with torch.no_grad():
            prediction = model(encoding["input_ids"], encoding["attention_mask"]).item()

        probability_fake = prediction
        probability_real = 1 - prediction
        predicted_label = "Fake" if prediction > 0.5 else "Real"

        st.subheader("🔍 Prediction Result")
        st.write(f"**Predicted Label**: {predicted_label}")
        st.write(f"**Fake Probability**: {probability_fake:.2%}")
        st.write(f"**Real Probability**: {probability_real:.2%}")

        collection.insert_one({
            "Claim": user_input,
            "ProbabilityFake": probability_fake,
            "ProbabilityReal": probability_real,
            "PredictedLabel": predicted_label
        })
        st.success("Claim added to MongoDB for tracking!")
    else:
        st.warning("Please enter a claim to analyze.")
