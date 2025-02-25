import streamlit as st
import pymongo
import os
import torch
from dotenv import load_dotenv
from classify_news import model, tokenizer  # Load model from classify_news.py
from preprocess import preprocess  # Load text preprocessing

# ✅ Load MongoDB credentials
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("❌ MongoDB connection string is missing! Check your .env file or GitHub Secrets.")
    st.stop()

# ✅ Connect to MongoDB
try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client["FakeNewsDB"]
    collection = db["PolitifactClaims"]
    st.success("✅ Connected to MongoDB!")
except Exception as e:
    st.error(f"❌ Failed to connect to MongoDB: {e}")
    st.stop()

# ✅ App Title
st.title("📰 Fake News Detector - Politifact")

# ==============================================
# 🔹 **Display Latest Fact-Checked Claims**
# ==============================================
st.header("🔍 Latest Fact-Checked Claims")
try:
    docs = list(collection.find().sort("_id", -1).limit(10))  # Get latest 10 records

    if docs:
        for doc in docs:
            st.subheader(f"📌 {doc.get('Claim', 'Unknown Claim')}")
            st.write(f"🗂 **Label**: {doc.get('predicted_label', 'Not classified')}")
            st.write(f"📊 **Fake Probability**: {doc.get('probability_fake', 0):.2%}")
            st.write("---")
    else:
        st.info("No classified claims found in MongoDB.")
except Exception as e:
    st.error(f"❌ Error fetching classified claims: {e}")

# ==============================================
# 📝 **Classify User's Input**
# ==============================================
st.header("📝 Classify Your Own Claim")
user_input = st.text_area("Enter a claim:")

if st.button("🔎 Analyze Claim"):
    if user_input:
        try:
            # ✅ Preprocess input
            cleaned_text = preprocess(user_input)
            encoding = tokenizer.encode_plus(
                cleaned_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
            )

            # ✅ Get Prediction
            with torch.no_grad():
                prediction = model(encoding["input_ids"], encoding["attention_mask"]).item()

            probability_fake = prediction
            probability_real = 1 - prediction
            predicted_label = "Fake" if prediction > 0.5 else "Real"

            # ✅ Display Results
            st.subheader("🔍 Prediction Result")
            st.write(f"**Predicted Label**: {predicted_label}")
            st.write(f"**Fake Probability**: {probability_fake:.2%}")
            st.write(f"**Real Probability**: {probability_real:.2%}")

            # ✅ Save to MongoDB
            collection.insert_one({
                "Claim": user_input,
                "probability_fake": probability_fake,
                "probability_real": probability_real,
                "predicted_label": predicted_label
            })
            st.success("✅ Claim added to MongoDB for tracking!")

        except Exception as e:
            st.error(f"❌ Error processing claim: {e}")
    else:
        st.warning("⚠️ Please enter a claim to analyze.")
