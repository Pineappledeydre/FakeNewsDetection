import streamlit as st
import pymongo
import torch
from dotenv import load_dotenv
import os
import sys

# ✅ Load environment variables
load_dotenv()

# ✅ Fetch `MONGO_URI`
MONGO_URI = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("❌ `MONGO_URI` is missing! Check Streamlit Secrets or GitHub Secrets.")
    st.stop()
else:
    st.success("✅ `MONGO_URI` Loaded!")

# ✅ Add `scripts` directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# ✅ Import necessary modules
from database import collection
from preprocess import preprocess
import classify_news

# ✅ Check if model was loaded successfully
if classify_news.model is None:
    st.error("❌ Model is missing! Please upload `models/bert_finetuned_model.pth`.")
    st.stop()
else:
    model = classify_news.model
    tokenizer = classify_news.tokenizer
    st.success("✅ Model and tokenizer loaded!")

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

# =====================================================
# 🔹 **Fetch New Claims from Politifact**
# =====================================================
if st.button("🔄 Fetch New Fact-Checked Claims"):
    st.info("⏳ Scraping new claims from Politifact...")
    
    try:
        fetch_new_politifact_claims()  # Run the scraper
        st.success("✅ New claims fetched and stored in MongoDB! Click 'Refresh' to update the list.")
    except Exception as e:
        st.error(f"❌ Error fetching claims: {e}")
        
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
