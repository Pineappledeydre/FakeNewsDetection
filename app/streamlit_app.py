import streamlit as st
import pymongo
import torch
from dotenv import load_dotenv
import sys
import os

# ✅ Load .env file (for local testing)
load_dotenv()

# ✅ Debug: Print all environment variables
st.write("🔍 **All Environment Variables:**", os.environ)

# ✅ Fetch MONGO_URI from Streamlit Secrets OR OS Environment
MONGO_URI = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")

# ✅ If still missing, show error
if not MONGO_URI:
    st.error("❌ `MONGO_URI` is missing! Check Streamlit Secrets or GitHub Secrets.")
    st.stop()
else:
    st.success("✅ `MONGO_URI` Loaded!")

# ✅ Add `scripts` directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

# ✅ Import only AFTER ensuring MONGO_URI is set
from database import collection
from preprocess import preprocess
import classify_news  # ✅ Load `model, tokenizer` from `classify_news`

model = classify_news.model
tokenizer = classify_news.tokenizer

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
