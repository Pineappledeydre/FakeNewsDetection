import streamlit as st
import pymongo
import torch
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ✅ Load Environment Variables
load_dotenv()

# ✅ Fetch MONGO_URI
MONGO_URI = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("❌ `MONGO_URI` is missing! Check Streamlit Secrets or GitHub Secrets.")
    st.stop()

# ✅ Import Necessary Modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from database import collection
from preprocess import preprocess
import classify_news
from scrape_politifact import fetch_new_politifact_claims  

# ✅ Load Model & Tokenizer
if classify_news.model is None:
    st.error("❌ Model is missing! Please upload `models/bert_finetuned_model.pth`.")
    st.stop()
model = classify_news.model
tokenizer = classify_news.tokenizer

# ✅ Connect to MongoDB
try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client["FakeNewsDB"]
    collection = db["PolitifactClaims"]
except Exception as e:
    st.error(f"❌ Failed to connect to MongoDB: {e}")
    st.stop()

# 🌟 **APP TITLE**
st.title("📰 Fake News Detector - Politifact")

# ==============================================
# 🔹 **Fetch New Claims from Politifact**
# ==============================================
st.header("🔄 Fetch New Fact-Checked Claims")
col1, col2 = st.columns(2)
with col1:
    min_claims = st.slider("Number of Claims:", 10, 100, 50, 10)
with col2:
    max_pages = st.slider("Max Pages to Search:", 1, 50, 10, 1)

if st.button("🔍 Fetch New Claims"):
    try:
        fetch_new_politifact_claims(min_claims=min_claims, max_pages=max_pages)
        st.success(f"✅ Scraped & Classified {min_claims} Claims from up to {max_pages} Pages!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error Fetching Claims: {e}")

# ==============================================
# 🔍 **Latest Fact-Checked Claims**
# ==============================================
st.header("🔍 Latest Fact-Checked Claims")
try:
    docs = list(collection.find().sort("_id", -1).limit(10))

    if docs:
        for doc in docs:
            st.subheader(f"📌 {doc.get('Claim', 'Unknown Claim')}")
            predicted_label = "Fake" if doc.get("is_fake", 1) == 1 else "Real"
            st.write(f"🗂 **Label**: {predicted_label}")
            st.write(f"📊 **Fake Probability**: {doc.get('probability_fake', 0):.2%}")
            st.divider()
    else:
        st.info("No classified claims found in MongoDB.")
except Exception as e:
    st.error(f"❌ Error Fetching Claims: {e}")

# ==============================================
# 📊 **Fake vs. Real News Distribution**
# ==============================================
st.header("📊 Fake vs. Real News Distribution")
try:
    df = pd.DataFrame(list(collection.find({}, {"Claim": 1, "probability_fake": 1, "probability_real": 1, "is_fake": 1})))

    if not df.empty:
        # ✅ Compute Predicted & Actual Labels
        df["predicted_label"] = df["probability_fake"].apply(lambda x: "Fake" if x > 0.5 else "Real")
        df["actual_label"] = df["is_fake"].apply(lambda x: "Fake" if x == 1 else "Real")

        # ✅ Calculate Accuracy
        df["correct"] = df["predicted_label"] == df["actual_label"]
        accuracy = df["correct"].mean() * 100  

        # ✅ Count Fake vs. Real for Visualization
        label_counts = df["predicted_label"].value_counts()

        # ✅ Pie Chart Visualization
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=["red", "green"], startangle=90)
        ax.set_title("Fake vs. Real News Distribution")
        st.pyplot(fig)

        # ✅ Display Model Accuracy
        st.metric(label="🎯 Model Accuracy", value=f"{accuracy:.2f}%", help="Percentage of correctly classified claims.")

        # ✅ Show Classification Table
        st.subheader("🔍 Full Classification Results")
        st.dataframe(df[["Claim", "predicted_label", "actual_label", "probability_fake", "probability_real"]])
    else:
        st.info("No classified claims found.")
except Exception as e:
    st.error(f"❌ Error Fetching Classified Claims: {e}")

# ==============================================
# 📝 **Classify Your Own Claim**
# ==============================================
st.header("📝 Classify Your Own Claim")
user_input = st.text_area("Enter a claim:")

if st.button("🔎 Analyze Claim"):
    if user_input:
        try:
            # ✅ Preprocess Input
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Label", value=predicted_label)
            with col2:
                st.metric(label="Fake Probability", value=f"{probability_fake:.2%}")

            # ✅ Save to MongoDB
            collection.insert_one({
                "Claim": user_input,
                "Label": "Not classified",
                "is_fake": 1 if predicted_label == "Fake" else 0,
                "clean_text": cleaned_text,
                "Source": "User Submitted",
                "Date": "Unknown",
                "probability_fake": probability_fake,
                "probability_real": probability_real,
                "predicted_label": predicted_label,
            })

            st.success("✅ Claim Added to MongoDB for Tracking!")

        except Exception as e:
            st.error(f"❌ Error Processing Claim: {e}")
    else:
        st.warning("⚠️ Please Enter a Claim to Analyze.")
