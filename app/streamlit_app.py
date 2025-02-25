import streamlit as st
import pymongo
import torch
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
# Fetch MONGO_URI
MONGO_URI = st.secrets.get("MONGO_URI") or os.getenv("MONGO_URI")

if not MONGO_URI:
    st.error("‼️ MONGO_URI is missing! Check Streamlit Secrets or GitHub Secrets.")
    st.stop()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from database import collection
from preprocess import preprocess
import classify_news
from scrape_politifact import fetch_new_politifact_claims  

if classify_news.model is None:
    st.error("‼️ Model is missing! Please upload models/bert_finetuned_model.pth.")
    st.stop()
model = classify_news.model
tokenizer = classify_news.tokenizer

# Connect to MongoDB
try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client["FakeNewsDB"]
    collection = db["PolitifactClaims"]
except Exception as e:
    st.error(f"‼️ Failed to connect to MongoDB: {e}")
    st.stop()

st.markdown("<h4 style='font-size:22px; font-weight:bold;'>📰 Fake News Detector - Covid 19</h4>", unsafe_allow_html=True)

# Fetch New Claims from Politifact
st.markdown("<h4 style='font-size:16px; font-weight:bold;'>🔄 Fetch New Fact-Checked Claims</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    min_claims = st.slider("Number of Claims:", 10, 100, 50, 10)
with col2:
    max_pages = st.slider("Max Pages to Search:", 1, 50, 10, 1)

if st.button("🔍 Fetch New Claims"):
    try:
        fetch_new_politifact_claims(min_claims=min_claims, max_pages=max_pages)
        st.success(f"🦖 Scraped & Classified {min_claims} Claims from up to {max_pages} Pages!")
        st.rerun()
    except Exception as e:
        st.error(f"‼️ Error Fetching Claims: {e}")

# Latest Fact-Checked Claims
st.markdown("<h4 style='font-size:12px; font-weight:bold;'>🔍 Latest Fact-Checked Claims</h4>", unsafe_allow_html=True)
try:
    docs = list(collection.find().sort("_id", -1).limit(10))

    if docs:
        for doc in docs:
            st.markdown(f"<p style='font-size:14px; font-weight:bold;'>🧷 {doc.get('Claim', 'Unknown Claim')}</p>", unsafe_allow_html=True)
            predicted_label = "Fake" if doc.get("is_fake", 1) == 1 else "Real"
            st.markdown(f"<p style='font-size:12px;'><b>Label:</b> {predicted_label}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:12px;'><b>Fake Probability:</b> {doc.get('probability_fake', 0):.2%}</p>", unsafe_allow_html=True)
            st.divider()

    else:
        st.info("No classified claims found in MongoDB.")
except Exception as e:
    st.error(f"‼️ Error Fetching Claims: {e}")

# Fake vs. Real News Distribution
st.markdown("<h4 style='font-size:16px; font-weight:bold;'>🦕 Fake vs. Real News Distribution 🦖</h4>", unsafe_allow_html=True)

try:
    df = pd.DataFrame(list(collection.find({}, {"Claim": 1, "probability_fake": 1, "probability_real": 1, "is_fake": 1})))

    if not df.empty:
        # ✅ Count Fake vs. Real for visualization
        label_counts = df["is_fake"].value_counts()
        labels = ["Fake", "Real"]
        sizes = [label_counts.get(1, 0), label_counts.get(0, 0)]  # Get counts safely
        colors = ["#E74C3C", "#2ECC71"]  # Red for Fake, Green for Real
        explode = [0.05, 0]  # Slightly separate Fake slice
    
        # ✅ **Create High-Quality Pie Chart**
        fig, ax = plt.subplots(figsize=(2, 2), dpi=300)  # Small but high-resolution
    
        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct='%1.0f%%',
            colors=colors,
            startangle=140,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'black'},
            textprops={'fontsize': 6, 'weight': 'bold'}  # Small, clear text
        )
    
        ax.set_title("Fake vs. Real News", fontsize=6, fontweight="bold", pad=2)
        ax.axis("equal")  # Ensures **perfect circle**
    
        # ✅ **Save as Image & Display in Streamlit**
        chart_path = "chart.png"
        fig.savefig(chart_path, bbox_inches="tight", dpi=300)  # High DPI for clarity
    
        st.image(chart_path, caption="Fake vs. Real News", use_container_width=True, width=120)  # Small, sharp image

        ############################################################################
        
        st.markdown(f"""
            <p style="font-size:16px; font-weight:bold; margin-bottom:2px;">Model Accuracy</p>
            <p style="font-size:14px; margin-top:0px;">{accuracy:.2f}%</p>
        """, unsafe_allow_html=True)

        st.markdown("<h4 style='font-size:16px; font-weight:bold;'>🔍 Full Classification Results</h4>", unsafe_allow_html=True)
        st.dataframe(df[["Claim", "predicted_label", "actual_label", "probability_fake", "probability_real"]])
    else:
        st.info("No classified claims found.")
except Exception as e:
    st.error(f"‼️ Error Fetching Classified Claims: {e}")

# Your Own Claim
st.markdown("<h4 style='font-size:16px; font-weight:bold;'>📝 Classify Your Own Claim</h4>", unsafe_allow_html=True)
user_input = st.text_area("Enter a claim:")

if st.button("🔎 Analyze Claim"):
    if user_input:
        try:
            cleaned_text = preprocess(user_input)
            encoding = tokenizer.encode_plus(
                cleaned_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
            )

            with torch.no_grad():
                prediction = model(encoding["input_ids"], encoding["attention_mask"]).item()

            probability_fake = prediction
            probability_real = 1 - prediction
            predicted_label = "Fake" if prediction > 0.5 else "Real"

            st.subheader("🔍 Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Label", value=predicted_label)
            with col2:
                st.metric(label="Fake Probability", value=f"{probability_fake:.2%}")

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

            st.success("🦖 Claim Added to MongoDB for Tracking!")

        except Exception as e:
            st.error(f"‼️ Error Processing Claim: {e}")
    else:
        st.warning("❔ Please Enter a Claim to Analyze.")
