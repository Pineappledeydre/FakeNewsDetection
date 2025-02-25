import streamlit as st
import pymongo
import torch
import pandas as pd
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
from scrape_politifact import fetch_new_politifact_claims  

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
# 🔹 **Fetch New Claims from Politifact (User Control)**
# =====================================================
st.header("🔄 Fetch New Fact-Checked Claims")

min_claims = st.slider("Select number of claims to fetch:", min_value=10, max_value=100, value=50, step=10)
max_pages = st.slider("Select maximum pages to search:", min_value=1, max_value=50, value=10, step=1)

if st.button("🔍 Fetch New Claims"):
    try:
        fetch_new_politifact_claims(min_claims=min_claims, max_pages=max_pages)  # ✅ Fetch & classify
        st.success(f"✅ Scraped and classified {min_claims} claims from up to {max_pages} pages!")
        st.rerun()  
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
            
            # Convert `is_fake` to 'Fake' or 'Real'
            predicted_label = "Fake" if doc.get("is_fake", 1) == 1 else "Real"
            st.write(f"🗂 **Label**: {predicted_label}")            
            st.write(f"📊 **Fake Probability**: {doc.get('probability_fake', 0):.2%}")
            st.write("---")
    else:
        st.info("No classified claims found in MongoDB.")
except Exception as e:
    st.error(f"❌ Error fetching classified claims: {e}")

# ==============================================
# 📊 **Visualization of Fake vs. Real News**
# ==============================================
import matplotlib.pyplot as plt

st.header("📊 Fake vs. Real News Distribution")

# ✅ Fetch latest predictions from MongoDB
try:
    df = pd.DataFrame(list(collection.find({}, {"Claim": 1, "probability_fake": 1, "probability_real": 1, "is_fake": 1})))

    if not df.empty:
        # ✅ Ensure 'predicted_label' is correctly derived
        df["predicted_label"] = df["probability_fake"].apply(lambda x: "Fake" if x > 0.5 else "Real")
        df["actual_label"] = df["is_fake"].apply(lambda x: "Fake" if x == 1 else "Real")

        # ✅ Calculate Accuracy
        df["correct"] = df["predicted_label"] == df["actual_label"]
        accuracy = df["correct"].mean() * 100  # Convert to percentage

        # ✅ Count Fake vs. Real for plotting
        label_counts = df["predicted_label"].value_counts()

        # ✅ Matplotlib Pie Chart
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=["red", "green"], startangle=90)
        ax.set_title("Fake vs. Real News Distribution")
        st.pyplot(fig)

        # ✅ Display Model Accuracy
        st.subheader(f"🎯 Model Accuracy: {accuracy:.2f}%")
        st.write("Accuracy is calculated as the percentage of correctly classified claims.")

        # ✅ Show table with results
        st.subheader("🔍 Full Classification Results")
        st.dataframe(df[["Claim", "predicted_label", "actual_label", "probability_fake", "probability_real"]])
    else:
        st.info("No classified claims found.")
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
                "Label": "Not classified",  # Default label from Politifact, will be updated later
                "is_fake": 1 if predicted_label == "Fake" else 0,
                "clean_text": cleaned_text,
                "Source": "User Submitted",
                "Date": "Unknown",
                "probability_fake": probability_fake,
                "probability_real": probability_real,
                "predicted_label": predicted_label,  # ✅ Now stored!
            })

            st.success("✅ Claim added to MongoDB for tracking!")

        except Exception as e:
            st.error(f"❌ Error processing claim: {e}")
    else:
        st.warning("⚠️ Please enter a claim to analyze.")
