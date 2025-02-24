import streamlit as st
import pymongo

MONGO_URI = "your_mongo_uri_here"

client = pymongo.MongoClient(MONGO_URI)
db = client["FakeNewsDB"]
collection = db["PolitifactClaims"]

st.title("📰 Fake News Detector - Politifact")

# Display latest classified claims
docs = list(collection.find().sort("_id", -1).limit(10))
if docs:
    for doc in docs:
        st.subheader(f"🔹 {doc['Claim']}")
        st.write(f"🔹 **Label**: {doc['predicted_label']}")
        st.write(f"🔹 **Fake Probability**: {doc['probability_fake']:.2%}")
        st.write("---")
else:
    st.write("No data available yet.")
