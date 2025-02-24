import pymongo
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file (for local) or GitHub Secrets (for cloud)
if os.path.exists(".env"):
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("🚨 ERROR: MONGO_URI is not set. Add it to your .env file or GitHub Secrets!")

try:
    client = pymongo.MongoClient(MONGO_URI)
    db = client["FakeNewsDB"]
    collection = db["PolitifactClaims"]
    print("✅ Connected to MongoDB Atlas!")
except pymongo.errors.ConnectionFailure as e:
    print(f"🚨 Connection failed: {e}")
    exit()

csv_path = "data/new_tweets_predictions.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"🚨 ERROR: {csv_path} not found!")

df = pd.read_csv(csv_path)

if df.empty:
    print("WARNING: No data found in CSV. Nothing to insert.")
else:
    documents = df.to_dict(orient="records")
    
    if collection.count_documents({}) > 0:
        print("WARNING: Collection already contains data. Skipping insertion to prevent duplicates.")
    else:
        collection.insert_many(documents)
        print(f"Inserted {len(documents)} records into MongoDB!")
