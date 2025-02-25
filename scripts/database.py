import pymongo
import os

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("❌ MONGO_URI is missing! Check your environment variables.")

try:
    client = pymongo.MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)  # ✅ Force TLS
    db = client["FakeNewsDB"]
    collection = db["PolitifactClaims"]
    print("✅ Connected to MongoDB!")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    raise
