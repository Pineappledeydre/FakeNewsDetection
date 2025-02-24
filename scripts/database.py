import pymongo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("ERROR: MONGO_URI is not set. Check your environment variables!")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["FakeNewsDB"]
collection = db["PolitifactClaims"]

print("✅ Connected to MongoDB Atlas!")
