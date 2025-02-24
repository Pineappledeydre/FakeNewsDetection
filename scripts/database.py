import pymongo
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("ERROR: MONGO_URI is not set. Add it to your .env file!")

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client["FakeNewsDB"]
collection = db["PolitifactClaims"]

# Load data
df = pd.read_csv("data/new_tweets_predictions.csv")

# Convert dataframe to MongoDB format
documents = df.to_dict(orient="records")
collection.insert_many(documents)

print("Data inserted into MongoDB!")
