import pandas as pd
import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv("MONGODB_URL")

client = pymongo.MongoClient(MONGO_URL)
db = client["ads_projects"]
collection = db["spam_ham"]

df = pd.read_csv("notebooks/spamham.csv")
print(f"✅ CSV loaded — {len(df)} rows")

df = df.rename(columns={'Label': 'label', 'Message': 'message'})
df = df.dropna(subset=['message'])

collection.delete_many({})
print("Old data cleared!")

records = df.to_dict(orient='records')
collection.insert_many(records)
print(f"✅ {len(records)} records inserted!")
print(f"Total in MongoDB: {collection.count_documents({})}")