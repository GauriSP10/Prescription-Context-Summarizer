import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env once at import time.
load_dotenv()

# Mongo basics.
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in .env")

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "patient_data")

# Collection names.
MONGO_COLL_TRAIN = os.getenv("MONGO_COLL_TRAIN", "train")
MONGO_COLL_FEATURES = os.getenv("MONGO_COLL_FEATURES", "features")
MONGO_COLL_PATIENT_NOTES = os.getenv("MONGO_COLL_PATIENT_NOTES", "patient_notes")
MONGO_COLL_NBME_NOTES = os.getenv("MONGO_COLL_NBME_NOTES", "nbme_notes")
MONGO_COLL_NBME_FEATURES = os.getenv("MONGO_COLL_NBME_FEATURES", "nbme_features")

# Data directory.
DATA_DIR = os.getenv("DATA_DIR", "./data")

# Makes sure that the data directory exists.
os.makedirs(DATA_DIR, exist_ok=True)

def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI)

def get_db():
    client = get_mongo_client()
    return client[MONGO_DB_NAME]
