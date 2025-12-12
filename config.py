import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load .env for local development.
load_dotenv()

# Try Streamlit secrets first (for deployment), fallback to .env (for local)
try:
    import streamlit as st
    MONGO_URI = st.secrets.get("MONGO_URI", os.getenv("MONGO_URI"))
    MONGO_DB_NAME = st.secrets.get("MONGO_DB_NAME", os.getenv("MONGO_DB_NAME", "patient_data"))
except:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "patient_data")

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in .env or Streamlit secrets")

# Collection names.
MONGO_COLL_TRAIN = os.getenv("MONGO_COLL_TRAIN", "train")
MONGO_COLL_FEATURES = os.getenv("MONGO_COLL_FEATURES", "features")
MONGO_COLL_PATIENT_NOTES = os.getenv("MONGO_COLL_PATIENT_NOTES", "patient_notes")
MONGO_COLL_NBME_NOTES = os.getenv("MONGO_COLL_NBME_NOTES", "nbme_notes")
MONGO_COLL_NBME_FEATURES = os.getenv("MONGO_COLL_NBME_FEATURES", "nbme_features")

# Data directory.
DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URI)

def get_db():
    client = get_mongo_client()
    return client[MONGO_DB_NAME]