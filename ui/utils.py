import os
from datetime import datetime
import streamlit as st
from pymongo import MongoClient
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "patient_data")
FEEDBACK_COLL = os.getenv("MONGO_COLL_FEEDBACK", "history_rx_feedback")

# Create a single MongoClient instance cached by Streamlit.
@st.cache_resource
def get_mongo_client():
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI)
        return client
    except Exception as e:
        st.sidebar.error(f"Mongo connection error: {e}")
        return None

# Get the feedback collection from MongoDB.
def get_feedback_collection():
    client = get_mongo_client()
    if client is None:
        return None
    db = client[MONGO_DB_NAME]
    return db[FEEDBACK_COLL]

# Log user feedback to MongoDB.
def log_feedback(note_text: str, prescription_text: str, result: dict, useful: bool, comments: str | None):
    coll = get_feedback_collection()
    if coll is None:
        return

    doc = {
        "timestamp": datetime.utcnow().isoformat(),
        "note_text": note_text,
        "prescription_text": prescription_text,
        "history_features": result.get("history_features", []),
        "prescriptions": result.get("prescriptions", []),
        "insights": result.get("insights", []),
        "useful": useful,
        "comments": comments or "",
    }
    coll.insert_one(doc)

# Extract text from a PDF uploaded via Streamlit.
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
    except Exception:
        return ""

    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n\n".join(texts).strip()