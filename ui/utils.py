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


@st.cache_resource
def get_mongo_client():
    """
    Create and cache a MongoDB client for the Streamlit application.
    Input: None (uses MONGO_URI from environment variables) | Output: MongoClient instance or None
    """
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI)
        return client
    except Exception as e:
        st.sidebar.error(f"Mongo connection error: {e}")
        return None


def get_feedback_collection():
    """
    Retrieve the MongoDB collection used to store user feedback.
    Input: None | Output: pymongo Collection object or None
    """
    client = get_mongo_client()
    if client is None:
        return None
    db = client[MONGO_DB_NAME]
    return db[FEEDBACK_COLL]


def log_feedback(note_text: str, prescription_text: str, result: dict, useful: bool, comments: str | None):
    """
    Log user feedback for history–prescription correlation results to MongoDB.
    Input: note_text (str), prescription_text (str), result (dict), useful (bool), comments (str | None)
    Output: None (writes feedback document to MongoDB)
    """
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


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text content from an uploaded PDF file for downstream NLP processing.
    Input: uploaded_file (Streamlit UploadedFile) | Output: Extracted text as a single string
    """
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
