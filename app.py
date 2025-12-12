import subprocess
import sys

# Install scispacy at runtime (avoids compilation issues)
try:
    import scispacy
except:
    subprocess.run([sys.executable, "-m", "pip", "install", "scispacy"], check=False)

# Install sci model at runtime
try:
    import spacy
    spacy.load("en_core_sci_sm")
except:
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
    ], check=False)


import os
import sys
import streamlit as st

# Configure Streamlit page.
st.set_page_config(
    page_title="AI Prescription Context Summarizer – Multi-Model App",
    layout="wide",
)

# Add models directory to path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

# Import page modules.
from ui.model1 import render_model1_page
from ui.model2 import render_model2_page


# Sidebar navigation.
st.sidebar.title("☰ Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "TF-IDF + UMLS Correlation Summarizer",
        "Abstractive Correlation Summarizer",
    ],
)

# Render selected page.
if page == "TF-IDF + UMLS Correlation Summarizer":
    render_model1_page()
else:
    render_model2_page()