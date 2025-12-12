import subprocess
import sys
import os


# Install all models at runtime (works on Streamlit Cloud)
def setup_dependencies():
    """Install spaCy models and scispacy at runtime."""
    try:
        # Install spacy models using spacy's built-in downloader
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                       capture_output=True, check=False)

        # Install scispacy
        subprocess.run([sys.executable, "-m", "pip", "install", "scispacy"],
                       capture_output=True, check=False)

        # Install sci model
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
        ], capture_output=True, check=False)

        print("[INFO] ✅ Dependencies installed")
    except Exception as e:
        print(f"[WARN] Dependency setup: {e}")


# Run setup on first import
setup_dependencies()
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