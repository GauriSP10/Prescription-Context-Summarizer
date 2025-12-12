# import os
# import sys
# import streamlit as st
#
# # Configure Streamlit page.
# st.set_page_config(
#     page_title="AI Prescription Context Summarizer – Multi-Model App",
#     layout="wide",
# )
#
# # Add models directory to path.
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
#
# if MODELS_DIR not in sys.path:
#     sys.path.append(MODELS_DIR)
#
# # Import page modules.
# from ui.model1 import render_model1_page
# from ui.model2 import render_model2_page
#
#
# # Sidebar navigation.
# st.sidebar.title("☰ Navigation")
# page = st.sidebar.radio(
#     "Go to:",
#     [
#         "TF-IDF + UMLS Correlation Summarizer",
#         "Abstractive Correlation Summarizer",
#     ],
# )
#
# # Render selected page.
# if page == "TF-IDF + UMLS Correlation Summarizer":
#     render_model1_page()
# else:
#     render_model2_page()

import os
import sys
import streamlit as st

st.set_page_config(
    page_title="AI Prescription Context Summarizer – Multi-Model App",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UI_DIR = os.path.join(BASE_DIR, "ui")

for p in [MODELS_DIR, UI_DIR, BASE_DIR]:
    if p not in sys.path:
        sys.path.append(p)

# Import page modules.
from ui.model1 import render_model1_page
from ui.model2 import render_model2_page

# Sidebar navigation.
st.sidebar.title("☰ Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Model 1 — TF-IDF + UMLS Correlation Summarizer",
        "Model 2 — Abstractive Correlation Summarizer",
    ],
)

if page.startswith("Model 1"):
    render_model1_page()
else:
    render_model2_page()
