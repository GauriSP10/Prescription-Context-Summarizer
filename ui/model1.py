from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pymongo import MongoClient
import PyPDF2
import os

load_dotenv()

# ----------------- Paths (Streamlit Cloud safe) -----------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(REPO_ROOT / "models")))

DEFAULT_MODEL_PATH = Path(os.getenv("HISTORY_MODEL_PATH", str(MODELS_DIR / "history_classifier.joblib")))
DEFAULT_MLB_PATH = Path(os.getenv("HISTORY_MLB_PATH", str(MODELS_DIR / "history_labels_mlb.joblib")))
DEFAULT_DATA_PATH = Path(os.getenv("HISTORY_DATA_PATH", str(DATA_DIR / "nbme_summarization_dataset.csv")))

# ----------------- Mongo (cached) -----------------
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "patient_data")
FEEDBACK_COLL = os.getenv("MONGO_COLL_FEEDBACK", "history_rx_feedback")


@st.cache_resource
def get_mongo_client():
    if not MONGO_URI:
        return None
    return MongoClient(MONGO_URI)


def get_feedback_collection():
    client = get_mongo_client()
    if client is None:
        return None
    return client[MONGO_DB_NAME][FEEDBACK_COLL]


def log_feedback(note_text, prescription_text, result, useful: bool, comments: str | None):
    coll = get_feedback_collection()
    if coll is None:
        return
    coll.insert_one(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "note_text": note_text,
            "prescription_text": prescription_text,
            "history_features": result.get("history_features", []),
            "prescriptions": result.get("prescriptions", []),
            "insights": result.get("insights", []),
            "useful": useful,
            "comments": comments or "",
        }
    )


# ----------------- PDF → TEXT -----------------
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
        if t.strip():
            texts.append(t)
    return "\n\n".join(texts).strip()


# ----------------- UI helpers -----------------
def metric_row(metrics: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro F1", f"{metrics.get('micro_f1', 0):.3f}")
    c2.metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
    c3.metric("Jaccard", f"{metrics.get('jaccard', 0):.3f}")
    c4.metric("Hamming Score", f"{metrics.get('hamming_score', 0):.3f}")


# ----------------- Main entry -----------------
def render_model1_page():
    st.title("Model 1 - TF-IDF + Rx Parser + UMLS Correlation")
    st.caption("Run analysis and evaluate history classifier performance.")
    st.markdown(
        "⚠️ **Disclaimer:** This is a research prototype. Outputs are model-generated and must not be used for real clinical decisions."
    )

    tab_run, tab_eval = st.tabs(["🧠 Run", "📊 Evaluate"])

    # =========================
    # TAB: RUN
    # =========================
    with tab_run:
        st.subheader("Run Correlation Pipeline")

        try:
            from models.correlationPipeline import correlate_history_and_prescription
        except Exception as e:
            st.error(f"Error importing correlation pipeline: {e}")
            st.stop()

        if "history_note" not in st.session_state:
            st.session_state.history_note = (
                "45-year-old female with insomnia and anxiety around work/family responsibilities."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Patient History Note")
            uploaded_pdf = st.file_uploader("Upload history PDF (optional)", type=["pdf"], key="m1_pdf")
            if uploaded_pdf is not None:
                with st.spinner("Extracting text from PDF..."):
                    txt = extract_text_from_pdf(uploaded_pdf)
                if txt:
                    st.session_state.history_note = txt
                    st.success("Loaded extracted text into the editor.")
                else:
                    st.warning("No text extracted (scanned PDFs won’t work with PyPDF2).")

            note_text = st.text_area("History Note", value=st.session_state.history_note, height=260)
            st.session_state.history_note = note_text

        with col2:
            st.markdown("### 💊 Prescription Text")
            default_rx = (
                "Sertraline 50 mg po once daily.\n"
                "Clonazepam 0.5 mg at night prn.\n"
                "Magnesium hydroxide 400mg/5ml suspension PO 30ml bid for the next 5 days."
            )
            prescription_text = st.text_area("Prescription", value=default_rx, height=260)

        st.markdown("---")

        a, b, c = st.columns(3)
        with a:
            threshold = st.slider("History label threshold", 0.05, 0.90, 0.30, 0.05, key="m1_thr")
        with b:
            top_k = st.number_input("Force include top-k labels", 1, 30, 8, key="m1_topk")
        with c:
            run_btn = st.button("Run Analysis", type="primary", key="m1_run")

        if run_btn:
            if not note_text.strip() or not prescription_text.strip():
                st.error("Please provide both history note and prescription text.")
            else:
                with st.spinner("Running pipeline..."):
                    result = correlate_history_and_prescription(
                        note_text=note_text,
                        prescription_text=prescription_text,
                        threshold=float(threshold),
                        top_k=int(top_k),
                    )

                st.success("Done")

                st.markdown("### Predicted History Features")
                feats = result.get("history_features", [])
                st.write(", ".join(feats) if feats else "_No strong features detected._")

                st.markdown("### Parsed Prescription Items")
                rx_list = result.get("prescriptions", [])
                if rx_list:
                    for i, rx in enumerate(rx_list, start=1):
                        label = rx.get("drug") or rx.get("raw") or f"Medication #{i}"
                        with st.expander(f"Medication #{i}: {label}", expanded=True):
                            st.json(rx)
                else:
                    st.write("_No medications parsed._")

                st.markdown("### 🔗 Correlation Insights")
                insights = result.get("insights", [])
                if insights:
                    for ins in insights:
                        st.markdown(f"- {ins}")
                else:
                    st.write("_No insights generated._")

                st.markdown("---")
                st.markdown("### Feedback")
                fb1, fb2 = st.columns([1, 3])
                with fb1:
                    useful_choice = st.radio("Useful?", ["Yes", "No"], index=0, key="m1_fb_useful")
                with fb2:
                    comments = st.text_input("Comments (optional)", key="m1_fb_comments")

                if st.button("Submit Feedback", key="m1_fb_submit"):
                    log_feedback(
                        note_text=note_text,
                        prescription_text=prescription_text,
                        result=result,
                        useful=(useful_choice == "Yes"),
                        comments=comments,
                    )
                    st.toast("Feedback saved ✅", icon="✅")

    # =========================
    # TAB: EVAL
    # =========================
    with tab_eval:
        st.subheader("Evaluation Dashboard (History Classifier)")
        st.caption("Metrics + threshold sweep + label maps.")

        try:
            from models.evalHistoryClassifier import evaluate_history_classifier
        except Exception as e:
            st.error(f"Could not import eval module: {e}")
            st.stop()

        with st.expander("Settings", expanded=True):
            r1, r2, r3 = st.columns(3)
            with r1:
                thr = st.slider("Threshold", 0.05, 0.90, 0.30, 0.05, key="m1_eval_thr")
            with r2:
                do_sweep = st.checkbox("Threshold sweep", value=True, key="m1_eval_sweep")
            with r3:
                sweep_step = st.selectbox("Sweep step", [0.01, 0.02, 0.05], index=2, key="m1_eval_step")

            r4 = st.columns(1)[0]
            heatmap_top = st.number_input("Heatmap: top labels", 10, 60, 30, key="m1_eval_heatmap")

            model_path = st.text_input("Model file", str(DEFAULT_MODEL_PATH), key="m1_eval_modelpath")
            mlb_path = st.text_input("Label binarizer file", str(DEFAULT_MLB_PATH), key="m1_eval_mlbpath")
            data_path = st.text_input("Eval dataset CSV", str(DEFAULT_DATA_PATH), key="m1_eval_datapath")

            run_eval = st.button("Run Evaluation", type="primary", key="m1_eval_run")

        if run_eval:
            with st.spinner("Evaluating..."):
                res = evaluate_history_classifier(
                    data_path=data_path,
                    model_path=model_path,
                    mlb_path=mlb_path,
                    threshold=float(thr),
                    sweep=bool(do_sweep),
                    sweep_step=float(sweep_step),
                    top_labels_for_heatmap=int(heatmap_top),
                )

            st.markdown("### Overall Metrics @ Threshold")
            metric_row(res["metrics_at_threshold"])

            if do_sweep and res.get("threshold_sweep"):
                st.markdown("### Threshold Sweep")
                df_sweep = pd.DataFrame(res["threshold_sweep"]).sort_values("threshold")
                st.dataframe(df_sweep, use_container_width=True)
                st.line_chart(df_sweep.set_index("threshold")[["micro_f1", "jaccard", "hamming_score"]])

            st.markdown("### Per-label Performance")
            df_labels = pd.DataFrame(res["label_stats"])

            search = st.text_input("Search label", value="", key="m1_label_search")
            if search.strip():
                df_show = df_labels[df_labels["label"].str.contains(search.strip(), case=False, na=False)]
            else:
                df_show = df_labels

            st.dataframe(df_show, use_container_width=True)

            st.markdown("### Label Frequency (Top 30)")
            top_freq = df_labels.sort_values("train_freq", ascending=False).head(30)
            st.bar_chart(top_freq.set_index("label")[["train_freq"]])

            st.markdown("### Label Co-occurrence Map")
            cooc = np.array(res["cooc_matrix"])
            hm_labels = res["cooc_labels"]

            fig = plt.figure()
            plt.imshow(cooc)
            plt.title("Label Co-occurrence (counts)")
            plt.xlabel("Label index (top labels)")
            plt.ylabel("Label index (top labels)")
            plt.colorbar()
            st.pyplot(fig, clear_figure=True)

            with st.expander("Heatmap label index legend"):
                for i, name in enumerate(hm_labels):
                    st.write(f"{i}: {name}")
