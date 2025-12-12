import streamlit as st
from ui.utils import extract_text_from_pdf, log_feedback

# Render the main model page with history-prescription correlation.
def render_model1_page():
    st.title("AI Prescription Context Summarizer")

    st.markdown(
        """
1. Predicts **clinical features** from a patient history (NBME-based multi-label classifier).
2. Parses the **prescription** using pattern-based NER.
3. Generates **model-based insights** linking the two.

> ⚠️ **Disclaimer:** This is a research prototype. Outputs are model-generated and **must not** be used for real clinical decisions.
"""
    )

    # Import correlation pipeline.
    try:
        from models.correlationPipeline import correlate_history_and_prescription
    except Exception as e:
        st.error(f"Error importing correlation pipeline: {e}")
        st.stop()

    # Initialize session state for history note.
    if "history_note" not in st.session_state:
        st.session_state.history_note = """Ms. Moore is a 45 year old female who presents to clinic complaining of increased nervousness.
She has been more nervous for the past two weeks, especially on Sunday night and Monday morning as the work week approaches.
She worries about caring for her children, in-laws, and elderly mother, and has trouble falling asleep.
She denies loss of interest, loss of energy, hopelessness, or slowed movements. Denies depressed mood."""

    # Input layout.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Patient History Note")

        uploaded_pdf = st.file_uploader(
            "Upload history as PDF (optional):",
            type=["pdf"],
            key="history_pdf_uploader",
        )

        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_pdf)
            if pdf_text:
                st.session_state.history_note = pdf_text
                st.success("Extracted text from PDF and loaded into the note box below.")
            else:
                st.warning("Could not extract any text from this PDF (it might be a scanned/image-only file).")

        note_text = st.text_area(
            "Paste, type, or edit the history note here:",
            value=st.session_state.history_note,
            key="history_note_textarea",
            height=260,
        )
        st.session_state.history_note = note_text

    with col2:
        st.subheader("💊 Prescription Text")
        default_rx = """Sertraline 50 mg po once daily.
Clonazepam 0.5 mg at night prn.
Magnesium hydroxide 400mg/5ml suspension PO 30ml bid for the next 5 days."""
        prescription_text = st.text_area(
            "Paste or type the prescription here:",
            value=default_rx,
            height=260,
        )

    st.markdown("---")

    # Controls.
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.3, 0.05)
    with col_b:
        top_k = st.number_input(
            "Max labels to force-include (top_k)",
            min_value=1,
            max_value=30,
            value=8,
            step=1
        )
    with col_c:
        run_button = st.button("🔍 Run Correlation", type="primary", key="run_my_model")

    result = None

    # Run pipeline.
    if run_button:
        if not note_text.strip() or not prescription_text.strip():
            st.error("Please provide both a history note and a prescription.")
        else:
            with st.spinner("Running models..."):
                try:
                    result = correlate_history_and_prescription(
                        note_text=note_text,
                        prescription_text=prescription_text,
                        threshold=threshold,
                        top_k=top_k,
                    )
                except Exception as e:
                    st.error(f"Error running correlation pipeline: {e}")
                    st.stop()

            st.success("✅ Done!")

            # Display results.
            _display_results(result)

            # Feedback section.
            _display_feedback_section(note_text, prescription_text, result)

# Display the correlation results.
def _display_results(result: dict):

    # Predicted history features.
    st.subheader("🧩 Predicted History Features")
    feats = result.get("history_features", [])
    if feats:
        st.write(", ".join(feats))
    else:
        st.write("_No strong features detected._")

    # Parsed prescriptions.
    st.subheader("💊 Parsed Prescription Items")
    rx_list = result.get("prescriptions", [])
    if rx_list:
        for i, rx in enumerate(rx_list, start=1):
            label = rx.get("drug") or rx.get("raw") or f"Medication #{i}"
            with st.expander(f"Medication #{i}: {label}", expanded=True):
                st.json(rx)
    else:
        st.write("_No medications parsed from the prescription text._")

    # Insights
    st.subheader("🔗 Correlation Insights")
    insights = result.get("insights", [])
    if insights:
        for ins in insights:
            st.markdown(f"- {ins}")
    else:
        st.write("_No insights generated._")

# Display the feedback collection section.
def _display_feedback_section(note_text: str, prescription_text: str, result: dict):

    st.markdown("---")
    st.subheader("🧪 Feedback (for future improvement)")

    fb_col1, fb_col2 = st.columns([1, 3])

    with fb_col1:
        useful_choice = st.radio(
            "Was this output useful?",
            ["Yes", "No"],
            index=0,
            key="fb_useful_my"
        )

    with fb_col2:
        comments = st.text_input("Any comments (optional):", key="fb_comments_my")

    if st.button("Submit feedback", key="fb_submit_my"):
        log_feedback(
            note_text=note_text,
            prescription_text=prescription_text,
            result=result,
            useful=(useful_choice == "Yes"),
            comments=comments,
        )
        st.toast("Feedback recorded. Thank you!", icon="✅")

    st.markdown("---")
    st.caption("⚠️ This is a research prototype. Do not use for diagnosis or treatment.")