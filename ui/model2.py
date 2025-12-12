# ui/page_teammate.py

import streamlit as st


def render_model2_page():
    """Render the teammate's model page (placeholder)."""

    st.title("👥 Abstractive Correlation Summarizer")

    st.markdown(
        """
This page is reserved for your **teammate's model**.

They can plug in:

- A different kind of model (e.g., risk scoring, clustering, another NLP task),
- Different input modalities (e.g., lab values, vitals),
- Or their own Streamlit layout.

Below is a placeholder skeleton they can modify.
"""
    )

    st.markdown("---")

    st.subheader("🔍 Example: Teammate Model Demo")

    input_text = st.text_area(
        "Teammate: put your input here (e.g., free text, JSON, CSV snippet):",
        height=200,
        key="teammate_input",
    )

    col1, col2 = st.columns(2)

    with col1:
        option = st.selectbox(
            "Teammate: model variant / mode",
            ["Mode A", "Mode B", "Mode C"],
            key="teammate_mode",
        )

    with col2:
        run_teammate = st.button("Run teammate model", type="primary", key="run_teammate")

    if run_teammate:
        if not input_text.strip():
            st.error("Please provide some input text.")
        else:
            st.info(f"[Placeholder] Teammate model would run in **{option}** on the provided input.")
            st.write("Raw input:")
            st.code(input_text)
            st.markdown(">🧪 This is a placeholder. Your teammate can replace this with real model logic.")