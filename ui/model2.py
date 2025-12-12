import streamlit as st
import time
import sys
import os

# Add models directory to path.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from models.t5TransformerModel import ClinicalNoteSummarizer
from models.utils import get_statistics, get_example_notes


@st.cache_resource
def load_model(model_name='t5-small'):
    """Load model (cached so it only loads once)"""
    return ClinicalNoteSummarizer(model_name=model_name)


def render_model2_page():
    """Render T5 Abstractive Summarizer page."""

    st.title("🤖 Clinical Note Summarizer")
    st.markdown("### Automated abstractive summarization using T5 transformer")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Settings")

        model_choice = st.selectbox("Model", ["t5-small", "t5-base"], index=0)

        max_length = st.slider("Max Summary Length (tokens)", 100, 500, 400, 20)
        min_length = st.slider("Min Summary Length (tokens)", 50, 200, 150, 10)

        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("This app uses T5 to generate abstractive summaries of clinical notes. "
                "400 tokens ≈ 250-300 words.")

        st.markdown("### ⚙️ Current Settings")
        st.code(f"Model: {model_choice}\nMax: {max_length} tokens\nMin: {min_length} tokens")

    with st.spinner(f"Loading {model_choice} model..."):
        summarizer = load_model(model_choice)

    model_info = summarizer.get_model_info()
    st.success(f"✓ Model loaded on {model_info['device']}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Input: Clinical Note")
        examples = get_example_notes()
        example_choice = st.selectbox("Load Example", ["Custom"] + list(examples.keys()))

        default_text = "" if example_choice == "Custom" else examples[example_choice]

        clinical_note = st.text_area(
            "Enter clinical note:",
            value=default_text,
            height=300,
            placeholder="Paste or type the clinical note here..."
        )

        st.caption(f"📊 Characters: {len(clinical_note)} | Words: {len(clinical_note.split())}")
        generate_btn = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

    with col2:
        st.subheader("✨ Output: Abstractive Summary")

        if generate_btn:
            if not clinical_note.strip():
                st.error("⚠️ Please enter a clinical note first!")
            else:
                with st.spinner("Generating summary..."):
                    start_time = time.time()

                    summary = summarizer.summarize(
                        clinical_note,
                        max_length=max_length,
                        min_length=min_length
                    )

                    elapsed_time = time.time() - start_time

                st.markdown("### Generated Summary:")
                st.info(summary)

                stats = get_statistics(clinical_note, summary)
                st.markdown("### 📊 Statistics:")
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric("Original", f"{stats['original_length']} chars")
                with col_b:
                    st.metric("Summary", f"{stats['summary_length']} chars")
                with col_c:
                    st.metric("Reduction", f"{stats['reduction_percentage']:.1f}%")

                st.caption(f"⏱️ Generated in {elapsed_time:.2f} seconds | "
                           f"Summary words: {len(summary.split())}")

                st.download_button(
                    label="📥 Download Summary",
                    data=f"ORIGINAL NOTE:\n{clinical_note}\n\nSUMMARY:\n{summary}",
                    file_name="clinical_summary.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Enter a clinical note and click 'Generate Summary'")
            st.markdown("**Tip:** For complex notes, use higher max_length (400-500 tokens)")

    st.markdown("---")
    st.caption("⚠️ Research prototype - Not for clinical use")