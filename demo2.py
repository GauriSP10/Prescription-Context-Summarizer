"""
NBME Medical Note Summarizer - Demonstration Script

This script demonstrates the complete clinical note summarization pipeline
using two modes:

1. Pre-defined example clinical notes
2. User-provided clinical note input

Usage:
    python demo1.py

Requirements:
    - Conda environment 'nbme_ml' activated
    - All dependencies installed (see requirements.txt)
"""

import sys
import os
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.t5TransformerModel import ClinicalNoteSummarizer
from models.utils import get_statistics, get_example_notes


def print_header(text):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)


def print_section(title, content):
    """Print formatted section with title and content."""
    print(f"\n{title}")
    print("-" * 80)
    print(content)


def get_user_input_note():
    """
    Collect multi-line clinical note input from user.
    User ends input with an empty line.
    """
    print("\n📝 Enter/paste a clinical note below.")
    print("Press ENTER on an empty line to finish.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return " ".join(lines)


def main():
    """
    Main demonstration function.

    Demonstrates:
    1. Pre-defined example note summarization
    2. User-provided note summarization
    """
    print_header("NBME MEDICAL NOTE SUMMARIZER - DEMO")

    print("This demo shows an automated clinical note summarization system")
    print("using a T5 transformer to generate structured SOAP summaries.\n")

    # Select demo mode
    print("Choose demo mode:")
    print("  1 - Use pre-defined example clinical note")
    print("  2 - Provide your own clinical note")

    choice = input("\nEnter choice (1 or 2): ").strip()

    # Load input note
    if choice == "2":
        example_note = get_user_input_note()
        if not example_note.strip():
            print("\n❌ No input provided. Exiting.")
            sys.exit(1)
        note_source = "USER-PROVIDED INPUT"
    else:
        examples = get_example_notes()
        example_note = examples["Chest Pain"]
        note_source = "PRE-DEFINED EXAMPLE (Chest Pain)"

    print_section(f"ORIGINAL CLINICAL NOTE ({note_source})", example_note)

    # Input statistics
    input_words = len(example_note.split())
    input_chars = len(example_note)
    print("\nInput Statistics:")
    print(f"  • Length: {input_chars} characters")
    print(f"  • Words: {input_words}")
    print(f"  • Sentences: {len(example_note.split('.'))}")

    # Initialize summarizer
    print("\n🔧 Initializing T5-base summarizer...")
    print("(This may take 20–30 seconds on first run)")

    summarizer = ClinicalNoteSummarizer(model_name="t5-base")

    model_info = summarizer.get_model_info()
    print(f"\n✓ Model: {model_info['model_name']}")
    print(f"✓ Device: {model_info['device']}")
    print(f"✓ Parameters: {model_info['parameters']:,}")

    # Generate summary
    print("\n⚙️  Generating abstractive summary...")
    start_time = time.time()

    summary = summarizer.summarize(
        example_note,
        max_length=350,
        min_length=120
    )

    processing_time = time.time() - start_time
    print(f"✓ Summary generated in {processing_time:.2f} seconds")

    # Display summary
    print_section("GENERATED SUMMARY (SOAP FORMAT)", summary)

    # Metrics
    stats = get_statistics(example_note, summary)

    print("\n📊 SUMMARY METRICS")
    print("-" * 80)
    print(f"Original Length:      {stats['original_length']} characters")
    print(f"Summary Length:       {stats['summary_length']} characters")
    print(f"Compression Ratio:    {stats['compression_ratio']:.2%}")
    print(f"Reduction:            {stats['reduction_percentage']:.1f}%")
    print(f"Processing Time:      {processing_time:.2f} seconds")

    # Comparison
    print_header("COMPARISON")

    print("\n📝 ORIGINAL (First 200 characters):")
    print(example_note[:200] + "...")

    print("\n✨ SUMMARY (Complete):")
    print(summary)

    # Final message
    print_header("DEMO COMPLETE")

    print("✓ Demonstrated:")
    print("  • Pre-defined and user-provided input support")
    print("  • T5-based abstractive summarization")
    print("  • SOAP-structured output")
    print(f"  • {stats['reduction_percentage']:.0f}% text reduction achieved")

    print("\n🌐 To run the full web interface:")
    print("  streamlit run app.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nPlease ensure:")
        print("  1. Conda environment 'nbme_ml' is activated")
        print("  2. Dependencies are installed: pip install -r requirements.txt")
        print("  3. macOS fix (if needed): export KMP_DUPLICATE_LIB_OK=TRUE")
        sys.exit(1)
