"""
Demo runner for Model 1: TF-IDF + UMLS Correlation Summarizer

Two ways to run:
1) User-provided input (interactive mode)
2) Pre-defined inputs (predefined mode)

Examples:
  python demo1.py --mode interactive
  python demo1.py --mode predefined
  python demo1.py --mode predefined --example 2
  python demo1.py --mode interactive --threshold 0.30 --top_k 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple, Dict, Any


def _try_import_pipeline():
    """
    Load correlation pipeline entrypoint for demo execution across different repo layouts.
    Input: None | Output: correlate_history_and_prescription (callable) or raises ImportError
    """
    """
    Robust import so demo1.py can be run from project root.
    Tries:
      - direct import (if demo1.py sits alongside correlationPipeline.py)
      - import from models/ (common structure in the repo)
    """
    # 1) Direct import
    try:
        from correlationPipeline import correlate_history_and_prescription  # type: ignore
        return correlate_history_and_prescription
    except Exception:
        pass

    # 2) Try from models/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    if os.path.isdir(models_dir) and models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    try:
        from correlationPipeline import correlate_history_and_prescription  # type: ignore
        return correlate_history_and_prescription
    except Exception as e:
        raise ImportError(
            "Could not import correlate_history_and_prescription from correlationPipeline.py.\n"
            "Make sure demo1.py is in the project root and correlationPipeline.py is either:\n"
            "  - in the same directory, or\n"
            "  - inside ./models/\n\n"
            f"Original import error: {e}"
        )


def _predefined_examples() -> Dict[int, Tuple[str, str]]:
    """
    Provide a small library of predefined (history_note, prescription_text) examples for consistent demos.
    Input: None | Output: Dict[int, Tuple[str, str]] mapping example_id -> (history_text, prescription_text)
    """
    """
    Pre-defined demo inputs (history_note, prescription_text).
    Add/edit examples freely.
    """
    ex1_history = (
        "25F presents with intermittent headaches and nausea for 2 weeks. "
        "Reports stress, poor sleep, and occasional palpitations. Denies fever. "
        "No focal neurological deficits. No known drug allergies."
    )
    ex1_rx = (
        "Propranolol 10 mg PO twice daily\n"
        "Ondansetron 4 mg tablet PO prn\n"
        "Ibuprofen 400 mg PO q6h prn for 5 days"
    )

    ex2_history = (
        "62M with history of type 2 diabetes and hypertension presents with "
        "chest tightness on exertion and shortness of breath. Smokes 1 ppd. "
        "BP elevated today. No syncope."
    )
    ex2_rx = (
        "Aspirin 81 mg tablet PO daily\n"
        "Atorvastatin 40 mg PO at night\n"
        "Nitroglycerin 0.4 mg SL prn chest pain"
    )

    ex3_history = (
        "34F reports dysuria and urinary frequency for 3 days, mild suprapubic discomfort. "
        "No flank pain, no vomiting, afebrile."
    )
    ex3_rx = (
        "Nitrofurantoin 100 mg capsule PO twice daily for 5 days\n"
        "Phenazopyridine 200 mg PO tid prn for 2 days"
    )

    return {
        1: (ex1_history, ex1_rx),
        2: (ex2_history, ex2_rx),
        3: (ex3_history, ex3_rx),
    }


def _prompt_multiline(prompt: str) -> str:
    """
    Read multi-line text input from the user until a blank line is entered (CLI interactive mode).
    Input: prompt (str) | Output: Combined multi-line string (str)
    """
    """
    Read multi-line user input until an empty line is entered.
    """
    print(prompt)
    print("(Enter an empty line to finish.)")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _run_demo(
    correlate_fn,
    history_text: str,
    prescription_text: str,
    threshold: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    """
    Execute the end-to-end correlation pipeline on a single (history, prescription) pair.
    Input: correlate_fn (callable), history_text (str), prescription_text (str), threshold (float), top_k (Optional[int])
    Output: result (Dict[str, Any]) containing history_features, parsed prescriptions, UMLS fields, and insights
    """
    """
    Calls your pipeline and returns the result dict.
    """
    return correlate_fn(
        note_text=history_text,
        prescription_text=prescription_text,
        threshold=threshold,
        top_k=top_k,
    )


def main():
    """
    Parse CLI arguments, load pipeline, run demo (interactive or predefined), and print JSON output.
    Input: None (reads CLI args) | Output: None (prints demo output; exits with nonzero code on failure)
    """
    parser = argparse.ArgumentParser(description="Demo runner for Model 1 correlation pipeline.")
    parser.add_argument(
        "--mode",
        choices=["interactive", "predefined"],
        default="predefined",
        help="Run demo in interactive mode or using predefined examples.",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=1,
        help="Predefined example number (only used when --mode predefined).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Probability threshold for multi-label history classifier.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional: restrict to top-k predicted labels.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    correlate_history_and_prescription = _try_import_pipeline()

    if args.mode == "interactive":
        history_text = _prompt_multiline("Paste / type the PATIENT HISTORY NOTE text:")
        prescription_text = _prompt_multiline("Paste / type the PRESCRIPTION text (each med on new line if possible):")

        if not history_text or not prescription_text:
            print("\n[ERROR] Both history note and prescription text are required.")
            sys.exit(1)

    else:
        examples = _predefined_examples()
        if args.example not in examples:
            print(f"\n[ERROR] Example {args.example} not found. Available: {sorted(examples.keys())}")
            sys.exit(1)

        history_text, prescription_text = examples[args.example]
        print(f"\n[INFO] Running predefined example #{args.example}")
        print("\n--- HISTORY NOTE ---")
        print(history_text)
        print("\n--- PRESCRIPTION ---")
        print(prescription_text)

    # Run pipeline
    try:
        result = _run_demo(
            correlate_fn=correlate_history_and_prescription,
            history_text=history_text,
            prescription_text=prescription_text,
            threshold=float(args.threshold),
            top_k=args.top_k,
        )
    except Exception as e:
        print("\n[ERROR] Demo failed while running pipeline.")
        print(f"Reason: {e}")
        sys.exit(2)

    # Display output
    print("\n\n==================== DEMO OUTPUT ====================")
    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    print("=====================================================\n")


if __name__ == "__main__":
    main()
