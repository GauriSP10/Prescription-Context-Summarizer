from dataclasses import asdict
from typing import Dict, Any, List

from historyInference import predict_history_features
from prescriptionParser import parse_prescription_text_ml, PrescriptionEntry
from umlsExtractor import extract_umls_concepts, summarize_semantic_types


def correlate_umls(history_text: str, prescription_text: str) -> Dict[str, Any]:
    """
    Performs UMLS semantic analysis on patient history and prescription texts to identify therapeutic alignment,
    shared medical concepts, and regimen complexity. Returns semantic profiles and clinical insights.
    Input: history_text (str), prescription_text (str) | Output: Dict with UMLS concepts, semantic types, and insights
    """
    try:
        # Extract UMLS concepts and semantic types from both texts.
        hist_umls = extract_umls_concepts(history_text)
        rx_umls = extract_umls_concepts(prescription_text)
    except Exception as e:
        print(f"[WARN] UMLS extraction failed: {e}")
        return {
            "history_umls": {"concepts": [], "semantic_type_counts": {}},
            "prescription_umls": {"concepts": [], "semantic_type_counts": {}},
            "umls_insights": ["UMLS semantic analysis unavailable"],
        }

    # Generate human-readable semantic profile summaries.
    hist_summary = summarize_semantic_types(hist_umls["semantic_type_counts"])
    rx_summary = summarize_semantic_types(rx_umls["semantic_type_counts"])

    insights = []

    # Semantic profile descriptions.
    insights.append(f"**Patient history semantic profile**: {hist_summary}.")
    insights.append(f"**Prescription semantic profile**: {rx_summary}.")

    # Analyze semantic type alignment between history and prescription.
    hist_types = set(hist_umls["semantic_type_counts"].keys())
    rx_types = set(rx_umls["semantic_type_counts"].keys())

    # Check for treatment-symptom alignment using UMLS semantic types.
    has_symptoms = any(
        t in hist_types for t in ["T184", "T048", "T033"])  # Signs/Symptoms, Mental disorders, Clinical findings
    has_drugs = any(t in rx_types for t in ["T121", "T200"])  # Pharmacologic substances, Clinical drugs

    if has_symptoms and has_drugs:
        insights.append(
            "**Therapeutic alignment**: The prescription contains pharmacologic interventions "
            "(UMLS types T121/T200) targeting documented symptoms and clinical findings "
            "(T184/T048/T033) in the patient history, suggesting appropriate symptom-treatment matching."
        )

    # Analyze concept-level overlap (exact UMLS CUI matches).
    hist_cuis = {c["cui"] for c in hist_umls["concepts"] if c.get("cui")}
    rx_cuis = {c["cui"] for c in rx_umls["concepts"] if c.get("cui")}
    shared_cuis = hist_cuis & rx_cuis

    if shared_cuis:
        insights.append(
            f"**Shared UMLS concepts**: {len(shared_cuis)} concept(s) appear in both "
            "history and prescription contexts, indicating direct semantic linkage between "
            "documented clinical presentation and prescribed treatment."
        )
    else:
        insights.append(
            "**Semantic divergence**: No identical UMLS concepts found in both texts. "
            "This is typical when patient-reported symptoms (e.g., 'nervousness') are addressed "
            "with standardized pharmaceutical names (e.g., 'Sertraline') that target the underlying condition."
        )

    # Analyze temporal complexity (medication schedule burden)
    rx_temporal = rx_umls["semantic_type_counts"].get("T079", 0)
    if rx_temporal >= 3:
        insights.append(
            f"**Regimen complexity**: The prescription contains {rx_temporal} temporal concepts "
            "(dosing schedules, durations), indicating a multi-frequency regimen that requires "
            "patient adherence monitoring."
        )

    return {
        "history_umls": hist_umls,
        "prescription_umls": rx_umls,
        "umls_insights": insights,
    }


def correlate_history_and_prescription(
        note_text: str,
        prescription_text: str,
        threshold: float = 0.3,
        top_k: int | None = None,
) -> Dict[str, Any]:
    """
    Main correlation pipeline combining NBME classifier, prescription parser, and UMLS semantic analysis.
    Input: note_text (str), prescription_text (str), threshold (float), top_k (int|None)
    Output: Dict with history_features (list), prescriptions (list), UMLS data (dict), and insights (list)
    """

    # 1. Extract structured clinical features using trained NBME classifier.
    history_features: List[str] = predict_history_features(
        note_text,
        threshold=threshold,
        top_k=top_k,
    )

    # 2. Parse prescription text into structured medication entries.
    rx_entries: List[PrescriptionEntry] = parse_prescription_text_ml(prescription_text)

    # 3. Perform UMLS-based semantic correlation analysis.
    umls_corr = correlate_umls(note_text, prescription_text)

    # 4. Combine structured features with semantic insights.
    all_insights: List[str] = []

    # Display NBME-derived structured features.
    if history_features:
        all_insights.append(
            "**Structured history features (NBME classifier)**: " + ", ".join(history_features) + "."
        )
    else:
        all_insights.append(
            "**Note**: No strong structured features detected at the current confidence threshold."
        )

    # Append UMLS semantic insights.
    all_insights.extend(umls_corr["umls_insights"])

    return {
        "history_features": history_features,
        "prescriptions": [asdict(e) for e in rx_entries],
        "history_umls": umls_corr["history_umls"],
        "prescription_umls": umls_corr["prescription_umls"],
        "insights": all_insights,
    }