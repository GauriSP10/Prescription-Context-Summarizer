from collections import Counter
from typing import Any, Dict, List
import spacy
from spacy.language import Language
import warnings
import os

_nlp = None
_linker = None


# Register the UMLS linker as a spaCy component.
@Language.factory("umls_linker")
def create_umls_linker(nlp, name):
    """Factory function to create UMLS linker component."""
    # IMPORTANT:
    # scispaCy registers the built-in component factory as "entity_linker".
    # We keep your custom factory name, but instantiate via the supported spaCy add_pipe approach
    # to avoid E002/E966 factory issues.
    try:
        import scispacy  # noqa: F401
        from scispacy.linking import EntityLinker  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"scispacy is not available: {e}")

    # Create and return an EntityLinker instance
    return EntityLinker(
        resolve_abbreviations=True,
        name=name,
        threshold=0.7
    )

# Load spaCy model and add UMLS linker.
def _build_nlp():
    """Load spaCy model from vendor_models directory (bundled with repo)."""
    global _nlp, _linker

    if _nlp is not None:
        return _nlp

    # Define path to vendored model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VENDOR_MODEL_PATH = os.path.join(BASE_DIR, "vendor_models", "en_core_sci_sm")

    # Try loading from vendor_models first
    try:
        if os.path.exists(VENDOR_MODEL_PATH):
            nlp = spacy.load(VENDOR_MODEL_PATH)
            print(f"[INFO] ✅ Loaded en_core_sci_sm from vendor_models/")
        else:
            # Fallback to installed model
            nlp = spacy.load("en_core_sci_sm")
            print("[INFO] Loaded en_core_sci_sm from system")
    except OSError:
        # Final fallback to web model
        nlp = spacy.load("en_core_web_sm")
        print("[INFO] Loaded en_core_web_sm (fallback)")

    # Add sentencizer
    if ("sentencizer" not in nlp.pipe_names) and ("parser" not in nlp.pipe_names):
        nlp.add_pipe("sentencizer", first=True)

    # Add UMLS linker
    try:
        print("[INFO] Initializing UMLS linker...")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            if "entity_linker" not in nlp.pipe_names:
                try:
                    nlp.add_pipe(
                        "entity_linker",
                        config={
                            "linker_name": "umls",
                            "resolve_abbreviations": True,
                            "threshold": 0.7,
                        },
                    )
                    _linker = nlp.get_pipe("entity_linker")
                except Exception:
                    if "umls_linker" not in nlp.pipe_names:
                        nlp.add_pipe("umls_linker")
                    _linker = nlp.get_pipe("umls_linker")
            else:
                _linker = nlp.get_pipe("entity_linker")

        print("[INFO] ✅ UMLS linker successfully initialized!")

    except Exception as e:
        print(f"[WARN] UMLS linker unavailable: {e}")
        print("[INFO] Continuing with basic biomedical NER only")
        _linker = None

    _nlp = nlp
    return nlp

# Extract UMLS-linked biomedical concepts.
def extract_umls_concepts(text: str) -> Dict[str, Any]:
    """Extract UMLS-linked biomedical concepts, filtering out generic ENTITY labels."""
    if not text or not text.strip():
        return {"concepts": [], "semantic_type_counts": {}}

    nlp = _build_nlp()
    doc = nlp(text)

    concepts = []
    semantic_types = []

    global _linker

    for ent in doc.ents:
        # Skip generic ENTITY labels - they're not informative
        if ent.label_ == "ENTITY":
            continue

        if _linker and hasattr(ent._, "kb_ents") and ent._.kb_ents:
            try:
                cui, score = ent._.kb_ents[0]
                kb_ent = _linker.kb.cui_to_entity[cui]
                sem_types = list(kb_ent.types) if hasattr(kb_ent, 'types') and kb_ent.types else []

                concepts.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "cui": cui,
                    "score": float(score),
                    "canonical_name": kb_ent.canonical_name,
                    "semantic_types": sem_types,
                })

                semantic_types.extend(sem_types)

            except (KeyError, AttributeError, IndexError):
                pass
        else:
            # Only add if it has a meaningful label
            if ent.label_ and ent.label_ != "ENTITY":
                concepts.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "cui": None,
                    "score": 1.0,
                    "canonical_name": ent.text,
                    "semantic_types": [ent.label_],
                })
                semantic_types.append(ent.label_)

    return {
        "concepts": concepts,
        "semantic_type_counts": dict(Counter(semantic_types)),
    }


# Complete UMLS Semantic Type labels.
UMLS_SEMANTIC_TYPE_LABELS = {
    # Organisms.
    "T001": "Organism",
    "T002": "Plant",
    "T004": "Fungus",
    "T005": "Virus",
    "T007": "Bacterium",

    # Anatomical Structures.
    "T017": "Anatomical Structure",
    "T021": "Fully Formed Anatomical Structure",
    "T022": "Body System",
    "T023": "Body Part, Organ, or Organ Component",
    "T024": "Tissue",
    "T025": "Cell",
    "T026": "Cell Component",
    "T029": "Body Location or Region",
    "T030": "Body Space or Junction",

    # Biological Function.
    "T032": "Organism Function",
    "T033": "Finding (Clinical findings)",
    "T034": "Laboratory or Test Result",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T039": "Physiologic Function",
    "T040": "Organism Function",
    "T041": "Mental Process",
    "T042": "Organ or Tissue Function",
    "T043": "Cell Function",
    "T044": "Molecular Function",
    "T045": "Genetic Function",

    # Chemicals & Drugs.
    "T103": "Chemical",
    "T109": "Organic Chemical",
    "T110": "Steroid",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T116": "Amino Acid, Peptide, or Protein",
    "T121": "Pharmacologic Substance (Drug)",
    "T122": "Biomedical or Dental Material",
    "T123": "Biologically Active Substance",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T127": "Vitamin",
    "T129": "Immunologic Factor",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T131": "Hazardous or Poisonous Substance",
    "T195": "Antibiotic",
    "T196": "Element, Ion, or Isotope",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",

    # Disorders.
    "T019": "Congenital Abnormality",
    "T020": "Acquired Abnormality",
    "T046": "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T049": "Cell or Molecular Dysfunction",
    "T050": "Experimental Model of Disease",
    "T184": "Sign or Symptom",
    "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process",

    # Procedures.
    "T058": "Health Care Activity",
    "T059": "Laboratory Procedure",
    "T060": "Diagnostic Procedure",
    "T061": "Therapeutic or Preventive Procedure",

    # Devices.
    "T074": "Medical Device",
    "T075": "Research Device",

    # Objects.
    "T071": "Entity",
    "T072": "Physical Object",
    "T073": "Manufactured Object",

    # Concepts & Ideas.
    "T077": "Conceptual Entity",
    "T078": "Idea or Concept",
    "T079": "Temporal Concept (Time-related)",
    "T080": "Qualitative Concept",
    "T081": "Quantitative Concept",
    "T082": "Spatial Concept",
    "T089": "Regulation or Law",
    "T090": "Occupation or Discipline",
    "T091": "Biomedical Occupation or Discipline",

    # Activities & Behaviors.
    "T052": "Activity",
    "T053": "Behavior",
    "T054": "Social Behavior",
    "T055": "Individual Behavior",
    "T056": "Daily or Recreational Activity",
    "T057": "Occupational Activity",

    # Organizations & Groups.
    "T092": "Organization",
    "T093": "Health Care Related Organization",
    "T094": "Professional Society",
    "T095": "Self-help or Relief Organization",
    "T096": "Group",
    "T097": "Professional or Occupational Group",
    "T098": "Population Group",
    "T099": "Family Group",
    "T100": "Age Group",
    "T101": "Patient or Disabled Group",

    # Geographic Areas.
    "T083": "Geographic Area",
}

# Convert UMLS semantic types to readable summary.
def summarize_semantic_types(semantic_type_counts: Dict[str, int]) -> str:
    """Convert UMLS semantic types to readable summary."""
    # Filter out ENTITY labels
    filtered_counts = {k: v for k, v in semantic_type_counts.items() if k != "ENTITY"}

    if not filtered_counts:
        return "containing general medical terminology without strongly typed biomedical concepts"

    sorted_types = sorted(
        filtered_counts.items(), key=lambda x: x[1], reverse=True
    )[:3]

    parts = []
    for sem_type, count in sorted_types:
        label = UMLS_SEMANTIC_TYPE_LABELS.get(sem_type, sem_type)
        parts.append(f"{label} ({count})")

    if len(parts) == 1:
        return f"primarily about {parts[0]}"
    elif len(parts) == 2:
        return f"largely about {parts[0]} and {parts[1]}"
    else:
        return f"largely about {parts[0]}, {parts[1]}, and {parts[2]}"