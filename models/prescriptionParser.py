import re
from dataclasses import dataclass
from typing import List, Optional

# Structured representation of a parsed prescription with extracted medication details (drug name, strength, dosage, route, frequency, duration).
@dataclass
class PrescriptionEntry:
    raw: str
    drug: Optional[str] = None
    strength: Optional[str] = None
    dosage: Optional[str] = None
    form: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None

# Parse prescription using pattern matching.
def parse_prescription_text_ml(text: str) -> List[PrescriptionEntry]:
    if not text or not text.strip():
        return []

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    entries = []

    dose_pattern = r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)/?\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml))?"
    freq_pattern = r"\b(once\s+daily|twice\s+daily|bid|tid|qid|q\d+h|at\s+night|daily|prn)\b"
    route_pattern = r"\b(po|oral(?:ly)?|iv|im|sc|subcut|topical)\b"
    form_pattern = r"\b(tablet|capsule|suspension|solution|injection|cream|syrup)\b"
    duration_pattern = r"for\s+(?:the\s+)?(?:next\s+)?(\d+)\s+(day|week|month)s?"

    for line in lines:
        entry = PrescriptionEntry(raw=line)

        dose_match = re.search(dose_pattern, line, re.IGNORECASE)
        if dose_match:
            if dose_match.group(3):
                entry.strength = f"{dose_match.group(1)} {dose_match.group(2)}/{dose_match.group(3)}"
            else:
                entry.strength = f"{dose_match.group(1)} {dose_match.group(2)}"

            before_dose = line[:dose_match.start()].strip()
            if before_dose:
                words = before_dose.split()
                entry.drug = " ".join(words[-2:]) if len(words) >= 2 else words[-1]
        else:
            words = line.split()
            if words:
                entry.drug = words[0]

        freq_match = re.search(freq_pattern, line, re.IGNORECASE)
        if freq_match:
            entry.frequency = freq_match.group(1).lower()

        route_match = re.search(route_pattern, line, re.IGNORECASE)
        if route_match:
            entry.route = route_match.group(1).lower()

        form_match = re.search(form_pattern, line, re.IGNORECASE)
        if form_match:
            entry.form = form_match.group(1).lower()

        duration_match = re.search(duration_pattern, line, re.IGNORECASE)
        if duration_match:
            entry.duration = f"for {duration_match.group(1)} {duration_match.group(2)}s"

        entries.append(entry)

    return entries if entries else [PrescriptionEntry(raw=text.strip())]