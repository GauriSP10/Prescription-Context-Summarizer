import os
import joblib
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "history_classifier.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "history_labels_mlb.joblib")

# Load the trained history classification model and label binarizer from disk.
def load_history_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"History model not found at {MODEL_PATH}")
    if not os.path.exists(MLB_PATH):
        raise FileNotFoundError(f"Label binarizer not found at {MLB_PATH}")
    clf = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    return clf, mlb

# Given a raw patient note, returns a list of predicted feature labels.
def predict_history_features(
    note_text: str,
    threshold: float = 0.3,
    top_k: int | None = None,
):
    """
    - Uses predict_proba, then thresholds per-label.
    - If top_k is provided, returns at most top_k labels by probability.
    """
    clf, mlb = load_history_model()

    # Get probabilities if available.
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([note_text])[0]  # shape: (n_labels,)
        # Thresholding.
        selected = probs >= threshold

        if top_k is not None:
            # Pick top_k labels by probability.
            top_idx = np.argsort(probs)[::-1][:top_k]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_idx] = True
            selected = np.logical_or(selected, mask)

        Y_pred_bin = selected.astype(int).reshape(1, -1)
    else:
        # Fall back to plain predict.
        Y_pred_bin = clf.predict([note_text])

    labels = mlb.inverse_transform(Y_pred_bin)[0]  # tuple of labels
    return list(labels)

# # Sanity Test.
# if __name__ == "__main__":
#     sample_note = """45-year-old female with trouble falling asleep and feeling very nervous about work and family care."""
#     feats = predict_history_features(sample_note)
#     print(feats)
