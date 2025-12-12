import os
import random
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DATA_PATH = os.path.join(DATA_DIR, "nbme_summarization_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "history_classifier.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "history_labels_mlb.joblib")

# Randomly sample one patient note with ground-truth summary from the NBME dataset for model testing.
def load_random_example():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH)
    df["note_text"] = df["note_text"].astype(str)
    df["summary"] = df["summary"].astype(str)
    return df.sample(n=1, random_state=random.randint(0, 99999)).iloc[0]

# Load the trained history classifier pipeline and MultiLabelBinarizer from disk for inference/testing.
def load_model_and_mlb():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")
    if not os.path.exists(MLB_PATH):
        raise FileNotFoundError(f"{MLB_PATH} not found. Train the model first.")
    clf = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    return clf, mlb

# Main execution: load random example, run trained classifier, and compare predicted features against ground truth.
if __name__ == "__main__":
    example = load_random_example()
    note_text = example["note_text"]
    true_summary = example["summary"]

    print("=== NOTE TEXT (truncated) ===")
    print(note_text[:800], "...\n")

    print("=== TRUE SUMMARY (from dataset) ===")
    print(true_summary, "\n")

    clf, mlb = load_model_and_mlb()

    # Predict.
    pred_probs = getattr(clf, "predict_proba", None)
    if pred_probs is not None:
        # If classifier supports predict_proba, we can threshold manually later.
        Y_pred_bin = clf.predict([note_text])
    else:
        # Fall back to direct predict.
        Y_pred_bin = clf.predict([note_text])

    labels_pred = mlb.inverse_transform(Y_pred_bin)[0]  # list of labels
    print("=== PREDICTED FEATURES ===")
    if labels_pred:
        for lbl in labels_pred:
            print(f"- {lbl}")
    else:
        print("(No labels predicted)")
