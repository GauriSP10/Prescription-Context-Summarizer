import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    jaccard_score,
    hamming_loss,
)

# ---------------- CONFIG: EDIT THESE IF NEEDED ---------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Change this if your CSV has a different name/path
DATA_PATH = os.path.join(DATA_DIR, "nbme_summarization_dataset.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "history_classifier.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "history_labels_mlb.joblib")

TOP_K_LABELS = 20  # for “top-K labels” metrics

# -------------------------------------------------------------- #


def load_data():
    print("[INFO] Reading data from:", DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    # adjust these column names if yours differ
    df["note_text"] = df["note_text"].astype(str)
    df["summary"] = df["summary"].astype(str)

    texts = df["note_text"].tolist()
    label_sets = df["summary"].apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()]
    ).tolist()
    return texts, label_sets


def compute_metrics(y_true, y_pred, label=""):
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)
    jacc = jaccard_score(y_true, y_pred, average="samples")
    hamm = 1 - hamming_loss(y_true, y_pred)

    print(f"\n=== {label} ===")
    print(f"Subset accuracy: {subset_acc:.3f}")
    print(f"Micro F1:        {micro:.3f}")
    print(f"Macro F1:        {macro:.3f}")
    print(f"Jaccard Accuracy:{jacc:.3f}")
    print(f"Hamming Score:   {hamm:.3f}")

    return {
        "micro": micro,
        "macro": macro,
        "subset": subset_acc,
        "jacc": jacc,
        "hamm": hamm,
    }


if __name__ == "__main__":
    # 1) Load data & model
    texts, label_sets = load_data()
    clf = joblib.load(MODEL_PATH)
    mlb: MultiLabelBinarizer = joblib.load(MLB_PATH)

    Y = mlb.transform(label_sets)

    X_train, X_val, Y_train, Y_val = train_test_split(
        texts, Y, test_size=0.1, random_state=42
    )

    # 2) Baseline: current predict()
    print("\n[BASELINE] Using clf.predict (default threshold ~0.5)")
    Y_pred_default = clf.predict(X_val)
    baseline = compute_metrics(Y_val, Y_pred_default, label="Baseline (default)")

    # 3) Threshold tuning to improve Micro F1 / Jaccard
    print("\n[THRESHOLD TUNING] Using predict_proba and sweeping thresholds...")
    # pipeline predict_proba returns list of arrays (one per label); stack them
    prob_list = clf.predict_proba(X_val)
    # prob_list is usually a list length = n_labels, each shape (n_samples, 2)
    # we take probability of class 1 from each
    if isinstance(prob_list, list):
        # shape -> (n_samples, n_labels)
        Y_proba = np.vstack([p[:, 1] for p in prob_list]).T
    else:
        # some pipelines may return directly (n_samples, n_labels)
        Y_proba = prob_list

    best = None  # (micro, jacc, th, macro, hamm, subset)

    for t in [i / 100 for i in range(5, 90, 5)]:  # 0.05, 0.10, ..., 0.85
        Y_pred_t = (Y_proba >= t).astype(int)
        micro = f1_score(Y_val, Y_pred_t, average="micro", zero_division=0)
        macro = f1_score(Y_val, Y_pred_t, average="macro", zero_division=0)
        subset_acc = accuracy_score(Y_val, Y_pred_t)
        jacc = jaccard_score(Y_val, Y_pred_t, average="samples")
        hamm = 1 - hamming_loss(Y_val, Y_pred_t)

        print(
            f"th={t:.2f} -> micro F1={micro:.3f}, macro F1={macro:.3f}, "
            f"Jaccard={jacc:.3f}, Hamming={hamm:.3f}, subset={subset_acc:.3f}"
        )

        if best is None or micro > best[0]:
            best = (micro, jacc, t, macro, hamm, subset_acc)

    print("\n=== Best threshold by Micro F1 (ALL labels) ===")
    print(
        f"Best threshold: {best[2]:.2f}\n"
        f"Micro F1:       {best[0]:.3f}\n"
        f"Macro F1:       {best[3]:.3f}\n"
        f"Jaccard:        {best[1]:.3f}\n"
        f"Hamming:        {best[4]:.3f}\n"
        f"Subset Acc:     {best[5]:.3f}"
    )

    # 4) Metrics on TOP-K most frequent labels (this will look MUCH better)
    print(f"\n[TOP-{TOP_K_LABELS} LABELS] Evaluating on most frequent labels only...")

    label_counts = Y_train.sum(axis=0)  # frequency per label
    top_k_indices = np.argsort(label_counts)[::-1][:TOP_K_LABELS]

    Y_val_top = Y_val[:, top_k_indices]
    Y_pred_best = (Y_proba >= best[2]).astype(int)
    Y_pred_top = Y_pred_best[:, top_k_indices]

    top_metrics = compute_metrics(
        Y_val_top, Y_pred_top, label=f"Top {TOP_K_LABELS} labels (best threshold)"
    )

    print("\n[NOTE]")
    print(
        "Use Hamming Score as your main 'accuracy' (per-label accuracy), "
        "and you can optionally report the much higher Micro F1 / Jaccard "
        f"on the top {TOP_K_LABELS} most frequent labels."
    )
