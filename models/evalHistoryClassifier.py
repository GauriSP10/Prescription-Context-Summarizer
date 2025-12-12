import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    hamming_loss,
    precision_recall_fscore_support,
)


def _load_dataset_csv(data_path: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if "note_text" not in df.columns or "summary" not in df.columns:
        raise ValueError("Dataset must contain columns: 'note_text' and 'summary'")

    df["note_text"] = df["note_text"].astype(str)
    df["summary"] = df["summary"].astype(str)

    texts = df["note_text"].tolist()
    label_sets = df["summary"].apply(lambda s: [x.strip() for x in s.split(";") if x.strip()]).tolist()
    return texts, label_sets


def _stack_proba(prob_list) -> np.ndarray:
    # If OneVsRest gives list of (n_samples,2) probs per label
    if isinstance(prob_list, list):
        return np.vstack([p[:, 1] for p in prob_list]).T
    return prob_list  # already (n_samples, n_labels)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    jacc = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    hamm = 1 - hamming_loss(y_true, y_pred)  # per-label "accuracy-ish"

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "jaccard": float(jacc),
        "hamming_score": float(hamm),
    }


def _label_cooccurrence(Y: np.ndarray, max_labels: int = 30) -> Tuple[np.ndarray, List[int]]:
    """
    Returns co-occurrence matrix for top-N labels by frequency.
    """
    freq = Y.sum(axis=0)
    top_idx = np.argsort(freq)[::-1][:max_labels]
    Yt = Y[:, top_idx]
    cooc = (Yt.T @ Yt).astype(int)
    return cooc, top_idx.tolist()


def evaluate_history_classifier(
    data_path: str,
    model_path: str,
    mlb_path: str,
    threshold: float = 0.30,
    sweep: bool = True,
    sweep_step: float = 0.05,
    test_size: float = 0.10,
    random_state: int = 42,
    top_labels_for_heatmap: int = 30,
) -> Dict[str, Any]:
    # Load dataset
    texts, label_sets = _load_dataset_csv(data_path)

    # Load artifacts
    clf = joblib.load(model_path)
    mlb = joblib.load(mlb_path)

    Y = mlb.transform(label_sets)

    X_train, X_val, Y_train, Y_val = train_test_split(
        texts, Y, test_size=test_size, random_state=random_state
    )

    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Model must support predict_proba for thresholding + eval.")

    proba_list = clf.predict_proba(X_val)
    Y_proba = _stack_proba(proba_list)
    Y_pred = (Y_proba >= threshold).astype(int)

    metrics_at_threshold = _compute_metrics(Y_val, Y_pred)

    # Threshold sweep
    threshold_sweep = []
    if sweep:
        for t in np.arange(0.05, 0.90, sweep_step):
            pred_t = (Y_proba >= t).astype(int)
            m = _compute_metrics(Y_val, pred_t)
            m["threshold"] = float(round(float(t), 2))
            threshold_sweep.append(m)

    # Per-label stats
    labels = list(mlb.classes_)
    label_counts = Y_train.sum(axis=0)

    prec, rec, f1, sup = precision_recall_fscore_support(
        Y_val, Y_pred, average=None, zero_division=0
    )

    label_stats = []
    for i, name in enumerate(labels):
        label_stats.append(
            {
                "label": name,
                "train_freq": int(label_counts[i]),
                "support_val": int(sup[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }
        )

    label_stats.sort(key=lambda d: d["train_freq"], reverse=True)

    # Co-occurrence heatmap data
    cooc, top_idx = _label_cooccurrence(Y_train, max_labels=top_labels_for_heatmap)
    heatmap_labels = [labels[i] for i in top_idx]

    return {
        "metrics_at_threshold": metrics_at_threshold,
        "threshold_sweep": threshold_sweep,
        "label_stats": label_stats,
        "cooc_matrix": cooc,
        "cooc_labels": heatmap_labels,
    }
