import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "nbme_summarization_dataset.csv")

# Load preprocessed NBME summarization dataset from CSV and ensure text columns are properly typed as strings.
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Did you run preprocessing?")
    df = pd.read_csv(DATA_PATH)
    # Make sure text fields are strings.
    df["note_text"] = df["note_text"].astype(str)
    df["summary"] = df["summary"].astype(str)
    return df

#     Convert 'summary' column like:
#       'feature1; feature2; feature3'
#     into list-of-labels for MultiLabelBinarizer.
def build_labels(df):
    label_sets = df["summary"].apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()]
    ).tolist()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_sets)
    print(f"[INFO] Number of unique labels (features): {len(mlb.classes_)}")
    return Y, mlb

#  Main training pipeline: loads NBME data, builds multi-label targets, trains TF-IDF+SVD+LogReg classifier,
#     evaluates performance, and saves the trained model artifacts to disk.
if __name__ == "__main__":
    print("[INFO] Loading data...")
    df = load_data()
    X = df["note_text"].tolist()
    Y, mlb = build_labels(df)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )
    print("[INFO] Train size:", len(X_train))
    print("[INFO] Val size:", len(X_val))

    # Build the pipeline:
    #   Text -> TF-IDF -> SVD -> One-vs-Rest Logistic Regression.
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    svd = TruncatedSVD(
        n_components=256,
        random_state=42,
    )
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=200,
            n_jobs=-1,
        )
    )

    text_clf = Pipeline([
        ("tfidf", tfidf),
        ("svd", svd),
        ("clf", clf),
    ])

    print("[INFO] Training classifier...")
    text_clf.fit(X_train, Y_train)

    print("[INFO] Evaluating on validation set...")
    Y_val_pred = text_clf.predict(X_val)
    print(classification_report(Y_val, Y_val_pred, zero_division=0))

    # Save everything: pipeline + label binarizer.
    model_path = os.path.join(MODEL_DIR, "history_classifier.joblib")
    mlb_path = os.path.join(MODEL_DIR, "history_labels_mlb.joblib")

    joblib.dump(text_clf, model_path)
    joblib.dump(mlb, mlb_path)

    print(f"[DONE] Saved model pipeline to: {model_path}")
    print(f"[DONE] Saved label binarizer to: {mlb_path}")
