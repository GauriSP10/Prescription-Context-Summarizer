import os
import pandas as pd

from config import (
    get_db,
    DATA_DIR,
    MONGO_COLL_TRAIN,
    MONGO_COLL_FEATURES,
    MONGO_COLL_PATIENT_NOTES,
    MONGO_COLL_NBME_NOTES,
    MONGO_COLL_NBME_FEATURES,
)

# Load all documents from a MongoDB collection into a pandas DataFrame, excluding the _id field.
def load_collection_as_df(db, collection_name: str) -> pd.DataFrame:
    coll = db[collection_name]
    docs = list(coll.find({}))
    if not docs:
        print(f"[WARN] Collection '{collection_name}' is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    print(f"[INFO] Loaded {len(df)} docs from '{collection_name}'")
    return df

# Write a pandas DataFrame to MongoDB collection, optionally clearing existing documents before insertion.
def write_df_to_mongo(df: pd.DataFrame, db, collection_name: str, clear_existing: bool = True):
    coll = db[collection_name]
    if clear_existing:
        coll.delete_many({})
        print(f"[INFO] Cleared '{collection_name}'")

    if df.empty:
        print(f"[WARN] No docs to insert into '{collection_name}' (empty DataFrame)")
        return

    docs = df.to_dict(orient="records")
    coll.insert_many(docs)
    print(f"[INFO] Inserted {len(docs)} docs into '{collection_name}'")

#  Merge NBME tables and construct a summarization dataset by aggregating positive feature annotations
#  per patient note into semicolon-separated summary strings for model training.
def build_summarization_dataset(train_df, features_df, notes_df) -> pd.DataFrame:
    required_train_cols = {"case_num", "pn_num", "feature_num", "annotation"}
    required_feat_cols = {"case_num", "feature_num", "feature_text"}
    required_note_cols = {"case_num", "pn_num", "pn_history"}

    if not required_train_cols.issubset(train_df.columns):
        raise ValueError(f"train_df missing: {required_train_cols - set(train_df.columns)}")
    if not required_feat_cols.issubset(features_df.columns):
        raise ValueError(f"features_df missing: {required_feat_cols - set(features_df.columns)}")
    if not required_note_cols.issubset(notes_df.columns):
        raise ValueError(f"notes_df missing: {required_note_cols - set(notes_df.columns)}")

    print("[INFO] Merging train + features + patient_notes ...")
    merged = (
        train_df
        .merge(features_df, on=["case_num", "feature_num"], how="left")
        .merge(notes_df, on=["case_num", "pn_num"], how="left")
    )
    print("[INFO] Merged shape:", merged.shape)

    # Keep only rows with positive annotations
    # If annotation stored as string "[]"/"[...]" this works; if list, cast to str.
    pos = merged[merged["annotation"].astype(str) != "[]"].copy()
    print("[INFO] Positive-labeled rows:", len(pos))

    print("[INFO] Grouping by note to build summaries ...")
    agg = (
        pos
        .groupby(["case_num", "pn_num", "pn_history"])["feature_text"]
        .agg(lambda xs: sorted({x for x in xs if isinstance(x, str)}))
        .reset_index()
    )

    agg = agg.rename(columns={"pn_history": "note_text"})
    agg["summary"] = agg["feature_text"].apply(lambda lst: "; ".join(lst))

    final = agg[["case_num", "pn_num", "note_text", "summary"]].copy()
    print("[INFO] Built summarization dataset with", len(final), "notes.")
    return final


def build_feature_vocab(features_df: pd.DataFrame) -> pd.DataFrame:
    if "feature_text" not in features_df.columns:
        raise ValueError("'feature_text' column missing in features_df")

    vocab = (
        features_df[["feature_text"]]
        .drop_duplicates()
        .sort_values("feature_text")
        .reset_index(drop=True)
    )
    print("[INFO] Built feature vocab of size", len(vocab))
    return vocab

# Main preprocessing pipeline: loads raw NBME data from MongoDB, merges and aggregates into
# note→summary pairs, builds feature vocabulary, and writes processed datasets back to MongoDB and CSV files.
if __name__ == "__main__":
    db = get_db()
    print(f"[INFO] Connected to MongoDB database: {db.name}")

    # 1. Load raw data.
    train_df = load_collection_as_df(db, MONGO_COLL_TRAIN)
    features_df = load_collection_as_df(db, MONGO_COLL_FEATURES)
    notes_df = load_collection_as_df(db, MONGO_COLL_PATIENT_NOTES)

    # 2. Build summarization dataset.
    summary_df = build_summarization_dataset(train_df, features_df, notes_df)

    # 3. Build feature vocab.
    vocab_df = build_feature_vocab(features_df)

    # 4. Write processed data back to Mongo.
    write_df_to_mongo(summary_df, db, MONGO_COLL_NBME_NOTES, clear_existing=True)
    write_df_to_mongo(vocab_df, db, MONGO_COLL_NBME_FEATURES, clear_existing=True)

    # 5. Save CSVs into /data.
    summary_csv_path = os.path.join(DATA_DIR, "nbme_summarization_dataset.csv")
    vocab_csv_path = os.path.join(DATA_DIR, "nbme_feature_vocab.csv")

    summary_df.to_csv(summary_csv_path, index=False)
    vocab_df.to_csv(vocab_csv_path, index=False)

    print("\n[DONE] Preprocessing complete.")
    print(f" - Summarization dataset rows: {len(summary_df)} → {summary_csv_path}")
    print(f" - Feature vocab size: {len(vocab_df)} → {vocab_csv_path}")
    print(f" - Mongo collections: '{MONGO_COLL_NBME_NOTES}', '{MONGO_COLL_NBME_FEATURES}'")

# python -c "
# import spacy
# from scispacy.linking import EntityLinker
#
# print('Loading model...')
# nlp = spacy.load('en_core_sci_sm')
#
# print('Adding UMLS linker (downloading ~1GB)...')
# linker = EntityLinker(resolve_abbreviations=True, name='scispacy_linker')
# nlp.add_pipe(linker)
#
# print('Testing...')
# doc = nlp('diabetes mellitus')
# print('UMLS setup complete!')
# "