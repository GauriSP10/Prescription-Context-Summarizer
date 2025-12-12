import os
import pandas as pd

from config import (
    get_db,
    DATA_DIR,
    MONGO_COLL_TRAIN,
    MONGO_COLL_FEATURES,
    MONGO_COLL_PATIENT_NOTES,
)

# CENTRALIZED FILE → COLLECTION MAPPING
UPLOAD_TARGETS = {
    "train.csv": MONGO_COLL_TRAIN,
    "features.csv": MONGO_COLL_FEATURES,
    "patient_notes.csv": MONGO_COLL_PATIENT_NOTES,
}

# CSV UPLOAD HELPER
def upload_csv_to_collection(db, csv_path: str, collection_name: str, clear_existing: bool = True):
    """
    Ingest a CSV file and upload its rows as documents into a target MongoDB collection.
    Input: db (MongoDB database), csv_path (str), collection_name (str), clear_existing (bool) | Output: None
    """
    """
    Reads a CSV file, converts to documents, and uploads to a MongoDB collection.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV file missing: {csv_path}")

    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[INFO] Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    coll = db[collection_name]
    if clear_existing:
        deleted = coll.delete_many({}).deleted_count
        print(f"[INFO] Cleared {deleted} old documents from '{collection_name}'")

    docs = df.to_dict(orient="records")
    if docs:
        coll.insert_many(docs)
        print(f"[INFO] Inserted {len(docs)} new documents into '{collection_name}'")
    else:
        print(f"[WARN] No rows found in {csv_path} — nothing inserted.")


# MAIN INGESTION PIPELINE
def upload_all_csvs():
    """
    Upload all required NBME CSV files into their configured MongoDB collections using UPLOAD_TARGETS mapping.
    Input: None | Output: None
    """
    """
    Walks through UPLOAD_TARGETS and uploads all required CSVs
    into their configured MongoDB collections.
    """
    print(f"[INFO] Loading database handle via get_db()...")
    db = get_db()
    print(f"[INFO] Connected to MongoDB database: {db.name}")

    for filename, collection in UPLOAD_TARGETS.items():
        csv_path = os.path.join(DATA_DIR, filename)

        try:
            upload_csv_to_collection(db, csv_path, collection)
        except Exception as e:
            print(f"[ERROR] Failed to upload {filename}: {e}")

    print("\n[DONE] All dataset uploads complete.")


# RUN SCRIPT
if __name__ == "__main__":
    upload_all_csvs()
