import os
import shutil
import spacy

MODEL_NAME = "en_core_sci_sm"
OUT_DIR = os.path.join("vendor_models", MODEL_NAME)

def main():
    nlp = spacy.load(MODEL_NAME)

    # Make sure output dir is clean.
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save the model to the repo folder.
    nlp.to_disk(OUT_DIR)

    print(f"[DONE] Saved {MODEL_NAME} to: {OUT_DIR}")
    print("You can now commit vendor_models/ to git and deploy without downloading.")

if __name__ == "__main__":
    main()
