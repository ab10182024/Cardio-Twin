import os
import zipfile
import urllib.request
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_PATH = "uci_har.zip"
OUT_DIR = "uci_har"


def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        print("Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)

    # Extract outer zip into OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Extracting outer zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(OUT_DIR)

    # Some mirrors include an inner zip: "UCI HAR Dataset.zip"
    inner_zip = os.path.join(OUT_DIR, "UCI HAR Dataset.zip")
    if os.path.exists(inner_zip):
        print("Found inner zip. Extracting inner dataset zip...")
        with zipfile.ZipFile(inner_zip, "r") as z:
            z.extractall(OUT_DIR)



def find_uci_har_base():
    """
    Find the folder that contains:
    train/X_train.txt and test/X_test.txt
    """
    for root, dirs, files in os.walk(OUT_DIR):
        if root.endswith(os.path.join("train")) and "X_train.txt" in files:
            base = os.path.dirname(root)  # strip /train
            print(f"Found UCI HAR base at: {base}")
            return base
    raise FileNotFoundError("UCI HAR Dataset not found after extraction.")


def load_data():
    base = find_uci_har_base()

    X_train = np.loadtxt(os.path.join(base, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base, "train", "y_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(base, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base, "test", "y_test.txt")).astype(int)

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # labels 1..6 → 0..5
    y = y - 1
    return X, y


def main():
    download_and_extract()
    X, y = load_data()

    print("Dataset loaded:", X.shape, y.shape)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(Xtr, ytr)

    preds = clf.predict(Xva)
    acc = accuracy_score(yva, preds)
    print(f"Validation accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/activity_rf_ucihar.pkl")

    print("✅ Activity model rebuilt and saved successfully.")


if __name__ == "__main__":
    main()
