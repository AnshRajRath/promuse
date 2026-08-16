from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "speech_features_normalized.csv"

OUTPUT_FILE = DATA_DIR / "speech_features_adapted.csv"


META_COLUMNS = [
    "file_id",
    "dataset",
    "file",
    "speaker",
    "emotion",
    "emotion_id"
]


def load_speech_data():

    df = pd.read_csv(INPUT_FILE)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLUMNS
    ]

    feature_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    return df, feature_cols


def domain_accuracy(df, feature_cols):

    X = df[feature_cols].values
    y = df["dataset"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    print("Domain classification accuracy:", accuracy)

    print("\nClassification report:")
    print(classification_report(y_test, pred))

    return accuracy


def coral_alignment(source, target):

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_cov = np.cov(source_centered, rowvar=False)
    target_cov = np.cov(target_centered, rowvar=False)

    eps = 1e-6

    source_cov += eps * np.eye(source_cov.shape[0])
    target_cov += eps * np.eye(target_cov.shape[0])

    source_eigenvalues, source_eigenvectors = np.linalg.eigh(
        source_cov
    )

    target_eigenvalues, target_eigenvectors = np.linalg.eigh(
        target_cov
    )

    source_eigenvalues = np.maximum(
        source_eigenvalues,
        eps
    )

    target_eigenvalues = np.maximum(
        target_eigenvalues,
        eps
    )

    source_inv_sqrt = (
        source_eigenvectors
        @ np.diag(1.0 / np.sqrt(source_eigenvalues))
        @ source_eigenvectors.T
    )

    target_sqrt = (
        target_eigenvectors
        @ np.diag(np.sqrt(target_eigenvalues))
        @ target_eigenvectors.T
    )

    transform = source_inv_sqrt @ target_sqrt

    aligned = source_centered @ transform

    aligned += target_mean

    return aligned


def perform_coral(df, feature_cols):

    crema_mask = df["dataset"] == "CREMA-D"
    ravdess_mask = df["dataset"] == "RAVDESS"

    crema = df.loc[crema_mask, feature_cols].values
    ravdess = df.loc[ravdess_mask, feature_cols].values

    print("CREMA-D samples:", len(crema))
    print("RAVDESS samples:", len(ravdess))

    print("\nPerforming CORAL alignment...")

    crema_aligned = coral_alignment(
        crema,
        ravdess
    )

    adapted = df.copy()

    adapted.loc[
        crema_mask,
        feature_cols
    ] = crema_aligned

    print("CORAL alignment complete.")

    return adapted


def save_adapted_data(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved adapted dataset:")
    print(OUTPUT_FILE)


def run():

    print("=" * 60)
    print("DOMAIN ADAPTATION")
    print("=" * 60)

    print("\nLoading speech features...")

    df, feature_cols = load_speech_data()

    print("Dataset shape:", df.shape)
    print("Features:", len(feature_cols))

    print("\nDataset distribution:")
    print(df["dataset"].value_counts())

    print("\n" + "=" * 60)
    print("BEFORE CORAL")
    print("=" * 60)

    before_accuracy = domain_accuracy(
        df,
        feature_cols
    )

    print("\n" + "=" * 60)
    print("CORAL ALIGNMENT")
    print("=" * 60)

    adapted_df = perform_coral(
        df,
        feature_cols
    )

    print("\n" + "=" * 60)
    print("AFTER CORAL")
    print("=" * 60)

    after_accuracy = domain_accuracy(
        adapted_df,
        feature_cols
    )

    save_adapted_data(
        adapted_df
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Before adaptation : {before_accuracy:.4f}"
    )

    print(
        f"After adaptation  : {after_accuracy:.4f}"
    )

    print(
        f"Reduction         : "
        f"{before_accuracy - after_accuracy:.4f}"
    )

    return adapted_df


if __name__ == "__main__":
    run()