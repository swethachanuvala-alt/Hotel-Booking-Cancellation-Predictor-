"""
Model utilities for the Hotel Booking Cancellation project.

Pure Python / scikit-learn only (no Streamlit import) so it can be used by the
web app, by train_model.py and in tests.

Pipeline (identical to notebook GGST_11.ipynb):
    drop Booking_ID + date of reservation
    -> LabelEncoder on categorical columns
    -> stratified 80/20 split (random_state=42)
    -> StandardScaler (fit on train)
    -> SMOTE on the training set
    -> tuned RandomForestClassifier

Target encoding (LabelEncoder, alphabetical):  0 = Canceled, 1 = Not_Canceled
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hotel_booking_cancellation.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

TARGET = "booking status"
DROP_COLS = ["Booking_ID", "date of reservation"]
CAT_COLS = ["type of meal", "room type", "market segment type"]
FEATURES = [
    "number of adults",
    "number of children",
    "number of weekend nights",
    "number of week nights",
    "type of meal",
    "car parking space",
    "room type",
    "lead time",
    "market segment type",
    "repeated",
    "P-C",
    "P-not-C",
    "average price",
    "special requests",
]

# Best parameters found by RandomizedSearchCV in the notebook
RF_PARAMS = dict(
    n_estimators=200,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

CANCELED_CLASS = 0  # LabelEncoder: 'Canceled' -> 0, 'Not_Canceled' -> 1

# Values used by the app's dropdowns (same values as the dataset)
MEAL_OPTIONS = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
ROOM_OPTIONS = [f"Room_Type {i}" for i in range(1, 8)]
SEGMENT_OPTIONS = ["Online", "Offline", "Corporate", "Complementary", "Aviation"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as provided."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# SMOTE (same algorithm as imblearn.over_sampling.SMOTE, k_neighbors=5)
# Implemented with scikit-learn only so the deployed app needs no extra
# dependency.
# --------------------------------------------------------------------------- #
def smote_resample(X: np.ndarray, y: np.ndarray, k: int = 5, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n_major = counts.max()
    X_out, y_out = [X], [y]
    for cls, cnt in zip(classes, counts):
        n_new = n_major - cnt
        if n_new <= 0:
            continue
        Xc = X[y == cls]
        kk = min(k, len(Xc) - 1)
        nn = NearestNeighbors(n_neighbors=kk + 1).fit(Xc)
        neighbours = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.randint(0, len(Xc), size=n_new)
        pick = neighbours[base, rng.randint(0, kk, size=n_new)]
        gap = rng.uniform(size=(n_new, 1))
        synthetic = Xc[base] + gap * (Xc[pick] - Xc[base])
        X_out.append(synthetic)
        y_out.append(np.full(n_new, cls, dtype=y.dtype))
    return np.vstack(X_out), np.concatenate(y_out)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_pipeline(df: pd.DataFrame | None = None):
    """Run the notebook pipeline. Returns (model, scaler, encoders, meta)."""
    if df is None:
        df = load_raw_data()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS + [TARGET]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_res, y_res = smote_resample(X_train_s, y_train.to_numpy())

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_res, y_res)

    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, list(model.classes_).index(1)]
    meta = {
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
        },
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "target_mapping": {"0": "Canceled", "1": "Not_Canceled"},
    }
    return model, scaler, encoders, meta


def save_artifacts(model, scaler, encoders, meta) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    # compress keeps the pickle small enough for GitHub (limit 100 MB / file)
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# Loading + prediction
# --------------------------------------------------------------------------- #
def default_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "number of adults": 2,
                "number of children": 0,
                "number of weekend nights": 1,
                "number of week nights": 2,
                "type of meal": "Meal Plan 1",
                "car parking space": 0,
                "room type": "Room_Type 1",
                "lead time": 60,
                "market segment type": "Online",
                "repeated": 0,
                "P-C": 0,
                "P-not-C": 0,
                "average price": 100.0,
                "special requests": 0,
            }
        ]
    )


def encode_features(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply the saved LabelEncoders to a raw booking DataFrame."""
    out = df[FEATURES].copy()
    for col in CAT_COLS:
        out[col] = encoders[col].transform(out[col].astype(str))
    return out.astype(float)


def predict_cancel_proba(model, scaler, encoders, df: pd.DataFrame) -> np.ndarray:
    """Probability that each booking will be CANCELED (0..1)."""
    X = encode_features(df, encoders)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xs = scaler.transform(X)
        proba = model.predict_proba(Xs)
    idx = list(model.classes_).index(CANCELED_CLASS)
    return proba[:, idx]


def load_artifacts(allow_retrain: bool = True):
    """
    Load the saved model files. If they are missing, were created with an
    incompatible scikit-learn version, or fail a smoke test, the model is
    re-trained from the bundled CSV so the app never crashes.

    Returns (model, scaler, encoders, meta, source)  source in {"saved","retrained"}
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            encoders = joblib.load(ENCODERS_PATH)
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        p = predict_cancel_proba(model, scaler, encoders, default_row())
        if not (0.0 <= float(p[0]) <= 1.0):
            raise ValueError("smoke test failed")
        return model, scaler, encoders, meta, "saved"
    except Exception:
        if not allow_retrain:
            raise
        model, scaler, encoders, meta = train_pipeline()
        try:
            save_artifacts(model, scaler, encoders, meta)
        except Exception:
            pass  # read-only file system is fine, model stays in memory
        return model, scaler, encoders, meta, "retrained"


def risk_level(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


LEAD_LABELS = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"]
_LEAD_BINS = [-1, 7, 30, 90, 180, 10_000]


def lead_bucket(values) -> pd.Series:
    """Group lead times (days) into readable buckets."""
    return pd.cut(pd.Series(values), bins=_LEAD_BINS, labels=LEAD_LABELS).astype(str)
