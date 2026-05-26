import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dating_data.csv"
MODEL_PATH = ROOT / "moonlight_model.pkl"
PIPE_PATH = ROOT / "pipeline.pkl"
OUT_DIR = ROOT / "assets" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
if not PIPE_PATH.exists():
    raise FileNotFoundError(f"Missing pipeline: {PIPE_PATH}")

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPE_PATH)
df = pd.read_csv(DATA_PATH)


def get_expected_feature_names(pipe_obj):
    try:
        if hasattr(pipe_obj, "feature_names_in_"):
            return list(pipe_obj.feature_names_in_)
        if hasattr(pipe_obj, "named_steps"):
            scaler = pipe_obj.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                return list(scaler.feature_names_in_)
    except Exception:
        pass
    return []


expected = get_expected_feature_names(pipeline)
if not expected:
    raise ValueError("Could not infer expected feature names from pipeline.")

# Direct aliases from app fields and dataset variations.
aliases = {
    "swipe_ratio": "swipe_right_ratio",
    "profile_pics": "profile_pics_count",
    "message_count": "message_sent_count",
    "emoji_rate": "emoji_usage_rate",
    "interests": "interest_tags",
}

# Build model input with exact expected columns.
X = pd.DataFrame(index=df.index)
for col in expected:
    if col in df.columns:
        X[col] = df[col]
        continue

    if col in aliases and aliases[col] in df.columns:
        X[col] = df[aliases[col]]
        continue

    # Handle one-hot style feature names like income_bracket_High
    matched = False
    for base in [
        "income_bracket",
        "education_level",
        "location_type",
        "relationship_intent",
        "body_type",
        "gender",
        "sexual_orientation",
        "zodiac_sign",
        "swipe_time_of_day",
        "app_usage_time_label",
        "swipe_right_label",
    ]:
        prefix = f"{base}_"
        if col.startswith(prefix) and base in df.columns:
            value = col[len(prefix):]
            X[col] = (df[base].astype(str) == value).astype(float)
            matched = True
            break

    if matched:
        continue

    # Numeric fallback
    X[col] = 0.0

# Ensure numeric conversion for pipeline/scaler compatibility.
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        # Use stable category codes when numeric conversion is not possible.
        numeric_try = pd.to_numeric(X[c], errors="coerce")
        if numeric_try.notna().sum() > 0:
            X[c] = numeric_try
        else:
            X[c] = pd.factorize(X[c].astype(str))[0].astype(float)
    else:
        X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.fillna(0.0)
Xp = pipeline.transform(X)

pred = model.predict(Xp)
pred_np = np.asarray(pred)
if pred_np.ndim == 1:
    pred_np = pred_np.reshape(-1, 1)

proba_raw = None
if hasattr(model, "predict_proba"):
    try:
        proba_raw = model.predict_proba(Xp)
    except Exception:
        proba_raw = None

# Titles for your 4 assignment targets.
target_titles = [
    "Target 1: Ghosting",
    "Target 2: Profile Verification (Bot/Fraud)",
    "Target 3: Ultimate Split Reason",
    "Target 4: Behavioral Personality Cluster",
]

# 1) Predicted class distributions per target (MODEL-BASED)
for idx, title in enumerate(target_titles):
    if idx >= pred_np.shape[1]:
        break
    series = pd.Series(pred_np[:, idx], name="prediction")
    counts = series.value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="viridis")
    plt.title(f"{title} — Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"model_target{idx+1}_pred_distribution.png", dpi=200)
    plt.close()

# 2) Confidence histograms from predict_proba (MODEL-BASED)
if isinstance(proba_raw, (list, tuple)):
    mean_conf = []
    labels = []
    for idx, probs in enumerate(proba_raw):
        if probs is None:
            continue
        arr = np.asarray(probs)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        conf = arr.max(axis=1)
        plt.figure(figsize=(8, 4))
        sns.histplot(conf, bins=30, kde=True, color="#6c8ef5")
        plt.title(f"{target_titles[idx]} — Prediction Confidence")
        plt.xlabel("Max Class Probability")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"model_target{idx+1}_confidence_hist.png", dpi=200)
        plt.close()

        labels.append(f"T{idx+1}")
        mean_conf.append(float(np.mean(conf)))

    if mean_conf:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=labels, y=mean_conf, palette="magma")
        plt.ylim(0, 1)
        plt.title("Model Confidence by Target (Mean)")
        plt.xlabel("Target")
        plt.ylabel("Mean Confidence")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "model_confidence_by_target.png", dpi=200)
        plt.close()

# 3) If model estimators have feature importances, show target-1 processed feature importances.
try:
    if hasattr(model, "estimators_") and len(model.estimators_) > 0:
        est0 = model.estimators_[0]
        if hasattr(est0, "feature_importances_"):
            fi = np.asarray(est0.feature_importances_)
            top_n = min(10, len(fi))
            idxs = np.argsort(fi)[::-1][:top_n]
            labels = [f"ProcessedFeature_{i}" for i in idxs]
            vals = fi[idxs]
            plt.figure(figsize=(9, 5))
            sns.barplot(x=vals, y=labels, palette="rocket")
            plt.title("Target 1 Model Top Processed Feature Importances")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "model_target1_top_processed_features.png", dpi=200)
            plt.close()
except Exception:
    pass

print("Generated model-based PNGs in:", OUT_DIR)
for f in sorted(OUT_DIR.glob("model_*.png")):
    print("-", f.name)
