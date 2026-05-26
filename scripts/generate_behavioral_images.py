import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dating_data.csv"
OUT_DIR = ROOT / "assets" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_cols = [
    "likes_received",
    "message_sent_count",
    "app_usage_time_min",
    "swipe_right_ratio",
    "emoji_usage_rate",
    "relationship_intent",
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Column missing from dataset: {c}")

# Build a simple engagement tier for behavioral charts.
score = (
    df["likes_received"].fillna(0) * 0.30
    + df["message_sent_count"].fillna(0) * 0.30
    + df["app_usage_time_min"].fillna(0) * 0.25
    + (df["swipe_right_ratio"].fillna(0) * 100.0) * 0.15
)

df_plot = df.copy()
df_plot["engagement_tier"] = pd.qcut(score.rank(method="first"), q=3, labels=["Low", "Medium", "High"])

# 1) Distribution of engagement tiers
plt.figure(figsize=(8, 5))
order = ["Low", "Medium", "High"]
sns.countplot(data=df_plot, x="engagement_tier", order=order, palette=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("Distribution of Engagement Tiers")
plt.xlabel("Engagement Tier")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "distribution_engagement_tiers.png", dpi=200)
plt.close()

# 2) Messages sent vs engagement tier
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="engagement_tier", y="message_sent_count", order=order, palette=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("Messages Sent Count vs Engagement Tier")
plt.xlabel("Engagement Tier")
plt.ylabel("Messages Sent Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "messages_sent_engagement.png", dpi=200)
plt.close()

# 3) Swipe right ratio vs engagement tier
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="engagement_tier", y="swipe_right_ratio", order=order, palette=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("Swipe Right Ratio vs Engagement Tier")
plt.xlabel("Engagement Tier")
plt.ylabel("Swipe Right Ratio")
plt.tight_layout()
plt.savefig(OUT_DIR / "swipe_right_ratio_engagement.png", dpi=200)
plt.close()

# 4) App usage time vs engagement tier
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="engagement_tier", y="app_usage_time_min", order=order, palette=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("App Usage Time vs Engagement Tier")
plt.xlabel("Engagement Tier")
plt.ylabel("App Usage Time (min)")
plt.tight_layout()
plt.savefig(OUT_DIR / "app_usage_time_engagement.png", dpi=200)
plt.close()

# 5) Emoji usage rate vs engagement tier
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="engagement_tier", y="emoji_usage_rate", order=order, palette=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("Emoji Usage Rate vs Engagement Tier")
plt.xlabel("Engagement Tier")
plt.ylabel("Emoji Usage Rate")
plt.tight_layout()
plt.savefig(OUT_DIR / "emoji_usage_engagement.png", dpi=200)
plt.close()

# 6) Relationship intent impact on engagement tier
rel_ct = pd.crosstab(df_plot["relationship_intent"], df_plot["engagement_tier"], normalize="index") * 100
rel_ct = rel_ct.reindex(columns=order)
rel_ct.plot(kind="bar", figsize=(10, 6), color=["#ff9aa2", "#ffd166", "#7db7ff"])
plt.title("Relationship Intent Impact on Engagement Tier")
plt.xlabel("Relationship Intent")
plt.ylabel("Percentage")
plt.xticks(rotation=35, ha="right")
plt.legend(title="Engagement Tier")
plt.tight_layout()
plt.savefig(OUT_DIR / "relationship_intent_engagement.png", dpi=200)
plt.close()

# 7) Top 10 feature importance (RF on engineered engagement tier)
feature_cols = [
    "age",
    "bio_length",
    "profile_pics_count",
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "mutual_matches",
    "message_sent_count",
    "emoji_usage_rate",
    "last_active_hour",
    "height_cm",
    "weight_kg",
    "income_bracket",
    "education_level",
    "relationship_intent",
    "location_type",
]
feature_cols = [c for c in feature_cols if c in df_plot.columns]

X = df_plot[feature_cols].copy()
y = df_plot["engagement_tier"].astype(str)

num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

rf = RandomForestClassifier(n_estimators=250, random_state=42)
pipe = Pipeline([("pre", pre), ("rf", rf)])
pipe.fit(X, y)

feature_names = []
if num_cols:
    feature_names.extend(num_cols)
if cat_cols:
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())

importances = pipe.named_steps["rf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(data=imp_df, y="feature", x="importance", palette="mako")
plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUT_DIR / "top10features.png", dpi=200)
plt.close()

print("Generated behavioral insight PNGs in:", OUT_DIR)
for f in sorted(OUT_DIR.glob("*.png")):
    print("-", f.name)
