import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data', 'dating_data.csv')
OUT = os.path.join(ROOT, 'assets', 'figs')
os.makedirs(OUT, exist_ok=True)

print('Loading data from', DATA)
df = pd.read_csv(DATA)

# 1) Match outcome distribution
plt.figure(figsize=(10,6))
counts = df['match_outcome'].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette='Set2')
plt.title('Match Outcome Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'match_outcome_distribution.png'))
plt.close()
print('Wrote match_outcome_distribution.png')

# 2) Ghosted vs Not Ghosted pie
ghost_counts = df['match_outcome'].apply(lambda x: 'Ghosted' if x == 'Ghosted' else 'Not Ghosted').value_counts()
plt.figure(figsize=(6,6))
plt.pie(ghost_counts.values, labels=ghost_counts.index, autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90)
plt.title('Ghosted vs Not Ghosted')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ghosted_pie.png'))
plt.close()
print('Wrote ghosted_pie.png')

# 3) Correlation heatmap on numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'correlation_heatmap.png'))
    plt.close()
    print('Wrote correlation_heatmap.png')
else:
    print('Not enough numeric columns for heatmap')

# 4) PCA + KMeans clusters using pipeline if available
pipeline_path = os.path.join(ROOT, 'pipeline.pkl')
if os.path.exists(pipeline_path):
    try:
        pipeline = joblib.load(pipeline_path)
        print('Loaded pipeline from', pipeline_path)
        # Build X (input features that app expects)
        feature_cols = ['age','swipe_right_ratio','interests','likes_received','mutual_matches','profile_pics_count','bio_length','message_sent_count','emoji_usage_rate','last_active_hour','height_cm','weight_kg']
        # some column name variations in CSV - map common names
        col_map = {
            'profile_pics_count': 'profile_pics_count',
            'profile_pics': 'profile_pics_count',
            'message_sent_count': 'message_sent_count',
            'message_count': 'message_sent_count',
            'swipe_right_ratio': 'swipe_right_ratio',
            'likes_received': 'likes_received'
        }
        # Build available features
        X = pd.DataFrame()
        for c in feature_cols:
            src = c
            if c not in df.columns and c in col_map:
                src = col_map[c]
            if src in df.columns:
                X[c] = df[src]
        # fill missing numeric
        X = X.fillna(X.median())
        X_proc = pipeline.transform(X)
        # If pipeline output is 2D array (PCA), use first two components
        if X_proc.ndim == 2 and X_proc.shape[1] >= 2:
            X_pca = X_proc
            # run KMeans for visualization
            km = KMeans(n_clusters=4, random_state=42, n_init=10)
            labels = km.fit_predict(X_pca)
            plt.figure(figsize=(8,6))
            palette = sns.color_palette('Set2', 8)
            plt.scatter(X_pca[:,0], X_pca[:,1], c=[palette[l] for l in labels], s=10, alpha=0.6)
            plt.title('PCA Clusters (visualization)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, 'pca_clusters.png'))
            plt.close()
            print('Wrote pca_clusters.png')
        else:
            print('Pipeline output not suitable for PCA scatter')
    except Exception as e:
        print('Failed to load/execute pipeline:', e)
else:
    print('No pipeline.pkl found, skipping PCA cluster plot')

# 5) Ghost probability histogram using model if available
model_path = os.path.join(ROOT, 'moonlight_model.pkl')
if os.path.exists(model_path) and os.path.exists(pipeline_path):
    try:
        model = joblib.load(model_path)
        print('Loaded model from', model_path)
        # use X and pipeline from above
        # generate synthetic samples jittering X
        n = 500
        if X.shape[0] > 0:
            base = X.iloc[:n].copy()
            # jitter
            for col in base.columns:
                if np.issubdtype(base[col].dtype, np.number):
                    base[col] = base[col] + np.random.normal(0, max(1.0, base[col].std())*0.05, size=len(base))
            Xs = pipeline.transform(base)
            proba_list = None
            try:
                proba_list = model.predict_proba(Xs)
            except Exception:
                # MultiOutputClassifier returns list-like
                try:
                    # if multioutput, it's list of arrays
                    proba_list = model.predict_proba(Xs)
                except Exception as e:
                    print('predict_proba failed:', e)
            ghost_probs = None
            if proba_list is not None:
                # if list-like, take first estimator's probs for class 1
                if isinstance(proba_list, (list, tuple)):
                    first = proba_list[0]
                    if first.ndim == 2 and first.shape[1] > 1:
                        ghost_probs = first[:,1]
                    else:
                        ghost_probs = first[:,0]
                else:
                    # single estimator
                    first = proba_list
                    if first.ndim == 2 and first.shape[1] > 1:
                        ghost_probs = first[:,1]
                    else:
                        ghost_probs = first[:,0]
            if ghost_probs is not None:
                plt.figure(figsize=(8,4))
                sns.histplot(ghost_probs, bins=30, kde=True, color='steelblue')
                plt.title('Ghosting Probability Distribution (synthetic samples)')
                plt.xlabel('Probability')
                plt.tight_layout()
                plt.savefig(os.path.join(OUT, 'ghost_probability_hist.png'))
                plt.close()
                print('Wrote ghost_probability_hist.png')
            else:
                print('No ghost probabilities available from model')
    except Exception as e:
        print('Failed to load model or compute probabilities:', e)
else:
    print('Model or pipeline missing, skipping ghost probability histogram')

print('Done. PNGs saved to', OUT)
