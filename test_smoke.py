import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

ROOT = Path(__file__).parent
print('Script directory (ROOT):', ROOT)
print('Files in script dir:', [p.name for p in ROOT.iterdir()])

# Load pipeline if exists
pipeline = None
p_path = ROOT / 'pipeline.pkl'
if p_path.exists():
    try:
        pipeline = joblib.load(p_path)
        print('Loaded pipeline from', p_path)
        print('Pipeline type:', type(pipeline))
    except Exception as e:
        print('Failed loading pipeline:', e)
        traceback.print_exc()
else:
    print('No pipeline.pkl found')

# Load model
model = None
m_path = ROOT / 'moonlight_model.pkl'
if m_path.exists():
    try:
        model = joblib.load(m_path)
        print('Loaded model from', m_path)
        print('Model type:', type(model))
        print('Model n_features_in_:', getattr(model, 'n_features_in_', None))
        print('Model feature_names_in_:', getattr(model, 'feature_names_in_', None))
    except Exception as e:
        print('Failed loading model:', e)
        traceback.print_exc()
else:
    print('No moonlight_model.pkl found')

# Build sample input
sample = pd.DataFrame([{
    'age': 25,
    'swipe_ratio': 0.5,
    'interests': 5,
    'likes_received': 10,
    'mutual_matches': 2,
    'profile_pics': 3,
    'bio_length': 150,
    'message_count': 20,
    'emoji_rate': 0.1,
    'last_active_hour': 20,
    'height_cm': 170.0,
    'weight_kg': 70.0
}])

print('\nSample input:\n', sample)

# Preprocess
Xp = None
if pipeline is not None:
    try:
        # sklearn pipeline-like object
        if hasattr(pipeline, 'transform'):
            try:
                Xp = pipeline.transform(sample)
                print('Pipeline.transform output shape:', getattr(Xp, 'shape', None))
            except Exception as e:
                print('pipeline.transform raised:', e)
                traceback.print_exc()
                # Inspect pipeline for clues
                try:
                    if hasattr(pipeline, 'named_steps'):
                        print('Pipeline steps:', list(pipeline.named_steps.keys()))
                    if hasattr(pipeline, 'get_feature_names_out'):
                        try:
                            fn = pipeline.get_feature_names_out(sample.columns)
                            print('Pipeline output feature names:', fn)
                        except Exception:
                            try:
                                fn = pipeline.get_feature_names_out()
                                print('Pipeline output feature names:', fn)
                            except Exception:
                                pass
                except Exception:
                    pass

        # dict-style artifact containing encoders/scaler/pca
        elif isinstance(pipeline, dict):
            art = pipeline
            df2 = sample.copy()
            encs = art.get('encoders')
            if encs:
                for col, le in encs.items():
                    if col in df2.columns:
                        df2[col] = le.transform(df2[col].astype(str))
            scaler = art.get('scaler')
            num_cols = art.get('numeric_cols') or [c for c in df2.columns if np.issubdtype(df2[c].dtype, np.number)]
            if scaler is not None and len(num_cols) > 0:
                try:
                    df2[num_cols] = scaler.transform(df2[num_cols])
                except Exception as e:
                    print('Scaler transform failed:', e)
                    traceback.print_exc()
            pca = art.get('pca')
            if pca is not None:
                Xp = pca.transform(df2.values)
            else:
                Xp = df2.values
            print('Prepared Xp from dict pipeline, shape:', getattr(Xp, 'shape', None))
    except Exception as e:
        print('Pipeline preprocessing failed:', e)
        traceback.print_exc()

if Xp is None:
    Xp = sample.values
    print('Using raw values, shape:', Xp.shape)

# Predict
if model is not None:
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xp)
            pred = model.predict(Xp)
            print('\nModel prediction:', pred)
            print('Model probabilities:', proba)
        else:
            pred = model.predict(Xp)
            print('\nModel prediction:', pred)
    except Exception as e:
        print('Model prediction failed:', e)
        traceback.print_exc()
else:
    print('No model to predict with.')
