import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from pathlib import Path

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False


st.set_page_config(
    page_title="🌕 Project Moonlight",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / 'assets' / 'figs'

INCOME_MAP = {"Low": 0, "Middle": 1, "Upper-Middle": 2, "High": 3, "Very High": 4, "Very Low": 0, "Lower-Middle": 1}
EDUCATION_MAP = {"No Formal Education": 0, "High School": 1, "Diploma": 2, "Associate’s": 3, "Bachelor’s": 4, "Master’s": 5, "MBA": 5, "PhD": 6, "Postdoc": 7}


@st.cache_resource
def load_resources():
    # Allow model to be provided via a public URL to avoid large Git LFS downloads during deploy.
    model_path = ROOT / 'moonlight_model.pkl'
    # Priority: Streamlit secrets -> environment variable
    model_url = None
    try:
        model_url = st.secrets.get('MODEL_URL') if hasattr(st, 'secrets') else None
    except Exception:
        model_url = None
    if not model_url:
        model_url = os.environ.get('MODEL_URL')

    if model_url and not model_path.exists():
        # Download model in streaming mode to avoid memory spikes
        tmp_path = model_path.with_suffix('.pkl.download')
        try:
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(model_path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to download model from MODEL_URL: {e}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}. Provide MODEL_URL in Streamlit secrets or env vars.")

    model_obj = joblib.load(model_path)
    pipeline_obj = joblib.load(ROOT / 'pipeline.pkl')
    return model_obj, pipeline_obj


@st.cache_data
def load_dataset():
    data_path = ROOT / 'data' / 'dating_data.csv'
    return pd.read_csv(data_path)


def get_expected_feature_names(pipeline_obj):
    try:
        if hasattr(pipeline_obj, 'feature_names_in_'):
            return list(pipeline_obj.feature_names_in_)
        if hasattr(pipeline_obj, 'named_steps'):
            scaler = pipeline_obj.named_steps.get('scaler')
            if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                return list(scaler.feature_names_in_)
    except Exception:
        pass
    return []


def build_feature_vector(inputs, feature_order):
    vector = pd.Series(0.0, index=feature_order, dtype=float)

    # Numeric fields that usually exist.
    direct_map = {
        'age': float(inputs['age']),
        'bio_length': float(inputs['bio_length']),
        'profile_pics_count': float(inputs['profile_pics_count']),
        'app_usage_time_min': float(inputs['app_usage_time_min']),
        'swipe_right_ratio': float(inputs['swipe_right_ratio']),
        'likes_received': float(inputs['likes_received']),
        'mutual_matches': float(inputs['mutual_matches']),
        'message_sent_count': float(inputs['message_sent_count']),
        'emoji_usage_rate': float(inputs['emoji_usage_rate']),
        'last_active_hour': float(inputs['last_active_hour']),
        'height_cm': float(inputs['height_cm']),
        'weight_kg': float(inputs['weight_kg']),
        'income_bracket': float(INCOME_MAP.get(inputs['income_bracket'], 1)),
        'education_level': float(EDUCATION_MAP.get(inputs['education_level'], 2)),
    }

    # Common aliases in your app/old code.
    aliases = {
        'swipe_ratio': 'swipe_right_ratio',
        'profile_pics': 'profile_pics_count',
        'message_count': 'message_sent_count',
        'emoji_rate': 'emoji_usage_rate',
        'interests': 'interest_tags_count',
    }

    for name, value in direct_map.items():
        if name in vector.index:
            vector.loc[name] = value

    for alias_name, source_name in aliases.items():
        if alias_name in vector.index and source_name in direct_map:
            vector.loc[alias_name] = direct_map[source_name]

    # If model expects one-hot encoded columns, attempt simple prefix matching.
    one_hot_inputs = {
        'income_bracket': inputs['income_bracket'],
        'education_level': inputs['education_level'],
        'location_type': inputs['location_type'],
        'relationship_intent': inputs['relationship_intent'],
    }
    for prefix, label in one_hot_inputs.items():
        target_col = f"{prefix}_{label}"
        if target_col in vector.index:
            vector.loc[target_col] = 1.0

    return pd.DataFrame([vector.values], columns=feature_order)


def preprocess_input(frame, pipeline_obj, feature_order):
    try:
        return pipeline_obj.transform(frame.reindex(columns=feature_order, fill_value=0.0))
    except Exception:
        return frame.reindex(columns=feature_order, fill_value=0.0).values


def get_target_pred_and_proba(model_obj, Xp, target_idx):
    pred = model_obj.predict(Xp)
    pred_arr = np.asarray(pred)
    pred_value = pred_arr[0, target_idx] if pred_arr.ndim == 2 else pred_arr[0]

    probs = None
    if hasattr(model_obj, 'predict_proba'):
        raw = model_obj.predict_proba(Xp)
        if isinstance(raw, (list, tuple)) and len(raw) > target_idx:
            probs = np.asarray(raw[target_idx][0], dtype=float)
        elif not isinstance(raw, (list, tuple)):
            probs = np.asarray(raw[0], dtype=float)
    return pred_value, probs


def render_target_charts(target_idx):
    chart_map = {
        1: [
            ("model_target1_pred_distribution.png", "Ghosting predictions across the dataset"),
            ("model_target1_confidence_hist.png", "Ghosting prediction confidence"),
            ("model_target1_top_processed_features.png", "Top processed features for Ghosting"),
        ],
        2: [
            ("model_target2_pred_distribution.png", "Bot/Fraud predictions across the dataset"),
            ("model_target2_confidence_hist.png", "Bot/Fraud prediction confidence"),
        ],
        3: [
            ("model_target3_pred_distribution.png", "Split Reason predictions across the dataset"),
            ("model_target3_confidence_hist.png", "Split Reason prediction confidence"),
        ],
        4: [
            ("model_target4_pred_distribution.png", "Personality Cluster predictions across the dataset"),
            ("model_target4_confidence_hist.png", "Personality Cluster prediction confidence"),
        ],
    }

    cards = chart_map.get(target_idx, [])
    if not cards:
        st.info("No charts configured for this target yet.")
        return

    st.markdown("<div class='section-chip'>Target Charts</div>", unsafe_allow_html=True)

    missing = []
    for filename, caption in cards:
        image_path = FIG_DIR / filename
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)
        else:
            st.warning(f"Missing image: {filename}")
            missing.append(filename)

    if missing:
        st.info("Run `python scripts/generate_model_based_pngs.py` to regenerate these target charts.")


def render_behavioral_insights_showcase():
    st.markdown("#### Behavioral Summary")
    st.write("This page combines model confidence with dataset-level behavior patterns so it stays distinct from the four target tabs.")

    image_path = FIG_DIR / "model_confidence_by_target.png"

    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Unable to load dataset for behavioral insights: {e}")
        df = None

    if df is not None:
        metric_cols = ["app_usage_time_min", "swipe_right_ratio", "likes_received", "message_sent_count"]
        summary = df[metric_cols].mean(numeric_only=True)
        top_intent_likes = df.groupby("relationship_intent")["likes_received"].mean().sort_values(ascending=False)
        top_intent_usage = df.groupby("relationship_intent")["app_usage_time_min"].mean().sort_values(ascending=False)
        top_intent_swipe = df.groupby("relationship_intent")["swipe_right_ratio"].mean().sort_values(ascending=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg app usage (min)", f"{summary['app_usage_time_min']:.1f}")
        c2.metric("Avg swipe-right ratio", f"{summary['swipe_right_ratio']:.2f}")
        c3.metric("Avg likes received", f"{summary['likes_received']:.1f}")
        c4.metric("Avg messages sent", f"{summary['message_sent_count']:.1f}")

        st.markdown("---")
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### Relationship Intent Patterns")
            intent_summary = (
                df.groupby("relationship_intent")
                .agg(
                    app_usage_time_min=("app_usage_time_min", "mean"),
                    swipe_right_ratio=("swipe_right_ratio", "mean"),
                    likes_received=("likes_received", "mean"),
                    message_sent_count=("message_sent_count", "mean"),
                )
                .reset_index()
            )

            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    intent_summary.sort_values("likes_received", ascending=False),
                    x="relationship_intent",
                    y="likes_received",
                    color="relationship_intent",
                    title="Average Likes Received by Relationship Intent",
                )
                fig.update_layout(showlegend=False, xaxis_title="Relationship Intent", yaxis_title="Average Likes Received")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(intent_summary.set_index("relationship_intent")["likes_received"])

        with right:
            st.markdown("#### Dataset Snapshot")
            st.dataframe(
                intent_summary.sort_values("likes_received", ascending=False).round(2),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Quick Takeaways")
        top_like_intent = top_intent_likes.index[0]
        top_usage_intent = top_intent_usage.index[0]
        top_swipe_intent = top_intent_swipe.index[0]
        st.info(
            f"• {top_like_intent} users receive the most likes on average.\n"
            f"• {top_usage_intent} users spend the most time in the app on average.\n"
            f"• {top_swipe_intent} users have the strongest swipe-right ratio on average."
        )

    st.markdown("---")
    st.markdown("#### Model Confidence Overview")
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
        st.caption("Mean model confidence across all four targets.")
    else:
        st.warning("Missing image: model_confidence_by_target.png")
        st.info("Run `python scripts/generate_model_based_pngs.py` to regenerate the summary chart.")

    st.markdown("#### Why this matters")
    st.write("Use these patterns to describe how behavior changes by relationship intent, then connect those observations back to the model results.")

    st.markdown("---")
    st.markdown("#### Group Members")
    st.markdown(
        """
        Fong Jun Toh<br>
        Mak Jia Hng<br>
        Chew Yi Yu<br>
        Tin Li Qi<br>
        Ee Mei Xuan<br>
        Ng Peng Han
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Closing")
    st.write("Thank you for viewing our project.")


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Playfair Display', serif; letter-spacing: -0.02em; }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 207, 220, 0.60), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 184, 200, 0.40), transparent 28%),
                linear-gradient(180deg, #fff8fb 0%, #fff4f7 45%, #fffdfd 100%);
        }
        .hero-shell {
            padding: 1.4rem 1.5rem;
            border-radius: 24px;
            border: 1px solid rgba(153, 61, 94, 0.12);
            background: linear-gradient(135deg, rgba(255,255,255,0.93), rgba(255,240,245,0.88));
            box-shadow: 0 18px 36px rgba(130, 54, 79, 0.10);
        }
        .section-chip {
            display: inline-block;
            padding: 0.34rem 0.7rem;
            margin: 0.9rem 0 0.8rem 0;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255, 127, 143, 0.16), rgba(125, 183, 255, 0.16));
            color: #6b2948;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Moonlight Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Live Predictor",
        "📊 Four Targets",
        "🧠 Behavioral Insights",
    ],
)

model = None
pipeline = None
feature_order = []
load_error = None
try:
    model, pipeline = load_resources()
    feature_order = get_expected_feature_names(pipeline)
except Exception as e:
    load_error = f"{e} | checked: {ROOT / 'moonlight_model.pkl'} and {ROOT / 'pipeline.pkl'}"

if not feature_order:
    feature_order = [
        'age', 'bio_length', 'profile_pics_count', 'income_bracket', 'education_level',
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour', 'height_cm', 'weight_kg'
    ]

st.sidebar.markdown("---")
if load_error:
    st.sidebar.error("Model load failed")
    st.sidebar.caption(load_error)
else:
    st.sidebar.success("Model and pipeline loaded")


def collect_inputs():
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 60, 28)
        bio_length = st.slider("Bio Length", 0, 500, 150)
        profile_pics_count = st.slider("Profile Pics Count", 0, 10, 4)
        app_usage_time_min = st.slider("App Usage Time (min)", 0, 300, 72)
        swipe_right_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.42, 0.01)
        likes_received = st.number_input("Likes Received", min_value=0, value=14, step=1)
        mutual_matches = st.number_input("Mutual Matches", min_value=0, value=5, step=1)

    with c2:
        message_sent_count = st.number_input("Messages Sent", min_value=0, value=28, step=1)
        emoji_usage_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.31, 0.01)
        last_active_hour = st.slider("Last Active Hour", 0, 23, 12)
        height_cm = st.slider("Height (cm)", 140, 210, 170)
        weight_kg = st.slider("Weight (kg)", 35, 140, 70)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Middle", "Upper-Middle", "High", "Very High"])
        education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor’s", "Master’s", "PhD"])

    relationship_intent = st.selectbox("Relationship Intent", ["Friends Only", "Casual Dating", "Serious Relationship", "Hookups", "Exploring", "Networking"])
    location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural", "Metro", "Small Town", "Remote Area"])

    return {
        'age': age,
        'bio_length': bio_length,
        'profile_pics_count': profile_pics_count,
        'app_usage_time_min': app_usage_time_min,
        'swipe_right_ratio': swipe_right_ratio,
        'likes_received': likes_received,
        'mutual_matches': mutual_matches,
        'message_sent_count': message_sent_count,
        'emoji_usage_rate': emoji_usage_rate,
        'last_active_hour': last_active_hour,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'income_bracket': income_bracket,
        'education_level': education_level,
        'relationship_intent': relationship_intent,
        'location_type': location_type,
    }


if page == "🔮 Live Predictor":
    st.markdown(
        """
        <div class="hero-shell">
            <div style="font-size:0.88rem;letter-spacing:0.12em;text-transform:uppercase;color:#a24c68;font-weight:700;">Live Predictor</div>
            <h1 style="margin:0.35rem 0 0.35rem 0;color:#3f1830;">🌕 Project Moonlight</h1>
            <div style="color: rgba(68,31,50,0.78);">Discover your dating archetype and predict your mutual matches.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-chip'>Profile Inputs</div>", unsafe_allow_html=True)
    inputs = collect_inputs()
    analyze = st.button("Analyze All 4 Targets", use_container_width=True)

    if analyze:
        if model is None or pipeline is None:
            st.error("Missing `moonlight_model.pkl` or `pipeline.pkl`.")
        else:
            raw_df = build_feature_vector(inputs, feature_order)
            Xp = preprocess_input(raw_df, pipeline, feature_order)

            target_meta = [
                (0, "Probability of Ghosting", "Binary Classification"),
                (1, "Profile Verification Flag (Bot/Fraud)", "Binary Classification"),
                (2, "Ultimate Split Reason Class", "Multiclass Classification"),
                (3, "Behavioral Personality Category", "Clustering"),
            ]

            rows = []
            for idx, title, ttype in target_meta:
                try:
                    pred_val, probs = get_target_pred_and_proba(model, Xp, idx)
                    rows.append({
                        'target': title,
                        'task': ttype,
                        'prediction': pred_val,
                        'confidence': float(np.max(probs)) if probs is not None and len(probs) > 0 else np.nan,
                    })
                except Exception as e:
                    rows.append({'target': title, 'task': ttype, 'prediction': f'Error: {e}', 'confidence': np.nan})

            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True)

            conf_df = result_df.dropna(subset=['confidence'])[['target', 'confidence']]
            if not conf_df.empty:
                if PLOTLY_AVAILABLE:
                    fig = px.bar(conf_df, x='target', y='confidence', color='target', title='Target Confidence')
                    fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(conf_df.set_index('target'))

elif page == "📊 Four Targets":
    st.markdown("<div class='section-chip'>Four Prediction Targets</div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs([
        "1. Ghosting",
        "2. Bot/Fraud",
        "3. Split Reason",
        "4. Personality Cluster",
    ])

    with t1:
        st.subheader("Target 1: Probability of Ghosting")
        st.write("Type: Binary Classification")
        render_target_charts(1)
    with t2:
        st.subheader("Target 2: Profile Verification Flag (Bot/Fraud)")
        st.write("Type: Binary Classification")
        render_target_charts(2)
    with t3:
        st.subheader("Target 3: Ultimate Split Reason Class")
        st.write("Type: Multiclass Classification")
        render_target_charts(3)
    with t4:
        st.subheader("Target 4: Behavioral Personality Category")
        st.write("Type: Clustering")
        render_target_charts(4)

else:
    st.markdown("<div class='section-chip'>Behavioral Insights</div>", unsafe_allow_html=True)
    st.subheader("Behavioral Insights Dashboard")
    render_behavioral_insights_showcase()
