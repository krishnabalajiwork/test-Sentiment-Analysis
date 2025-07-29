# ─────────────────────────────────────────────────────────────
#  Simple BTS Sentiment Analyzer  |  Streamlit 2025
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pathlib

# ───────────────────────────────
#  Page & global style
# ───────────────────────────────
st.set_page_config(
    page_title="💜 BTS Sentiment",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BTS-purple everywhere
PRIMARY_PURPLE = "#8E44AD"

st.markdown(
    f"""
    <style>
        /* --- GLOBAL --- */
        .stApp {{background-color:#F8F5FB;}}

        /* --- SIDEBAR --- */
        [data-testid="stSidebar"] > div:first-child {{
            background-color:{PRIMARY_PURPLE};
            color:#FFFFFF;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] .stMarkdown p {{
            color:#FFFFFF !important;
        }}

        /* --- HEADER BANNER --- */
        .bts-header {{
            background:{PRIMARY_PURPLE};
            padding:2rem 1rem;
            border-radius:10px;
            text-align:center;
            color:#FFFFFF;
            margin-bottom:1.5rem;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────
#  Helper functions
# ───────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["comment_text"]).reset_index(drop=True)

    # Pre-compute polarity & sentiment label
    df["polarity"] = df["comment_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["sentiment"] = df["polarity"].apply(
        lambda p: "Positive" if p > 0 else ("Negative" if p < 0 else "Neutral")
    )
    return df

@st.cache_resource(show_spinner=False)
def train(df: pd.DataFrame):
    X = CountVectorizer(stop_words="english", max_features=3000).fit_transform(df.comment_text)
    y = LabelEncoder().fit_transform(df.sentiment)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc

def predict_one(text: str):
    X = vectorizer.transform([text])
    pred_idx = model.predict(X)[0]
    label = label_encoder.inverse_transform([pred_idx])[0]
    return label

# ───────────────────────────────
#  Data & model
# ───────────────────────────────
DATA_CANDIDATES = ["bts_2021_1.csv", "data/bts_2021_1.csv"]
csv_path = next((p for p in DATA_CANDIDATES if pathlib.Path(p).exists()), None)

if not csv_path:
    st.error("Dataset **bts_2021_1.csv** not found in repo 😢")
    st.stop()

with st.spinner("Loading data & training…"):
    df = load_data(csv_path)
    vectorizer = CountVectorizer(stop_words="english", max_features=3000)
    X_all = vectorizer.fit_transform(df.comment_text)
    label_encoder = LabelEncoder().fit(df.sentiment)
    model, accuracy = train(df)

# ───────────────────────────────
#  Layout
# ───────────────────────────────
st.markdown(
    """
    <div class="bts-header">
        <h1>💜 BTS Comment Sentiment Analyzer</h1>
        <p>Type any sentence about BTS and get instant sentiment feedback!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar –- very simple stats
with st.sidebar:
    st.subheader("📊  Dataset")
    st.metric("Comments", f"{len(df):,}")
    st.metric("Model accuracy", f"{accuracy*100:.1f}%")

# Main input
user_text = st.text_input("📝 Enter a comment", placeholder="E.g. ‘I love BTS so much!’")

if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please type something first.")
    else:
        sentiment = predict_one(user_text)
        emoji = {"Positive":"😊", "Neutral":"😐", "Negative":"😞"}[sentiment]
        st.success(f"{emoji} **{sentiment}**")

        # Optional explanation
        pol = TextBlob(user_text).sentiment.polarity
        st.caption(f"Polarity score ≈ {pol:.3f}")

# ───────────────────────────────
#  Footer
# ───────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TextBlob – © 2025")

