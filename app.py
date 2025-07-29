# ─────────────────────────────────────────────────────────────
#  BTS Sentiment Analyzer – Colour-Fix & Sidebar-Fix Edition
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import nltk, pathlib
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ─── Theme values ────────────────────────────────────────────
PRIMARY_PURPLE = "#8E44AD"
DARK_TEXT      = "#2C2C2C"          # universal on-light text
LAVENDER_BG    = "#F8F5FB"

st.set_page_config(page_title="💜 BTS Sentiment",
                   page_icon="🎤",
                   layout="wide")

# ─── GLOBAL STYLE FIXES (all colour problems resolved) ───────
st.markdown(f"""
<style>
/* ---------- GLOBAL COLOURS ---------- */
.stApp {{background:{LAVENDER_BG}; color:{DARK_TEXT};}}
h1,h2,h3,h4,h5,h6, .markdown-text-container * {{color:{DARK_TEXT} !important;}}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"]>div:first-child {{
    background:{PRIMARY_PURPLE};
    color:#FFF;
}}
[data-testid="stSidebar"]>div:not(:first-child) {{
    background:#EFE6F7;
}}
[data-testid="stSidebar"] * {{color:{DARK_TEXT}!important;}}

/* ---------- METRICS (Fix for App Statistics) ---------- */
[data-testid="stMetric"] {{
    color:{DARK_TEXT} !important;
}}
[data-testid="stMetric"] * {{
    color:{DARK_TEXT} !important;
}}
[data-testid="stMetric"] div {{
    white-space: normal;
    color:{DARK_TEXT} !important;
}}

/* ---------- HEADER ---------- */
.bts-header {{
  background:{PRIMARY_PURPLE};
  padding:1.6rem 1rem;
  border-radius:10px;
  text-align:center;
  color:#FFF;
  margin-bottom:1.6rem;
}}

/* ---------- RESULT CARD ---------- */
.result-card {{
  width:100%;
  border-radius:16px;
  padding:2.4rem;
  margin-top:1rem;
  text-align:center;
  color:#FFF;
  font-size:2.2rem;
  font-weight:800;
  box-shadow:0 2px 16px #0002;
}}

/* ---------- INPUT FIELD ---------- */
input[type='text'] {{
  background:#EFE6F7;
  border:2px solid {PRIMARY_PURPLE};
  border-radius:12px;
  padding:0.6rem 1rem;
  font-size:1.05rem;
  color:{DARK_TEXT};
}}
input[type='text']::placeholder {{
  color:#6E5A80;
  opacity:1;
}}
input[type='text']:focus {{
  outline:none;
  border:2px solid #9054D6;
}}
</style>
""", unsafe_allow_html=True)


# ─── Helper: VADER prediction ────────────────────────────────
def vader_predict(text):
    score = sia.polarity_scores(text)["compound"]
    if score > 0.2:
        return "Positive", score
    if score < -0.2:
        return "Negative", score
    return "Neutral", score

# ─── Optional dataset stats ──────────────────────────────────
csv_path = next((p for p in ["bts_2021_1.csv","data/bts_2021_1.csv"]
                 if pathlib.Path(p).exists()), None)
total_comments = 0
if csv_path:
    try:
        df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["comment_text"])
        total_comments = len(df)
    except Exception:
        pass

# ─── HEADER ─────────────────────────────────────────────────
st.markdown("""
<div class="bts-header">
  <h1>🎤 Sentiment Analysis</h1>
  <p>Type any sentence about BTS and get instant feedback!</p>
</div>
""", unsafe_allow_html=True)

# ─── LAYOUT ─────────────────────────────────────────────────
left, right = st.columns([2.5, 0.8])

with left:
    st.markdown("### 📝 Enter your comment about BTS here...")
    user_text = st.text_input("", key="input")
    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.warning("Please type something first!")
        else:
            sentiment, comp = vader_predict(user_text)
            emoji = {"Positive":"😊","Neutral":"😐","Negative":"😞"}[sentiment]
            grad = {
                "Positive":"linear-gradient(135deg,#2ecc71 0%,#8E44AD 100%)",
                "Neutral" :"linear-gradient(135deg,#ffffff 0%,#8E44AD 100%)",
                "Negative":"linear-gradient(135deg,#e74c3c 0%,#8E44AD 100%)"
            }[sentiment]
            st.markdown(
                f'<div class="result-card" style="background:{grad};">'
                f'{emoji} {sentiment}'
                '</div>',
                unsafe_allow_html=True)
            st.caption(f"VADER compound score: {comp:+.3f}")

    st.markdown("---")
    st.markdown("## How It Works")
    st.write(
        "This app uses **VADER** (Valence Aware Dictionary and Sentiment Reasoner) "
        "for accurate sentiment analysis. VADER is designed for social-media-style "
        "text and understands emojis, slang and punctuation.")
    st.markdown("## Tips for Better Results")
    st.write(
        "- Use full sentences for clearer context.\n"
        "- Emojis and exclamation marks help convey tone.\n"
        "- Very short fragments may be classified as Neutral.")

with right:
    st.markdown("## App Statistics")
    st.metric("Dataset Size", f"{total_comments:,}")
    st.metric("Model", "VADER")

# ─── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8E44AD;'>"
    "Built with 💜 using Streamlit & VADER • © 2025"
    "</div>",
    unsafe_allow_html=True)
