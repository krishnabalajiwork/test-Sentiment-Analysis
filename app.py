# ─────────────────────────────────────────────────────────────
#  BTS Comment Sentiment Analyzer – cartoon edition (2025)
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
from textblob import TextBlob
import pathlib

# ─── Page & theme ────────────────────────────────────────────
st.set_page_config(page_title="💜 BTS Sentiment",   page_icon="🎤",
                   layout="wide", initial_sidebar_state="expanded")
PRIMARY_PURPLE = "#8E44AD"

st.markdown(f"""
<style>
/* overall colours */
.stApp {{background-color:#F8F5FB;}}
[data-testid="stSidebar"] > div:first-child {{background:{PRIMARY_PURPLE};color:#FFF;}}
[data-testid="stSidebar"] * {{color:#FFF!important;}}

/* header banner */
.bts-header {{
  background:{PRIMARY_PURPLE};
  padding:2rem 1rem;border-radius:10px;
  text-align:center;color:#FFF;margin-bottom:1.5rem;
}}

/* result card */
.result-card {{
  width:100%;border-radius:15px;padding:2rem;
  text-align:center;color:#FFF;font-size:2rem;font-weight:700;
}}
</style>
""", unsafe_allow_html=True)

# ─── Helper: ultra-light sentiment classifier ────────────────
def classify(text: str) -> tuple[str, float]:
    """Return ('Positive'|'Neutral'|'Negative', polarity_score)."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.20:
        return "Positive", polarity
    if polarity < -0.20:
        return "Negative", polarity
    return "Neutral", polarity

# ─── Optional dataset stats for sidebar ──────────────────────
DATA_CANDIDATES = ["bts_2021_1.csv", "data/bts_2021_1.csv"]
csv_path = next((p for p in DATA_CANDIDATES if pathlib.Path(p).exists()), None)
total_comments = 0
if csv_path:
    try:
        df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["comment_text"])
        total_comments = len(df)
    except Exception:
        pass  # file unreadable; keep stats at zero

# ─── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="bts-header">
  <h1>💜 BTS Comment Sentiment Analyzer</h1>
  <p>Type any sentence about BTS and get instant feedback!</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar: stats + cartoon strip ──────────────────────────
with st.sidebar:
    st.subheader("📊 Dataset")
    st.metric("Comments", f"{total_comments:,}")
    st.markdown("---")

    img_dir = pathlib.Path("images")
    cartoon_imgs = sorted(img_dir.glob("*.*"))
    if cartoon_imgs:
        st.markdown("### BTS Cartoons")
        for img in cartoon_imgs:
            st.image(str(img), use_column_width=True)

# ─── Main input & prediction card ────────────────────────────
user_text = st.text_input(
    "📝 Enter a comment", placeholder="E.g. “I love BTS so much!”")

if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.warning("Please type something first.")
    else:
        sentiment, pol = classify(user_text)
        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[sentiment]
        grad = {
            "Positive": "linear-gradient(135deg,#2ecc71 0%,#8E44AD 100%)",
            "Neutral":  "linear-gradient(135deg,#ffffff 0%,#8E44AD 100%)",
            "Negative": "linear-gradient(135deg,#e74c3c 0%,#8E44AD 100%)"
        }[sentiment]

        st.markdown(
            f'<div class="result-card" style="background:{grad};">'
            f'{emoji} {sentiment}'
            '</div>', unsafe_allow_html=True)
        st.caption(f"TextBlob polarity ≈ {pol:+.3f}")

# ─── Main-page cartoon gallery (fills empty space) ───────────
if cartoon_imgs:
    st.markdown("### 🎨 BTS Cartoon Gallery")
    for img in cartoon_imgs:
        st.image(str(img), use_column_width=True)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TextBlob  |  © 2025")
