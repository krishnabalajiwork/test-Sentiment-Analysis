# ─────────────────────────────────────────────────────────────
#  BTS Comment Sentiment Analyzer – Clean & Professional UI
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pathlib

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ─── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="💜 BTS Sentiment", 
    page_icon="🎤", 
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY_PURPLE = "#8E44AD"
LIGHT_PURPLE = "#9B59B6"

# ─── Enhanced Styling ────────────────────────────────────────
st.markdown(f"""
<style>
/* Global App Styling */
.stApp {{
    background-color: #F8F5FB;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

/* Sidebar Styling */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg, {PRIMARY_PURPLE} 0%, {LIGHT_PURPLE} 100%);
    color: #FFFFFF;
}}
[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}

/* Header Banner */
.bts-header {{
    background: linear-gradient(135deg, {PRIMARY_PURPLE} 0%, {LIGHT_PURPLE} 100%);
    padding: 2.5rem 2rem;
    border-radius: 15px;
    text-align: center;
    color: #FFFFFF;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(142, 68, 173, 0.3);
}}

.bts-header h1 {{
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}

.bts-header p {{
    font-size: 1.2rem;
    opacity: 0.9;
}}

/* Input Field Styling */
.stTextInput > div > div > input {{
    background: linear-gradient(90deg, #ffffff 0%, #f7e9fb 100%);
    border: 2px solid {PRIMARY_PURPLE};
    border-radius: 12px;
    padding: 1rem;
    font-size: 1.1rem;
    color: #333;
}}

.stTextInput > div > div > input:focus {{
    border-color: {LIGHT_PURPLE};
    box-shadow: 0 0 0 3px rgba(142, 68, 173, 0.2);
}}

/* Result Card */
.result-card {{
    width: 100%;
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    color: #FFFFFF;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 2rem 0;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    animation: slideIn 0.5s ease-out;
}}

@keyframes slideIn {{
    from {{
        opacity: 0;
        transform: translateY(20px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

/* Button Styling */
.stButton > button {{
    background: linear-gradient(135deg, {PRIMARY_PURPLE} 0%, {LIGHT_PURPLE} 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(142, 68, 173, 0.4);
}}
</style>
""", unsafe_allow_html=True)

# ─── VADER Sentiment Classification ──────────────────────────
def classify_sentiment(text):
    """Classify sentiment using VADER with improved thresholds."""
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.15:
        return "Positive", compound, scores
    elif compound <= -0.15:
        return "Negative", compound, scores
    else:
        return "Neutral", compound, scores

# ─── Dataset Loading ─────────────────────────────────────────
DATA_CANDIDATES = ["bts_2021_1.csv", "data/bts_2021_1.csv"]
csv_path = next((p for p in DATA_CANDIDATES if pathlib.Path(p).exists()), None)
total_comments = 0

if csv_path:
    try:
        df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["comment_text"])
        total_comments = len(df)
    except Exception:
        df = pd.DataFrame()

# ─── Header Section ──────────────────────────────────────────
st.markdown("""
<div class="bts-header">
    <h1>💜 BTS Comment Sentiment Analyzer</h1>
    <p>Advanced AI-powered sentiment analysis for BTS fans worldwide</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar: Clean Stats & Info ─────────────────────────────
with st.sidebar:
    st.markdown("## 📊 App Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset Size", f"{total_comments:,}")
    with col2:
        st.metric("Model", "VADER")
    
    st.markdown("---")
    
    st.markdown("## 💜 About BTS")
    st.markdown("""
    **Members:** RM, Jin, Suga, J-Hope, Jimin, V, Jungkook
    
    **Debut:** June 13, 2013
    
    **Fandom:** ARMY 💜
    
    **Genre:** K-pop, Hip hop, R&B
    """)

# ─── Main Content Area ───────────────────────────────────────
st.markdown("## 💬 Sentiment Analysis")

# Input section
user_text = st.text_input(
    "",
    placeholder="Enter your comment about BTS here... (e.g., 'I love their new album so much!')",
    help="Type any comment about BTS and get instant sentiment analysis",
    key="user_input"
)

# Prediction button and results
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("⚠️ Please enter a comment to analyze.")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment, compound, detailed_scores = classify_sentiment(user_text)
                
                # Choose emoji and gradient based on sentiment
                sentiment_config = {
                    "Positive": {
                        "emoji": "😊",
                        "gradient": "linear-gradient(135deg, #2ecc71 0%, #8E44AD 100%)"
                    },
                    "Negative": {
                        "emoji": "😞", 
                        "gradient": "linear-gradient(135deg, #e74c3c 0%, #8E44AD 100%)"
                    },
                    "Neutral": {
                        "emoji": "😐",
                        "gradient": "linear-gradient(135deg, #95a5a6 0%, #8E44AD 100%)"
                    }
                }
                
                config = sentiment_config[sentiment]
                
                # Display result card
                st.markdown(f'''
                <div class="result-card" style="background: {config['gradient']};">
                    {config['emoji']} {sentiment}
                </div>
                ''', unsafe_allow_html=True)
                
                # Additional details
                st.markdown("### 📈 Detailed Analysis")
                
                col_pos, col_neu, col_neg = st.columns(3)
                with col_pos:
                    st.metric("Positive Score", f"{detailed_scores['pos']:.3f}")
                with col_neu:
                    st.metric("Neutral Score", f"{detailed_scores['neu']:.3f}")
                with col_neg:
                    st.metric("Negative Score", f"{detailed_scores['neg']:.3f}")
                
                st.metric("Compound Score", f"{compound:+.3f}", 
                         help="Overall sentiment score (-1 to +1)")

# ─── Information Section ─────────────────────────────────────
st.markdown("---")
st.markdown("## ℹ️ How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🧠 VADER Sentiment Analysis
    This app uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for accurate sentiment analysis:
    
    - **Social Media Optimized**: Perfect for fan comments and casual language
    - **Emoji Recognition**: Understands emojis and emoticons
    - **Punctuation Aware**: Considers exclamation marks and capitalization
    - **Real-time Processing**: Instant results with high accuracy
    """)

with col2:
    st.markdown("""
    ### 💡 Tips for Better Results
    - Use complete sentences for more accurate analysis
    - Include emojis to express your feelings
    - Be specific about what you liked or disliked
    - Natural language works best!
    
    **Example:**
    - ✅ "I absolutely love their new choreography! 💜"
    - ❌ "good song"
    """)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #8E44AD;'>
    <p style='font-size: 1.2rem; font-weight: 600;'>
        Built with 💜 for ARMY by ARMY
    </p>
    <p style='opacity: 0.8;'>
        Powered by Streamlit & VADER Sentiment Analysis | © 2025
    </p>
</div>
""", unsafe_allow_html=True)