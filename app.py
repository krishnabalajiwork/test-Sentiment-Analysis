# ─────────────────────────────────────────────────────────────
#  Sentiment-Analysis Streamlit App  |  krishnabalajiwork 2025
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
import pathlib

# ───────────────────────────────
#  Page config
# ───────────────────────────────
st.set_page_config(
    page_title="🎤 BTS Comment Sentiment Analyzer",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────
#  CSS Styling
# ───────────────────────────────
st.markdown("""
<style>
/* BTS-inspired purple theme */
.main-header {
    background: linear-gradient(90deg, #8E44AD, #9B59B6);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #8E44AD, #9B59B6);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #2C3E50;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #ffffff !important;
}

.music-emoji {
    font-size: 2rem;
    animation: bounce 2s infinite;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-10px);
    }
    60% {
        transform: translateY(-5px);
    }
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────
#  Helper functions
# ───────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """Load BTS YouTube comments CSV and drop missing rows."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        return df.dropna(subset=["comment_text"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def polarity_subjectivity(text: str) -> tuple:
    """Calculate sentiment polarity and subjectivity using TextBlob."""
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except:
        return 0.0, 0.0

def label_from_polarity(p: float) -> str:
    """Convert polarity score to sentiment label."""
    if p > 0:
        return "Positive"
    elif p < 0:
        return "Negative"
    else:
        return "Neutral"

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """Vectorize, encode, split, and train Logistic Regression model."""
    if df.empty:
        return None, None, None, None
    
    # Compute polarity/subjectivity + rule-based label
    sentiment_data = df["comment_text"].apply(
        lambda x: pd.Series(polarity_subjectivity(x))
    )
    df[["polarity", "subjectivity"]] = sentiment_data
    df["sentiment"] = df["polarity"].apply(label_from_polarity)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])

    # Vectorize text
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["comment_text"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "conf_mat": confusion_matrix(y_test, y_pred),
        "classes": le.classes_
    }
    
    return model, vectorizer, le, metrics

def predict_comments(comments: list, model, vectorizer, le):
    """Predict sentiment for new comments."""
    if not comments or not model:
        return pd.DataFrame()
    
    try:
        X_new = vectorizer.transform(comments)
        preds = le.inverse_transform(model.predict(X_new))
        pol_sub = [polarity_subjectivity(c) for c in comments]

        rows = []
        for cmt, pred, (pol, sub) in zip(comments, preds, pol_sub):
            # Add emoji based on sentiment
            emoji = "😊" if pred == "Positive" else "😐" if pred == "Neutral" else "😞"
            rows.append({
                "Comment": cmt[:100] + "..." if len(cmt) > 100 else cmt,
                "Sentiment": f"{emoji} {pred}",
                "Polarity": f"{pol:.3f}",
                "Subjectivity": f"{sub:.3f}"
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return pd.DataFrame()

# ───────────────────────────────
#  Main App
# ───────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 BTS Comment Sentiment Analyzer</h1>
        <p>AI-powered sentiment analysis for YouTube comments</p>
        <div class="music-emoji">🎶</div>
    </div>
    """, unsafe_allow_html=True)

    # File check & data load
    DATA_PATHS = ["bts_2021_1.csv", "data/bts_2021_1.csv"]
    csv_file = next((p for p in DATA_PATHS if pathlib.Path(p).exists()), None)

    if not csv_file:
        st.error("""
        **Dataset not found!** 
        
        Please upload the **bts_2021_1.csv** file to your repository root or create a **data/** folder and place it there.
        
        You can download the dataset from: https://www.kaggle.com/datasets/seungguini/bts-youtube-comments
        """)
        st.stop()

    # Load data and train model
    with st.spinner("Loading dataset and training model..."):
        df = load_data(csv_file)
        if df.empty:
            st.error("Failed to load dataset or dataset is empty.")
            st.stop()
        
        model, vectorizer, le, metrics = train_model(df)
        if not model:
            st.error("Failed to train model.")
            st.stop()

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Analysis Tools")
        mode = st.radio(
            "Select analysis mode:",
            ["Single Comment", "Batch Analysis", "Model Performance"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Comments", f"{len(df):,}")
        st.metric("Model Accuracy", f"{metrics['accuracy']:.1%}")
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        st.markdown("### 📈 Sentiment Distribution")
        for sentiment, count in sentiment_counts.items():
            emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
            st.write(f"{emoji} {sentiment}: {count:,} ({count/len(df)*100:.1f}%)")

    # Mode 1: Single Comment Analysis
    if mode == "Single Comment":
        st.subheader("🔍 Analyze Individual Comment")
        
        # Sample comments for quick testing
        sample_comments = [
            "I love BTS so much! Their music changed my life! 💜",
            "This video was okay, nothing special.",
            "I hate this song, it's terrible and boring."
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area(
                "Enter a comment to analyze:",
                placeholder="Type or paste a YouTube comment here...",
                height=100
            )
        
        with col2:
            st.write("**Quick Test:**")
            for i, sample in enumerate(sample_comments):
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    user_input = sample
                    st.rerun()

        if st.button("🎯 Analyze Sentiment", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a comment to analyze.")
            else:
                with st.spinner("Analyzing sentiment..."):
                    result_df = predict_comments([user_input], model, vectorizer, le)
                    if not result_df.empty:
                        result = result_df.iloc[0]
                        
                        # Display results in cards
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Sentiment", result["Sentiment"])
                        with col2:
                            st.metric("Polarity", result["Polarity"])
                        with col3:
                            st.metric("Subjectivity", result["Subjectivity"])
                        
                        # Explanation
                        polarity_val = float(result["Polarity"])
                        subjectivity_val = float(result["Subjectivity"])
                        
                        st.markdown("### 📝 Analysis Explanation")
                        
                        # Polarity explanation
                        if polarity_val > 0.1:
                            pol_text = "**Positive** - The comment expresses favorable feelings"
                        elif polarity_val < -0.1:
                            pol_text = "**Negative** - The comment expresses unfavorable feelings"
                        else:
                            pol_text = "**Neutral** - The comment is neither clearly positive nor negative"
                        
                        # Subjectivity explanation
                        if subjectivity_val > 0.5:
                            sub_text = "**Subjective** - The comment expresses personal opinions/emotions"
                        else:
                            sub_text = "**Objective** - The comment states facts or neutral information"
                        
                        st.write(f"**Polarity ({polarity_val:.3f}):** {pol_text}")
                        st.write(f"**Subjectivity ({subjectivity_val:.3f}):** {sub_text}")

    # Mode 2: Batch Analysis
    elif mode == "Batch Analysis":
        st.subheader("📂 Batch Comment Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with comments",
            type="csv",
            help="CSV should contain a 'comment_text' column"
        )
        
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                if "comment_text" not in batch_df.columns:
                    st.error("❌ CSV must contain a 'comment_text' column")
                else:
                    st.success(f"✅ Loaded {len(batch_df)} comments")
                    
                    if st.button("🚀 Analyze All Comments", type="primary"):
                        with st.spinner("Processing batch analysis..."):
                            comments = batch_df["comment_text"].dropna().tolist()
                            results = predict_comments(comments, model, vectorizer, le)
                            
                            if not results.empty:
                                st.success("Analysis complete!")
                                
                                # Show summary
                                sentiment_summary = results["Sentiment"].str.split().str[1].value_counts()
                                col1, col2, col3 = st.columns(3)
                                
                                for i, (sentiment, count) in enumerate(sentiment_summary.items()):
                                    emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
                                    with [col1, col2, col3][i % 3]:
                                        st.metric(f"{emoji} {sentiment}", count)
                                
                                # Show results table
                                st.dataframe(results, use_container_width=True)
                                
                                # Download button
                                csv = results.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results as CSV",
                                    data=csv,
                                    file_name="sentiment_analysis_results.csv",
                                    mime="text/csv"
                                )
                                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Mode 3: Model Performance
    else:
        st.subheader("📈 Model Performance Dashboard")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.1%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.1%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.1%}")
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        conf_fig = px.imshow(
            metrics["conf_mat"],
            text_auto=True,
            x=metrics["classes"],
            y=metrics["classes"],
            color_continuous_scale="Purples",
            title="Model Prediction Accuracy by Class",
            labels={"x": "Predicted", "y": "Actual"}
        )
        conf_fig.update_layout(height=500)
        st.plotly_chart(conf_fig, use_container_width=True)
        
        # Polarity Distribution
        st.markdown("### 📊 Polarity Score Distribution")
        pol_fig = px.histogram(
            df, 
            x="polarity", 
            color="sentiment",
            nbins=50,
            title="Distribution of Sentiment Polarity Scores",
            color_discrete_map={
                "Positive": "#8E44AD",
                "Neutral": "#95A5A6", 
                "Negative": "#E74C3C"
            }
        )
        pol_fig.update_layout(height=400)
        st.plotly_chart(pol_fig, use_container_width=True)
        
        # Dataset insights
        st.markdown("### 💡 Dataset Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_polarity = df.groupby('sentiment')['polarity'].mean()
            st.write("**Average Polarity by Sentiment:**")
            for sentiment, avg_pol in avg_polarity.items():
                emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
                st.write(f"{emoji} {sentiment}: {avg_pol:.3f}")
        
        with col2:
            avg_subjectivity = df.groupby('sentiment')['subjectivity'].mean()
            st.write("**Average Subjectivity by Sentiment:**")
            for sentiment, avg_sub in avg_subjectivity.items():
                emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
                st.write(f"{emoji} {sentiment}: {avg_sub:.3f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7F8C8D;'>
        <p>🎵 <strong>BTS Comment Sentiment Analyzer</strong> | Built with ❤️ using Streamlit & TextBlob</p>
        <p>© 2025 Krishna Balaji | AI-powered sentiment analysis for K-pop fans</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
