# 🎤 BTS Comment Sentiment Analyzer

An AI-powered Streamlit web application that performs sentiment analysis on YouTube comments from BTS videos. This project uses machine learning to classify comments as Positive, Neutral, or Negative, providing insights into fan reactions and engagement.

## 🌟 Features

- **Single Comment Analysis**: Analyze individual comments with detailed sentiment breakdown
- **Batch Processing**: Upload CSV files to analyze multiple comments at once
- **Model Performance Dashboard**: Visualize accuracy metrics and confusion matrices
- **Interactive Visualizations**: Beautiful charts powered by Plotly
- **BTS-Inspired Design**: Purple-themed UI matching K-pop aesthetics
- **Export Results**: Download analysis results as CSV files

## 🚀 Live Demo

Visit the live application: [BTS Sentiment Analyzer](your-streamlit-url-here)

## 📊 Model Performance

- **Overall Accuracy**: ~88%
- **Positive Sentiment**: 90% precision, 88% recall
- **Neutral Sentiment**: 88% precision, 94% recall  
- **Negative Sentiment**: 81% precision, 56% recall

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **TextBlob**: Natural language processing and sentiment analysis
- **scikit-learn**: Machine learning algorithms (Logistic Regression)
- **Plotly**: Interactive data visualizations
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Access to the BTS YouTube comments dataset

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/krishnabalajiwork/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the BTS YouTube comments dataset from [Kaggle](https://www.kaggle.com/datasets/seungguini/bts-youtube-comments) and place the `bts_2021_1.csv` file in your project root directory.

### 4. Run the Application

```bash
streamlit run sentiment_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Sentiment-Analysis/
├── sentiment_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── bts_2021_1.csv       # Dataset (download separately)
└── data/                # Optional: place dataset here
    └── bts_2021_1.csv
```

## 🎯 Usage

### Single Comment Analysis
1. Navigate to the "Single Comment" mode in the sidebar
2. Enter a YouTube comment in the text area
3. Click "Analyze Sentiment" to get results
4. View sentiment classification, polarity, and subjectivity scores

### Batch Analysis
1. Switch to "Batch Analysis" mode
2. Upload a CSV file with a `comment_text` column
3. Click "Analyze All Comments" to process the batch
4. Download results as CSV for further analysis

### Model Performance
1. Select "Model Performance" mode to view:
   - Accuracy, precision, recall, and F1-score metrics
   - Confusion matrix heatmap
   - Polarity distribution charts
   - Dataset insights and statistics

## 📊 Understanding the Metrics

- **Polarity**: Ranges from -1 (negative) to +1 (positive)
- **Subjectivity**: Ranges from 0 (objective) to 1 (subjective)
- **Sentiment Categories**:
  - **Positive**: Polarity > 0
  - **Neutral**: Polarity = 0
  - **Negative**: Polarity < 0

## 🎨 Customization

### Changing the Theme
Modify the CSS in `sentiment_app.py` to customize colors and styling:

```python
# Update the color scheme in the CSS section
.main-header {
    background: linear-gradient(90deg, #your-color1, #your-color2);
}
```

### Adding New Features
The modular structure makes it easy to add new functionality:
- Additional sentiment analysis models
- Different visualization types
- Export formats (JSON, Excel)
- Real-time comment scraping

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `sentiment_app.py` as the main file
5. Deploy and share your live URL

### Local Network Deployment

```bash
streamlit run sentiment_app.py --server.address 0.0.0.0 --server.port 8501
```

## 📈 Dataset Information

The application uses the BTS YouTube Comments dataset which contains:
- **Source**: YouTube comments from BTS videos
- **Size**: ~28,000 comments
- **Columns**: comment_text, comment_author, comment_date, likes, etc.
- **Language**: Primarily English
- **Time Period**: 2021 data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Seungguini](https://www.kaggle.com/seungguini) for the BTS YouTube comments dataset
- BTS and ARMY community for inspiring this project
- Streamlit team for the amazing framework
- TextBlob developers for sentiment analysis capabilities

## 📧 Contact

**Krishna Balaji**
- GitHub: [@krishnabalajiwork](https://github.com/krishnabalajiwork)
- Email: your-email@example.com

## 🔮 Future Enhancements

- [ ] Add emotion detection beyond sentiment
- [ ] Implement real-time YouTube comment scraping
- [ ] Support for multiple languages
- [ ] Advanced NLP models (BERT, RoBERTa)
- [ ] Comment trend analysis over time
- [ ] Integration with other social media platforms

---

⭐ **Star this repository if you found it helpful!** ⭐