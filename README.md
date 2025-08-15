💜 BTS Comment Sentiment Analyzer
![BTS Banner](https://img.shields.io/badge/BTS-Comment%20Sentiment%20Analyzer-9B59B6?style=for-the-badge&logo=github](https://img.shields.io/badge/Live%20Demo-Available-brightgreen?style=for-the-badgetest-sentiment-analysis-bts](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=pythonlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColorg.shields.io/badge/NLP-VADER%20Sentiment-6C3483?style=.com/cjhutto/vaderSentiment Overview

BTS Comment Sentiment Analyzer is a Streamlit-powered web application for analyzing the sentiment behind BTS-related comments. Utilizing VADER, it provides instant sentiment breakdowns that help ARMY fans understand the tone of conversations and feedback.

🎯 Problem Statement
BTS fans want to understand the overall positivity, negativity, and neutrality in online discussions. However, social media language is challenging for standard sentiment tools.

Varying emotional tone in fandom

Fast-changing online slang

Need for easy-to-understand sentiment results

💡 Solution
An ARMY-themed interface where you can:

Instantly analyze BTS comments for sentiment (positive, neutral, negative)

See overall and detailed breakdowns

Get easy-to-read, visually appealing results

🚀 Live Demo
🌐 Try it now: https://test-sentiment-analysis-bts.streamlit.app/

Key Features
💜 VADER Sentiment Analysis – Social-media optimized NLP

📊 Detailed Sentiment Breakdown – See positivity, neutrality, negativity scores

🎨 Purple BTS Theme – Fandom-inspired design

✨ Mobile Responsive – Use it anywhere!

🏗️ Technical Architecture
Sentiment Analysis Flow
text
graph TD;
    Input[Comment Input] --> VADER[Run VADER Sentiment]
    VADER --> Output[Display Sentiment Scores]
Technology Stack
Frontend: Streamlit

Backend: Python 3.8+

NLP: VADER Sentiment Analysis

Data: BTS comments dataset (bts_2021_1.csv by seungguini)

Version Control: Git, GitHub

Deployment: Streamlit Cloud

🗃️ Dataset
bts_2021_1.csv: Real-world BTS comments sourced/curated by seungguini
(Credit for data collection and project inspiration!)

🛠️ Installation & Setup
Prerequisites
Python 3.8+

pip

Local Development
bash
git clone https://github.com/yourusername/bts-sentiment-analyzer.git
cd bts-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
Open your browser: http://localhost:8501

Dependencies
text
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
vaderSentiment>=3.3.2
📈 Usage
Input BTS comments (or use the included dataset)

Analyze sentiment – see instant breakdowns

Interpret results

Positive: Encouraging, happy tone

Neutral: Balanced, informational

Negative: Critical or sad tone

🙏 Credits
seungguini – For providing the BTS comments dataset and inspiring this sentiment analysis project!

ARMY Community – For unending support, creativity, and feedback.

VADER & Streamlit Teams – For building the scientific and developer tools behind this project.

🤝 Contributing
Contributions welcome! Please submit issues or Pull Requests.

📄 License
MIT License. See LICENSE for details.

📞 Contact
GitHub: yourusername

Live Demo: test-sentiment-analysis-bts.streamlit.app

Portfolio: View All Projects

⭐ If you love this project, give it a star!

Built with 💜 for the BTS ARMY and powered by VADER NLP.
