# 📁 Deployment Instructions

## Quick Setup Guide

### Step 1: Download Files
You have been provided with 3 essential files:
1. **sentiment_app.py** - Main Streamlit application
2. **requirements.txt** - Python dependencies 
3. **README.md** - Project documentation

### Step 2: Upload to GitHub
1. Go to your GitHub repository: https://github.com/krishnabalajiwork/Sentiment-Analysis
2. Upload all 3 files to the repository root
3. Make sure to also upload your **bts_2021_1.csv** dataset file

### Step 3: Deploy on Streamlit Cloud
1. Visit: https://share.streamlit.io
2. Click "New App"
3. Select your repository: `krishnabalajiwork/Sentiment-Analysis`
4. Set main file path: `sentiment_app.py`
5. Click "Deploy"

### Step 4: Access Your Live App
After deployment (2-3 minutes), you'll get a URL like:
`https://sentiment-analysis-bts.streamlit.app/`

## File Structure
Your repository should look like this:
```
Sentiment-Analysis/
├── sentiment_app.py          # ← Main app file
├── requirements.txt          # ← Dependencies
├── README.md                # ← Documentation  
├── bts_2021_1.csv           # ← Your dataset
└── deployment-guide.md      # ← This file
```

## Important Notes
- Make sure the **bts_2021_1.csv** file is in the repository root
- All files must be committed to GitHub before deployment
- The app will automatically install dependencies from requirements.txt
- First load may take longer as the model trains on your dataset

## Features Your App Will Have
✅ Single comment sentiment analysis  
✅ Batch CSV upload and processing  
✅ Model performance dashboard  
✅ Interactive visualizations  
✅ BTS-inspired purple theme  
✅ Export results as CSV  

## Troubleshooting
- If deployment fails, check that all files are properly uploaded
- Ensure your dataset file is named exactly `bts_2021_1.csv`
- Check the Streamlit Cloud logs for any error messages

Your sentiment analysis app will be ready to share with the world! 🎉