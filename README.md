# Sentiment-Analysis-on-Twitter-Data
# Twitter Sentiment Analysis (NLP) The project follows an end-to-end ML workflow similar to a production-style pipeline.
## Overview
This project performs sentiment analysis on Twitter data using Natural Language Processing (NLP) and Machine Learning techniques.  
The goal is to classify tweets into **positive**, **negative**, or **neutral** categories.

## Dataset
**Source:** Kaggle – Twitter US Airline Sentiment  
**Size:** 14,640 tweets  

Each tweet includes:
- Text content
- Sentiment label (positive / negative / neutral)
- Airline information
- Confidence scores

Dataset is stored in:


---

## Project Structure
twitter-sentiment-analysis/
│
├── data/
│ └── raw/
│ └── Tweets.csv
│
├── notebooks/
│ └── 01_twitter_sentiment_eda_modeling.ipynb
│
├── model/
│ ├── sentiment_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── README.md
└── requirements.txt


---

## Methodology
### 1. Exploratory Data Analysis (EDA)
- Sentiment distribution analysis
- Tweet length analysis
- Class imbalance inspection

### 2. Text Preprocessing
- Lowercasing
- Removing URLs, mentions, punctuation
- Stopword removal
- Lemmatization

### 3. Feature Engineering
- TF-IDF Vectorization

### 4. Modeling
- Logistic Regression
- Linear Support Vector Classifier (LinearSVC)

### 5. Evaluation
- Precision, Recall, F1-score
- Comparison of multiple models

---

## Results
- Achieved ~77% accuracy
- LinearSVC showed strong performance on negative sentiment detection
- Clean and reproducible pipeline

---

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn

---

## How to Run
```bash
pip install -r requirements.txt
## Author
Ivan Lucas

