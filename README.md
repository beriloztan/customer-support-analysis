# Customer Support Analysis

NLP-based analysis of 2.8M+ Twitter customer support tweets. Classifies complaints into 7 categories and predicts sentiment using machine learning.

![Dashboard](assets/dashboard.png)

## Overview

Companies receive thousands of support requests daily. This project automates complaint categorization and sentiment detection to help support teams prioritize and route tickets efficiently.

**Dataset:** [Twitter Customer Support](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 2,811,774 tweets from 108 companies

## Features

- **7-Category Classifier** — Shipping, Billing, Technical, Account, Product, General, Refund
- **Sentiment Analysis** — Positive / Negative prediction per tweet
- **Live Tweet Tool** — Enter any tweet and get instant predictions
- **Interactive Dashboard** — Built with Streamlit + Plotly

## Results

| Model | Task | F1 Score |
|-------|------|----------|
| TF-IDF + Logistic Regression | Category Classification | ~0.98 |
| TF-IDF + Logistic Regression | Sentiment Analysis | ~0.99 |

## Project Structure

```
customer-support-analysis/
├── notebooks/
│   ├── 1_EDA.ipynb              # Text cleaning & keyword-based classification
│   ├── 2_Sentiment_Model.ipynb  # Category classification model
│   └── 3_Sentiment.ipynb        # Sentiment model
├── dashboard.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

> **Note:** Raw data files are excluded from this repo due to size. Download `twcs.csv` from Kaggle and place it in the `data/` folder.

## How to Run

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Tech Stack

Python · Scikit-learn · NLTK · TF-IDF · Streamlit · Plotly · Pandas
