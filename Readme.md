Financial News Sentiment Analysis

Project Overview:

This project builds a financial news sentiment analysis system step by step.

The goal is to classify financial news text into Positive, Negative, or Neutral sentiment.


Week 1: NLP Fundamentals & Lexicon-Based Sentiment Analysis

In Week 1, a rule-based sentiment analysis system was built.

Text preprocessing:
Tokenization,
Stopword removal,
Lemmatization

Sentiment analysis using:
VADER,
TextBlob,
Custom financial lexicon,
Ensemble scoring of sentiment

Model evaluation using:
Accuracy,
Precision, Recall, F1-score
Confusion Matrix


Week 2: Supervised Machine Learning

Improve sentiment classification performance using machine learning models.

Feature Engineering:
Bag-of-Words (BoW),
TF-IDF ,
Vocabulary size limited to 5,000 features

Models Trained:
Logistic Regression ,
Naive Bayes ,
K-Nearest Neighbors (KNN)

Model Evaluation: Performance comparison using standard classification metrics

Machine learning models outperformed lexicon-based approaches by learning sentiment directly from data.

TF-IDF features performed better than Bag-of-Words due to better handling of important financial terms.

Logistic Regression achieved the best overall balance between accuracy and F1-score among ML models.


Week 3: Transformers & FinBERT 

Use transformer-based deep learning to further improve sentiment analysis.

Transformer Basics & FinBERT:

Loaded a pre-trained FinBERT model from Hugging Face,
Performed zero-shot sentiment analysis (no training on our dataset).

Fine-tuning FinBERT on Financial PhraseBank:

Fine-tuned FinBERT on the processed Financial PhraseBank dataset ,
Used transfer learning to adapt the model to our specific data ,
Implemented training using Hugging Face’s Trainer API ,
Evaluated performance on a validation set ,

FinBERT handled financial context, negation, and subtle sentiment cues better than lexicon and ML models.


Week 4: Sentiment vs Stock Prices

Validate sentiment models on real financial news and market data

Real-time stock price pipeline using yfinance

Financial news dataset ingestion and preprocessing (Kaggle)

Sentiment scoring of real news using:

VADER,
Logistic Regression,
FinBERT


Week 5: Sentiment-Driven Trading & Backtesting

Custom backtesting engine for historical evaluation

Comparative analysis of:

VADER-based strategy

Logistic Regression-based strategy

----------------------------------------------------------------
Overall :

Model performance improved progressively from lexicon-based → ML → transformer-based approaches
