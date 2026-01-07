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

Model Evaluation

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

