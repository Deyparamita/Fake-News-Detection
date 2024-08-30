# Fake News Detection Using Logistic Regression

## Overview

This project involves developing a Fake News Detection model using Logistic Regression. The model classifies news articles as either real or fake based on their textual content. The dataset used contains 20,800 news articles, and the model achieves high accuracy, demonstrating its effectiveness in distinguishing between genuine and misleading news.
- **0**: Real news
- **1**: Fake news

## Key Features

- **Model**: Logistic Regression
- **Text Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset**: 20,800 news articles
- **Accuracy**:
  - Training Accuracy: 98.66%
  - Test Accuracy: 97.91%

## Libraries and Modules Used

- **NumPy**: `import numpy as np` – For numerical operations.
- **Pandas**: `import pandas as pd` – For data manipulation and analysis.
- **Regular Expressions**: `import re` – For text preprocessing.
- **NLTK**: 
  - `from nltk.corpus import stopwords` – To handle common stopwords.
  - `from nltk.stem.porter import PorterStemmer` – For stemming words.
- **Scikit-Learn**: 
  - `from sklearn.feature_extraction.text import TfidfVectorizer` – For converting text into numerical feature vectors.
  - `from sklearn.model_selection import train_test_split` – To split the data into training and test sets.
  - `from sklearn.linear_model import LogisticRegression` – To create the logistic regression model.
  - `from sklearn.metrics import accuracy_score` – To evaluate the model's performance.

## How to Run

1. **Install Dependencies**: Make sure you have the required libraries installed. You can install them using pip:
   ```bash
   pip install numpy pandas scikit-learn nltk